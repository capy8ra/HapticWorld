# world_ft — box-flip ACT pipeline (data → training → real-world rollout)

Self-contained reference for the `world_ft` project. `ft` = force/torque. Goal: train an ACT
policy to flip a box with the **right OpenArm**, and study whether **force/torque** and **world-model
simulation data** help. This doc covers data collection (real + sim), every training script, and how
the real-world eval rollout works.

Everything lives under `world_ft/` (top-level teleop/collection scripts) and
`world_ft/learning/` (dataset, models, training, eval). The world model itself lives in a
**separate repo**: `HapticWorld/worldsim`.

---

## 0. Environment

One conda env for everything: **`hapticworld`** (`envs/environment.yaml`) — torch + cv2 + av +
wandb + pinocchio (with the MJCF loader) + python-`can` + redis. Training, the world-model
worker, the inference worker, robot workers, and the eval controller all run in it.

Gotchas:
- `nova` is imported via `sys.path` (scripts insert `robot/` themselves); no pip install needed.
- Run robot-side commands from **`robot/`** and worldsim-side commands from **`worldsim/`**
  (relative paths in configs assume those working directories).

Hardware: bimanual OpenArm (leader + follower), but the box-flip task uses the **right arm only**.
Head camera = Intel RealSense (BGR, 640×480). CAN channels (both dongles connected, per
`collect_data.yaml`): leader_left=can0, leader_right=can1, follower_left=can2, follower_right=can3.
Unplugging the follower dongle renumbers the leader — **always verify CAN channels**.

---

## 1. Data format (real and sim are the SAME layout)

One folder per episode/trajectory. A dataset directory (e.g. `data/real_training/`) holds many:

```
<data_dir>/<timestamp>/
    camera_head.mp4                         # RGB video (real: RealSense 640x480 ~30fps; sim: WM render 128x128 @10fps)
    low_dim_npys/
        follower_right_mit_command:q.npy          (N, 8)  ACTION target (commanded joint pos, 7 arm + gripper)
        follower_right_mit_command:timestamp.npy  (N,)
        ...                                       (see per-source streams below)
```

Alignment is by **timestamp**: `VideoMp4Dataset` reads mp4 frame timestamps; `LowDimDataset`
interpolates each `.npy` stream to requested times. (Colons in filenames are intentional.)

**Real** episodes additionally have measured state + interaction torque:
`follower_{l,r}_state:{actual_angle_rad, actual_torque_nm, actual_velocity_radps, timestamp}`,
`follower_controller_state:{tau_interaction_L, tau_interaction_R, timestamp}`, plus leader states.

**Sim** episodes have **no measured follower state** (the follower is virtual) — only:
`follower_right_mit_command:{q,timestamp}` and
`follower_controller_state:{torque_pred_nm, timestamp}` (the world model's predicted torque),
plus `follower_controller_state:tau_interaction_R` written by the §2b post-process.

> Key consequence: sim data has **no joint angle**, and its raw torque `torque_pred_nm` is the WM's
> *total* predicted torque (the stage-2 head was trained on measured `actual_torque_nm`), whereas
> real torque is `tau_interaction_R` (interaction = measured − gravity comp). The post-process
> (`postprocess_sim_tau_interaction.py`) converts sim to the real definition; the dataset
> auto-selects the torque stream, preferring `tau_interaction` (see §3).

Data dirs: `data/real_training/` (~50 real episodes), `data/sim_training/` (75 trunk-WM episodes,
collected WITH force feedback), `data/sim_training_wo_ft/` (56 trunk-WM episodes collected WITHOUT
feedback, 2026-07-24 — the claim-2b comparison set), `data/sim_state_training/` (state-WM pipeline),
`data/collecting/` (raw real dump, ~630), `data/eval_rollouts/<spec>/` (recorded real eval episodes,
`--save`), `data/sim_training_quarantine/` (corrupted episodes pulled from training). Checkpoints:
`data/model_ckpts/<exp>_<ts>/`. Seeds for in-sim eval: `worldsim/data/wm_seed_live/`.

---

## 2. Data collection

### 2a. Real teleop collection — `collect_data.yaml`
Human moves the **leader**; the **follower mirrors it** (teleop), while real camera + follower state
are recorded. Run:

```bash
process-compose -f world_ft/collect_data.yaml
```
Key processes: `perception_redis`(6380) + `camera_head`(RealSense); `control_redis`(6379);
`follower_mit_controller_{left,right}` (can2/can3) + `leader_mit_controller_{left,right}` (can0/can1);
`follower_solver` (`nova/robot/openarm1/teleop_follower_solver.py`, follower q ← leader q + gravity comp);
`leader_solver`; `tau_interaction_publisher.py` (publishes `tau_interaction = actual_torque − mit_command:tau`);
`video_recorder.py` + `low_dim_npy_recorder.py` + `collect_data.py`. → writes episodes to
`data/collecting/`. `data/real_training/` is a curated subset used for training.

### 2b. World-model "sim" collection — 3 terminals
Here there is **no real follower and no real box**: a trained **world model** plays the follower +
scene. The human moves the real **right leader**, the WM renders the predicted follower camera view
and predicts its torque, and that (image + leader action + predicted torque) is recorded as ACT data.

- **Term 1** — control bus + right-leader controller (can1):
  ```bash
  process-compose -f world_ft/teleop_wm_leader_only.yaml
  ```
  Variant `teleop_wm_leader_follower.yaml` *also* runs the real right follower + `teleop_follower_solver_right.py`
  so the physical follower mirrors the leader while you drive the sim (does not change what's recorded).

- **Term 2** — homes the leader to the seed pose, floats it, and reflects the WM's
  predicted interaction torque back to the leader as **force feedback**:
  ```bash
  python world_ft/leader_wm_force_feedback.py --seed-episode <SAME hdf5>
  ```
  Torque math: `tau_interaction = torque_pred[:7] − gravity_comp(leader_q)`;
  `reflected = ramp·k·LPF(tau_interaction)` (clamped ±8 N·m); the leader feels `reflected`.
  (`k`/LPF/clamp are for the felt force only — never a model input.)

- **Term 3** (run from `worldsim/`) — the world model:
  ```bash
  PYTHONPATH=HapticWorld/robot python scripts/inference/wm_follower_worker.py \
      --ckpt world_ft_ckpts/stage2_trunk/checkpoints/last.ckpt \
      --stage3-ckpt world_ft_ckpts/stage3/checkpoints/last.ckpt \
      --seed-episode data/world_ft_seed_0713/val/episode_0.hdf5 --publish-torque --dtype bfloat16
  ```
  It reads the leader joint angles as the "action", rolls a latent dynamics model, decodes the
  predicted camera view, and predicts the follower torque (`virtual_follower_right_state:torque_pred_nm`).

**Recording controls** (with the WM prediction window focused; multi-episode since 2026-07-24):
`ENTER` = start recording · `s` = queue the episode for **background save**, reset the sim to the
seed, hold `--resume-delay-s` (3 s, time to move the leader back; ENTER skips the hold), then the
sim goes **LIVE — recording starts on the next ENTER**. Nothing blocks on the arm pose: if the
leader is > `--rehome-eps-deg` (5°) from the seed pose when the sim resumes it only prints a
WARNING. `q` = quit (discards an in-progress recording; queued saves are drained before exit).
The window shows a red **REC** / green **IDLE** indicator plus the pending-save count; torque
publishing pauses during the hold, so the force-feedback watchdog zeroes the reflected force and
the leader moves freely. All decodes (display / saver / seed) share `_DECODE_LOCK` — the diffusion
decoder is not thread-safe (concurrent decodes corrupt frames).
Latents are buffered during the run and decoded to **clean, exactly-paired** frames at save time
(on the saver thread, own CUDA stream — avoids labeled/async-lagged frames);
low-dim `.npy`s are saved truncated to the frames actually written so counts always match. Output →
`data/sim_training/<ts>/` with `camera_head.mp4` + `follower_right_mit_command:q` + `torque_pred_nm`.
Image quality knob: `--dec-infer-steps` (default 3; higher = sharper decoded frames).

**STATE-pipeline collection** (2026-07-27): `wm_follower_worker_state.py` now has the same
multi-episode ACT recording (identical keys/flow, background saves, REC HUD). It uses
`stage2_state` + `stage3_state` (stage-1 `2026-07-16_14-12-06`) and writes to
**`data/sim_state_training/`** (kept separate from the trunk-collected `sim_training/`):
```bash
PYTHONPATH=HapticWorld/robot python scripts/inference/wm_follower_worker_state.py \
    --ckpt world_ft_ckpts/stage2_state/checkpoints/last.ckpt \
    --stage3-ckpt world_ft_ckpts/stage3_state/checkpoints/last.ckpt \
    --seed-episode data/world_ft_seed_0713/val/episode_0.hdf5 --publish-torque
```
(fp32 default = the validated coupled rollout; if the live loop can't hold 10 Hz, try
`--dtype bfloat16` and eyeball the output. Session recordings -> `wm_sessions_state/`.)

**After each sim-collection batch** (before training), run:
```bash
python world_ft/postprocess_sim_tau_interaction.py
```
It writes `follower_controller_state:tau_interaction_R = torque_pred_nm − gravity(commanded q)` into
each new episode (idempotent; `--force` to redo), converting the WM's total torque to the real-data
interaction definition. Training auto-selects it; a half-processed folder fails at dataset init
(LowDimDataset requires identical stream sets), so a forgotten run is caught loudly.

**World-model checkpoints (in `worldsim/`):** 3 training stages — stage1 (autoencoder → latent space +
frozen encoder), stage2 (`--ckpt`, dynamics + torque head), stage3 (`--stage3-ckpt`, decoder-only
fine-tune for sharper decode of drifted latents). **stage2 and stage3 must share the same stage1**
(check `.hydra/config.yaml` `load_ae`). Two families exist (all on ModelScope `ShaotingPeng/world_ft_ckpts`):
- **trunk**: `stage2_trunk` + `stage3` share stage1 `2026-07-15_02-35-54`.
- **state** (dynamics conditioned on `[q_des(8), q_measured(8)]`, `LatentWorldModelState`,
  autoregressive GRU state branch): `stage2_state` (run `2026-07-17_15-47-53`) + `stage3_state`
  (run `2026-07-21_22-54-36`, downloaded 2026-07-22) share stage1 `2026-07-16_14-12-06`.
  Never mix decoders across families — the latent spaces differ.

**State-model teleop variant:** same 3 terminals, but Term 3 runs
`scripts/inference/wm_follower_worker_state.py` instead:
```bash
PYTHONPATH=HapticWorld/robot python scripts/inference/wm_follower_worker_state.py \
    --ckpt world_ft_ckpts/stage2_state/checkpoints/last.ckpt \
    --stage3-ckpt world_ft_ckpts/stage3_state/checkpoints/last.ckpt \
    --seed-episode data/world_ft_seed_0713/val/episode_0.hdf5 --publish-torque
```
`q_measured` is not read from any robot — the model's GRU state branch predicts it autoregressively
(the `coupled_rollout` deployment path, stepped live). Same redis keys as the trunk worker, so
Terms 1–2 are unchanged. Defaults to float32 (the validated precision; ~0.04 s/step, fine for 10 Hz).
Seed episodes without `obs/qpos_right` (e.g. `world_ft_seed_0713`, pre-state-converter) fall back to
`action[0]` for the initial measured state with a warning — fine at the homed pose. NOTE: this worker
records sessions (`wm_sessions/`) but does **not** yet have the `_ACTRecorder` ENTER/`s`/`q` sim-episode
recording — teleop/force-feedback only until that's ported.

---

## 3. Models & dataset (`learning/`)

**ACT** = DETR-VAE (ResNet-18 backbone + transformer, ~31M params). Predicts a **chunk** of future
commanded joint positions (`action_horizon=20` over `action_duration=2.0s` → 10 Hz), L1 loss on
normalized actions. Vendored under `nova/learning/model/act/`.

- **`model.py` — `ACT_Abs`**: image + `joint_state`. `build_model_and_optimizer(..., use_angle, use_torque)`
  → obs_dim = per-arm (8·use_angle + 8·use_torque). Feeds `joint_state` as the ACT **proprio token**.
- **`model_image_only.py` — `ACT_ImageOnly`**: image only. Built with `use_proprio=False`, which
  **drops the proprio token** from the DETR decoder (added a `use_proprio` flag to the shared
  `detr_vae.py`/`transformer.py`, default `True` so existing models are unchanged).
- **`dataset.py` — `OpenArmDataset`**: loads `camera_head.mp4` (`center_crop_resize` → 224×224, RGB
  [0,1]) + `joint_state` + the action chunk (`follower_right_mit_command:q`). Flags:
  - `image_only=True` → no `joint_state` at all.
  - `use_angle` → include `follower_{arm}_state:actual_angle_rad` (8-d).
  - `use_torque` → include torque (8-d), **auto-selecting** the stream: `tau_interaction_{R}` if
    present (real, and sim after the §2b post-process) else `torque_pred_nm` (raw sim).
  - Guards: rejects stray non-dir entries and incomplete/aborted episodes with a clear error.

Checkpoints save `{model.pth, optimizer.pth, info.yaml}` every 5000 steps + `ckpt_latest`. `info.yaml`
records the full CONFIG — the eval worker reads it to auto-configure (image_only / use_torque / etc.).

---

## 4. Training scripts

Shared loop lives in **`train_core.py`** (`run_training(config)` + `BASE_CONFIG`); each entry file is
~10 lines that set the variant + dataset and call it. Run any with:
```bash
python world_ft/learning/<script>.py
# WANDB_DISABLED=1 to skip logging. Checkpoints -> data/model_ckpts/<exp_prefix>_<ts>/
```

**Input-ablation matrix (current, canonical set):**

| script | input | data | obs `joint_state` | torque source |
|---|---|---|---|---|
| `train_image_only_real.py` | image only | real | none | — |
| `train_image_only_sim.py` | image only | sim | none | — |
| `train_image_torque_real.py` | image + torque (no angle) | real | (8) | `tau_interaction_R` |
| `train_image_torque_sim.py` | image + torque (no angle) | sim | (8) | `tau_interaction_R` (post-processed) |
| `train_image_state_torque_real.py` | image + angle + torque | real | (16) | `tau_interaction_R` |

(sim × image+state+torque is intentionally omitted — sim has no measured joint angle.)

**Feedback-condition comparison (claim 2b)** — `train_image_torque_sim_wo_t.py` (exp_prefix
`it_sim_wo_t`): same recipe as `train_image_torque_sim.py` but on `data/sim_training_wo_ft`
(collected WITHOUT force feedback). Trained 2026-07-24 to **budget parity** with `it_sim`
(auto-stopped at ckpt_step_5000 vs it_sim's 5106 steps; final loss 0.056 vs 0.060). Collection-time
behavioral difference already measured: mean |torque| 1.35 N·m (no feedback) vs ~2.5 (with).
Caveat for the comparison: 56 vs 75 episodes — consider a 56-episode with-feedback retrain to
match count.

**Sharpness-augmented real variants** — `train_image_only_real_blur.py` / `train_image_torque_real_blur.py`
(exp_prefix `*_real_blur`): identical to the real entries plus `blur_aug_res=128, blur_aug_p=0.5`
(50% of samples get the image resampled 224→128→224, the exact deploy `--match-blur-res` op).
Makes the policy sharpness-invariant so real-trained policies can ALSO be evaluated inside the
world model (soft 128-px renders — plain real-trained policies oscillate there; see
[WM_EVAL.md](WM_EVAL.md)). Measured: aug drops Laplacian variance ~8× (690→82), matching the
real-vs-render gap. NOTE: `ckpt_select` matches by input flags + data domain, so once a `_blur`
run finishes it becomes the `it_real`/`i_real` "latest" — use explicit paths to pin the old ones.

All default to: right arm, `obs_horizon=1`, `action_horizon=20`, `image_size=224`, `batch_size=128`,
`num_workers=20`. wandb project `world_ft-act-ablation`.

**Older ad-hoc training files (removed 2026-07-22; superseded by the matrix above):**
`train_box_flip.py` (= image_state_torque_real), `train_box_flip_no_torque.py` (image+angle, no torque),
`train_box_flip_image_only.py` (= image_only_real), `train_image_only.py` (= image_only_sim). Their
checkpoints under `model_ckpts/` remain valid; each `info.yaml` records the exact config used.

---

## 5. Real-world eval rollout

Two terminals. The stack is **right-arm only, no leader**; the policy replaces the human.

**Term 1** — background workers; pick the checkpoint via **`WORLD_FT_MODEL`**
(process-compose takes no extra flags — the env var is substituted into the worker command):
```bash
WORLD_FT_MODEL=it_sim process-compose -f world_ft/learning/eval.yaml
```
Processes: `perception_redis`(6380) + `camera_head`(RealSense); `control_redis`(6379);
`follower_mit_controller_right`(can3); `model_inference_worker` (launched
runs `eval_inference_worker.py`). `WORLD_FT_MODEL` (default `it_sim`) takes an explicit ckpt path or
a selector for the **latest** matching checkpoint (matched by `info.yaml` config, not folder names):
`i` = image-only, `it` = image+torque, `its` = image+state+torque, with an optional
training-domain suffix `_real` / `_sim` — e.g. `it_sim`, `i_real`. Bare `i|it|its` = latest from
either domain, so **use the suffixed form once both domains exist** (a newly trained ckpt silently
becomes "latest" otherwise). Non-variant runs are selected by **exp_prefix** (e.g. `it_sim_wo_t` →
newest `model_ckpts/it_sim_wo_t_*/ckpt_latest`); domain suffixes match the dataset dir basename
EXACTLY, so e.g. `sim_training_wo_ft` runs never hijack the `_sim` specs.

**Term 2** — the interactive controller, with the **same selector**:
```bash
python world_ft/learning/eval_controller.py --redis-port 6379 --model it_sim
```
`--model` resolves the same checkpoint and auto-derives the home reference episode from its
training dataset (closest-to-mean start pose; shared logic in `learning/ckpt_select.py`).

**Operate** (with the `eval_controller` terminal focused; hand on the e-stop):
1. On launch the arm is **limp** (kp=kd=0 + gravity comp). **Place it on the table by hand** near the
   box-flip start pose.
2. **`h`** → home: speed-limited ramp to the reference start pose, then hold.
3. **`ENTER`** (or **`g`**) → rollout: right arm tracks the policy (ramps into the trajectory).
4. **`s`** → stop / limp.

### `eval_inference_worker.py` (~10 Hz)
Loads the ckpt and **auto-detects `image_only` / `use_torque` from `info.yaml`**. Each loop: read
`camera_head:image` (BGR→RGB, `center_crop_resize`→224) [+ `follower_right_state:actual_angle_rad`
and `tau_interaction_R` for state models], run the policy, publish `model_inference:action_desired_qpos_right`
(a 20-step q chunk) + `action_timestamp`. Flags:
- `--match-blur-res`: downsample the real 224 image to this res then back to 224 so its **sharpness
  matches the soft sim renders** (real camera is ~12× sharper than sim; this closes the main sim→real
  image gap). Default **auto**: sim-trained ckpt → 128, real-trained → 0 (from the ckpt's
  `dataset_path`); pass an explicit value to override.

### `eval_controller.py` (~2 kHz)
Reads `follower_right_state` + the model's action chunk; interpolates it and commands
`follower_right_mit_command` at high rate with `FOLLOWER_KP/KD` + gravity-comp feedforward. Also
publishes `follower_controller_state:tau_interaction_R` itself (= measured torque − gravity comp), so
state+torque models get their torque obs without a separate publisher. Notable behavior:
- **Home reference** = first `follower_right_mit_command:q` of the reference episode. **This MUST
  match the policy's training data** (sim-trained → a `sim_training` episode; real-trained → a
  `real_training` episode) or the arm jumps when the policy engages. With `--model i|it|its` this is
  automatic (episode closest to the dataset-mean start pose, re-derived every launch, so it stays
  correct as data changes); the manual `reference_episode_lowdim_path` is only a fallback.
- **Ramp-in on ENTER**: ramps from the current pose into the first predicted chunk over
  `stitch_duration` (0.45 s) so engaging doesn't snap the arm.
- **`action_z_offset_m`** (deploy-time IK correction, default 0; was −1 cm for the image-only
  policy's height bias): lowers/raises **every
  predicted reach pose** by N meters in world Z. Used to fix a systematic vertical bias (the
  image-only policy reaching above a short box). Applied via Jacobian IK only when (re)building the
  trajectory (Enter + ~every 2.45 s renewal), not per control tick.
- **`h` homing** is a straight-line joint ramp (not collision-aware) — hand-place near start first.
- **`--random-home`** (robustness probe): each `h` homes to the reference pose shifted by a fresh
  uniform EE offset (x,y ±`home_rand_xy_m`=0.1 m, z [`home_rand_z_min_m`,`home_rand_z_max_m`] =
  [−0.02, 0.03] m; position-only IK). Sampled + achieved offsets are printed per trial for logging.
- **`--save`** (rollout recording): each ENTER→`s` (or `h`/exit) rollout is saved as one episode to
  `data/eval_rollouts/<--model spec>/<timestamp>/` — `camera_head.mp4` + `low_dim_npys/` (right-arm
  state/commands, `tau_interaction_R`, the policy's raw 20-step chunks; stream list =
  `eval_recording_config.yaml`) + `meta.yaml` (spec, resolved ckpt, reference episode, home offset,
  controller params). Uses the collect_data recorder workers, which `eval.yaml` now runs idle on
  both buses — their output dir comes from `$WORLD_FT_MODEL`, so **launch the stack with the same
  spec as `--model`**. The controller warns if no recorder is detected.

---

## 5b. Policy rollout INSIDE the world model (offline, no robot)

> Rollout **realism** (does the policy push the WM off-manifold?) is measured with the
> box-sharpness metric — definition, pass gate, and the 2026-07-23 i_sim vs it_sim results live in
> **[WM_EVAL.md](WM_EVAL.md)**. Headline: on the 11–17 s flip window, 16/20 `i_sim` rollouts fail
> the gate vs 4/20 for `it_sim`.

`worldsim/scripts/inference/rollout_policy_in_wm.py` (from `worldsim/`) closes the loop
policy↔WM. `--wm {trunk,state}` selects the world model (state = `stage2_state`+`stage3_state`
coupled rollout, ckpt defaults switch automatically; use `--dtype float32`, its validated
precision, and a separate `--out` e.g. `wm_policy_rollouts_state`). Loop: decode latent → policy obs (torque policies get `torque_pred − gravity(q_cmd)` via a
gravity/IK server auto-spawned as a subprocess — `learning/gravity_ik_server.py`; its
"pinocchio" is a stub) → feed the chunk's first `--exec-steps` (5 = 0.5 s receding horizon) back
into the dynamics. Per rollout a **seed episode is sampled** (seeded RNG → the (seed, offset)
sequence is identical across policies: rollout #k is a paired comparison). Outputs 20 s videos +
`.npz` logs under `wm_policy_rollouts/<policy>/`; success is scored by eye.

**Seeds** (initial scene = frame 0 + arm pose `action[0]`, BOTH required) are captured live with
varied box + arm poses — 2 terminals. Like `collect_data.yaml`, the stack floats the
leader (gravity comp, right-only `teleop_leader_float_right.py`) and the follower syncs to it on
startup (it JUMPS to the leader pose — put the leader near the follower before launching):
```bash
process-compose -f world_ft/capture_wm_seeds.yaml   # camera + right-arm teleop
python world_ft/capture_wm_seeds.py   # pose arm (via leader) + box, ENTER, x20
```
→ `worldsim/data/wm_seed_live/episode_*.hdf5` (the rollout's default `--seed-episodes`). The tool
prefers `follower_right_mit_command:q` (commanded = WM action semantics; measured fallback for a
limp arm) and only captures when the camera + pose streams are live. Because arm poses vary in the
seeds, `--random-home` is OFF by default in the rollout (flag still exists for pinned-seed runs).

## 6. Key lessons / gotchas (why things are the way they are)

- **State vs image-only**: state-conditioned ACT works well because `action ≈ current joint state +
  small delta` (robust). Image-only must infer everything (esp. **height/depth**) from one head-cam
  frame → brittle (covariate shift), and systematically imprecise on the reach height for a short box.
- **sim→real image gap is sharpness**, not color/framing: measured real ≈ 12× sharper than sim renders
  (color/brightness/framing match). Fix = `--match-blur-res 128` at deploy (or raise `--dec-infer-steps`
  when collecting). This was the main cause of "loss drops but rollout is random".
- **Home pose must match the policy's data** — a ~20° mismatch caused a "bump" when the policy engaged.
- **DETR builder calls `argparse.parse_args()` on `sys.argv`** → any script that builds the ACT model
  AND has its own CLI flags must blank `sys.argv` during the build (done in `eval_inference_worker`).
- Everything (robot control, IK, torch) lives in the single `hapticworld` env.
- **Sim torque ≠ real torque definition** (resolved 2026-07-22): the WM predicts *total* torque
  (`torque_pred_nm`, label = measured `actual_torque_nm`), real deploys with `tau_interaction`
  (interaction). `postprocess_sim_tau_interaction.py` writes `tau_interaction_R = torque_pred −
  gravity(commanded q)` into sim episodes so both domains train/deploy on the same quantity —
  re-run it after every sim-collection batch.

---

## 7. Known open items / TODO
- `obs_horizon=2` retrain (temporal/parallax cues) to help the image-only depth/height problem; a
  wrist camera would help most.
```

# World-Model Teleoperation — System & Run Guide

Drive a trained world model as a **virtual follower**: a human moves the real OpenArm
**right leader**, and the world simulator's follower moves on screen (Milestone 1), and
the model's **predicted interaction torque** is reflected back to the leader as force
feedback (Milestone 2). The real follower, its camera, and the left arm are **not** used —
the model generates the image and the torque.

This is the master run guide. Milestone-1 background lives in
[`teleop_wm_milestone1.md`](teleop_wm_milestone1.md); training in
[`training_world_ft.md`](training_world_ft.md).

---

## 1. Two repos, one redis bus

| Repo / env | Role |
|---|---|
| **IWS_test** (`iws` conda env) | World model + inference. `scripts/inference/wm_follower_worker.py` (the sim worker), `rollout_world_ft.py` (offline eval), `replay_leader_to_redis.py` (replay). Torch + `interactive_world_sim`. |
| **robot/** | Real robot. Damiao CAN controllers, `OpenArm1Solver` (pinocchio), and the leader-side scripts under `world_ft/`. |

They never import each other's heavy code — the **only** coupling is **localhost redis
(port 6379)**, using `nova.redis_client` (redis streams, `maxlen=1` for latest-value). The
the worldsim worker imports it via a sys.path insert of `robot/`.

### Data flow (force-feedback / Milestone 2)
```
human moves real RIGHT leader (CAN can1)
  │  leader_right_state:actual_angle_rad  (redis, ~1kHz, published by the damiao controller)
  ▼
wm_follower_worker.py            (GPU, ~10Hz)
  action = [leader 7 joints, seed gripper] → normalize
  z ← dynamics_forward(z, action)                    # one sim step from a fixed seed latent
  decode(z) → on-screen image                        # async, background CUDA stream
  torque_pred = torque_head(z, action)               # TOTAL joint torque, raw N·m
  publish virtual_follower_right_state:torque_pred_nm (+ :timestamp)
  ▲                                              │
  │ leader angle                                 ▼ torque_pred_nm
leader_wm_force_feedback.py       (nova, ~1kHz)
  tau_interaction = torque_pred[:7] − gravity_comp(leader_q)   # nova has pinocchio; worker does not
  reflected = ramp·k · LPF(tau_interaction), rate-limited, CLAMPED ±8 N·m, watchdog→0 if stale
  command leader_right_mit_command: tau = gravity_comp − reflected   (kp=kd=0, gripper limp)
```
For **visual-only** (Milestone 1), drop the torque path: the worker runs without
`--publish-torque`, and the leader side just homes + gravity-comp floats
(`teleop_wm.yaml` → `leader_home_and_float.py`).

---

## 2. Prerequisites (once)

- **Checkpoints** (retrained, under `worldsim/world_ft_ckpts/`):
  - `stage2/checkpoints/last.ckpt` — dynamics + torque head + encoder.
  - `stage3/checkpoints/last.ckpt` — fine-tuned **decoder** (robust to latent drift).
  - Stage 1 is **not** needed at inference (its frozen encoder is inside stage 2, and stage 3
    is the fine-tuned stage-1 decoder). Each ckpt needs its sibling `.hydra/config.yaml`.
- **Seed episode**: `worldsim/data/world_ft_seed_0713/val/episode_0.hdf5` — defines the fixed
  start pose (`action[0]`) the leader homes to **and** the sim's frame-0 scene. **Both sides
  must use the same seed episode.** (Regenerate from a raw collection: see §7.)
- **CAN up** (right leader = **can1** with followers connected — see §6):
  ```bash
  sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on
  sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on
  ```
- **Env deps** (`h5py`, `redis`) ship in the `hapticworld` env.

---

## 3. Which experiment? (quick reference)

| Goal | Robot? | Section | Terminals |
|---|---|---|---|
| Validate the model offline (pred vs GT video + torque RMSE) | no | §4.A | 1 |
| Leader drives the sim, **visual only** | yes | §4.B | 2 |
| Leader drives the sim, **with force feedback** | yes | §4.C | 3 |
| Replay a recorded episode's actions to the sim (no arm) | no (redis only) | §4.D | 2 |
| Drive the physical leader along a recorded trajectory (hardware-in-the-loop) | yes | §4.E | 3 |

All GPU runs default to `--dtype bfloat16` below (≈1.9× faster dyn+torque on the 5080; see §9).

---

## 4. The experiments

### A. Offline rollout — validate the model (no robot)
Rolls the *recorded* actions of converted episodes through the model and writes
`episode_N_rollout.mp4` (pred | gt) + `episode_N_torque.png` (pred-vs-gt torque + RMSE).
```bash
conda activate hapticworld
cd HapticWorld/worldsim
python scripts/inference/rollout_world_ft.py \
  --ckpt checkpoints/wm/boxflip/stage2/checkpoints/last.ckpt \
  --stage3-ckpt checkpoints/wm/boxflip/stage3/checkpoints/last.ckpt \
  --data_dir data/world_ft_eval_0713_rollout --split val \
  --out wm_rollout_0713
```
(Latest 10-episode run mean torque RMSE ≈ **0.13 N·m**.)

### B. Visual teleop — leader controls the sim (no torque)
**Term 1 (nova) — home the leader on can1, then float:**
```bash
cd HapticWorld/robot
process-compose -f world_ft/teleop_wm.yaml
```
Wait for `leader float ready`.

**Term 2 — the sim worker:**
```bash
conda activate hapticworld
cd HapticWorld/worldsim
PYTHONPATH=HapticWorld/robot python scripts/inference/wm_follower_worker.py --ckpt checkpoints/wm/boxflip/stage2/checkpoints/last.ckpt --stage3-ckpt checkpoints/wm/boxflip/stage3/checkpoints/last.ckpt --seed-episode data/world_ft_seed_0713/val/episode_0.hdf5 --dtype bfloat16
```
Order: **Term 1 (wait "leader float ready") → Term 2.** Move the leader → the sim follows.
Press `q` (or Ctrl-C) to quit.

### C. Force-feedback teleop — leader controls the sim AND feels it
**Term 1 (nova) — leader-only stack (redis + can1 controller):**
```bash
cd HapticWorld/robot
process-compose -f world_ft/teleop_wm_leader_only.yaml
```

**Term 2 (nova) — force-feedback driver (homes → holds → auto-engages). Start at `--k 0` to observe:**
```bash
cd HapticWorld/robot
python world_ft/leader_wm_force_feedback.py --seed-episode HapticWorld/worldsim/data/world_ft_seed_0713/val/episode_0.hdf5 --k 0
```

**Term 3 — the sim worker, publishing torque:**
```bash
conda activate hapticworld
cd HapticWorld/worldsim
PYTHONPATH=HapticWorld/robot python scripts/inference/wm_follower_worker.py --ckpt checkpoints/wm/boxflip/stage2/checkpoints/last.ckpt --stage3-ckpt checkpoints/wm/boxflip/stage3/checkpoints/last.ckpt --seed-episode data/world_ft_seed_0713/val/episode_0.hdf5 --publish-torque --dtype bfloat16
```
Order: **Term 1 → Term 2 (homes + holds at the seed pose) → Term 3.** When the worker starts
publishing, Term 2 auto-prints `worker detected … switching to float + reflect` and the arm
goes backdrivable.

**Tuning the feedback strength (`--k`):** `k` scales the reflected interaction torque
(`reflected = k·tau_interaction`, then clamped to ±8 N·m). Start at `0` (observe `|tau_int|max`
— should be ~0 in free space, rising on sim contact), then rerun Term 2 ramping up
`--k 0.3 → 0.6 → 0.9`, watching for oscillation/buzz. `k≈1` ≈ 1:1 with the collected force;
the original real teleop used `0.9` for stability. To feel *small* forces at high `k`, leader
friction/stiction compensation still needs to be added (currently deferred).

### D. Replay recorded actions to the sim (no arm, redis only)
Feeds a converted episode's actions straight onto redis as the leader angle, so the
**unchanged** worker runs on known-good input (isolates the worker from the live arm). Redis
must be running (start one: `redis-server --port 6379 --save "" --appendonly no`).

**Worker, then replay (either order — replay auto-waits for the worker):**
```bash
# worker
conda activate hapticworld && cd HapticWorld/worldsim
PYTHONPATH=HapticWorld/robot python scripts/inference/wm_follower_worker.py --ckpt checkpoints/wm/boxflip/stage2/checkpoints/last.ckpt --stage3-ckpt checkpoints/wm/boxflip/stage3/checkpoints/last.ckpt --seed-episode data/world_ft_eval_0713_rollout/val/episode_0.hdf5 --dtype bfloat16

# replay (nova env)
cd HapticWorld/robot
PYTHONPATH=HapticWorld/robot python HapticWorld/worldsim/scripts/inference/replay_leader_to_redis.py --episode HapticWorld/worldsim/data/world_ft_eval_0713_rollout/val/episode_0.hdf5
```

### E. Hardware-in-the-loop replay — drive the physical leader along a recorded trajectory
Actively position-controls the real leader through a recorded trajectory; the damiao
controller's **measured** angle then drives the sim (exercises the full real path). 3
terminals: Term 1 = `teleop_wm_leader_only.yaml` (as in §4.C), Term 3 = the worker (as in
§4.B). Term 2:
```bash
cd HapticWorld/robot
python world_ft/leader_replay_trajectory.py --seed-episode HapticWorld/worldsim/data/world_ft_seed_0713/val/episode_0.hdf5
```
⚠️ This drives the arm on its own — hand on the e-stop; it aborts if a recorded frame jumps
more than `--max-step-deg` (20°).

### F. State-conditioned model (`stage2_state`) — same teleop, swap the worker
The ModelScope `stage2_state` checkpoint is a **different architecture**
(`LatentWorldModelState`): the dynamics condition is 16-d `[q_des, q_measured]` and an
autoregressive GRU **state branch** predicts the measured follower state internally, so
there is still **no real follower**. The leader side is **unchanged** — only the worldsim
worker differs (`wm_follower_worker_state.py` instead of `wm_follower_worker.py`).

Prereq (one-time): the state-model code from `stage2_state_validation/payload/` must be
installed into the repo (algorithm/dataset/experiment + configs + the two
`experiments/__init__.py` registry edits). The checkpoint lives at
`world_ft_ckpts/stage2_state/checkpoints/last.ckpt` (+ its `.hydra/config.yaml`).

Run exactly like §4.B (visual) or §4.C (force) — Term 1/Term 2 on nova are identical;
only the worldsim worker command changes. Visual (§4.B) Term 2:
```bash
conda activate hapticworld
cd HapticWorld/worldsim
PYTHONPATH=HapticWorld/robot python scripts/inference/wm_follower_worker_state.py --ckpt world_ft_ckpts/stage2_state/checkpoints/last.ckpt --seed-episode data/world_ft_seed_0713/val/episode_0.hdf5 --dtype bfloat16
```
Force feedback (§4.C) Term 3: add `--publish-torque` (the leader `leader_wm_force_feedback.py`
side is unchanged — it still reads `virtual_follower_right_state:torque_pred_nm`).

Notes:
- **No `--stage3-ckpt` here.** The released stage-3 decoder was fine-tuned from a
  *different* stage-1 (`2026-07-15`) than `stage2_state` (`2026-07-16`), so its latent space
  does not match — grafting it corrupts the decode. The checkpoint's built-in decoder is
  used (verified clean; see `wm_state_verify/`). The flag stays for a future state-matched
  stage-3.
- **Seed state.** The worker seeds the measured state from `obs/qpos_right` if the seed
  episode has it (v4/v5 state datasets), else from `action[0]` (the homed pose) with a
  warning. `world_ft_seed_0713` predates the state converter, so it uses the `action[0]`
  proxy — fine, since measured ≈ commanded at the seed pose.
- `--dtype bfloat16` runs ~15 Hz on the 5080 (fp32 is the safe default; the state branch is
  always kept fp32 internally regardless).

---

## 5. `wm_follower_worker.py` options

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | (required) | Stage-2 (torque) checkpoint: dynamics + torque head + encoder |
| `--stage3-ckpt` | none | Stage-3 checkpoint; its decoder is grafted over stage-2 (sharper, drift-robust) |
| `--seed-episode` | (required) | hdf5 for the fixed start pose + frame-0 scene; **same on both sides** |
| `--dtype` | `float32` | `bfloat16` = tensor cores + flash attention (~1.9× faster dyn+torque, clean); `float16` also available |
| `--publish-torque` | off | Milestone 2: predict + publish `virtual_follower_right_state:torque_pred_nm` |
| `--dec-infer-steps` | `3` | decoder denoising steps (the original's value; `1` is fast but grainy) |
| `--async-decode` / `--no-async-decode` | on | decode on a background CUDA stream so the sim/torque loop keeps ~10 Hz |
| `--record` / `--no-record` | on | save the session (see §8) |
| `--record-dir` | `wm_sessions` | parent dir for recordings |
| `--hz` | `10.0` | sim step rate (= training cadence) |
| `--display-size` | `512` | window size |
| `--device` | `cuda:0` | inference device |

The decode is **display-only** (it never feeds the latent rollout), which is why async decode
+ few-step decode are safe for the sim. `--display-every-n` only applies with `--no-async-decode`.

---

## 6. CAN mapping (the recurring gotcha)

CAN interface numbers are assigned by **USB enumeration order**, so unplugging an arm or
rebooting can renumber them. Current wiring (both dongles connected):

| CAN | dongle (USB port) | arm |
|---|---|---|
| can0 | `3-1.2` | leader_left |
| **can1** | `3-1.2` | **leader_right** ← used by the WM teleop |
| can2 | `3-1.3.2` | follower_left |
| can3 | `3-1.3.2` | follower_right |

`teleop_wm.yaml` / `teleop_wm_leader_only.yaml` use **can1** for the right leader;
`collect_data.yaml` uses can2/can3 for followers. **Verify** on the first run that the *right*
arm homes (if the left moves, swap to `can0`). A durable fix (pin names by USB port +
`dev_id`) is drafted at `robot/world_ft/99-openarm-can.rules` — see its header to
install.

---

## 7. Data conversion (raw collection → hdf5)

Raw collections live under `data/collecting/<timestamp>_PDT/`
(camera mp4 + `low_dim_npys/`). Convert with `scripts/data_collection/convert_world_ft.py`,
which globs `<src>/*_PDT` and writes `<out>/{train,val}/episode_N.hdf5`. To convert a subset
(e.g. one seed episode, or N random eval episodes), symlink the chosen `*_PDT` dirs into a
scratch dir and point `--src` at it:
```bash
conda activate hapticworld && cd HapticWorld/worldsim
python scripts/data_collection/convert_world_ft.py --src <dir_of_symlinks> --out data/<name> --n_val <N>
```
The recorded `action` is `follower_right_mit_command:q` = the leader's measured angle (identity
in this teleop), so it doubles as both the homing target and the sim action.

---

## 8. Outputs / recordings

Every worker run (unless `--no-record`) saves to `wm_sessions/<timestamp>/` on quit:
- `rollout.mp4` — the predicted frames as displayed (H.264, browser-playable).
- `torque.png` — 8-joint **predicted** torque vs time (no GT — live teleop has no real follower).
- `session.npz` — `t`, `torque_pred_nm` (T×8), `action` (T×8) for analysis.

Torque is computed for recording even without `--publish-torque`. Files finalize on `q` or Ctrl-C.

---

## 9. Key design decisions / what we learned

- **Stage-3 decoder graft.** Stage 3 fine-tunes only the decoder; the worker/rollout load
  stage-2 (encoder+dynamics+torque) and graft stage-3's decoder over it. The latent space is
  shared (both froze stage-1's encoder).
- **Torque head predicts TOTAL joint torque** (`actual_torque_nm`), not interaction torque. So
  interaction is isolated on the **nova side**: `tau_interaction = torque_pred − gravity_comp(q)`
  (via the shared `OpenArm1Solver`).
- **±8 N·m reflected-torque clamp** — the hard safety cap on the human-facing force, applied
  after LPF/rate-limit, plus a **staleness watchdog** (decay to 0 if WM torque is stale) and a
  **gain ramp** from 0. The world model is open-loop and drifts — these are what stand between a
  bad prediction and the hand.
- **10 Hz cadence matters.** Below ~10 Hz each step advances the leader by a larger-than-trained
  joint delta → out-of-distribution → drift/blur. Keep the sim loop at 10 Hz.
- **`dec_infer_steps`.** The original codebase decodes at **3** steps; `1` (which we briefly
  used for speed) leaves visible speckle noise. Default is 3.
- **Async decode.** The 3-step decode (~115 ms) runs in a background thread on its own CUDA
  stream, so the sim/torque loop holds ~10 Hz while the image renders crisply (~4 Hz refresh).
- **bf16 via autocast.** Casting the whole module breaks the time-embedding; instead the model
  stays fp32 and forwards run under `torch.autocast`, keeping latents in fp32. On the RTX 5080
  this ~1.9× the dyn+torque loop (87 → 46 ms) with clean output.
- **Seed consistency + CAN verification** are the two most common setup mistakes — same
  `--seed-episode` on both sides, and confirm the right arm is on `can1`.

---

## 10. Safety

- Hand on the **e-stop** for any run that moves the arm (homing, force feedback, hardware replay).
- **Hand-place the leader near the seed pose** first (`world_ft_seed_0713` action[0] ≈
  `[-7.5, 27.2, -2.0, 67.7, 17.0, 27.0, 33.9]°`) so the homing move is small.
- Force feedback: ramp `--k` from 0, watch for oscillation, keep sessions short (open-loop drift).
- Confirm the **right** leader is the arm that homes before trusting the CAN mapping.

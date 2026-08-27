# world_ft — complete task lifecycle (box-flip task, concluded 2026-07-27)

**Read this first.** It is the top-level map of the whole project: how a manipulation task goes
from raw teleop data → a torque-augmented interactive world simulator (the "world_ft" WM) → ACT
policies (± torque) → rollouts in sim and on the real robot. Detail lives in three companion docs:

| doc | contents |
|---|---|
| [PIPELINE.md](PIPELINE.md) | operational detail for every robot-side step (env, streams, commands, flags, gotchas) |
| [WM_EVAL.md](WM_EVAL.md) | evaluation methodology + dated results (realism metric, gate, what didn't work) |
| `worldsim/docs/training_world_ft.md` | the WM's own stage 1→2→3 training runbook (hydra commands, W&B, caching) |
| [WIPE_TASK.md](WIPE_TASK.md) | follow-up task (2026-07-28→): whiteboard-wipe — collection rig, ACT matrix, rollout tooling, findings |
| [WIPE_WM.md](WIPE_WM.md) | wipe task → world-simulator training runbook (collection → upload → server conversion → WM stages 1-3) |
| [MICROWAVE_TASK.md](MICROWAVE_TASK.md) | third task (2026-08-11→): microwave door opening — collection, WM, ACT matrix, real + in-WM rollouts, paper video set |

**The project in one sentence:** take the vision-only Interactive World Simulator (IWS) and close
the force loop — the WM *predicts joint torque* (calibrated against reality), the human *feels* it
during in-sim teleop, the collected data *contains* it, and force-conditioned policies can be
*trained and evaluated* on it.

**Two repos, two conda envs** (get this right first — PIPELINE §0):
- `robot/` — robot stack, ACT learning, eval. Single env `hapticworld` (CAN, MJCF
  pinocchio, torch): robot workers, eval_controller, training, inference worker. Anything
  needing dynamics/IK
  through `learning/gravity_ik_server.py` (subprocess bridge, auto-spawned).
- `HapticWorld/worldsim` — the world model (training, checkpoints, in-sim workers). Scripts there
  need `PYTHONPATH=HapticWorld/robot` for `nova.redis_client`.

---

## Phase A — real teleop data for the WM

1. Bimanual teleop collection: `process-compose -f world_ft/collect_data.yaml`
   (leader/follower MIT controllers on can0-3, mirror solvers, RealSense `camera_head`,
   `tau_interaction_publisher.py`, video/low-dim recorders). Episodes → `data/collecting/<ts>/`
   as `camera_head.mp4` + `low_dim_npys/*.npy` (timestamp-aligned streams).
2. Convert a curated batch to WM hdf5s: `worldsim/scripts/data_collection/convert_world_ft.py
   --src <batch dir> --out data/world_ft_vN --n_val 5 --target_hz 10`.
   **Label decision:** `obs/joint_torque` (the stage-2 target) = `actual_torque_nm` — the TOTAL
   measured motor torque, NOT interaction torque (`tau_interaction` is stored but unused).
   Everything downstream converts between the two definitions explicitly.
3. **Vary the box pose across episodes.** The single most expensive lesson of this project:
   with a fixed box, policies learn visually-clocked replay and ignore the box entirely.

## Phase B — WM training (worldsim/)

Three stages via `main.py` + Hydra; launchers in `scripts/train/` (`run_stage{1,2,3}.sh`,
`train_world_ft_stage*.sh`); full runbook in `worldsim/docs/training_world_ft.md`:
- **Stage 1** — autoencoder (encoder + diffusion decoder), self-supervised, actions ignored.
- **Stage 2** — freeze enc/dec, train latent dynamics + the **right-arm joint-torque head**
  (raw N·m; this head is the core world_ft addition over IWS).
- **Stage 3** — decoder-only finetune on noised latents (sharper decode of drifted rollout
  latents). **Compatibility rule:** stage 2 and stage 3 must share the same stage-1 run
  (check `.hydra/config.yaml: load_ae`).
Production checkpoints: `checkpoints/wm/boxflip/{stage1, stage2, stage3}` (stage-1
lineage `2026-07-15_02-35-54`). A state-conditioned variant (`stage2_state`+`stage3_state`)
was built and abandoned — it underperformed the plain trunk+torque-head (WM_EVAL §3c; keep as
the justified-simplicity ablation, don't build on it).

## Phase C — WM evaluation

- Open-loop, per-episode: `worldsim/scripts/inference/rollout_world_ft.py --ckpt ... --stage3-ckpt
  ... --data_dir ... --split val` → pred|gt video + 8-joint torque-RMSE plots.
- Torque calibration facts (measured, quotable): WM-predicted contact torque matches real within
  ~10% on the load-bearing joints (contact p95 ≈ 7-8 N·m on joints 1/3); free-space baselines and
  10 Hz temporal statistics also match. See WM_EVAL and the 2026-07-22/23 analyses.

## Phase D — ACT training data (both domains)

**Real**: same `collect_data.yaml` rig → curate into
`data/real_training/` (~50 episodes, box pose varied). Torque stream:
`follower_controller_state:tau_interaction_R` = measured − gravity-comp feedforward.

**Sim** (3 terminals — PIPELINE §2b): the WM plays follower + scene; the human drives the real
right leader and *feels* the WM's predicted torque:
1. `process-compose -f world_ft/teleop_wm_leader_only.yaml` (control bus + leader can1;
   `_leader_follower` variant also mirrors the real follower).
2. `python world_ft/leader_wm_force_feedback.py --seed-episode <hdf5>` (homes leader,
   floats, reflects `torque_pred − gravity(leader q)` with LPF/clamp/watchdog).
3. `python scripts/inference/wm_follower_worker.py --ckpt <stage2 last.ckpt>
   --stage3-ckpt <stage3> --seed-episode <same hdf5> --publish-torque --dtype bfloat16`
   — **multi-episode flow**: ENTER = record · `s` = background save + reset + 3 s hold →
   live → ENTER again · `q` = quit (drains saves). REC/IDLE HUD; ~20-25 s/episode sustained
   (≈5× the old save-and-relaunch flow). All decodes share `_DECODE_LOCK` — the diffusion
   decoder is NOT thread-safe (concurrent decodes corrupt frames, measured).
   Episodes → `data/sim_training/` (with feedback) / `data/sim_training_wo_ft/` (feedback off —
   the collection-condition comparison; operators press ~2.5 vs ~1.35 N·m mean with/without).

**After every sim batch, mandatory:** `python world_ft/postprocess_sim_tau_interaction.py
[--data-dir ...]` — writes `tau_interaction_R = torque_pred_nm − gravity(commanded q)`
so sim and real torque are the SAME quantity for training and deploy. A mixed folder fails loudly
at dataset init (LowDimDataset requires identical stream sets). Sanity-check new episodes for
corruption/length (see the checker pattern in the 2026-07-24 session; quarantine to
`data/sim_training_quarantine/`).

## Phase E — ACT policy training (from `robot/`)

`world_ft/learning/`: `dataset.py` (OpenArmDataset — timestamp-interpolated mp4+npy,
auto-selects `tau_interaction` over `torque_pred_nm`, `blur_aug_res` sharpness augmentation),
`model.py` (`ACT_Abs`, image+state token) / `model_image_only.py` (proprio token dropped),
`train_core.py` (shared loop; ~10-line entry scripts define the matrix):

| entry script | input | data |
|---|---|---|
| `train_image_only_{real,sim}.py` | image | real / sim |
| `train_image_torque_{real,sim}.py` | image + tau_interaction (8-d) | real / sim |
| `train_image_state_torque_real.py` | image + angle + torque (16-d) | real |
| `train_image_{only,torque}_real_blur.py` | + 224→128→224 aug on 50% of samples | real |
| `train_image_torque_sim_wo_t.py` | image + torque | sim, no-feedback collection |

- `_blur` variants make real-trained policies **sharpness-invariant** so they can also run inside
  the WM (renders are ~12× softer than the camera; there is no deploy-time fix in that direction).
- For condition comparisons, train to **budget parity** (stop at the same step count — see the
  auto-stop-at-ckpt_step_5000 wrapper pattern, 2026-07-24).
- Checkpoint selection everywhere (`learning/ckpt_select.py`): variant specs `i|it|its` ×
  `_real|_sim` (matched by info.yaml config + dataset-dir basename), exp-prefix specs
  (`it_sim_wo_t`), or explicit paths.

## Phase F — policy rollouts

**Real robot** (PIPELINE §5; two terminals):
```bash
WORLD_FT_MODEL=<spec> process-compose -f world_ft/learning/eval.yaml
python world_ft/learning/eval_controller.py --model <spec> [--random-home] [--save]
```
- Worker (`eval_inference_worker.py`): auto-configures from info.yaml (image_only/use_angle/
  use_torque), auto `--match-blur-res` (sim-trained→128, real→0), ~10 Hz chunk publishing.
- Controller (`eval_controller.py`, 2 kHz): limp → `h` home (auto reference episode =
  closest-to-mean start of the ckpt's own dataset) → ENTER rollout (receding horizon; use
  `--chunk-renew-every-s 0.5` for reactivity) → `s` stop. Publishes `tau_interaction_R` itself.
- `--random-home`: fresh EE offset per trial (x,y ±0.10 m, z −0.02..0.03 m).
- `--save`: records each ENTER→s rollout (video + full robot streams + policy chunks + meta.yaml)
  to `data/eval_rollouts/<spec>/` via the collect_data recorder workers (in eval.yaml).

**Inside the WM** (offline, no robot; WM_EVAL + PIPELINE §5b):
1. Capture seed scenes (box+arm varied): `process-compose -f world_ft/capture_wm_seeds.yaml`
   + `python world_ft/capture_wm_seeds.py` (ENTER per scene → minimal hdf5s in
   `worldsim/data/wm_seed_live/`; a seed needs the FRAME and the ARM POSE, both captured live).
2. `worldsim/scripts/inference/rollout_policy_in_wm.py` — policy↔WM closed loop, 0.5 s receding
   horizon, one seed sampled per rollout from a job list built ONCE (rollout #k is paired across
   all policies), torque policies get `torque_pred − gravity(q_cmd)` via the gravity server.
   `--replay <npz list>` re-runs recorded rollouts (longer horizon / retries, never overwrites).
3. Score realism: `worldsim/scripts/inference/score_box_sharpness.py --window-s 11 17` — the
   box-integrity gate (sharp/sat/area/valid vs pre-contact baseline). Do NOT use image-space
   cycle-consistency (reconstruction error rewards blur — measured dead end, WM_EVAL §4).
   Success itself is scored by eye from the videos.

## Key findings (what the experiments established)

1. **Low-variation demos ⇒ visually-clocked replay.** Policies trained on fixed-scene data ignore
   the box completely (remove it — same motion). Vision becomes causal only if the box pose varies
   in training. Verify with the box-removal probe before trusting any policy.
2. **Force is encoded in position commands.** Demos command ~7° past the box surface; effective
   joint stiffness ≈52 N·m/rad reproduces the demonstrated force open-loop. Torque input only
   matters when required force varies (mass/friction/pose/slip) — design evals accordingly.
3. **In WM-generated data, torque labels derive from the same latent as the image** (informational
   redundancy) — yet torque conditioning still helps at deploy (it_sim > i_sim robustness with
   varied poses) and keeps rollouts scene-consistent.
4. **Contextually-OOD policies break the WM** (box smears): i_sim fails the realism gate 16/20 vs
   ~4/20 for it_sim / i_real / it_real. Marginal action stats cannot detect this. Realism-gate
   every in-WM evaluation.
5. **The sharpness gap is bidirectional**: real→sim-policy fixed by `--match-blur-res 128`;
   sim→real-policy has no deploy fix — requires train-time blur augmentation.
6. **Collection condition changes demonstrations measurably** (2.5 vs 1.35 N·m mean pressing
   with/without force feedback) — the mechanism behind the data-quality claim.

## Paper skeleton (2026-07-24 discussion; sources searched)

Thesis: *force is the missing modality in interactive world simulators*. Claims: (1) first
calibrated force-in-the-loop world simulator; (2) feedback → collection throughput + demo quality
(user study + torque-profile evidence); (3) force-labeled synthetic data → force policies with
real gains; (4) force-aware policy evaluation with realism gating and sim-real agreement.
Framing rules: never claim a comparison the baseline can't enter (IWS cannot run torque policies —
that's a COVERAGE claim); fidelity = SIMPLER/WorldEval-style sim-real correlation across
policies/checkpoints. Still open: user study, it_sim_wo_t real-robot comparison (count-matched
retrain recommended: 56 vs 75 episodes), checkpoint-sweep correlation, optional stage-2
torque-loss ablation. Baseline paper: IWS (arXiv 2603.08546); related: FACTR, SIMPLER
(2405.05941), WorldEval (2505.19017), haptic-data-quality (2211.03020).

## Gotchas quicklist (each cost real time once)

- Single env `hapticworld` carries CAN + MJCF pinocchio + torch. Run robot-side commands from
  `robot/`, worldsim-side from `worldsim/`. `nova` is imported via sys.path inserts.
- DETR's builder calls `argparse.parse_args()` on bare `sys.argv` — blank argv around model builds.
- `--help` may crash for typer scripts on some typer/click version combos (seen with typer 0.15 / click 8.4) —
  scripts run fine.
- CAN channels renumber when dongles are re-plugged — verify before driving motors.
- "Frame didn't arrive within 5000" = camera on a USB-2 link, not a code bug.
- NVIDIA unattended driver updates break CUDA for every process until reboot
  (`nvidia-smi` → "Driver/library version mismatch").
- The WM diffusion decoder is not thread-safe — hold `_DECODE_LOCK` for every decode.
- LowDimDataset requires identical stream sets across episodes — post-process before training.
- eval.yaml + controller must use the SAME checkpoint spec (env var + --model).

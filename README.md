# HapticWorld

A force-aware interactive world model for contact-rich robot manipulation.
HapticWorld extends a latent-diffusion world simulator with an interaction-torque
head, so the learned simulator predicts not just the next camera frame but also
the force the robot feels — enabling force-feedback teleoperation *inside the
simulator*, torque-conditioned policy training, and no-hardware policy evaluation.

This release ships the **box-flipping** task end to end: data collection rigs,
world-model training (3 stages), ACT policy training (with and without torque),
and closed-loop policy rollout inside the trained simulator.

## Repository layout

| dir | contents |
|---|---|
| `worldsim/` | the world model: training (stages 1–3), inference workers, in-simulator policy rollout |
| `robot/` | OpenArm robot stack (`nova/`) + task rigs (`world_ft/`): collection, teleop, ACT training, real-robot eval |
| `checkpoints/` | download script for pretrained checkpoints (HuggingFace) |
| `docs/` | task walkthrough (`BOXFLIP_TASK.md`), operations manual (`PIPELINE.md`), model architecture |
| `envs/` | conda environment |

## Install

```bash
conda env create -f envs/environment.yaml
conda activate hapticworld
python checkpoints/download_assets.py        # pretrained WM + policies + seeds (~2.3 GB)
```

(`process-compose`, used only by the real-robot rigs, is a
[standalone binary](https://github.com/F1bonacc1/process-compose).)

## 1. No robot needed — evaluate policies inside the simulator

The trained world model plays the environment; a trained ACT policy plays the
human. Closed-loop, offline, no hardware:

```bash
cd worldsim
python scripts/inference/rollout_policy_in_wm.py --policies i_sim,it_sim --n-rollouts 5
```

Each rollout saves `wm_policy_rollouts/<policy>/rollout_<k>.mp4` + a `.npz` log.
`i_sim` is the no-torque policy, `it_sim` the torque-conditioned one; seeds
(initial scenes) are sampled from `data/wm_seed_live/`.

## 2. Train it yourself

World model (stages: autoencoder → latent dynamics + torque head → decoder
finetune), from `worldsim/`:

```bash
bash scripts/train/run_stage1.sh
bash scripts/train/run_stage2.sh
bash scripts/train/run_stage3.sh
```

ACT policies, from `robot/`:

```bash
python world_ft/learning/train_image_only_sim.py     # no-torque (i_sim)
python world_ft/learning/train_image_torque_sim.py   # torque (it_sim)
```

See `docs/BOXFLIP_TASK.md` for the full walkthrough including dataset
conversion, and `worldsim/docs` for training details.

## 3. Full hardware loop (OpenArm)

With a bimanual OpenArm rig (leader + follower, Damiao CAN motors, RealSense
head camera) you can collect data, teleoperate the *simulator* with force
feedback on the real leader arm, and run real-robot policy evals. Start at
`docs/PIPELINE.md`; the rigs live in `robot/world_ft/*.yaml`.

## Acknowledgements

The world-model backbone builds on the
[Interactive World Simulator](https://github.com/WangYixuan12/interactive_world_sim)
(Wang et al.) — see `LICENSE` / `LEGAL.md`. The torque head, force-feedback
teleop loop, robot stack, and task pipelines are new in HapticWorld.

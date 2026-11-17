# HapticWorld

Code and checkpoints for **HapticWorld: an Interactive World Simulator with Real-time Torque
Feedback**. HapticWorld is a learned world model that predicts the next camera frame together
with the joint torque the robot would feel. This repository covers three tasks (box flipping,
whiteboard wiping, microwave door opening) and lets you:

1. roll out the released torque-conditioned ACT policies inside HapticWorld,
2. train the policies on the released demonstrations, and
3. train HapticWorld on play data, using the released sample data or your own recordings.

## Layout

```
worldsim/                the world model
  hapticworld/           model code: consistency decoder, latent dynamics, torque head, data loader
  configurations/        Hydra configs
  main.py                training entry (Hydra)
  scripts/train/         run_stage{1,2,3}.sh, train_and_eval.sh
  scripts/inference/     rollout_policy_in_wm.py (policy inside HapticWorld), rollout_hapticworld.py (open-loop validation)
policy/                  the ACT policy: model (act/), data loader (data/), training entries, arm model for gravity compensation (robot_model/)
checkpoints/             download_assets.py (fetches everything below from HuggingFace)
data/                    created by the download: seeds/, demos/, play/
envs/environment.yaml    conda environment
```

## Install

```bash
conda env create -f envs/environment.yaml
conda activate hapticworld
python checkpoints/download_assets.py            # all tasks; --task wipe for one task
```

Per task this fetches:

| path | contents |
|---|---|
| `checkpoints/wm/<task>/stage{1,2,3}` | the HapticWorld model |
| `checkpoints/policies/<task>/<task>_it_sim` | the ACT policy |
| `data/seeds/<task>` | start scenes for policy rollouts: a first frame and a start pose each |
| `data/demos/<task>` | the demonstrations the policy was trained on, collected inside HapticWorld |
| `data/play/<task>` | sample play data (8 train + 2 val episodes) in the format world-model training expects |

The demonstrations were recorded while an operator teleoperated HapticWorld: the video is
HapticWorld's rendered frames, the torque is HapticWorld's predicted torque, and the actions are
the operator's joint commands. No real-robot data is used to train the released policies.

## Roll out a policy inside HapticWorld

```bash
cd worldsim
python scripts/inference/rollout_policy_in_wm.py --task boxflip --n-rollouts 20    # wipe | microwave
```

Each rollout starts from a seed scene and runs the policy closed-loop for 20 s at 10 Hz: the
policy sees the rendered frame and the predicted interaction torque. Videos and `.npz` logs go to
`wm_policy_rollouts/<task>/it_sim/`. `--policies <path to model.pth>` rolls out your own
checkpoint; `--ckpt/--stage3-ckpt` select another world model.

## Train the policy

```bash
python policy/train_boxflip_it_sim.py          # wipe | microwave; from the repository root
```

Trains ACT (image + torque input) on `data/demos/<task>` up to the step the paper used.
Checkpoints go to `checkpoints/policies/<task>/<task>_it_sim_<timestamp>/`; roll one out with
`--policies <task>_it_sim`.

## Train HapticWorld

```bash
cd worldsim
bash scripts/train/train_and_eval.sh boxflip    # wipe | microwave
```

This trains the three stages on `data/play/<task>` (autoencoder, latent dynamics + torque head,
decoder fine-tune), then rolls the result out on the validation episodes (predicted vs recorded
frames and torque) and runs the released policy inside it. Outputs go to `worldsim/outputs/`.
The stages can also be run one by one with `scripts/train/run_stage{1,2,3}.sh`.

Options, as environment variables: `DATA` (dataset directory), `NGPU`, `BATCH` (per GPU,
default 16), `STEPS` (stop each stage after this many steps; by default training runs until
stopped), `WANDB_MODE` (default offline). If you run out of GPU memory, lower `BATCH`,
`VAL_HORIZON` (frames per validation rollout, default 200) and `RENDER_BATCH` (default 16):

```bash
STEPS=200 BATCH=2 VAL_HORIZON=50 RENDER_BATCH=8 bash scripts/train/train_and_eval.sh boxflip
```

To use a trained model in the rollout script by default, copy the stage-2 and stage-3 runs
(`checkpoints/last.ckpt` and `.hydra/config.yaml`) to `checkpoints/wm/<task>/stage2` and `stage3`.

## Data format

Play data for the world model is one hdf5 file per episode, `<DATA>/{train,val}/episode_<i>.hdf5`:

```
action                     (T, 8)        float32   commanded right-arm joint position at 10 Hz (7 joints + gripper)
obs/images/camera_1_color  (T, H, W, 3)  uint8     head-camera RGB; center-cropped to a square and resized to 128x128 on load
obs/joint_torque           (T, 8)        float32   measured joint torque in N·m, gravity included (the torque head's target)
```

Validation episodes must be at least `VAL_HORIZON` frames long. For wipe, training also needs a
static background image of the scene: `python scripts/data_collection/make_static_background.py
--data_dir <DATA>` (`train_and_eval.sh` runs it if the file is missing).

Demonstrations for the policy are one folder per episode with `camera_head.mp4` and
`low_dim_npys/`: `follower_right_mit_command:q.npy` (N, 8) is the action and
`follower_controller_state:tau_interaction_R.npy` (N, 8) the torque input (predicted torque minus
gravity), each with its `:timestamp.npy`.

## Acknowledgements

The world-model backbone builds on the
[Interactive World Simulator](https://github.com/WangYixuan12/interactive_world_sim) (Wang et al.),
see `LICENSE`. The torque head, the torque-conditioned rollout and the task pipelines are new.

We used Claude (Anthropic) to help organize this code for release: restructuring the repository,
removing unused code, writing documentation and testing the release scripts.

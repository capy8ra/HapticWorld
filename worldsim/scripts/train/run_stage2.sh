#!/usr/bin/env bash
# Stage 2: action-conditioned latent dynamics + the torque head, trained jointly on top of the
# frozen stage-1 autoencoder (the HapticWorld model). See scripts/train/_common.sh for variables.
#   TASK=wipe NGPU=4 bash scripts/train/run_stage2.sh [extra hydra overrides]
#   AE_CKPT=<stage-1 last.ckpt>   defaults to the newest outputs/hapticworld_<task>_stage_1 run;
#                                 use ../checkpoints/wm/<task>/stage1/checkpoints/last.ckpt to
#                                 build on the released autoencoder instead of training stage 1.
source "$(dirname "$0")/_common.sh"
NAME="hapticworld_${TASK}_stage_2"
AE_CKPT="${AE_CKPT:-$(latest_ckpt "hapticworld_${TASK}_stage_1")}"
[[ -n "$AE_CKPT" ]] || { echo "[stage2] no stage-1 checkpoint for $TASK: run run_stage1.sh or set AE_CKPT"; exit 1; }
RESUME=""; [[ "${FRESH:-0}" == 1 ]] || RESUME="$(latest_ckpt "$NAME")"
echo "[stage2] task=$TASK data=$DATA GPUs=$CUDA_VISIBLE_DEVICES batch/gpu=$BATCH load_ae=$AE_CKPT ${RESUME:+resume=$RESUME}"
python main.py +name="$NAME" \
  "hydra.run.dir=outputs/$NAME/\${now:%Y-%m-%d}_\${now:%H-%M-%S}" \
  ${RESUME:+load=$RESUME} \
  experiment=exp_latent_dyn_torque algorithm=latent_world_model_torque dataset=hapticworld_dyn_dataset \
  "dataset.dataset_dir=$DATA" \
  dataset.obs_keys=[camera_1_color] \
  '~dataset.shape_meta.obs.camera_0_color' \
  dataset.horizon=10 dataset.val_horizon=200 \
  experiment.training.precision=bf16-mixed \
  experiment.training.batch_size="$BATCH" \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.training.checkpointing.every_n_train_steps=0 \
  experiment.training.checkpointing.every_n_epochs=1 \
  +experiment.training.checkpointing.save_on_train_epoch_end=true \
  +experiment.training.checkpointing.monitor=step \
  +experiment.training.checkpointing.mode=max \
  +experiment.training.checkpointing.save_top_k=3 \
  +experiment.training.checkpointing.save_last=true \
  experiment.validation.limit_batch=1.0 experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step="$(val_every 6000)" \
  algorithm.training_stage=2 \
  "algorithm.load_ae=$AE_CKPT" \
  algorithm.latent_dim=512 algorithm.action_dim=8 algorithm.torque_dim=8 \
  algorithm.sampling_strategy=terminal_only \
  algorithm.noise_scheduler.loss_weighting=uniform \
  experiment.num_devices="$NGPU" \
  wandb.mode="$WANDB_MODE" \
  "algorithm.metrics=$METRICS" \
  "${EXTRA[@]}" "$@"

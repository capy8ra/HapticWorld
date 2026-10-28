#!/usr/bin/env bash
# Stage 1: observation autoencoder (encoder + consistency decoder), self-supervised on the play
# data; actions are not used. See scripts/train/_common.sh for the environment variables.
#   TASK=wipe NGPU=4 bash scripts/train/run_stage1.sh [extra hydra overrides]
source "$(dirname "$0")/_common.sh"
NAME="hapticworld_${TASK}_stage_1"
RESUME=""; [[ "${FRESH:-0}" == 1 ]] || RESUME="$(latest_ckpt "$NAME")"
TASK_OVERRIDES=""
if [[ "$TASK" == wipe ]]; then
  TASK_OVERRIDES="$(wipe_recon_overrides) algorithm.cond_lr_mult=5.0 algorithm.warmup_steps=2000 dataset.val_horizon=200 experiment.validation.batch_size=1"
fi
echo "[stage1] task=$TASK data=$DATA GPUs=$CUDA_VISIBLE_DEVICES batch/gpu=$BATCH ${RESUME:+resume=$RESUME}"
python main.py +name="$NAME" \
  "hydra.run.dir=outputs/$NAME/\${now:%Y-%m-%d}_\${now:%H-%M-%S}" \
  ${RESUME:+load=$RESUME} \
  algorithm=latent_world_model experiment=exp_latent_dyn dataset=play_dataset \
  "dataset.dataset_dir=$DATA" \
  dataset.horizon=1 dataset.val_horizon=1 \
  dataset.obs_keys=[camera_1_color] dataset.action_mode=right_qpos \
  '~dataset.shape_meta.obs.camera_0_color' \
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
  experiment.validation.limit_batch=1.0 experiment.validation.batch_size=10 \
  experiment.validation.val_every_n_step="$(val_every 5000)" \
  algorithm.latent_dim=512 algorithm.action_dim=8 algorithm.training_stage=1 \
  experiment.num_devices="$NGPU" \
  wandb.mode="$WANDB_MODE" \
  "algorithm.metrics=$METRICS" \
  $TASK_OVERRIDES "${EXTRA[@]}" "$@"

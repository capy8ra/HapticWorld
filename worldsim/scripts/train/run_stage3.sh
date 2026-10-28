#!/usr/bin/env bash
# Stage 3: decoder fine-tune. The encoder stays frozen; the decoder is retrained at 0.1x the
# learning rate on latents with small Gaussian noise, so it stays sharp on the slightly drifted
# latents that stage 2 produces during long rollouts. Loads the STAGE-1 checkpoint (stage 2 does
# not change the autoencoder), so stages 2 and 3 share one latent space by construction.
#   TASK=wipe NGPU=4 bash scripts/train/run_stage3.sh [extra hydra overrides]
#   AE_CKPT=<stage-1 last.ckpt>   defaults to the newest outputs/hapticworld_<task>_stage_1 run
source "$(dirname "$0")/_common.sh"
NAME="hapticworld_${TASK}_stage_3"
AE_CKPT="${AE_CKPT:-$(latest_ckpt "hapticworld_${TASK}_stage_1")}"
[[ -n "$AE_CKPT" ]] || { echo "[stage3] no stage-1 checkpoint for $TASK: run run_stage1.sh or set AE_CKPT"; exit 1; }
RESUME=""; [[ "${FRESH:-0}" == 1 ]] || RESUME="$(latest_ckpt "$NAME")"
TASK_OVERRIDES=""
[[ "$TASK" == wipe ]] && TASK_OVERRIDES="$(wipe_recon_overrides)"
echo "[stage3] task=$TASK data=$DATA GPUs=$CUDA_VISIBLE_DEVICES batch/gpu=$BATCH load_ae=$AE_CKPT ${RESUME:+resume=$RESUME}"
python main.py +name="$NAME" \
  "hydra.run.dir=outputs/$NAME/\${now:%Y-%m-%d}_\${now:%H-%M-%S}" \
  ${RESUME:+load=$RESUME} \
  algorithm=latent_world_model experiment=exp_latent_dyn dataset=play_dataset \
  "dataset.dataset_dir=$DATA" \
  dataset.horizon=1 dataset.val_horizon=200 \
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
  experiment.validation.limit_batch=1.0 experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step="$(val_every 30000)" \
  algorithm.latent_dim=512 algorithm.action_dim=8 algorithm.training_stage=3 \
  "algorithm.load_ae=$AE_CKPT" \
  algorithm.sampling_strategy=terminal_only \
  algorithm.noise_scheduler.loss_weighting=uniform \
  experiment.num_devices="$NGPU" \
  wandb.mode="$WANDB_MODE" \
  "algorithm.metrics=$METRICS" \
  $TASK_OVERRIDES "${EXTRA[@]}" "$@"

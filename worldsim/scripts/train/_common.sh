# Shared setup for run_stage{1,2,3}.sh. Not meant to be run directly.
#
#   TASK=boxflip|wipe|microwave   which task (default boxflip)
#   DATA=<dir>                    play dataset (default <repo>/data/play/$TASK)
#   NGPU=<n>                      number of GPUs (default 1); or set CUDA_VISIBLE_DEVICES yourself
#   BATCH=<n>                     per-GPU batch size (default 16, as in the paper)
#   STEPS=<n>                     stop after n training steps (default: train until stopped)
#   VAL_HORIZON=<n>               frames per validation rollout (default 200); lower it, with BATCH and
#   RENDER_BATCH=<n>              frames decoded at once in validation (default 16), if you run out of GPU memory
#   WANDB_MODE=online|offline     default offline (set wandb.entity in configurations/config.yaml for online)
#   METRICS='[fvd]'               FVD needs metrics/i3d_torchscript.pt (download_assets.py); default '[]'
#   FRESH=1                       start a new run instead of resuming the newest one of this stage
#   env.local.sh                  optional file at worldsim/ that is sourced first (API keys, module loads)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/env.local.sh" ]] && { set +u; source "$ROOT/env.local.sh"; set -u; }
TASK="${TASK:-boxflip}"
case "$TASK" in boxflip|wipe|microwave) ;; *) echo "TASK must be boxflip, wipe or microwave (got '$TASK')"; exit 1 ;; esac
DATA="${DATA:-$ROOT/../data/play/$TASK}"
[[ -d "$DATA/train" ]] || { echo "no play dataset at $DATA (expected $DATA/{train,val}/episode_*.hdf5; python checkpoints/download_assets.py --what play)"; exit 1; }
NGPU="${NGPU:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
BATCH="${BATCH:-16}"
METRICS="${METRICS:-[]}"
WANDB_MODE="${WANDB_MODE:-offline}"
# optional overrides appended after each stage's own settings (Hydra: the last one wins)
EXTRA=()
[[ -n "${STEPS:-}" ]] && EXTRA+=("experiment.training.max_steps=$STEPS")
[[ -n "${VAL_HORIZON:-}" ]] && EXTRA+=("dataset.val_horizon=$VAL_HORIZON" "experiment.validation.batch_size=1")
[[ -n "${RENDER_BATCH:-}" ]] && EXTRA+=("algorithm.render_batch_size=$RENDER_BATCH")
# validation interval: the stage default, or the end of a shorter run
val_every() { if [[ -n "${STEPS:-}" && "$STEPS" -lt "$1" ]]; then echo "$STEPS"; else echo "$1"; fi; }
latest_ckpt() { ls -t "outputs/$1"/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true; }
# The wipe scene is nearly static (a board with a few marks), so the stage-1/3 reconstruction
# loss is re-weighted toward pixels that differ from a static background (dyn_loss_weight) and
# stage 1 additionally trains the conditioning path with a higher learning rate (cond_lr_mult).
# Box flip and microwave use the plain objective.
wipe_recon_overrides() {
  [[ -f "$DATA/background_128.npy" ]] || { echo "wipe needs $DATA/background_128.npy -- run: python scripts/data_collection/make_static_background.py --data_dir $DATA"; exit 1; }
  echo "algorithm.dyn_loss_weight=9.0 algorithm.dyn_background_path=$DATA/background_128.npy"
}

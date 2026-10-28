#!/usr/bin/env bash
# Train HapticWorld for one task (stage 1 -> stage 2 -> stage 3), then evaluate the result:
#   * open-loop rollout on the validation episodes: predicted vs recorded frames and torque (RMSE)
#   * closed-loop rollout of the released policy inside the new model, seeded from the validation episodes
#
#   cd worldsim && bash scripts/train/train_and_eval.sh boxflip      # or wipe | microwave
#
# Uses data/play/<task> (DATA=<dir> for your own data) and the options of run_stage*.sh
# (NGPU, BATCH, STEPS, VAL_HORIZON, RENDER_BATCH, ...). Without STEPS each stage trains until stopped.
# Outputs: outputs/hapticworld_<task>_stage_<n>/<timestamp>/ and outputs/hapticworld_<task>_eval/.
set -euo pipefail
TASK="${1:-boxflip}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
DATA="${DATA:-$ROOT/../data/play/$TASK}"
if [[ "$TASK" == wipe && ! -f "$DATA/background_128.npy" ]]; then
  python scripts/data_collection/make_static_background.py --data_dir "$DATA"
fi
export TASK DATA FRESH=1
bash scripts/train/run_stage1.sh
bash scripts/train/run_stage2.sh
bash scripts/train/run_stage3.sh
S2="$(ls -t outputs/hapticworld_${TASK}_stage_2/*/checkpoints/last.ckpt | head -n1)"
S3="$(ls -t outputs/hapticworld_${TASK}_stage_3/*/checkpoints/last.ckpt | head -n1)"
OUT="outputs/hapticworld_${TASK}_eval"; rm -rf "$OUT"
python scripts/inference/rollout_hapticworld.py --ckpt "$S2" --stage3-ckpt "$S3" \
  --data_dir "$DATA" --split val --out "$OUT/rollout_val"
for EP in "$DATA"/val/episode_*.hdf5; do   # one policy rollout per validation episode
  python scripts/inference/rollout_policy_in_wm.py --task "$TASK" --ckpt "$S2" --stage3-ckpt "$S3" \
    --seed-episodes "$EP" --n-rollouts 1 --out "$OUT/policy_rollout/$(basename "$EP" .hdf5)"
done
echo "done: stage 2 $S2, stage 3 $S3, evaluation in $OUT/"

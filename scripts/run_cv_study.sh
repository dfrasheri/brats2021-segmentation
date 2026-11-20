#!/usr/bin/env bash
# Full-scale study: every config x 5 folds x 3 seeds on all 1,251 patients.
#
#   python scripts/build_cache.py --n 1251       # once, ~15 min, ~40 GB
#   ITERS=20000 CONFIGS="nnunet_style segresnet swin_unetr" ./scripts/run_cv_study.sh
#   python scripts/evaluate_cv.py
#
# Budget: 5 configs x 5 folds x 3 seeds = 75 runs. At ITERS=20000 on one A100
# that is roughly 75 x 45 min ~ 56 GPU-hours; set CONFIGS / SEEDS / FOLDS to
# run a slice per machine and parallelise across GPUs. Finished runs are
# skipped, so the script is safe to re-launch after a pre-empted spot VM.
set -euo pipefail
cd "$(dirname "$0")/.."

ITERS="${ITERS:-20000}"
CONFIGS="${CONFIGS:-baseline_unet nnunet_style vnet segresnet swin_unetr}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SEEDS="${SEEDS:-42 43 44}"
EXTRA="${EXTRA:---mlflow}"

for cfg in $CONFIGS; do
  for fold in $FOLDS; do
    for seed in $SEEDS; do
      ckpt="checkpoints/${cfg}_f${fold}_s${seed}_best.pt"
      hist="logs/${cfg}_f${fold}_s${seed}_history.json"
      if [[ -f "$ckpt" && -f "$hist" ]]; then
        echo "skip $cfg fold=$fold seed=$seed (done)"; continue
      fi
      echo "=== $cfg fold=$fold seed=$seed iters=$ITERS ==="
      python scripts/train.py --config "$cfg" --fold "$fold" --seed "$seed" \
        --iters "$ITERS" $EXTRA
    done
  done
done

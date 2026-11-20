#!/usr/bin/env bash
# Sequential training of all three configurations.
set -e
PY=./.venv/Scripts/python.exe
ITERS=${ITERS:-2000}
for cfg in nnunet_style vnet baseline_unet; do
  echo "=============================================================="
  echo "TRAINING $cfg  ($(date +%H:%M:%S))"
  echo "=============================================================="
  $PY scripts/train.py --config "$cfg" --iters "$ITERS" \
      --val-every 400 --val-patients 8 --workers 4
done
echo "ALL TRAINING COMPLETE ($(date +%H:%M:%S))"

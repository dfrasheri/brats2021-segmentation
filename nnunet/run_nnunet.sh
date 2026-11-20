#!/usr/bin/env bash
# Official nnU-Net v2 as a reference, on exactly the same folds as this repo.
#
# Cost: preprocessing ~1 h CPU; 3d_fullres training ~12-24 h per fold on one
# A100 at the default 1000 epochs. Set NNUNET_EPOCHS=250 and use the
# nnUNetTrainer_250epochs variant for a cheaper but still standard reference.
set -euo pipefail
cd "$(dirname "$0")/.."

export nnUNet_raw="${nnUNet_raw:-$HOME/nnunet_data/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnunet_data/preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnunet_data/results}"
TRAINER="${TRAINER:-nnUNetTrainer}"
FOLDS="${FOLDS:-0 1 2 3 4}"
DS=137

pip install -q "nnunetv2>=2.5"

python nnunet/convert.py export
nnUNetv2_plan_and_preprocess -d $DS -c 3d_fullres --verify_dataset_integrity
python nnunet/convert.py splits         # overwrite nnU-Net's random folds with ours

for f in $FOLDS; do
  nnUNetv2_train $DS 3d_fullres "$f" -tr "$TRAINER"
done

# Each fold's validation/ folder holds predictions on that fold's held-out
# patients = our test fold. Convert labels back and score with our metrics.
python nnunet/convert.py import "$nnUNet_results/Dataset137_BraTS2021/${TRAINER}__nnUNetPlans__3d_fullres" \
  nnunet/predictions_brats
python scripts/evaluate_predictions.py --pred nnunet/predictions_brats --name nnunetv2 --cases-from-split cv

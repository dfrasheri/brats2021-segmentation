"""Score a folder of NIfTI predictions against ground truth with this repo's metrics.

Used to put external systems (the official nnU-Net v2, or anything else) on
exactly the same footing as the models trained here: same patients, same
region definitions, same Dice and HD95 code.

  python scripts/evaluate_predictions.py --pred nnunet/predictions --name nnunetv2 \
      --gt-root ~/datasets/brats2021 --cases-from-split test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, DATA_ROOT, N_FOLDS, SEED, CV_VAL_RATIO
from src.data import split_cases, kfold_split
from src.metrics import evaluate_segmentation, aggregate, REGIONS
from src.nifti_io import modality_paths, load_seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="folder of <case_id>.nii.gz predictions")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gt-root", default=str(DATA_ROOT), help="folder containing case directories")
    ap.add_argument("--cases-from-split", choices=["test", "all", "cv"], default="test",
                    help="'test' = the README's 45-patient test split; 'cv' = all folds")
    args = ap.parse_args()

    pred_dir = Path(args.pred).expanduser()
    gt_root = Path(args.gt_root).expanduser()
    available = sorted(p.name.replace(".nii.gz", "") for p in pred_dir.glob("*.nii.gz"))
    if args.cases_from_split == "test":
        wanted = set(split_cases()[2])
    elif args.cases_from_split == "cv":
        wanted = {c for f in range(N_FOLDS)
                  for c in kfold_split(fold=f, n_folds=N_FOLDS, seed=SEED, val_ratio=CV_VAL_RATIO)[2]}
    else:
        wanted = set(available)
    cases = [c for c in available if c in wanted]
    if not cases:
        sys.exit("no predictions overlap the requested case set")

    import nibabel as nib
    rows, per_case = [], []
    for c in cases:
        gt_dir = next(gt_root.rglob(c), None)
        if gt_dir is None:
            print(f"  {c}: ground truth not found, skipped")
            continue
        seg = load_seg(modality_paths(gt_dir)["seg"])
        pred = np.asanyarray(nib.load(str(pred_dir / f"{c}.nii.gz")).dataobj).astype(np.uint8)
        res = evaluate_segmentation(pred, seg)      # metrics treat label 4 and 3 alike
        per_case.append(res)
        rows.append({"config": args.name, "case_id": c,
                     **{f"{r}_{m}": res[r][m] for r in REGIONS for m in ("dice", "hd95")}})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"external_{args.name}_per_case.csv", index=False)
    agg = aggregate(per_case)
    with open(RESULTS_DIR / f"external_{args.name}_summary.json", "w") as f:
        json.dump({"name": args.name, "n_cases": len(per_case), "metrics": agg}, f, indent=2)
    print("  ".join(f"{r} Dice {agg[r]['dice_mean']:.4f}  HD95 {agg[r]['hd95_mean']:.2f}"
                    for r in REGIONS))


if __name__ == "__main__":
    main()

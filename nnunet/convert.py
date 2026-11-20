"""Bridge between this repo and the official nnU-Net v2 framework.

  python nnunet/convert.py export        # BraTS -> nnUNet_raw/Dataset137_BraTS2021
  python nnunet/convert.py splits        # our exact folds -> splits_final.json
  python nnunet/convert.py import <dir>  # nnU-Net predictions -> BraTS labels

Follows nnU-Net's own Dataset137 conversion: region-based training on
WT/TC/ET with labels 0 bg, 1 oedema, 2 necrosis, 3 enhancing. That label
order differs from this repo's (1 necrosis, 2 oedema), so `import` maps
predictions back to BraTS labels before they are scored.

The splits file is the point of this script: nnU-Net otherwise draws its own
random folds, and a comparison on different patients would be meaningless.
Folds 0-4 match scripts/run_cv_study.sh; fold 5 reproduces the README's
original train+val / 45-patient test split.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from src.config import MANIFEST_CSV, N_FOLDS, SEED, CV_VAL_RATIO
from src.data import kfold_split, split_cases

DATASET = "Dataset137_BraTS2021"
# nnU-Net channel order, as in its reference BraTS conversion.
CHANNELS = {"0": "T1", "1": "T1ce", "2": "T2", "3": "Flair"}
CHANNEL_FILES = ["t1", "t1ce", "t2", "flair"]


def _env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"set {name} (see nnunet/run_nnunet.sh)")
    return Path(v)


def brats_to_nnunet(seg):
    out = np.zeros_like(seg)
    out[seg == 2] = 1
    out[seg == 1] = 2
    out[seg == 4] = 3
    return out


def nnunet_to_brats(seg):
    out = np.zeros_like(seg)
    out[seg == 1] = 2
    out[seg == 2] = 1
    out[seg == 3] = 4
    return out


def export(args):
    import nibabel as nib
    root = _env("nnUNet_raw") / DATASET
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(MANIFEST_CSV)
    cases = cases[cases["complete"]]
    if args.limit:
        cases = cases.head(args.limit)
    for i, row in enumerate(cases.to_dict("records"), 1):
        cid = row["case_id"]
        for ch, mod in enumerate(CHANNEL_FILES):
            dst = root / "imagesTr" / f"{cid}_{ch:04d}.nii.gz"
            if not dst.exists():
                shutil.copy(row[mod], dst)
        dst = root / "labelsTr" / f"{cid}.nii.gz"
        if not dst.exists():
            nii = nib.load(row["seg"])
            seg = brats_to_nnunet(np.asanyarray(nii.dataobj).astype(np.uint8))
            nib.save(nib.Nifti1Image(seg, nii.affine, nii.header), str(dst))
        if i % 100 == 0:
            print(f"  {i}/{len(cases)}", flush=True)
    meta = {"channel_names": CHANNELS,
            "labels": {"background": 0, "whole tumor": [1, 2, 3],
                       "tumor core": [2, 3], "enhancing tumor": [3]},
            "regions_class_order": [1, 2, 3],
            "numTraining": len(cases), "file_ending": ".nii.gz"}
    (root / "dataset.json").write_text(json.dumps(meta, indent=2))
    print(f"exported {len(cases)} cases to {root}")


def splits(args):
    out = _env("nnUNet_preprocessed") / DATASET
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(pd.read_csv(MANIFEST_CSV).query("complete")["case_id"])
    result = []
    for f in range(N_FOLDS):
        train, val, test = kfold_split(ids, fold=f, n_folds=N_FOLDS, seed=SEED, val_ratio=CV_VAL_RATIO)
        # nnU-Net has no separate model-selection set and trains to the final
        # checkpoint, so our val patients join its training set.
        result.append({"train": train + val, "val": test})
    tr, va, te = split_cases()          # the README's 300-patient split
    result.append({"train": tr + va, "val": te})
    (out / "splits_final.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {len(result)} folds to {out / 'splits_final.json'}")


def import_predictions(args):
    import nibabel as nib
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("*.nii.gz"))
    # Inside an nnU-Net results folder, only fold_*/validation/ holds predictions.
    if any(p.parent.name == "validation" for p in files):
        files = [p for p in files if p.parent.name == "validation"]
    for p in files:
        nii = nib.load(str(p))
        seg = nnunet_to_brats(np.asanyarray(nii.dataobj).astype(np.uint8))
        nib.save(nib.Nifti1Image(seg, nii.affine, nii.header), str(dst / p.name))
    print(f"converted {len(files)} predictions to BraTS labels in {dst}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--limit", type=int); e.set_defaults(fn=export)
    s = sub.add_parser("splits"); s.set_defaults(fn=splits)
    i = sub.add_parser("import"); i.add_argument("src"); i.add_argument("dst", nargs="?",
                                                                      default="nnunet/predictions_brats")
    i.set_defaults(fn=import_predictions)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

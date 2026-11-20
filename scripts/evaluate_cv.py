"""Evaluate the k-fold x multi-seed study.

Every patient is in exactly one test fold, so each seed yields one
out-of-fold prediction per patient over the whole dataset. Reported per config:

  * Dice/HD95 mean over all out-of-fold patients, averaged over seeds
  * seed-to-seed std (the error bar the single-seed study could not give)
  * fold-to-fold std (sensitivity to which patients were held out)

Outputs results/cv_metrics_per_case.csv and results/cv_summary.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import pandas as pd

from src.config import CONFIGS, RESULTS_DIR, N_FOLDS, SEED, CV_VAL_RATIO, get_device
from src.checkpoint import load_model, find_checkpoints, seed_of
from src.data import kfold_split, load_case
from src.metrics import evaluate_segmentation, REGIONS
from src.postprocess import predict_probs, postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="patients per fold (smoke test)")
    ap.add_argument("--out-prefix", default="cv")
    args = ap.parse_args()

    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg_name in args.configs:
        for fold in range(args.n_folds):
            ckpts = find_checkpoints(cfg_name, fold)
            if not ckpts:
                print(f"{cfg_name} fold {fold}: no checkpoints, skipped", flush=True)
                continue
            _, _, test_ids = kfold_split(fold=fold, n_folds=args.n_folds, seed=SEED,
                                         val_ratio=CV_VAL_RATIO)
            test_ids = test_ids[:args.limit] if args.limit else test_ids
            for ck in ckpts:
                model, cfg = load_model(ck, device)
                for case_id in test_ids:
                    img, seg = load_case(case_id)
                    pred = predict_probs(model, img, cfg["normalization"], tta=args.tta,
                                         device=device).argmax(0).astype(np.uint8)
                    if args.postprocess:
                        pred = postprocess(pred)
                    res = evaluate_segmentation(pred, seg)
                    rows.append({"config": cfg_name, "fold": fold, "seed": seed_of(ck),
                                 "case_id": case_id,
                                 **{f"{r}_{m}": res[r][m] for r in REGIONS
                                    for m in ("dice", "hd95")}})
                print(f"{cfg_name} fold {fold} seed {seed_of(ck)}: {len(test_ids)} cases", flush=True)

    if not rows:
        print("nothing evaluated - train with scripts/run_cv_study.sh first")
        return
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{args.out_prefix}_metrics_per_case.csv", index=False)

    summary = {}
    for cfg_name, g in df.groupby("config"):
        entry = {"n_patients": int(g["case_id"].nunique()),
                 "n_seeds": int(g["seed"].nunique()),
                 "n_folds": int(g["fold"].nunique())}
        for r in REGIONS:
            for m in ("dice", "hd95"):
                col = f"{r}_{m}"
                per_seed = g.groupby("seed")[col].mean()
                per_fold = g.groupby("fold")[col].mean()
                entry[col] = {"mean": float(per_seed.mean()),
                              "seed_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
                              "fold_std": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else None,
                              "patient_std": float(g[col].std())}
        summary[cfg_name] = entry
        print(f"{cfg_name}: " + "  ".join(
            f"{r} {entry[f'{r}_dice']['mean']:.4f}"
            + (f"±{entry[f'{r}_dice']['seed_std']:.4f}" if entry[f'{r}_dice']['seed_std'] else "")
            for r in REGIONS))

    with open(RESULTS_DIR / f"{args.out_prefix}_summary.json", "w") as f:
        json.dump({"tta": args.tta, "postprocess": args.postprocess, "configs": summary}, f, indent=2)


if __name__ == "__main__":
    main()

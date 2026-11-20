"""Test-time augmentation, ensembling and ET post-processing: what each adds.

Runs on the original single-split checkpoints (checkpoints/<config>_best.pt),
so it needs no new training. Protocol:

  1. On the VALIDATION patients, pick the ET suppression threshold that
     maximises mean Dice of the full pipeline. The test set plays no part.
  2. On the TEST patients, score every variant:
        <model>                single model, as in the README table
        <model>+tta            8-way mirror TTA
        ensemble               soft vote of all available models
        ensemble+tta
        ...+pp                 each of the above plus ET suppression
  3. Paired Wilcoxon of every variant against the best single model.

Outputs results/postproc_per_case.csv and results/postproc_summary.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import RESULTS_DIR, get_device
from src.checkpoint import load_model, find_checkpoints
from src.data import split_cases, load_case
from src.metrics import evaluate_segmentation, REGIONS
from src.postprocess import predict_probs, ensemble_probs, postprocess

ET_GRID = [0, 100, 250, 500, 1000, 2000]


def case_probs(models, img, device):
    """{model_name: {"plain": probs, "tta": probs}} for one case."""
    out = {}
    for name, (model, cfg) in models.items():
        out[name] = {"plain": predict_probs(model, img, cfg["normalization"], device=device),
                     "tta": predict_probs(model, img, cfg["normalization"], tta=True, device=device)}
    return out


def variants(probs, et_min):
    """Every (variant name -> label map) for one case."""
    preds = {}
    for name, p in probs.items():
        preds[name] = p["plain"].argmax(0).astype(np.uint8)
        preds[f"{name}+tta"] = p["tta"].argmax(0).astype(np.uint8)
    if len(probs) > 1:
        preds["ensemble"] = ensemble_probs([p["plain"] for p in probs.values()]).argmax(0).astype(np.uint8)
        preds["ensemble+tta"] = ensemble_probs([p["tta"] for p in probs.values()]).argmax(0).astype(np.uint8)
    for k in list(preds):
        preds[f"{k}+pp"] = postprocess(preds[k], et_min_total=et_min)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=["baseline_unet", "nnunet_style", "vnet"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=20)
    args = ap.parse_args()

    device = get_device()
    models = {}
    for c in args.configs:
        ck = find_checkpoints(c)
        if ck:
            models[c] = load_model(ck[0], device)
    if not models:
        sys.exit("no checkpoints found in checkpoints/ - train first")
    print(f"models: {list(models)}", flush=True)

    _, val_ids, test_ids = split_cases()
    val_ids = val_ids[:args.val_limit]
    test_ids = test_ids[:args.limit] if args.limit else test_ids
    final = "ensemble+tta" if len(models) > 1 else f"{next(iter(models))}+tta"

    # 1. Tune the ET threshold on validation patients only.
    val_probs = [(case_probs(models, img, device), seg)
                 for img, seg in (load_case(c) for c in val_ids)]
    tuning = {}
    for t in ET_GRID:
        scores = []
        for probs, seg in val_probs:
            pred = variants(probs, t)[f"{final}+pp"]
            r = evaluate_segmentation(pred, seg, compute_hd95=False)
            scores.append(np.mean([r[k]["dice"] for k in REGIONS]))
        tuning[t] = float(np.mean(scores))
        print(f"  val ET threshold {t:>5}: mean Dice {tuning[t]:.4f}", flush=True)
    et_min = max(tuning, key=tuning.get)
    print(f"selected ET threshold: {et_min} voxels (on {len(val_ids)} val patients)", flush=True)

    # 2. Score every variant on the test set.
    rows = []
    for i, case_id in enumerate(test_ids, 1):
        img, seg = load_case(case_id)
        for name, pred in variants(case_probs(models, img, device), et_min).items():
            res = evaluate_segmentation(pred, seg)
            rows.append({"variant": name, "case_id": case_id,
                         **{f"{r}_{m}": res[r][m] for r in REGIONS for m in ("dice", "hd95")}})
        print(f"  test {i}/{len(test_ids)}", flush=True)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "postproc_per_case.csv", index=False)

    # 3. Paired comparison against the best single model (by mean test Dice of the plain models).
    means = df.groupby("variant")[[f"{r}_dice" for r in REGIONS]].mean()
    singles = [m for m in models if m in means.index]
    ref = means.loc[singles].mean(axis=1).idxmax()
    piv = {r: df.pivot(index="case_id", columns="variant", values=f"{r}_dice") for r in REGIONS}
    summary = {"et_threshold": et_min, "et_tuning_val": tuning, "reference": ref,
               "n_test": len(test_ids), "variants": {}}
    for v in means.index:
        entry = {}
        for r in REGIONS:
            d = (piv[r][v] - piv[r][ref]).dropna()
            p = float(wilcoxon(d).pvalue) if v != ref and (d != 0).any() else None
            entry[r] = {"dice": float(piv[r][v].mean()), "delta_vs_ref": float(d.mean()),
                        "wilcoxon_p": p, "hd95": float(df[df.variant == v][f"{r}_hd95"].mean())}
        summary["variants"][v] = entry
        print(f"{v:<28} " + "  ".join(
            f"{r} {entry[r]['dice']:.4f} ({entry[r]['delta_vs_ref']:+.4f})" for r in REGIONS))
    with open(RESULTS_DIR / "postproc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

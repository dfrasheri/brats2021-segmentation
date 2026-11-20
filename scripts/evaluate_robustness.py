"""Robustness: missing MRI sequences, external data, and whether uncertainty
flags the cases that go wrong.

  python scripts/evaluate_robustness.py --checkpoint checkpoints/nnunet_style_best.pt
  python scripts/evaluate_robustness.py --checkpoint ... --external-dir ~/datasets/brats2023

For each test patient (or each external case) it runs 8-way TTA and records:
  * Dice per region with all four sequences
  * Dice with each sequence removed (zero-filled, and mean-imputed)
  * mean predictive entropy and mutual information over the predicted tumour
and reports the Spearman correlation between case uncertainty and Dice. A
useful uncertainty estimate correlates negatively: unsure where it is wrong.

With --save-maps N it writes uncertainty-map figures for the N least and most
certain cases to results/figures/uncertainty_*.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import RESULTS_DIR, MODALITIES, get_device
from src.checkpoint import load_model
from src.data import split_cases, load_case
from src.metrics import evaluate_segmentation, REGIONS
from src.nifti_io import modality_paths, load_volumes, load_seg, crop
from src.postprocess import ALL_FLIPS
from src.inference import sliding_window_probs
from src.preprocessing import normalize_volume
from src.uncertainty import predictive_entropy, mutual_information, case_uncertainty, drop_modality


def tta_samples(model, img, normalization, device):
    """[S,K,H,W,D]: one softmax map per flip, each flipped back."""
    x = normalize_volume(img, method=normalization)
    out = []
    for axes in ALL_FLIPS:
        sp = tuple(a + 1 for a in axes)
        xi = np.ascontiguousarray(np.flip(x, axis=sp)) if axes else x
        p = sliding_window_probs(model, xi, device=device)
        out.append(np.flip(p, axis=sp) if axes else p)
    return np.stack(out)


def external_cases(root, limit):
    """Yield (case_id, img, seg) from a folder of BraTS-format case directories."""
    dirs = sorted(d for d in Path(root).expanduser().iterdir() if d.is_dir())
    for d in dirs[:limit] if limit else dirs:
        paths = modality_paths(d)
        if "seg" not in paths:
            continue
        img, _, _ = load_volumes(paths)
        seg = load_seg(paths["seg"])
        img, bb = crop(img)
        yield d.name, img, seg[bb]


def save_maps(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(records), 4, figsize=(12, 3 * len(records)), squeeze=False)
    for row, (cid, img, seg, pred, ent, dice) in zip(axes, records):
        z = int(np.argmax((seg > 0).sum(axis=(0, 1)))) if seg.any() else seg.shape[2] // 2
        for ax, data, title, cmap in zip(
                row, [img[0][..., z], seg[..., z], pred[..., z], ent[..., z]],
                [f"{cid} FLAIR", "ground truth", f"prediction (WT Dice {dice:.2f})", "entropy"],
                ["gray", "viridis", "viridis", "magma"]):
            ax.imshow(np.rot90(data), cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--external-dir", default=None, help="BraTS 2023/2024 style case folders")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-dropout-test", action="store_true", help="skip missing-modality runs")
    ap.add_argument("--save-maps", type=int, default=2)
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_model(args.checkpoint, device)
    norm = cfg["normalization"]
    tag = "external" if args.external_dir else "internal"

    if args.external_dir:
        cases = external_cases(args.external_dir, args.limit)
    else:
        _, _, test_ids = split_cases()
        test_ids = test_ids[:args.limit] if args.limit else test_ids
        cases = ((c, *load_case(c)) for c in test_ids)

    rows, kept = [], []
    for case_id, img, seg in cases:
        samples = tta_samples(model, img, norm, device)
        pred = samples.mean(0).argmax(0).astype(np.uint8)
        ent, mi = predictive_entropy(samples), mutual_information(samples)
        res = evaluate_segmentation(pred, seg)
        row = {"case_id": case_id, "condition": "all",
               "entropy": case_uncertainty(ent, pred), "mutual_info": case_uncertainty(mi, pred),
               **{f"{r}_dice": res[r]["dice"] for r in REGIONS}}
        rows.append(row)
        kept.append((case_id, img, seg, pred, ent, res["WT"]["dice"]))
        kept = sorted(kept, key=lambda k: k[-1])
        kept = kept[:args.save_maps] + kept[-args.save_maps:] if len(kept) > 2 * args.save_maps else kept

        if not args.no_dropout_test:
            for m in MODALITIES:
                for fill in ("zero", "mean"):
                    p = sliding_window_probs(model, normalize_volume(drop_modality(img, m, fill), norm),
                                             device=device).argmax(0).astype(np.uint8)
                    r2 = evaluate_segmentation(p, seg, compute_hd95=False)
                    rows.append({"case_id": case_id, "condition": f"no_{m}_{fill}",
                                 **{f"{r}_dice": r2[r]["dice"] for r in REGIONS}})
        print(f"  {case_id}: WT {res['WT']['dice']:.3f}  entropy {row['entropy']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"robustness_{tag}_per_case.csv", index=False)

    full = df[df.condition == "all"]
    summary = {"checkpoint": str(args.checkpoint), "n_cases": len(full),
               "conditions": df.groupby("condition")[[f"{r}_dice" for r in REGIONS]].mean().to_dict("index"),
               "uncertainty_vs_dice_spearman": {}}
    for unc in ("entropy", "mutual_info"):
        for r in REGIONS:
            if len(full) > 2:
                rho, p = spearmanr(full[unc], full[f"{r}_dice"])
                summary["uncertainty_vs_dice_spearman"][f"{unc}_{r}"] = {"rho": float(rho), "p": float(p)}
    with open(RESULTS_DIR / f"robustness_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["conditions"], indent=1))
    print(json.dumps(summary["uncertainty_vs_dice_spearman"], indent=1))

    if kept:
        out = RESULTS_DIR / "figures" / f"uncertainty_{tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_maps(kept, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

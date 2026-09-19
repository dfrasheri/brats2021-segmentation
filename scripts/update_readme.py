"""Inject measured results into README.md at the RESULTS_TABLE marker.

Keeps the README honest by construction: the table is generated from
results/summary.json, so it cannot drift from what was actually measured.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, LOG_DIR, PROJECT_ROOT

NAMES = {"nnunet_style": "nnU-Net-style U-Net", "vnet": "V-Net", "baseline_unet": "Baseline U-Net"}
ORDER = ["nnunet_style", "vnet", "baseline_unet"]
MARKER = "<!-- RESULTS_TABLE -->"


def build():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    order = [c for c in ORDER if c in summary] + [c for c in summary if c not in ORDER]
    n_test = summary[order[0]]["n_test_patients"]

    out = []
    out.append(f"Held-out test set: **{n_test} patients**, never seen during training or model "
               "selection. Dice is reported as mean ± standard deviation across those patients.\n")
    out.append("| Model | Params | Dice WT | Dice TC | Dice ET | HD95 WT (vox) | Inference |")
    out.append("|---|---|---|---|---|---|---|")
    for c in order:
        s, m = summary[c], summary[c]["metrics"]
        out.append("| {} | {:.1f}M | **{:.3f}** ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} "
                   "| {:.1f} | {:.2f}s |".format(
                       NAMES.get(c, c), s["parameters"] / 1e6,
                       m["WT"]["dice_mean"], m["WT"]["dice_std"],
                       m["TC"]["dice_mean"], m["TC"]["dice_std"],
                       m["ET"]["dice_mean"], m["ET"]["dice_std"],
                       m["WT"]["hd95_mean"], s["inference_seconds_mean"]))

    # Bundled ablation: same architecture, different preprocessing/loss/augmentation.
    if {"baseline_unet", "nnunet_style"} <= set(summary):
        b = summary["baseline_unet"]["metrics"]
        n = summary["nnunet_style"]["metrics"]
        out.append("\n### What the nnU-Net-style configuration buys\n")
        out.append("Same architecture and same iteration budget as the baseline. The only "
                   "differences are per-modality z-scoring, Dice+CE instead of plain CE, "
                   "augmentation, and deep supervision — changed together, so this is the "
                   "effect of the bundle, not of any single factor.\n")
        out.append("| Region | Baseline U-Net | nnU-Net-style | Change |")
        out.append("|---|---|---|---|")
        for r in ("WT", "TC", "ET"):
            d = n[r]["dice_mean"] - b[r]["dice_mean"]
            out.append("| {} | {:.3f} | {:.3f} | {:+.3f} |".format(
                r, b[r]["dice_mean"], n[r]["dice_mean"], d))

    out.append("\n### Training cost\n")
    out.append("| Config | Iterations | Wall clock | Best val Dice |")
    out.append("|---|---|---|---|")
    total = 0.0
    for c in order:
        f = LOG_DIR / f"{c}_history.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        total += h["train_minutes"]
        out.append("| {} | {} | {:.1f} min | {:.3f} |".format(
            NAMES.get(c, c), h["hyperparameters"]["iterations"],
            h["train_minutes"], h["best_val_mean_dice"]))
    out.append(f"\n**Total training time: {total:.0f} minutes** on one RTX 4070 Laptop.\n")

    figs = sorted((RESULTS_DIR / "figures").glob("*.png")) if (RESULTS_DIR / "figures").exists() else []
    if figs:
        out.append("### Figures\n")
        for p in figs:
            title = p.stem.replace("_", " ").title()
            out.append(f"**{title}**\n")
            out.append(f"![{title}](results/figures/{p.name})\n")

    return "\n".join(out)


def main():
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if MARKER not in text:
        print("marker not found; README already filled?")
        return
    readme.write_text(text.replace(MARKER, build()), encoding="utf-8")
    print("README results section updated")


if __name__ == "__main__":
    main()

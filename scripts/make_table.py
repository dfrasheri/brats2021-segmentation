"""Render results/summary.json as the markdown table that goes into the README."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, LOG_DIR

NAMES = {"nnunet_style": "nnU-Net-style U-Net", "vnet": "V-Net",
         "baseline_unet": "Baseline U-Net"}
ORDER = ["nnunet_style", "vnet", "baseline_unet"]


def main():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    order = [c for c in ORDER if c in summary] + [c for c in summary if c not in ORDER]

    print("| Model | Params | Dice WT | Dice TC | Dice ET | HD95 WT | Inference |")
    print("|---|---|---|---|---|---|---|")
    for c in order:
        s, m = summary[c], summary[c]["metrics"]
        print("| {} | {:.1f}M | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} "
              "| {:.1f} | {:.2f}s |".format(
                  NAMES.get(c, c), s["parameters"] / 1e6,
                  m["WT"]["dice_mean"], m["WT"]["dice_std"],
                  m["TC"]["dice_mean"], m["TC"]["dice_std"],
                  m["ET"]["dice_mean"], m["ET"]["dice_std"],
                  m["WT"]["hd95_mean"], s["inference_seconds_mean"]))

    print("\n\n### Training cost\n")
    print("| Config | Iterations | Wall clock | Best val Dice |")
    print("|---|---|---|---|")
    for c in order:
        f = LOG_DIR / f"{c}_history.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        print("| {} | {} | {:.1f} min | {:.3f} |".format(
            NAMES.get(c, c), h["hyperparameters"]["iterations"],
            h["train_minutes"], h["best_val_mean_dice"]))


if __name__ == "__main__":
    main()

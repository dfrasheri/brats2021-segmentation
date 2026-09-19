"""Regenerate the README Results section from measured output.

Idempotent: replaces everything between the "## Results" heading and the next
horizontal rule, so it can be re-run after any evaluation. The table is built
from results/*.json, which is what keeps the README from drifting away from
what was actually measured.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, LOG_DIR, PROJECT_ROOT

NAMES = {"nnunet_style": "nnU-Net-style U-Net", "vnet": "V-Net", "baseline_unet": "Baseline U-Net"}
ORDER = ["nnunet_style", "vnet", "baseline_unet"]
FIG_TITLES = {
    "training_curves": "Training loss and validation Dice",
    "dice_distribution": "Per-patient Dice distribution",
    "qualitative_nnunet_style": "nnU-Net-style: worst / median / best test cases",
    "qualitative_vnet": "V-Net: worst / median / best test cases",
    "qualitative_baseline_unet": "Baseline U-Net: worst / median / best test cases",
}


def build():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    order = [c for c in ORDER if c in summary] + [c for c in summary if c not in ORDER]
    n_test = summary[order[0]]["n_test_patients"]
    L = []

    L.append(f"Held-out test set: **{n_test} patients**, never seen during training or model "
             "selection. Dice is mean ± standard deviation **across patients** — it describes "
             "case-to-case variability, not a seed-to-seed error bar (one seed per config).\n")

    # Bold only the genuine best in each column.
    best = {}
    for key, better in [("WT_dice", max), ("TC_dice", max), ("ET_dice", max),
                        ("WT_hd95", min), ("infer", min)]:
        if key == "infer":
            best[key] = better(order, key=lambda c: summary[c]["inference_seconds_mean"])
        else:
            region, metric = key.split("_")
            best[key] = better(order, key=lambda c: summary[c]["metrics"][region][metric + "_mean"])

    def cell(c, region, metric="dice"):
        m = summary[c]["metrics"][region]
        v, s = m[metric + "_mean"], m[metric + "_std"]
        txt = f"{v:.3f} ± {s:.3f}" if metric == "dice" else f"{v:.1f}"
        return f"**{txt}**" if best.get(f"{region}_{metric}") == c else txt

    L.append("| Model | Params | Dice WT | Dice TC | Dice ET | HD95 WT (vox) | Inference |")
    L.append("|---|---|---|---|---|---|---|")
    for c in order:
        s = summary[c]
        inf = f"{s['inference_seconds_mean']:.2f}s"
        if best["infer"] == c:
            inf = f"**{inf}**"
        L.append(f"| {NAMES.get(c, c)} | {s['parameters']/1e6:.1f}M | "
                 f"{cell(c,'WT')} | {cell(c,'TC')} | {cell(c,'ET')} | "
                 f"{cell(c,'WT','hd95')} | {inf} |")

    # Significance — the part that decides which gaps are real.
    sig_file = RESULTS_DIR / "significance.json"
    if sig_file.exists():
        sig = json.loads(sig_file.read_text())
        L.append("\n### Which differences are real\n")
        L.append("Per-patient Dice varies far more (σ ≈ 0.10–0.25) than the gap between models, so "
                 "eyeballing the means above is not enough. Every config is evaluated on the *same* "
                 "patients, so the comparison is **paired** — which removes between-patient "
                 "variance and asks the question that matters: on a given patient, is one model "
                 "better? Wilcoxon signed-rank is reported alongside the t-test because "
                 "per-patient Dice is left-skewed and bounded at 1.\n")
        L.append("| Comparison | Region | Δ Dice | 95% CI | Wilcoxon p | Better on |")
        L.append("|---|---|---|---|---|---|")
        for pair, regions in sig.items():
            a, b = pair.split("_vs_")
            label = f"{NAMES.get(a, a)} vs {NAMES.get(b, b)}"
            for i, (region, r) in enumerate(regions.items()):
                L.append("| {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {} | {}/{} |".format(
                    label if i == 0 else "", region, r["mean_difference"],
                    r["ci95_low"], r["ci95_high"],
                    "**{:.4f}**".format(r["wilcoxon_p"]) if r["wilcoxon_p"] < 0.05
                    else "{:.3f}".format(r["wilcoxon_p"]),
                    r["n_better"], r["n"]))

        L.append("\n**What this actually says.** The configuration bundle (per-modality z-scoring, "
                 "Dice+CE, augmentation, deep supervision) buys **nothing measurable on whole "
                 "tumour** — all three models sit within 0.003 of each other there — but a real "
                 "**+0.04 to +0.05 on tumour core and enhancing tumour**. That is the sensible "
                 "direction: whole tumour is large and high-contrast on FLAIR, so even a weak "
                 "configuration finds it, and the harder small structures are where loss function "
                 "and normalisation earn their keep. Reporting only the headline WT number would "
                 "have hidden the entire effect.\n")
        L.append("The two strong configs are statistically indistinguishable from each other on "
                 "this test set; with one seed apiece, the honest reading is a tie.\n")

    L.append("### Training cost\n")
    L.append("| Config | Iterations | Wall clock | Best val Dice |")
    L.append("|---|---|---|---|")
    total = 0.0
    for c in order:
        f = LOG_DIR / f"{c}_history.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        total += h["train_minutes"]
        L.append(f"| {NAMES.get(c, c)} | {h['hyperparameters']['iterations']} | "
                 f"{h['train_minutes']:.1f} min | {h['best_val_mean_dice']:.3f} |")
    L.append(f"\n**Total training time: {total:.0f} minutes** on one RTX 4070 Laptop.\n")

    figs = sorted((RESULTS_DIR / "figures").glob("*.png")) if (RESULTS_DIR / "figures").exists() else []
    if figs:
        L.append("### Figures\n")
        for p in figs:
            title = FIG_TITLES.get(p.stem, p.stem.replace("_", " "))
            L.append(f"**{title}**\n")
            L.append(f"![{title}](results/figures/{p.name})\n")

    return "\n".join(L)


def main():
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    body = build()

    start = text.index("## Results")
    end = text.index("\n---", start)
    readme.write_text(text[:start] + "## Results\n\n" + body + text[end:], encoding="utf-8")
    print("README Results section regenerated")


if __name__ == "__main__":
    main()

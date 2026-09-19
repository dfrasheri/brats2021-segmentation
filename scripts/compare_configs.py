"""Paired statistical comparison between configurations.

Every configuration is evaluated on the SAME held-out patients, so the
comparison is paired. That matters: per-patient Dice varies far more
(std ~0.10-0.25) than the difference between models does, so comparing two
independent means would drown a real effect in between-patient variance.
Pairing removes that variance and asks the right question - on a given
patient, is one model better than the other?

Reports mean difference, a 95% confidence interval, a paired t-test, and a
Wilcoxon signed-rank test (which does not assume normality; per-patient Dice
is left-skewed and bounded at 1, so the t-test alone would be optimistic).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from scipy import stats

from src.config import RESULTS_DIR
from src.metrics import REGIONS

BASE = "baseline_unet"


def compare(df, a_name, b_name):
    out = {}
    for region in REGIONS:
        a = df[df.config == a_name].set_index("case_id")[f"{region}_dice"]
        b = df[df.config == b_name].set_index("case_id")[f"{region}_dice"]
        common = a.index.intersection(b.index)
        if len(common) < 3:
            continue
        a, b = a[common], b[common]
        d = (a - b).values

        se = d.std(ddof=1) / np.sqrt(len(d))
        t_p = stats.ttest_rel(a, b).pvalue
        # Wilcoxon is undefined when every difference is zero.
        w_p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0

        out[region] = {
            "n": int(len(d)),
            "mean_difference": float(d.mean()),
            "ci95_low": float(d.mean() - 1.96 * se),
            "ci95_high": float(d.mean() + 1.96 * se),
            "paired_t_p": float(t_p),
            "wilcoxon_p": float(w_p),
            "n_better": int((d > 0).sum()),
        }
    return out


def main():
    df = pd.read_csv(RESULTS_DIR / "metrics_per_case.csv")
    configs = [c for c in df.config.unique() if c != BASE]

    results = {}
    for c in configs:
        results[f"{c}_vs_{BASE}"] = compare(df, c, BASE)

    with open(RESULTS_DIR / "significance.json", "w") as f:
        json.dump(results, f, indent=2)

    for pair, regions in results.items():
        print(f"\n{pair}  (paired, same patients)")
        for region, r in regions.items():
            verdict = "significant" if r["wilcoxon_p"] < 0.05 else "not significant"
            print(f"  {region}: {r['mean_difference']:+.4f} "
                  f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  "
                  f"t p={r['paired_t_p']:.4f}  wilcoxon p={r['wilcoxon_p']:.4f}  "
                  f"better on {r['n_better']}/{r['n']}  -> {verdict}")
    print(f"\nWrote {RESULTS_DIR / 'significance.json'}")


if __name__ == "__main__":
    main()

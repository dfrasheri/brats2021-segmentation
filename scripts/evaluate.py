"""Evaluate trained checkpoints on the held-out test patients.

Reports per-region Dice / HD95 / sensitivity / specificity as mean +/- std
ACROSS PATIENTS for a single training run. That spread describes case-to-case
variability, not seed-to-seed variability - we do not have the budget for
multiple seeds, and the README says so rather than implying error bars we
did not measure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from src.config import CONFIGS, CHECKPOINT_DIR, RESULTS_DIR, NUM_CLASSES, get_device
from src.data import split_cases, load_case
from src.models import create_model, count_parameters
from src.metrics import evaluate_segmentation, aggregate, REGIONS
from src.inference import predict_case


def load_checkpoint(config_name, device):
    ckpt_path = CHECKPOINT_DIR / f"{config_name}_best.pt"
    if not ckpt_path.exists():
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("settings", CONFIGS[config_name])
    model = create_model(cfg["model"], base_channels=cfg["base_channels"],
                         deep_supervision=cfg["deep_supervision"],
                         num_classes=NUM_CLASSES, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--limit", type=int, default=None, help="evaluate only N test patients")
    args = ap.parse_args()

    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, _, test_ids = split_cases()
    if args.limit:
        test_ids = test_ids[:args.limit]
    print(f"Evaluating on {len(test_ids)} held-out test patients, device={device}", flush=True)

    rows, summary = [], {}
    for config_name in args.configs:
        model, cfg = load_checkpoint(config_name, device)
        if model is None:
            print(f"  {config_name}: no checkpoint, skipped", flush=True)
            continue

        print(f"\n{config_name}: {cfg['description']}", flush=True)
        per_case, times = [], []
        for i, case_id in enumerate(test_ids, 1):
            img, seg = load_case(case_id)
            t0 = time.time()
            pred = predict_case(model, img, normalization=cfg["normalization"], device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)

            res = evaluate_segmentation(pred, seg)
            per_case.append(res)
            rows.append({"config": config_name, "case_id": case_id,
                         **{f"{r}_{m}": res[r][m] for r in REGIONS
                            for m in ("dice", "hd95", "sensitivity", "specificity")}})
            if i % 10 == 0:
                print(f"  {i}/{len(test_ids)}", flush=True)

        agg = aggregate(per_case)
        summary[config_name] = {
            "description": cfg["description"],
            "parameters": count_parameters(model),
            "n_test_patients": len(test_ids),
            "inference_seconds_mean": float(np.mean(times)),
            "inference_seconds_std": float(np.std(times)),
            "metrics": agg,
        }
        print(f"  Dice  WT {agg['WT']['dice_mean']:.4f}+/-{agg['WT']['dice_std']:.4f}  "
              f"TC {agg['TC']['dice_mean']:.4f}+/-{agg['TC']['dice_std']:.4f}  "
              f"ET {agg['ET']['dice_mean']:.4f}+/-{agg['ET']['dice_std']:.4f}", flush=True)
        print(f"  HD95  WT {agg['WT']['hd95_mean']:.2f}  TC {agg['TC']['hd95_mean']:.2f}  "
              f"ET {agg['ET']['hd95_mean']:.2f}   inference {np.mean(times):.2f}s/patient", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "metrics_per_case.csv", index=False)
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {RESULTS_DIR / 'metrics_per_case.csv'} and summary.json", flush=True)


if __name__ == "__main__":
    main()

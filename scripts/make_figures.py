"""Qualitative figures: best and worst test cases, plus training curves.

The worst cases matter more than the best ones. A repo that only shows its wins
is not evidence of anything; showing where the model breaks, and being able to
say why, is the part that distinguishes understanding from luck.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.config import RESULTS_DIR, LOG_DIR, CONFIGS, NUM_CLASSES, get_device, CHECKPOINT_DIR
from src.data import load_case
from src.models import create_model
from src.inference import predict_case

FIG_DIR = RESULTS_DIR / "figures"
LABEL_CMAP = ListedColormap(["#e41a1c", "#4daf4a", "#ffd92f"])
LEGEND = [Patch(color="#e41a1c", label="1 necrotic core"),
          Patch(color="#4daf4a", label="2 oedema"),
          Patch(color="#ffd92f", label="3 enhancing")]


def busiest_slice(seg):
    per_slice = (seg > 0).sum(axis=(0, 1))
    return int(per_slice.argmax()) if per_slice.max() > 0 else seg.shape[2] // 2


def overlay(ax, base, labels, z, title):
    ax.imshow(np.rot90(base[:, :, z]), cmap="gray")
    masked = np.ma.masked_equal(np.rot90(labels[:, :, z]), 0)
    ax.imshow(masked, cmap=LABEL_CMAP, vmin=1, vmax=3, alpha=0.6, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def qualitative(config_name, device):
    df = pd.read_csv(RESULTS_DIR / "metrics_per_case.csv")
    df = df[df["config"] == config_name].sort_values("WT_dice")
    if df.empty:
        return
    picks = [("worst", df.iloc[0]), ("median", df.iloc[len(df) // 2]), ("best", df.iloc[-1])]

    ckpt = torch.load(CHECKPOINT_DIR / f"{config_name}_best.pt", map_location=device,
                      weights_only=False)
    cfg = ckpt["settings"]
    model = create_model(cfg["model"], base_channels=cfg["base_channels"],
                         deep_supervision=cfg["deep_supervision"],
                         num_classes=NUM_CLASSES, device=device)
    model.load_state_dict(ckpt["model"])

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for row, (kind, rec) in enumerate(picks):
        img, seg = load_case(rec["case_id"])
        pred = predict_case(model, img, normalization=cfg["normalization"], device=device)
        z = busiest_slice(seg)
        flair = img[0].astype(np.float32)

        axes[row, 0].imshow(np.rot90(flair[:, :, z]), cmap="gray")
        axes[row, 0].set_title(f"{kind}: {rec['case_id']}\nFLAIR", fontsize=10)
        axes[row, 0].axis("off")
        overlay(axes[row, 1], flair, seg, z, "ground truth")
        overlay(axes[row, 2], flair, pred, z, f"prediction (WT Dice {rec['WT_dice']:.3f})")

    axes[0, 2].legend(handles=LEGEND, loc="lower right", fontsize=7)
    fig.suptitle(f"{config_name}: worst / median / best test cases by whole-tumour Dice",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / f"qualitative_{config_name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, flush=True)


def curves():
    histories = {}
    for f in sorted(LOG_DIR.glob("*_history.json")):
        h = json.loads(f.read_text())
        histories[h["config"]] = h
    if not histories:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in histories.items():
        if h["loss_curve"]:
            ax1.plot([p["iter"] for p in h["loss_curve"]],
                     [p["loss"] for p in h["loss_curve"]], label=name)
        if h["val_curve"]:
            ax2.plot([p["iter"] for p in h["val_curve"]],
                     [p["mean"] for p in h["val_curve"]], marker="o", label=name)
    ax1.set(xlabel="iteration", ylabel="training loss", title="Training loss")
    ax2.set(xlabel="iteration", ylabel="mean Dice (WT/TC/ET)",
            title="Validation Dice (full-volume, sliding window)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "training_curves.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, flush=True)


def per_region_box():
    df = pd.read_csv(RESULTS_DIR / "metrics_per_case.csv")
    configs = list(df["config"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, region in zip(axes, ["WT", "TC", "ET"]):
        data = [df[df["config"] == c][f"{region}_dice"].values for c in configs]
        ax.boxplot(data, labels=configs, showmeans=True)
        ax.set_title(f"{region} Dice")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    axes[0].set_ylabel("Dice")
    fig.suptitle("Per-patient Dice distribution on held-out test set", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "dice_distribution.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    curves()
    if (RESULTS_DIR / "metrics_per_case.csv").exists():
        per_region_box()
        for name in CONFIGS:
            if (CHECKPOINT_DIR / f"{name}_best.pt").exists():
                qualitative(name, device)

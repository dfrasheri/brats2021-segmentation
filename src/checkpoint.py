"""Load a trained checkpoint into a ready-to-run model."""
from pathlib import Path

import torch

from .config import CONFIGS, CHECKPOINT_DIR, NUM_CLASSES
from .models import create_model


def load_model(path, device="cpu"):
    """Returns (model in eval mode, settings dict) from a checkpoint written by train.py."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg = ckpt.get("settings") or CONFIGS[ckpt["config"]]
    model = create_model(cfg["model"], base_channels=cfg["base_channels"],
                         deep_supervision=cfg["deep_supervision"],
                         num_classes=NUM_CLASSES, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def find_checkpoints(config_name, fold=None):
    """Checkpoints for one config: the single-split one, or every seed of one fold."""
    if fold is None:
        p = CHECKPOINT_DIR / f"{config_name}_best.pt"
        return [p] if p.exists() else []
    return sorted(CHECKPOINT_DIR.glob(f"{config_name}_f{fold}_s*_best.pt"))


def seed_of(path):
    """Parse the seed from a `<config>_f<fold>_s<seed>_best.pt` name."""
    stem = Path(path).stem
    return int(stem.split("_s")[-1].split("_")[0])

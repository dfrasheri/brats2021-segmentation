"""Central configuration. Every tunable lives here so runs are reproducible."""
from pathlib import Path

# ---- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path.home() / "datasets" / "brats2021"
MANIFEST_CSV = DATA_ROOT / "manifest_task1_training.csv"
CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- Data --------------------------------------------------------------
FULL_SHAPE = (240, 240, 155)
MODALITIES = ["flair", "t1", "t1ce", "t2"]
NUM_MODALITIES = 4
NUM_CLASSES = 4          # 0 background, 1 necrotic, 2 edema, 3 enhancing (BraTS label 4 remapped)

# Subset size. Patch-based training is iteration-bound, not epoch-bound, so a
# larger cache would not improve a short run - it would only cost disk and time.
SUBSET_SIZE = 300
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# ---- Training ----------------------------------------------------------
SEED = 42
PATCH_SIZE = (128, 128, 128)   # must be divisible by 2**depth
BATCH_SIZE = 2                 # 8.6 GB VRAM ceiling with AMP at 128^3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 12.0
WARMUP_ITERS = 100
VAL_EVERY = 250
VAL_PATIENTS = 10              # subset of val split used for in-training monitoring
FG_PATCH_PROB = 0.5            # fraction of patches centred on tumour

# ---- Inference ---------------------------------------------------------
SW_OVERLAP = 0.5               # sliding-window overlap


def get_device():
    """Resolved at call time, not import time, so `import src.config` never fails."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Per-configuration definitions. These drive both the model comparison and the
# bundled preprocessing/loss/augmentation ablation.
CONFIGS = {
    "baseline_unet": {
        "model": "unet",
        "normalization": "minmax",
        "loss": "ce",
        "augment": False,
        "deep_supervision": False,
        "base_channels": 16,
        "description": "Plain 3D U-Net, min-max scaling, cross-entropy only, no augmentation.",
    },
    "nnunet_style": {
        "model": "unet",
        "normalization": "zscore",
        "loss": "dice_ce",
        "augment": True,
        "deep_supervision": True,
        "base_channels": 16,
        "description": "nnU-Net design principles: per-modality z-score, Dice+CE, deep supervision, augmentation.",
    },
    "vnet": {
        "model": "vnet",
        "normalization": "zscore",
        "loss": "dice_ce",
        "augment": True,
        "deep_supervision": False,
        "base_channels": 16,
        "description": "V-Net style residual encoder-decoder, same preprocessing as nnunet_style.",
    },
}

"""Patch-based dataset over the preprocessed .npy cache.

Full 240x240x155x4 volumes do not fit in 8 GB of VRAM, so training samples
128^3 patches. Sampling is biased toward tumour: purely random 128^3 patches in
a mostly-background volume would spend a short run learning to predict zeros.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    CACHE_DIR, MANIFEST_CSV, PATCH_SIZE, SEED, FG_PATCH_PROB,
    TRAIN_RATIO, VAL_RATIO,
)
from .preprocessing import normalize_volume, augment_patch


def cached_case_ids():
    """Case ids present in the cache, in deterministic order."""
    csv = CACHE_DIR / "cached_cases.csv"
    if csv.exists():
        return sorted(pd.read_csv(csv)["case_id"].tolist())
    return sorted(p.name.replace("_img.npy", "") for p in CACHE_DIR.glob("*_img.npy"))


def split_cases(case_ids=None, seed=SEED):
    """Deterministic patient-level split.

    Splitting by patient (never by slice or patch) is what keeps the test score
    honest: two patches from one brain are near-duplicates, so a slice-level
    split leaks the test set into training.
    """
    ids = sorted(case_ids if case_ids is not None else cached_case_ids())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    ids = [ids[i] for i in perm]

    n = len(ids)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def load_case(case_id):
    """Return (img float32 [4,H,W,D], seg uint8 [H,W,D]) straight from cache."""
    img = np.load(CACHE_DIR / f"{case_id}_img.npy").astype(np.float32)
    seg = np.load(CACHE_DIR / f"{case_id}_seg.npy")
    return img, seg


def _random_patch(img, seg, patch_size, rng, fg_prob):
    """Crop one patch, padding first if the volume is smaller than the patch."""
    _, H, W, D = img.shape
    ph, pw, pd = patch_size

    pad = [max(0, ph - H), max(0, pw - W), max(0, pd - D)]
    if any(pad):
        img = np.pad(img, ((0, 0), (0, pad[0]), (0, pad[1]), (0, pad[2])))
        seg = np.pad(seg, ((0, pad[0]), (0, pad[1]), (0, pad[2])))
        _, H, W, D = img.shape

    centre = None
    if rng.rand() < fg_prob:
        fg = np.argwhere(seg > 0)
        if len(fg):
            centre = fg[rng.randint(len(fg))]

    if centre is None:
        start = [rng.randint(0, max(1, H - ph + 1)),
                 rng.randint(0, max(1, W - pw + 1)),
                 rng.randint(0, max(1, D - pd + 1))]
    else:
        start = [int(np.clip(centre[i] - p // 2, 0, max(0, s - p)))
                 for i, (p, s) in enumerate(zip((ph, pw, pd), (H, W, D)))]

    sl = tuple(slice(s, s + p) for s, p in zip(start, (ph, pw, pd)))
    return img[(slice(None),) + sl], seg[sl]


class PatchDataset(Dataset):
    """Yields random patches. Length is an iteration budget, not a patient count."""

    def __init__(self, case_ids, iterations, normalization="zscore",
                 augment=False, patch_size=PATCH_SIZE, seed=SEED, fg_prob=FG_PATCH_PROB):
        self.case_ids = list(case_ids)
        self.iterations = int(iterations)
        self.normalization = normalization
        self.augment = augment
        self.patch_size = patch_size
        self.fg_prob = fg_prob
        self.seed = seed
        if not self.case_ids:
            raise ValueError("PatchDataset got an empty case list - is the cache built?")

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        # Seeded per item so a run is reproducible across workers and restarts.
        rng = np.random.RandomState((self.seed + idx) % (2 ** 31 - 1))
        case_id = self.case_ids[rng.randint(len(self.case_ids))]
        img, seg = load_case(case_id)

        img, seg = _random_patch(img, seg, self.patch_size, rng, self.fg_prob)
        img = normalize_volume(img, method=self.normalization)
        if self.augment:
            img, seg = augment_patch(img, seg, rng)

        return (torch.from_numpy(np.ascontiguousarray(img)).float(),
                torch.from_numpy(np.ascontiguousarray(seg)).long())

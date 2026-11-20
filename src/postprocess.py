"""Test-time augmentation, probability ensembling and label post-processing.

All functions use the internal label convention {0 bg, 1 necrotic, 2 oedema,
3 enhancing}; probabilities are softmax maps [K,H,W,D].
"""
from itertools import combinations

import numpy as np
from scipy import ndimage

from .inference import sliding_window_probs
from .preprocessing import normalize_volume

# All 8 combinations of flips over the three spatial axes (identity included).
ALL_FLIPS = [c for r in range(4) for c in combinations((0, 1, 2), r)]


def tta_probs(model, image, flips=ALL_FLIPS, **kwargs):
    """Mirror test-time augmentation on a normalised [C,H,W,D] volume.

    Each flipped copy is predicted and its prediction flipped back before
    averaging. Forgetting the flip-back is the classic bug, and it is
    invisible on a symmetric phantom, so the test uses an asymmetric one.
    """
    acc = None
    for axes in flips:
        spatial = tuple(a + 1 for a in axes)        # +1 skips the channel axis
        x = np.flip(image, axis=spatial) if axes else image
        p = sliding_window_probs(model, np.ascontiguousarray(x), **kwargs)
        p = np.flip(p, axis=spatial) if axes else p
        acc = p if acc is None else acc + p
    return acc / len(flips)


def predict_probs(model, img, normalization="zscore", tta=False, **kwargs):
    """Normalise a raw [4,H,W,D] volume and return softmax probabilities."""
    x = normalize_volume(img, method=normalization)
    return tta_probs(model, x, **kwargs) if tta else sliding_window_probs(model, x, **kwargs)


def ensemble_probs(prob_maps, weights=None):
    """Weighted mean of probability maps from several models (soft voting).

    Soft voting keeps the confidence that the per-voxel majority vote in
    `inference.ensemble_predictions` discards, and it cannot tie.
    """
    maps = np.stack([np.asarray(p, dtype=np.float32) for p in prob_maps], axis=0)
    w = np.ones(len(maps), np.float32) if weights is None else np.asarray(weights, np.float32)
    return np.tensordot(w / w.sum(), maps, axes=1)


def remove_small_components(mask, min_voxels):
    """Drop connected components (26-connectivity) smaller than `min_voxels`."""
    mask = np.asarray(mask, dtype=bool)
    comp, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask.copy()
    sizes = np.bincount(comp.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[comp]


def suppress_small_et(pred, min_total_voxels=500, relabel_to=1):
    """If the total predicted enhancing tumour is tiny, relabel it to necrosis.

    BraTS scores an empty ET prediction on an ET-free case as Dice 1 and any
    false-positive voxel as Dice 0, so a few spurious ET voxels on a
    non-enhancing tumour cost the whole case. Relabelling to necrosis keeps
    those voxels inside tumour core, so TC and WT are unchanged. Tune the
    threshold on validation data, never on the test set.
    """
    out = np.asarray(pred).copy()
    n_et = int((out == 3).sum())
    if 0 < n_et < min_total_voxels:
        out[out == 3] = relabel_to
    return out


def postprocess(pred, et_min_total=500, wt_min_component=0):
    """Standard chain: suppress tiny ET, then optionally drop small WT islands
    (the contralateral false positives seen in the failure analysis)."""
    out = suppress_small_et(pred, et_min_total) if et_min_total else np.asarray(pred).copy()
    if wt_min_component:
        out[~remove_small_components(out > 0, wt_min_component)] = 0
    return out

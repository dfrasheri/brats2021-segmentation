"""Voxel-wise uncertainty and missing-modality robustness.

Uncertainty comes from disagreement between several predictions of the same
case (TTA flips, ensemble members, or both). Two maps are reported:

  * predictive entropy   H[mean p]               total uncertainty
  * mutual information   H[mean p] - mean H[p]   the part due to disagreement
                                                 (epistemic), which should rise
                                                 on unfamiliar data
"""
import numpy as np

from .config import MODALITIES

EPS = 1e-8


def entropy(probs, axis=0):
    p = np.clip(np.asarray(probs, dtype=np.float32), EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def predictive_entropy(prob_samples):
    """[S,K,H,W,D] -> [H,W,D] in nats; bounded by log K."""
    return entropy(np.mean(prob_samples, axis=0), axis=0)


def mutual_information(prob_samples):
    """Epistemic part of the uncertainty; exactly zero when all samples agree."""
    samples = np.asarray(prob_samples, dtype=np.float32)
    mean_h = np.mean([entropy(s, axis=0) for s in samples], axis=0)
    return np.maximum(predictive_entropy(samples) - mean_h, 0.0)


def case_uncertainty(unc_map, pred):
    """Scalar per case: mean uncertainty over predicted tumour (whole map if none).

    Used to test whether uncertainty flags bad cases: a useful map should
    correlate negatively with Dice across patients.
    """
    mask = np.asarray(pred) > 0
    return float(unc_map[mask].mean()) if mask.any() else float(np.mean(unc_map))


def drop_modality(img, modality, fill="zero"):
    """Simulate a missing MRI sequence in a raw [4,H,W,D] volume.

    fill="zero" blanks the channel (what a naive pipeline sees);
    fill="mean" imputes the mean of the remaining sequences inside the brain.
    """
    idx = MODALITIES.index(modality) if isinstance(modality, str) else int(modality)
    out = np.asarray(img, dtype=np.float32).copy()
    if fill == "zero":
        out[idx] = 0.0
    elif fill == "mean":
        others = [c for c in range(out.shape[0]) if c != idx]
        out[idx] = out[others].mean(axis=0)
    else:
        raise ValueError("fill must be 'zero' or 'mean'")
    return out

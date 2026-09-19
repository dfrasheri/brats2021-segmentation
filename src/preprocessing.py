"""Intensity normalisation, brain cropping and augmentation."""
import numpy as np


def brain_bbox(vol, pad=4):
    """Bounding box of non-zero voxels, padded and clipped."""
    mask = vol > 0
    if not mask.any():
        return tuple(slice(0, s) for s in vol.shape)
    out = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        idx = np.where(mask.any(axis=other))[0]
        out.append(slice(max(int(idx[0]) - pad, 0),
                         min(int(idx[-1]) + 1 + pad, vol.shape[axis])))
    return tuple(out)


def crop_to_brain(img, seg=None, pad=4):
    """Crop a [C,H,W,D] volume (and matching seg) to the brain region of channel 0."""
    bb = brain_bbox(img[0], pad=pad)
    cropped = img[(slice(None),) + bb]
    return (cropped, seg[bb]) if seg is not None else (cropped, None)


def normalize_volume(img, method="zscore", clip_percentile=(1, 99)):
    """Normalise each modality independently.

    MRI intensity has no absolute scale - it varies by scanner and sequence - so
    per-modality standardisation over brain voxels only is what makes patients
    comparable. Background stays exactly 0 so it never shifts the statistics.
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        ch = img[c].astype(np.float32)
        brain = ch > 0
        if not brain.any():
            continue

        vals = ch[brain]
        if clip_percentile is not None:
            lo, hi = np.percentile(vals, clip_percentile)
            ch = np.clip(ch, lo, hi)
            vals = ch[brain]

        if method == "zscore":
            std = vals.std()
            ch = (ch - vals.mean()) / std if std > 0 else ch - vals.mean()
        elif method == "minmax":
            rng = vals.max() - vals.min()
            ch = (ch - vals.min()) / rng if rng > 0 else ch - vals.min()
        else:
            raise ValueError("unknown normalization: " + str(method))

        ch[~brain] = 0.0
        out[c] = ch
    return out


def augment_patch(img, seg, rng):
    """Flips plus mild intensity jitter.

    Flips are safe here because tumours are not lateralised in any way the model
    should rely on. Intensity jitter stands in for scanner-to-scanner variation.
    """
    for axis in range(3):
        if rng.rand() < 0.5:
            img = np.flip(img, axis=axis + 1)
            seg = np.flip(seg, axis=axis)

    if rng.rand() < 0.3:
        img = img * rng.uniform(0.9, 1.1)
    if rng.rand() < 0.3:
        img = img + rng.uniform(-0.1, 0.1)

    return np.ascontiguousarray(img), np.ascontiguousarray(seg)


def create_segmentation_target(seg):
    """BraTS 2021 labels the enhancing tumour 4 and has no 3. Remap to contiguous 0-3."""
    out = np.asarray(seg).copy().astype(np.int64)
    out[out == 4] = 3
    return out

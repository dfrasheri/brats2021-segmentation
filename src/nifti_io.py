"""Raw NIfTI in, NIfTI out, in the original 240x240x155 space.

The training cache stores brain-cropped arrays and discards the crop box, which
is fine for scoring against cropped labels but not for anything leaving the
pipeline: the official BraTS validation server, the inference API, and
MONAILabel all expect a mask aligned voxel-for-voxel with the input scans.
This module keeps the crop box and the affine so predictions can be put back.

Two naming conventions are recognised:
  BraTS 2021        BraTS2021_00000_flair / _t1 / _t1ce / _t2 / _seg
  BraTS 2023/2024   BraTS-GLI-00000-000-t2f / -t1n / -t1c / -t2w / -seg
Both are mapped onto the internal channel order MODALITIES = flair,t1,t1ce,t2.
"""
from pathlib import Path

import numpy as np

from .config import MODALITIES
from .preprocessing import brain_bbox

SUFFIXES = {
    "brats2021": {"flair": "_flair", "t1": "_t1", "t1ce": "_t1ce", "t2": "_t2", "seg": "_seg"},
    "brats2023": {"flair": "-t2f", "t1": "-t1n", "t1ce": "-t1c", "t2": "-t2w", "seg": "-seg"},
}
# Enhancing tumour is label 4 in BraTS 2021 and label 3 from BraTS 2023 onwards.
ET_LABEL = {"brats2021": 4, "brats2023": 3}


def detect_convention(case_dir):
    case_dir = Path(case_dir)
    names = [p.name for p in case_dir.glob("*.nii*")]
    if any(n.endswith(("-t1c.nii.gz", "-t1c.nii")) for n in names):
        return "brats2023"
    if any(n.endswith(("_t1ce.nii.gz", "_t1ce.nii")) for n in names):
        return "brats2021"
    raise FileNotFoundError(f"no recognisable BraTS modality files in {case_dir}")


def modality_paths(case_dir, convention=None):
    """{modality: path} for one case directory (plus 'seg' if present)."""
    case_dir = Path(case_dir)
    convention = convention or detect_convention(case_dir)
    case_id = case_dir.name
    out = {}
    for key, suffix in SUFFIXES[convention].items():
        for ext in (".nii.gz", ".nii"):
            p = case_dir / f"{case_id}{suffix}{ext}"
            if p.exists():
                out[key] = p
                break
    missing = [m for m in MODALITIES if m not in out]
    if missing:
        raise FileNotFoundError(f"{case_id}: missing {missing}")
    return out


def load_volumes(paths):
    """Load 4 modality files (dict or list in MODALITIES order).

    Returns (img [4,H,W,D] float32, affine, header) in the original space.
    """
    import nibabel as nib
    if isinstance(paths, dict):
        paths = [paths[m] for m in MODALITIES]
    nii = [nib.load(str(p)) for p in paths]
    shapes = {n.shape for n in nii}
    if len(shapes) != 1:
        raise ValueError(f"modalities have different shapes: {shapes}")
    img = np.stack([np.asanyarray(n.dataobj).astype(np.float32) for n in nii], axis=0)
    return img, nii[0].affine, nii[0].header


def load_seg(path):
    import nibabel as nib
    seg = np.asanyarray(nib.load(str(path)).dataobj).astype(np.uint8)
    seg[seg == 4] = 3
    return seg


def crop(img, pad=4):
    """Crop to the brain (FLAIR extent, as in build_cache). Returns (cropped, bbox)."""
    bb = brain_bbox(img[0], pad=pad)
    return img[(slice(None),) + bb], bb


def uncrop(pred, bbox, full_shape):
    """Place a cropped [h,w,d] label map back into a zero volume of `full_shape`."""
    out = np.zeros(full_shape, dtype=pred.dtype)
    out[bbox] = pred
    return out


def to_challenge_labels(pred, convention="brats2021"):
    """Internal {0,1,2,3} -> challenge label set (ET=4 for 2021, 3 for 2023)."""
    out = np.asarray(pred, dtype=np.uint8).copy()
    out[out == 3] = ET_LABEL[convention]
    return out


def save_prediction(pred_full, affine, header, path, convention="brats2021"):
    """Write a full-size internal-label prediction as a challenge-format NIfTI."""
    import nibabel as nib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(to_challenge_labels(pred_full, convention), affine, hdr), str(path))
    return path

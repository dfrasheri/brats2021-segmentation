"""BraTS evaluation metrics.

BraTS is not scored per label. It is scored on three nested regions, so that is
what this module reports:

    WT (whole tumour)     labels 1 + 2 + 3   everything
    TC (tumour core)      labels 1 + 3       tumour without the surrounding oedema
    ET (enhancing tumour) label  3           the contrast-enhancing active part

Labels are assumed already remapped to the contiguous set {0,1,2,3}; raw BraTS
label 4 is folded into 3 on entry so a prediction and a target can never be
interpreted under different label conventions.
"""
import numpy as np

REGIONS = ("WT", "TC", "ET")


def _canonical(seg):
    """Fold raw BraTS label 4 into 3 so pred and target share one convention."""
    out = np.asarray(seg).astype(np.int16, copy=True)
    out[out == 4] = 3
    return out


def region_mask(seg, region):
    """Binary mask for one BraTS region. One definition, used for pred and target
    alike, so the two can never drift apart."""
    seg = _canonical(seg)
    if region == "WT":
        return seg > 0
    if region == "TC":
        return np.isin(seg, [1, 3])
    if region == "ET":
        return seg == 3
    raise ValueError("unknown region: " + str(region))


def brats_regions(seg):
    return {r: region_mask(seg, r) for r in REGIONS}


def dice(pred_mask, target_mask, smooth=1e-7):
    """Dice = 2|X and Y| / (|X| + |Y|).

    Convention: both empty scores 1.0. A patient with no enhancing tumour that
    the model correctly leaves empty is a success, not a zero.
    """
    p = np.asarray(pred_mask, dtype=bool)
    t = np.asarray(target_mask, dtype=bool)
    p_sum, t_sum = p.sum(), t.sum()
    if p_sum == 0 and t_sum == 0:
        return 1.0
    return float((2.0 * np.logical_and(p, t).sum() + smooth) / (p_sum + t_sum + smooth))


def _surface_points(mask):
    from scipy import ndimage
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.empty((0, 3))
    border = mask & ~ndimage.binary_erosion(mask)
    return np.argwhere(border)


def hausdorff95(pred_mask, target_mask):
    """95th-percentile symmetric surface distance, in voxels (1 mm isotropic).

    HD95 rather than the true maximum because a single stray voxel would
    otherwise dominate the score; this is the BraTS convention. Uses a KD-tree,
    since the dense pairwise form is O(N^2) and hangs on real surfaces.
    """
    from scipy.spatial import cKDTree

    p = _surface_points(pred_mask)
    t = _surface_points(target_mask)
    if len(p) == 0 and len(t) == 0:
        return 0.0
    if len(p) == 0 or len(t) == 0:
        return float("nan")   # undefined, not zero - excluded from averages

    d_pt = cKDTree(t).query(p)[0]
    d_tp = cKDTree(p).query(t)[0]
    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)))


def sensitivity_specificity(pred_mask, target_mask):
    p = np.asarray(pred_mask, dtype=bool)
    t = np.asarray(target_mask, dtype=bool)
    tp = np.logical_and(p, t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    fp = np.logical_and(p, ~t).sum()
    sens = float(tp / (tp + fn)) if (tp + fn) else 1.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 1.0
    return sens, spec


def evaluate_segmentation(pred, target, compute_hd95=True):
    """Per-region metrics for one patient.

    Returns {"WT": {"dice":..., "hd95":..., "sensitivity":..., "specificity":...}, ...}
    """
    pred = _canonical(pred)
    target = _canonical(target)

    results = {}
    for region in REGIONS:
        p = region_mask(pred, region)
        t = region_mask(target, region)
        sens, spec = sensitivity_specificity(p, t)
        results[region] = {
            "dice": dice(p, t),
            "hd95": hausdorff95(p, t) if compute_hd95 else float("nan"),
            "sensitivity": sens,
            "specificity": spec,
        }
    return results


def aggregate(per_case):
    """Mean and std across patients, ignoring NaN (undefined HD95)."""
    out = {}
    for region in REGIONS:
        out[region] = {}
        for metric in ("dice", "hd95", "sensitivity", "specificity"):
            vals = np.array([c[region][metric] for c in per_case], dtype=float)
            vals = vals[~np.isnan(vals)]
            out[region][metric + "_mean"] = float(vals.mean()) if len(vals) else float("nan")
            out[region][metric + "_std"] = float(vals.std()) if len(vals) else float("nan")
            out[region][metric + "_n"] = int(len(vals))
    return out

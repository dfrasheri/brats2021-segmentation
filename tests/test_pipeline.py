"""Tests for preprocessing, splitting, models and inference.

These cover the invariants that would otherwise fail silently: a split that leaks,
a normalisation that shifts background, an ensemble that votes over the wrong axis.
"""
import numpy as np
import pytest
import torch

from src.preprocessing import normalize_volume, brain_bbox, crop_to_brain, create_segmentation_target
from src.data import split_cases
from src.models import create_model, count_parameters
from src.inference import ensemble_predictions, sliding_window_inference


@pytest.fixture
def volume():
    """A 4-channel volume with a bright blob on a zero background, one scale per channel."""
    rng = np.random.RandomState(0)
    img = np.zeros((4, 32, 32, 32), dtype=np.float32)
    for c in range(4):
        # Each modality gets a deliberately different intensity scale.
        img[c, 8:24, 8:24, 8:24] = rng.uniform(50, 150) + rng.randn(16, 16, 16) * 10
    return img


# ------------------------------------------------------------ Normalisation

def test_normalize_leaves_background_at_zero(volume):
    """Background must stay exactly 0, or it starts contributing signal."""
    for method in ("zscore", "minmax"):
        out = normalize_volume(volume, method=method)
        assert np.all(out[volume == 0] == 0.0), f"{method} moved the background"


def test_zscore_standardises_brain_voxels(volume):
    out = normalize_volume(volume, method="zscore", clip_percentile=None)
    for c in range(4):
        brain = out[c][volume[c] > 0]
        assert brain.mean() == pytest.approx(0.0, abs=1e-4)
        assert brain.std() == pytest.approx(1.0, abs=1e-4)


def test_zscore_removes_between_modality_scale_differences(volume):
    """The point of per-modality normalisation: scanner scale must not survive it."""
    raw_means = [volume[c][volume[c] > 0].mean() for c in range(4)]
    assert max(raw_means) / min(raw_means) > 1.2      # the fixture really does differ

    out = normalize_volume(volume, method="zscore", clip_percentile=None)
    norm_means = [out[c][volume[c] > 0].mean() for c in range(4)]
    assert max(abs(m) for m in norm_means) < 1e-4     # all collapsed to ~0


def test_minmax_maps_brain_into_unit_range(volume):
    out = normalize_volume(volume, method="minmax", clip_percentile=None)
    for c in range(4):
        brain = out[c][volume[c] > 0]
        assert brain.min() == pytest.approx(0.0, abs=1e-5)
        assert brain.max() == pytest.approx(1.0, abs=1e-5)


def test_normalize_handles_empty_channel():
    """An all-zero modality must not divide by zero or emit NaN."""
    img = np.zeros((4, 8, 8, 8), dtype=np.float32)
    img[0, 2:6, 2:6, 2:6] = 100.0
    out = normalize_volume(img, method="zscore")
    assert np.isfinite(out).all()
    assert np.all(out[1] == 0)


def test_unknown_normalization_raises(volume):
    with pytest.raises(ValueError):
        normalize_volume(volume, method="quantile")


# ---------------------------------------------------------------- Cropping

def test_brain_bbox_covers_all_nonzero():
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[5:12, 7:15, 3:9] = 1.0
    bb = brain_bbox(vol, pad=0)
    assert vol[bb].sum() == vol.sum()
    assert vol[bb].shape == (7, 8, 6)


def test_brain_bbox_pad_is_clipped_to_bounds():
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[0:3, 0:3, 0:3] = 1.0
    bb = brain_bbox(vol, pad=50)
    assert vol[bb].shape == (10, 10, 10)     # padding cannot escape the array


def test_crop_to_brain_keeps_image_and_label_aligned():
    img = np.zeros((4, 20, 20, 20), dtype=np.float32)
    seg = np.zeros((20, 20, 20), dtype=np.uint8)
    img[:, 6:14, 6:14, 6:14] = 1.0
    seg[8:12, 8:12, 8:12] = 1

    cropped_img, cropped_seg = crop_to_brain(img, seg, pad=0)
    assert cropped_img.shape[1:] == cropped_seg.shape
    assert cropped_seg.sum() == seg.sum()    # no labels lost


def test_empty_volume_bbox_does_not_crash():
    bb = brain_bbox(np.zeros((6, 6, 6), dtype=np.float32))
    assert np.zeros((6, 6, 6))[bb].shape == (6, 6, 6)


# ------------------------------------------------------------ Label remap

def test_label_4_becomes_3():
    seg = np.array([0, 1, 2, 4, 4, 1], dtype=np.uint8)
    out = create_segmentation_target(seg)
    assert set(np.unique(out).tolist()) == {0, 1, 2, 3}
    assert (out == 3).sum() == 2
    assert 4 not in np.unique(out)


def test_label_remap_does_not_mutate_input():
    seg = np.array([0, 4, 4], dtype=np.uint8)
    create_segmentation_target(seg)
    assert 4 in np.unique(seg), "input array was modified in place"


# ---------------------------------------------------------------- Splitting

def test_splits_are_disjoint_by_patient():
    """The property the whole evaluation rests on."""
    ids = [f"case_{i:04d}" for i in range(200)]
    train, val, test = split_cases(ids)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == len(ids)


def test_split_is_deterministic():
    ids = [f"case_{i:04d}" for i in range(120)]
    assert split_cases(ids, seed=42) == split_cases(ids, seed=42)


def test_split_changes_with_seed():
    ids = [f"case_{i:04d}" for i in range(120)]
    assert split_cases(ids, seed=1)[0] != split_cases(ids, seed=2)[0]


def test_split_is_independent_of_input_order():
    """Shuffling the input list must not change who lands in the test set."""
    ids = [f"case_{i:04d}" for i in range(120)]
    shuffled = list(reversed(ids))
    assert sorted(split_cases(ids)[2]) == sorted(split_cases(shuffled)[2])


# ------------------------------------------------------------------ Models

@pytest.mark.parametrize("name,deep_supervision", [("unet", False), ("unet", True), ("vnet", False)])
def test_model_output_shape(name, deep_supervision):
    model = create_model(name, num_classes=4, base_channels=4, deep_supervision=deep_supervision)
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    primary = out[0] if isinstance(out, list) else out
    assert primary.shape == (1, 4, 32, 32, 32)


def test_deep_supervision_returns_multiple_heads():
    model = create_model("unet", base_channels=4, deep_supervision=True)
    with torch.no_grad():
        out = model(torch.randn(1, 4, 32, 32, 32))
    assert isinstance(out, list) and len(out) > 1
    # Auxiliary heads are at progressively coarser resolution.
    sizes = [o.shape[-1] for o in out]
    assert sizes == sorted(sizes, reverse=True)


def test_indivisible_input_fails_loudly():
    """155 slices is not divisible by 16 - this must raise, not silently misalign."""
    model = create_model("unet", base_channels=4)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 4, 32, 32, 30))


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        create_model("transformer")


def test_count_parameters_is_positive():
    assert count_parameters(create_model("unet", base_channels=4)) > 0


# --------------------------------------------------------------- Ensembling

def test_ensemble_votes_per_voxel():
    """The bug this guards against: a global bincount returns one label for the
    entire volume instead of voting at each voxel independently."""
    a = np.zeros((4, 4, 4), np.uint8); a[0, 0, 0] = 1; a[1, 1, 1] = 3
    b = a.copy(); b[2, 2, 2] = 2
    c = a.copy()

    out = ensemble_predictions([a, b, c])

    assert out.shape == (4, 4, 4)
    assert out[0, 0, 0] == 1          # unanimous
    assert out[1, 1, 1] == 3          # unanimous
    assert out[2, 2, 2] == 0          # 2 of 3 say background, so background wins
    assert len(np.unique(out)) > 1    # not collapsed to a single label


def test_ensemble_of_identical_predictions_is_unchanged():
    p = np.random.RandomState(0).randint(0, 4, (6, 6, 6)).astype(np.uint8)
    assert np.array_equal(ensemble_predictions([p, p, p]), p)


def test_ensemble_majority_beats_minority():
    shape = (3, 3, 3)
    majority = np.full(shape, 2, np.uint8)
    minority = np.full(shape, 1, np.uint8)
    out = ensemble_predictions([majority, majority, minority])
    assert np.all(out == 2)


# ------------------------------------------------------- Sliding window

def test_sliding_window_returns_input_shape():
    """A model trained on patches must still return a full-size label map."""
    model = create_model("unet", num_classes=4, base_channels=4)
    img = np.random.RandomState(0).randn(4, 40, 36, 44).astype(np.float32)

    pred = sliding_window_inference(model, img, patch_size=(32, 32, 32),
                                    overlap=0.5, num_classes=4, amp=False)

    assert pred.shape == (40, 36, 44)
    assert pred.dtype == np.uint8
    assert pred.max() < 4


def test_sliding_window_pads_volumes_smaller_than_patch():
    model = create_model("unet", num_classes=4, base_channels=4)
    img = np.random.RandomState(0).randn(4, 16, 16, 16).astype(np.float32)

    pred = sliding_window_inference(model, img, patch_size=(32, 32, 32),
                                    overlap=0.5, num_classes=4, amp=False)

    assert pred.shape == (16, 16, 16)

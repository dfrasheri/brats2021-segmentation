"""Tests for the evaluation metrics.

These matter more than most tests in this project: a silently wrong metric does not
crash, it just reports a number that looks plausible. Every result in the README
depends on these functions being right.
"""
import numpy as np
import pytest

from src.metrics import (
    REGIONS, dice, hausdorff95, sensitivity_specificity,
    region_mask, brats_regions, evaluate_segmentation, aggregate,
)


@pytest.fixture
def phantom():
    """A small volume with all three label classes, nested like a real tumour."""
    seg = np.zeros((48, 48, 48), dtype=np.int16)
    seg[10:40, 10:40, 10:40] = 2   # oedema, outermost
    seg[18:32, 18:32, 18:32] = 1   # necrotic core
    seg[22:28, 22:28, 22:28] = 3   # enhancing, innermost
    return seg


# --------------------------------------------------------------------- Dice

def test_dice_identical_is_one(phantom):
    for region in REGIONS:
        m = region_mask(phantom, region)
        assert dice(m, m) == pytest.approx(1.0, abs=1e-6)


def test_dice_disjoint_is_zero():
    a = np.zeros((10, 10, 10), bool); a[0:3, 0:3, 0:3] = True
    b = np.zeros((10, 10, 10), bool); b[7:10, 7:10, 7:10] = True
    assert dice(a, b) == pytest.approx(0.0, abs=1e-6)


def test_dice_both_empty_is_one():
    """A patient with no enhancing tumour, correctly predicted empty, is a success.

    Returning 0.0 here would punish the model for being right, and would drag the
    ET average down on exactly the patients it handled correctly.
    """
    empty = np.zeros((8, 8, 8), bool)
    assert dice(empty, empty) == 1.0


def test_dice_false_positive_on_empty_target_is_zero():
    """Predicting a region that is not there must score 0, not 1."""
    pred = np.zeros((8, 8, 8), bool); pred[2:4, 2:4, 2:4] = True
    assert dice(pred, np.zeros((8, 8, 8), bool)) == pytest.approx(0.0, abs=1e-3)


def test_dice_half_overlap():
    a = np.zeros((10, 10, 10), bool); a[0:4, :, :] = True
    b = np.zeros((10, 10, 10), bool); b[2:6, :, :] = True
    # |A|=|B|=400, intersection 200 -> 2*200 / 800 = 0.5
    assert dice(a, b) == pytest.approx(0.5, abs=1e-4)


def test_dice_is_symmetric(phantom):
    a = region_mask(phantom, "WT")
    b = region_mask(np.roll(phantom, 3, axis=0), "WT")
    assert dice(a, b) == pytest.approx(dice(b, a), abs=1e-9)


# ------------------------------------------------------------------- Labels

def test_label_4_and_3_are_equivalent(phantom):
    """BraTS 2021 labels the enhancing tumour 4; this project remaps it to 3.

    A prediction using one convention and a target using the other must not be
    silently scored as a total mismatch.
    """
    raw = phantom.copy()
    raw[raw == 3] = 4
    for region in REGIONS:
        assert np.array_equal(region_mask(raw, region), region_mask(phantom, region))

    results = evaluate_segmentation(raw, phantom)
    for region in REGIONS:
        assert results[region]["dice"] == pytest.approx(1.0, abs=1e-6)


def test_regions_are_nested(phantom):
    """ET must sit inside TC, which must sit inside WT."""
    r = brats_regions(phantom)
    assert (r["ET"] & ~r["TC"]).sum() == 0, "ET voxels found outside TC"
    assert (r["TC"] & ~r["WT"]).sum() == 0, "TC voxels found outside WT"
    assert r["WT"].sum() > r["TC"].sum() > r["ET"].sum()


def test_region_definitions(phantom):
    assert np.array_equal(region_mask(phantom, "WT"), phantom > 0)
    assert np.array_equal(region_mask(phantom, "TC"), np.isin(phantom, [1, 3]))
    assert np.array_equal(region_mask(phantom, "ET"), phantom == 3)
    # Oedema is deliberately excluded from the core.
    assert not region_mask(phantom, "TC")[phantom == 2].any()


def test_unknown_region_raises():
    with pytest.raises(ValueError):
        region_mask(np.zeros((4, 4, 4), int), "NOPE")


# ---------------------------------------------------------------- Hausdorff

def test_hd95_identical_is_zero(phantom):
    m = region_mask(phantom, "WT")
    assert hausdorff95(m, m) == pytest.approx(0.0, abs=1e-6)


def test_hd95_recovers_a_known_shift(phantom):
    """A rigid shift of n voxels should give a surface distance of about n."""
    m = region_mask(phantom, "WT")
    shifted = np.roll(m, 5, axis=0)
    assert hausdorff95(shifted, m) == pytest.approx(5.0, abs=1.0)


def test_hd95_undefined_when_one_side_empty():
    """NaN, not 0. Zero would read as a perfect boundary match."""
    full = np.zeros((8, 8, 8), bool); full[2:6, 2:6, 2:6] = True
    assert np.isnan(hausdorff95(full, np.zeros((8, 8, 8), bool)))
    assert np.isnan(hausdorff95(np.zeros((8, 8, 8), bool), full))


def test_hd95_both_empty_is_zero():
    assert hausdorff95(np.zeros((8, 8, 8), bool), np.zeros((8, 8, 8), bool)) == 0.0


def test_hd95_grows_with_error(phantom):
    m = region_mask(phantom, "WT")
    near = hausdorff95(np.roll(m, 2, axis=0), m)
    far = hausdorff95(np.roll(m, 8, axis=0), m)
    assert far > near


# ------------------------------------------------ Sensitivity / specificity

def test_sensitivity_specificity_perfect(phantom):
    m = region_mask(phantom, "WT")
    sens, spec = sensitivity_specificity(m, m)
    assert sens == pytest.approx(1.0)
    assert spec == pytest.approx(1.0)


def test_sensitivity_zero_when_nothing_predicted(phantom):
    m = region_mask(phantom, "WT")
    sens, spec = sensitivity_specificity(np.zeros_like(m), m)
    assert sens == pytest.approx(0.0)
    assert spec == pytest.approx(1.0)   # no false positives either


def test_specificity_drops_with_false_positives(phantom):
    m = region_mask(phantom, "WT")
    over = m.copy()
    over[0:6, 0:6, 0:6] = True          # a blob where there is no tumour
    sens, spec = sensitivity_specificity(over, m)
    assert sens == pytest.approx(1.0)   # still finds everything real
    assert spec < 1.0                   # but is now wrong about background


# ------------------------------------------------------------- Integration

def test_evaluate_segmentation_shape(phantom):
    results = evaluate_segmentation(phantom, phantom)
    assert set(results) == set(REGIONS)
    for region in REGIONS:
        assert set(results[region]) == {"dice", "hd95", "sensitivity", "specificity"}


def test_evaluate_segmentation_degrades_with_error(phantom):
    perfect = evaluate_segmentation(phantom, phantom)
    shifted = evaluate_segmentation(np.roll(phantom, 4, axis=0), phantom)
    for region in REGIONS:
        assert shifted[region]["dice"] < perfect[region]["dice"]
        assert shifted[region]["hd95"] > perfect[region]["hd95"]


def test_aggregate_ignores_nan():
    """Undefined HD95 must be excluded from the average rather than poisoning it."""
    per_case = [
        {r: {"dice": 0.8, "hd95": 4.0, "sensitivity": 0.9, "specificity": 0.99} for r in REGIONS},
        {r: {"dice": 0.6, "hd95": np.nan, "sensitivity": 0.7, "specificity": 0.98} for r in REGIONS},
    ]
    agg = aggregate(per_case)
    assert agg["WT"]["dice_mean"] == pytest.approx(0.7)
    assert agg["WT"]["dice_n"] == 2
    assert agg["WT"]["hd95_mean"] == pytest.approx(4.0)
    assert agg["WT"]["hd95_n"] == 1          # the NaN case was dropped
    assert not np.isnan(agg["WT"]["hd95_mean"])

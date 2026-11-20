"""Tests for cross-validation, TTA/ensembling, post-processing, uncertainty,
NIfTI round-tripping, the end-to-end predictor and the HTTP API."""
import importlib.util

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data import kfold_split
from src.inference import sliding_window_probs, sliding_window_inference
from src.postprocess import (ALL_FLIPS, tta_probs, ensemble_probs, suppress_small_et,
                             remove_small_components, postprocess)
from src.uncertainty import predictive_entropy, mutual_information, drop_modality, entropy
from src.nifti_io import crop, uncrop, to_challenge_labels

PATCH = (32, 32, 32)
IDS = [f"case_{i:03d}" for i in range(53)]


class Pointwise(nn.Module):
    """1x1x1 conv: flip-equivariant, so correct TTA must equal plain inference."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv3d(4, 4, 1)

    def forward(self, x):
        return self.conv(x)


# ---- k-fold ---------------------------------------------------------------

def test_kfold_test_folds_partition_all_patients():
    tests = [kfold_split(IDS, fold=f, n_folds=5)[2] for f in range(5)]
    flat = [c for t in tests for c in t]
    assert sorted(flat) == sorted(IDS) and len(flat) == len(set(flat))


@pytest.mark.parametrize("fold", range(5))
def test_kfold_sets_are_disjoint(fold):
    tr, va, te = kfold_split(IDS, fold=fold, n_folds=5)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(IDS)
    assert len(va) >= 1


def test_kfold_is_deterministic_and_order_independent():
    a = kfold_split(IDS, fold=2)
    b = kfold_split(list(reversed(IDS)), fold=2)
    assert a == b


def test_kfold_rejects_bad_fold():
    with pytest.raises(ValueError):
        kfold_split(IDS, fold=5, n_folds=5)


# ---- probabilities, TTA, ensembling ----------------------------------------

@pytest.fixture
def asym_volume():
    rng = np.random.RandomState(0)
    x = rng.randn(4, 20, 24, 18).astype(np.float32)
    x[:, :5] += 3.0                       # strongly asymmetric along every axis
    x[:, :, :4] -= 2.0
    x[:, :, :, :3] += 1.5
    return x


def test_probs_sum_to_one(asym_volume):
    p = sliding_window_probs(Pointwise(), asym_volume, patch_size=PATCH, device=torch.device("cpu"))
    assert p.shape == (4,) + asym_volume.shape[1:]
    np.testing.assert_allclose(p.sum(0), 1.0, atol=1e-5)


def test_labels_are_argmax_of_probs(asym_volume):
    kw = dict(patch_size=PATCH, device=torch.device("cpu"))
    m = Pointwise()
    np.testing.assert_array_equal(sliding_window_inference(m, asym_volume, **kw),
                                  sliding_window_probs(m, asym_volume, **kw).argmax(0))


def test_tta_flips_predictions_back(asym_volume):
    kw = dict(patch_size=PATCH, device=torch.device("cpu"))
    m = Pointwise()
    plain = sliding_window_probs(m, asym_volume, **kw)
    tta = tta_probs(m, asym_volume, **kw)
    np.testing.assert_allclose(tta, plain, atol=1e-4)


def test_all_eight_flips():
    assert len(ALL_FLIPS) == 8 and () in ALL_FLIPS and (0, 1, 2) in ALL_FLIPS


def test_ensemble_probs_is_weighted_mean():
    a = np.zeros((2, 2, 2, 2), np.float32); a[0] = 1
    b = np.zeros((2, 2, 2, 2), np.float32); b[1] = 1
    np.testing.assert_allclose(ensemble_probs([a, b])[0], 0.5)
    np.testing.assert_allclose(ensemble_probs([a, b], weights=[3, 1])[0], 0.75)


# ---- post-processing --------------------------------------------------------

def test_small_et_is_relabelled_to_necrosis_not_background():
    pred = np.zeros((20, 20, 20), np.uint8)
    pred[5:10, 5:10, 5:10] = 2
    pred[6:8, 6:8, 6:8] = 3                              # 8 ET voxels
    out = suppress_small_et(pred, min_total_voxels=50)
    assert (out == 3).sum() == 0
    assert (out == 1).sum() == 8
    assert ((out > 0) == (pred > 0)).all()               # WT unchanged
    assert (np.isin(out, (1, 3)) == np.isin(pred, (1, 3))).all()   # TC unchanged


def test_large_et_is_kept():
    pred = np.zeros((20, 20, 20), np.uint8)
    pred[2:12, 2:12, 2:12] = 3
    np.testing.assert_array_equal(suppress_small_et(pred, 500), pred)


def test_remove_small_components_keeps_large():
    m = np.zeros((30, 30, 30), bool)
    m[2:12, 2:12, 2:12] = True                           # 1000 voxels
    m[25:27, 25:27, 25:27] = True                        # 8 voxels, separate
    out = remove_small_components(m, 100)
    assert out[5, 5, 5] and not out[26, 26, 26]
    assert out.sum() == 1000


def test_diagonal_neighbours_are_one_component():
    m = np.zeros((5, 5, 5), bool)
    m[1, 1, 1] = m[2, 2, 2] = True
    assert remove_small_components(m, 2).sum() == 2


def test_postprocess_wt_islands():
    pred = np.zeros((30, 30, 30), np.uint8)
    pred[2:12, 2:12, 2:12] = 2
    pred[25:27, 25:27, 25:27] = 2
    out = postprocess(pred, et_min_total=0, wt_min_component=100)
    assert out[26, 26, 26] == 0 and out[5, 5, 5] == 2


# ---- uncertainty and missing modalities --------------------------------------

def test_entropy_bounds():
    k = 4
    uniform = np.full((k, 3, 3, 3), 1 / k, np.float32)
    onehot = np.zeros((k, 3, 3, 3), np.float32); onehot[0] = 1
    np.testing.assert_allclose(entropy(uniform), np.log(k), rtol=1e-5)
    np.testing.assert_allclose(entropy(onehot), 0, atol=1e-5)


def test_mutual_information_zero_when_samples_agree():
    rng = np.random.RandomState(0)
    p = rng.dirichlet(np.ones(4), size=(3, 3, 3)).transpose(3, 0, 1, 2).astype(np.float32)
    samples = np.stack([p, p, p])
    np.testing.assert_allclose(mutual_information(samples), 0, atol=1e-5)


def test_mutual_information_positive_when_samples_disagree():
    a = np.zeros((2, 1, 1, 1), np.float32); a[0] = 1
    b = np.zeros((2, 1, 1, 1), np.float32); b[1] = 1
    mi = mutual_information(np.stack([a, b]))
    np.testing.assert_allclose(mi, np.log(2), rtol=1e-4)
    assert predictive_entropy(np.stack([a, b])).item() == pytest.approx(np.log(2), rel=1e-4)


def test_drop_modality():
    img = np.stack([np.full((4, 4, 4), v, np.float32) for v in (1, 2, 3, 4)])
    z = drop_modality(img, "t1ce", "zero")
    assert (z[2] == 0).all() and (z[[0, 1, 3]] == img[[0, 1, 3]]).all()
    m = drop_modality(img, "t1ce", "mean")
    np.testing.assert_allclose(m[2], (1 + 2 + 4) / 3)
    assert (img[2] == 3).all()                           # input not mutated


# ---- NIfTI space round trip ---------------------------------------------------

def test_crop_uncrop_round_trip():
    img = np.zeros((4, 40, 40, 30), np.float32)
    img[:, 10:25, 8:30, 5:20] = 1.0
    cropped, bb = crop(img)
    pred = (cropped[0] > 0).astype(np.uint8)
    back = uncrop(pred, bb, img.shape[1:])
    np.testing.assert_array_equal(back, (img[0] > 0).astype(np.uint8))


@pytest.mark.parametrize("conv,et", [("brats2021", 4), ("brats2023", 3)])
def test_challenge_labels(conv, et):
    out = to_challenge_labels(np.array([0, 1, 2, 3]), conv)
    assert list(out) == [0, 1, 2, et]


@pytest.mark.skipif(importlib.util.find_spec("nibabel") is None, reason="nibabel not installed")
def test_nifti_save_and_detect(tmp_path):
    import nibabel as nib
    from src.nifti_io import detect_convention, modality_paths, load_volumes, save_prediction
    case = tmp_path / "BraTS2021_00001"
    case.mkdir()
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    for i, m in enumerate(("flair", "t1", "t1ce", "t2")):
        nib.save(nib.Nifti1Image(np.full((8, 8, 8), i + 1, np.float32), affine),
                 str(case / f"BraTS2021_00001_{m}.nii.gz"))
    assert detect_convention(case) == "brats2021"
    img, aff, hdr = load_volumes(modality_paths(case))
    assert img.shape == (4, 8, 8, 8) and img[2].max() == 3
    pred = np.zeros((8, 8, 8), np.uint8); pred[2:4, 2:4, 2:4] = 3
    out = save_prediction(pred, aff, hdr, tmp_path / "pred.nii.gz")
    back = nib.load(str(out))
    assert np.asanyarray(back.dataobj).max() == 4
    np.testing.assert_array_equal(back.affine, affine)


# ---- end-to-end predictor and API -----------------------------------------------

@pytest.fixture
def tiny_checkpoint(tmp_path, monkeypatch):
    """A real UNet checkpoint, made small enough for CPU tests by shrinking the patch."""
    import src.inference as inf
    fn = inf.sliding_window_probs.__wrapped__          # under the torch.no_grad decorator
    monkeypatch.setattr(fn, "__defaults__", (PATCH,) + fn.__defaults__[1:])
    from src.config import CONFIGS
    from src.models import create_model
    cfg = dict(CONFIGS["nnunet_style"], base_channels=4)
    model = create_model("unet", base_channels=4, deep_supervision=True)
    path = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "settings": cfg, "config": "nnunet_style"}, path)
    return path


def test_predictor_returns_full_size_labels(tiny_checkpoint):
    from src.predictor import Predictor, volumes_ml
    img = np.zeros((4, 30, 34, 26), np.float32)
    img[:, 5:25, 6:28, 4:22] = np.random.RandomState(0).rand(4, 20, 22, 18) + 0.1
    res = Predictor([tiny_checkpoint], et_min_total=100).predict(img, return_uncertainty=True)
    assert res["labels"].shape == img.shape[1:]
    assert res["labels"].max() <= 3
    _, bb = crop(img)
    outside = np.ones(img.shape[1:], bool)
    outside[bb] = False
    assert (res["labels"][outside] == 0).all()           # nothing outside the brain crop
    assert res["entropy"].shape == img.shape[1:]
    assert set(volumes_ml(res["labels"])) == {"WT", "TC", "ET"}


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None or
                    importlib.util.find_spec("nibabel") is None or
                    importlib.util.find_spec("httpx") is None, reason="API deps not installed")
def test_api_predict_round_trip(tiny_checkpoint, tmp_path, monkeypatch):
    import gzip
    import io
    import nibabel as nib
    monkeypatch.setenv("MODEL_CHECKPOINTS", str(tiny_checkpoint))
    monkeypatch.setenv("ET_MIN_TOTAL", "0")
    from fastapi.testclient import TestClient
    import serve.app as api
    api._state.clear()
    files = {}
    for m in ("flair", "t1", "t1ce", "t2"):
        vol = np.zeros((24, 24, 20), np.float32)
        vol[4:20, 4:20, 3:17] = np.random.RandomState(1).rand(16, 16, 14) + 0.1
        p = tmp_path / f"{m}.nii.gz"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        files[m] = (p.name, p.read_bytes(), "application/gzip")
    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict?output=json", files=files)
        assert r.status_code == 200 and set(r.json()["volumes_ml"]) == {"WT", "TC", "ET"}
        r = client.post("/predict", files=files)
        assert r.status_code == 200
        seg = nib.Nifti1Image.from_bytes(gzip.decompress(r.content))
        assert seg.shape == (24, 24, 20)
        bad = dict(files, t2=("t2.nii.gz", b"not a nifti", "application/gzip"))
        assert client.post("/predict", files=bad).status_code == 400


@pytest.mark.skipif(importlib.util.find_spec("monai") is None, reason="MONAI not installed")
@pytest.mark.parametrize("name", ["segresnet", "swin_unetr"])
def test_monai_models_output_shape(name):
    from src.config import CONFIGS
    from src.models import create_model
    m = create_model(name, base_channels=8 if name == "segresnet" else 12).eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 4, 64, 64, 64))
    assert out.shape == (1, 4, 64, 64, 64)

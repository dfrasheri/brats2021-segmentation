"""End-to-end predictor for raw, full-size scans.

One class used by the validation-server export, the HTTP API and MONAILabel,
so all three run exactly the same inference:

    raw [4,240,240,155] -> crop to brain -> per-model normalisation
    -> sliding window (+ optional mirror TTA) -> soft-vote ensemble
    -> ET suppression -> uncrop -> [240,240,155] internal labels
"""
import time

import numpy as np

from .checkpoint import load_model
from .nifti_io import crop, uncrop
from .postprocess import predict_probs, ensemble_probs, postprocess
from .uncertainty import predictive_entropy


class Predictor:
    def __init__(self, checkpoints, device="cpu", tta=False, et_min_total=500):
        if not checkpoints:
            raise ValueError("at least one checkpoint is required")
        import torch
        self.device = torch.device(device)      # accept "cpu"/"cuda" strings too
        self.tta = tta
        self.et_min_total = et_min_total
        self.models = [load_model(p, device) for p in checkpoints]
        self.checkpoints = [str(p) for p in checkpoints]

    def predict(self, img_full, return_uncertainty=False):
        """img_full: raw [4,H,W,D] in MODALITIES order. Returns dict with
        'labels' [H,W,D] (internal 0-3), optional 'entropy', and timings."""
        t0 = time.time()
        img, bbox = crop(np.asarray(img_full, dtype=np.float32))
        probs = [predict_probs(m, img, cfg["normalization"], tta=self.tta, device=self.device)
                 for m, cfg in self.models]
        mean = ensemble_probs(probs) if len(probs) > 1 else probs[0]
        labels = mean.argmax(0).astype(np.uint8)
        if self.et_min_total:
            labels = postprocess(labels, et_min_total=self.et_min_total)

        out = {"labels": uncrop(labels, bbox, img_full.shape[1:]),
               "seconds": time.time() - t0}
        if return_uncertainty:
            samples = np.stack(probs) if len(probs) > 1 else mean[None]
            out["entropy"] = uncrop(predictive_entropy(samples).astype(np.float32),
                                    bbox, img_full.shape[1:])
        return out


def volumes_ml(labels, voxel_volume_ml=0.001):
    """Region volumes in millilitres (BraTS is resampled to 1 mm^3 = 0.001 ml)."""
    labels = np.asarray(labels)
    return {"WT": float((labels > 0).sum() * voxel_volume_ml),
            "TC": float(np.isin(labels, (1, 3)).sum() * voxel_volume_ml),
            "ET": float((labels == 3).sum() * voxel_volume_ml)}

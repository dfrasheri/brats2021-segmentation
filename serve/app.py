"""HTTP inference service.

  POST /predict   multipart: flair, t1, t1ce, t2 (.nii / .nii.gz, same grid)
                  ?output=nifti (default) -> segmentation NIfTI, BraTS labels
                  ?output=json            -> region volumes and timing only
  GET  /health    liveness/readiness (also used by Cloud Run)
  GET  /model     which checkpoints are loaded and how inference is configured

Configuration (environment):
  MODEL_CHECKPOINTS  comma-separated .pt paths  (default: checkpoints/nnunet_style_best.pt)
  TTA                "1" to enable mirror test-time augmentation (8x slower)
  ET_MIN_TOTAL       ET suppression threshold in voxels (0 disables)
  PORT               listening port (Cloud Run sets this)

Research software. Not a medical device; outputs must not be used for diagnosis.
"""
import gzip
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from src.config import MODALITIES, get_device
from src.nifti_io import load_volumes, to_challenge_labels
from src.predictor import Predictor, volumes_ml

DISCLAIMER = "Research use only. Not a medical device. Not for diagnosis or treatment decisions."
MAX_UPLOAD_MB = 200

_state = {}


@asynccontextmanager
async def lifespan(_app):
    # Fail fast at start-up if the model cannot load, rather than on the first request.
    if os.environ.get("SKIP_MODEL_LOAD") != "1":
        get_predictor()
    yield


app = FastAPI(title="BraTS 2021 tumour segmentation", version="1.0.0",
              description=DISCLAIMER, lifespan=lifespan)


def get_predictor():
    if "predictor" not in _state:
        ckpts = [p.strip() for p in os.environ.get(
            "MODEL_CHECKPOINTS", str(ROOT / "checkpoints" / "nnunet_style_best.pt")).split(",") if p.strip()]
        missing = [p for p in ckpts if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"checkpoint(s) not found: {missing}")
        _state["predictor"] = Predictor(ckpts, device=get_device(),
                                        tta=os.environ.get("TTA", "0") == "1",
                                        et_min_total=int(os.environ.get("ET_MIN_TOTAL", "500")))
    return _state["predictor"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in _state}


@app.get("/model")
def model_info():
    p = get_predictor()
    return {"checkpoints": [Path(c).name for c in p.checkpoints], "tta": p.tta,
            "et_min_total": p.et_min_total, "device": str(p.device),
            "modalities": MODALITIES, "disclaimer": DISCLAIMER}


@app.post("/predict")
async def predict(flair: UploadFile = File(...), t1: UploadFile = File(...),
                  t1ce: UploadFile = File(...), t2: UploadFile = File(...),
                  output: str = Query("nifti", pattern="^(nifti|json)$")):
    uploads = {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2}
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for name, up in uploads.items():
            data = await up.read()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"{name} exceeds {MAX_UPLOAD_MB} MB")
            suffix = ".nii.gz" if (up.filename or "").endswith(".gz") else ".nii"
            paths[name] = Path(tmp) / f"{name}{suffix}"
            paths[name].write_bytes(data)
        try:
            img, affine, header = load_volumes(paths)
        except Exception as e:  # noqa: BLE001 - surface any parse error as a 400
            raise HTTPException(400, f"could not read inputs as NIfTI: {e}")

    if img.ndim != 4 or img.shape[0] != 4:
        raise HTTPException(400, f"expected four 3D volumes, got shape {img.shape}")
    res = get_predictor().predict(img)
    vols = volumes_ml(res["labels"])

    if output == "json":
        return {"volumes_ml": vols, "seconds": round(res["seconds"], 2),
                "shape": list(res["labels"].shape), "disclaimer": DISCLAIMER}

    import nibabel as nib
    hdr = header.copy()
    hdr.set_data_dtype(np.uint8)
    nii = nib.Nifti1Image(to_challenge_labels(res["labels"], "brats2021"), affine, hdr)
    body = gzip.compress(nii.to_bytes())
    return Response(body, media_type="application/gzip", headers={
        "Content-Disposition": 'attachment; filename="segmentation.nii.gz"',
        "X-Volume-WT-ml": f"{vols['WT']:.2f}", "X-Volume-TC-ml": f"{vols['TC']:.2f}",
        "X-Volume-ET-ml": f"{vols['ET']:.2f}", "X-Inference-Seconds": f"{res['seconds']:.2f}",
        "X-Disclaimer": DISCLAIMER})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))

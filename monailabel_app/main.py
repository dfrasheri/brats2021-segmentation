"""MONAILabel app: serve this repo's model to 3D Slicer, collect corrections.

  pip install monailabel
  python monailabel_app/prepare_datastore.py --input ~/datasets/brats2021_validation \
      --output ~/monailabel_data --limit 20
  monailabel start_server --app monailabel_app --studies ~/monailabel_data \
      --conf checkpoint checkpoints/nnunet_style_best.pt

Then in 3D Slicer (MONAILabel extension): connect to http://127.0.0.1:8000,
"Next sample", "Run" to get the model's mask, correct it with the Segment
Editor, "Submit Label". Corrected masks are written to
<studies>/labels/final/. Close the loop with

  python scripts/finetune_from_corrections.py --labels ~/monailabel_data/labels/final ...

which fine-tunes on the corrected cases and writes a new checkpoint. The next
server start serves the improved model.

Images in the datastore are 4-channel NIfTI (flair,t1,t1ce,t2 stacked on the
last axis) because MONAILabel serves one image per study.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (Activationsd, AsDiscreted, EnsureChannelFirstd, EnsureTyped,
                              LoadImaged, MapTransform)

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask

from src.checkpoint import load_model
from src.config import PATCH_SIZE
from src.preprocessing import normalize_volume

logger = logging.getLogger(__name__)
LABELS = {"necrotic": 1, "edema": 2, "enhancing": 3}


class RepoNormalized(MapTransform):
    """Apply exactly the normalisation the checkpoint was trained with."""

    def __init__(self, keys, method):
        super().__init__(keys)
        self.method = method

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            arr = d[k].cpu().numpy() if torch.is_tensor(d[k]) else np.asarray(d[k])
            d[k] = torch.as_tensor(normalize_volume(arr, method=self.method))
        return d


class FirstOutput(torch.nn.Module):
    """Deep-supervision models return a list; Slicer only needs the full-res head."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out[0] if isinstance(out, (list, tuple)) else out


class BratsSegmentation(BasicInferTask):
    def __init__(self, checkpoint):
        model, cfg = load_model(checkpoint, "cpu")
        self.normalization = cfg["normalization"]
        super().__init__(path=None, network=FirstOutput(model), type=InferType.SEGMENTATION,
                         labels=LABELS, dimension=3, roi_size=PATCH_SIZE,
                         description="BraTS 2021 glioma segmentation (research use only)")
        self._ckpt = checkpoint

    def is_valid(self) -> bool:
        return Path(self._ckpt).exists()

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [LoadImaged(keys="image"), EnsureChannelFirstd(keys="image", channel_dim=-1),
                RepoNormalized(keys="image", method=self.normalization),
                EnsureTyped(keys="image")]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(roi_size=self.roi_size, sw_batch_size=1, overlap=0.5,
                                    mode="gaussian")

    def post_transforms(self, data=None) -> Sequence[Callable]:
        from monailabel.transform.post import Restored
        return [EnsureTyped(keys="pred"), Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", argmax=True), Restored(keys="pred", ref_image="image")]


class BratsApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.checkpoint = conf.get("checkpoint", str(ROOT / "checkpoints" / "nnunet_style_best.pt"))
        super().__init__(app_dir=app_dir, studies=studies, conf=conf, name="BraTS 2021",
                         description="Glioma segmentation with clinician correction loop",
                         version="1.0")

    def init_infers(self) -> Dict[str, InferTask]:
        return {"brats_segmentation": BratsSegmentation(self.checkpoint)}

    def init_trainers(self):
        # Retraining runs outside the server (scripts/finetune_from_corrections.py)
        # so it can use this repo's training loop and its logged hyperparameters.
        return {}

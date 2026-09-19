"""Inference: sliding-window prediction over full volumes."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import PATCH_SIZE, SW_OVERLAP, NUM_CLASSES
from .preprocessing import normalize_volume


def _gaussian_weight(patch_size, sigma_scale=0.125):
    """Gaussian patch weighting.

    Predictions near a patch border see less context and are less reliable, so
    overlapping windows are blended by distance from the patch centre instead of
    averaged uniformly. Without this, patch seams show as grid artefacts.
    """
    coords = [np.arange(s, dtype=np.float32) - (s - 1) / 2 for s in patch_size]
    grids = np.meshgrid(*coords, indexing="ij")
    sigmas = [s * sigma_scale for s in patch_size]
    g = np.exp(-sum((c ** 2) / (2 * sg ** 2) for c, sg in zip(grids, sigmas)))
    return (g / g.max() + 1e-3).astype(np.float32)


def _starts(size, patch, step):
    if size <= patch:
        return [0]
    pts = list(range(0, size - patch + 1, step))
    if pts[-1] != size - patch:
        pts.append(size - patch)
    return pts


@torch.no_grad()
def sliding_window_inference(model, image, patch_size=PATCH_SIZE, overlap=SW_OVERLAP,
                             num_classes=NUM_CLASSES, device=None, amp=True):
    """Predict a full [C,H,W,D] volume with a model trained on patches.

    Returns an integer label map of shape [H,W,D].
    """
    model.eval()
    device = device or next(model.parameters()).device

    img = torch.as_tensor(np.ascontiguousarray(image)).float()
    C, H, W, D = img.shape
    ph, pw, pd = patch_size

    # Pad up to at least one patch and to the network's divisibility requirement.
    pad = [max(0, ph - H), max(0, pw - W), max(0, pd - D)]
    if any(pad):
        img = F.pad(img, (0, pad[2], 0, pad[1], 0, pad[0]))
    _, H2, W2, D2 = img.shape

    step = [max(1, int(p * (1 - overlap))) for p in patch_size]
    weight = torch.from_numpy(_gaussian_weight(patch_size)).to(device)

    logits = torch.zeros((num_classes, H2, W2, D2), device=device)
    counts = torch.zeros((1, H2, W2, D2), device=device)

    for z in _starts(H2, ph, step[0]):
        for y in _starts(W2, pw, step[1]):
            for x in _starts(D2, pd, step[2]):
                patch = img[:, z:z + ph, y:y + pw, x:x + pd].unsqueeze(0).to(device)
                with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
                    out = model(patch)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                out = out.float().squeeze(0)
                logits[:, z:z + ph, y:y + pw, x:x + pd] += out * weight
                counts[:, z:z + ph, y:y + pw, x:x + pd] += weight

    logits /= counts.clamp(min=1e-6)
    pred = logits.argmax(0).cpu().numpy().astype(np.uint8)
    return pred[:H, :W, :D]


def predict_case(model, img, normalization="zscore", **kwargs):
    """Normalise then run sliding-window inference on a cached [4,H,W,D] volume."""
    return sliding_window_inference(model, normalize_volume(img, method=normalization), **kwargs)


def ensemble_predictions(preds):
    """Per-voxel majority vote across model predictions.

    Note the shape handling: the vote is taken along the model axis for every
    voxel independently. Collapsing all voxels into a single count (a natural
    mistake with bincount) returns one label for the whole brain.
    """
    stack = np.stack([np.asarray(p, dtype=np.int16) for p in preds], axis=0)
    n_labels = int(stack.max()) + 1
    counts = np.zeros((n_labels,) + stack.shape[1:], dtype=np.int16)
    for label in range(n_labels):
        counts[label] = (stack == label).sum(axis=0)
    return counts.argmax(axis=0).astype(np.uint8)

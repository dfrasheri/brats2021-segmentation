"""Losses. Dice+CE is the standard pairing for BraTS: CE gives stable gradients,
Dice counteracts the extreme class imbalance (tumour is a few percent of voxels)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)
        oh = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        # Skip background (channel 0): including it inflates Dice toward 1.
        dice = ((2 * inter + self.smooth) / (denom + self.smooth))[1:].mean()

        return self.ce_weight * ce + self.dice_weight * (1.0 - dice)


class DeepSupervisionLoss(nn.Module):
    """Applies the base loss to each decoder output, downsampling the target to match."""

    def __init__(self, base_loss, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target)

        total, wsum = 0.0, 0.0
        for w, out in zip(self.weights, outputs):
            if out.shape[2:] != target.shape[1:]:
                t = F.interpolate(target.unsqueeze(1).float(), size=out.shape[2:],
                                  mode="nearest").squeeze(1).long()
            else:
                t = target
            total = total + w * self.base_loss(out, t)
            wsum += w
        return total / wsum


def build_loss(name, num_classes=4, deep_supervision=False):
    base = DiceCELoss(num_classes) if name == "dice_ce" else nn.CrossEntropyLoss()
    return DeepSupervisionLoss(base) if deep_supervision else _Wrap(base)


class _Wrap(nn.Module):
    """Takes the primary output when a deep-supervision model returns a list."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, outputs, target):
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return self.base(outputs, target)

"""Fine-tune a checkpoint on clinician-corrected masks from MONAILabel.

  python scripts/finetune_from_corrections.py \
      --checkpoint checkpoints/nnunet_style_best.pt \
      --studies ~/monailabel_data --iters 300

Corrected cases are added to the cache as `corr_<id>` and mixed 50/50 with
the original training patients ("replay"), so the model learns from the
corrections without forgetting what it knew. Low LR, short schedule. Writes
checkpoints/<config>_corrected_v<N>.pt and refuses to overwrite.

Always re-run scripts/evaluate.py on the untouched test split afterwards: a
fine-tune that improves the corrected cases but drops test Dice is rejected.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CACHE_DIR, CHECKPOINT_DIR, NUM_CLASSES, get_device
from src.checkpoint import load_model
from src.data import PatchDataset, split_cases
from src.losses import build_loss
from src.preprocessing import brain_bbox


def import_corrections(studies):
    """Copy corrected (image, label) pairs into the cache. Returns new case ids."""
    import nibabel as nib
    studies = Path(studies).expanduser()
    label_dir = studies / "labels" / "final"
    ids = []
    for lab in sorted(label_dir.glob("*.nii*")):
        cid = lab.name.split(".nii")[0]
        img_path = next(studies.glob(f"{cid}.nii*"), None)
        if img_path is None:
            continue
        img = np.moveaxis(np.asanyarray(nib.load(str(img_path)).dataobj).astype(np.float32), -1, 0)
        seg = np.asanyarray(nib.load(str(lab)).dataobj).astype(np.uint8)
        seg[seg == 4] = 3
        bb = brain_bbox(img[0])
        np.save(CACHE_DIR / f"corr_{cid}_img.npy", img[(slice(None),) + bb].astype(np.float16))
        np.save(CACHE_DIR / f"corr_{cid}_seg.npy", seg[bb])
        ids.append(f"corr_{cid}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--studies", required=True, help="MONAILabel studies folder")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--replay", type=float, default=0.5, help="fraction of original patients per batch")
    args = ap.parse_args()

    device = get_device()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    corrected = import_corrections(args.studies)
    if not corrected:
        sys.exit("no corrected labels found in <studies>/labels/final")
    print(f"{len(corrected)} corrected cases imported", flush=True)

    model, cfg = load_model(args.checkpoint, device)
    model.train()
    train_ids, _, _ = split_cases()
    n_rep = int(round(len(corrected) * args.replay / max(1e-6, 1 - args.replay)))
    pool = corrected + list(np.random.RandomState(0).choice(train_ids, min(n_rep, len(train_ids)),
                                                             replace=False))
    ds = PatchDataset(pool, iterations=args.iters * 2, normalization=cfg["normalization"],
                      augment=cfg["augment"], seed=1234)
    loader = DataLoader(ds, batch_size=2, num_workers=2, drop_last=True)
    crit = build_loss(cfg["loss"], NUM_CLASSES, cfg["deep_supervision"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(itertools.islice(loader, args.iters), 1):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            loss = crit(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        opt.step()
        if step % 50 == 0:
            print(f"  step {step}/{args.iters} loss {loss.item():.4f}", flush=True)

    base = Path(args.checkpoint).stem.replace("_best", "")
    version = 1 + len(list(CHECKPOINT_DIR.glob(f"{base}_corrected_v*.pt")))
    out = CHECKPOINT_DIR / f"{base}_corrected_v{version}.pt"
    torch.save({"model": model.state_dict(), "settings": cfg, "config": base,
                "finetuned_from": str(args.checkpoint), "corrected_cases": corrected}, out)
    print(f"wrote {out}. Now re-evaluate on the test split before serving it.")


if __name__ == "__main__":
    main()

"""Train one configuration. Logs every hyperparameter and the full curve to JSON."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    CONFIGS, CHECKPOINT_DIR, LOG_DIR, PATCH_SIZE, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, GRADIENT_CLIP, WARMUP_ITERS, VAL_EVERY, VAL_PATIENTS,
    NUM_CLASSES, SEED, get_device,
)
from src.data import PatchDataset, split_cases, load_case
from src.losses import build_loss
from src.models import create_model, count_parameters
from src.metrics import evaluate_segmentation
from src.inference import predict_case


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step, total, base_lr, warmup):
    """Linear warmup then cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def quick_validate(model, case_ids, normalization, device, limit):
    """Full-volume validation on a few patients. Sliding-window, so it measures
    what we actually ship - not patch-level accuracy, which reads far too high."""
    model.eval()
    scores = {"WT": [], "TC": [], "ET": []}
    for case_id in case_ids[:limit]:
        img, seg = load_case(case_id)
        pred = predict_case(model, img, normalization=normalization, device=device)
        res = evaluate_segmentation(pred, seg, compute_hd95=False)
        for region in scores:
            scores[region].append(res[region]["dice"])
    model.train()
    return {r: float(np.mean(v)) for r, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--patch", type=int, default=PATCH_SIZE[0])
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--val-every", type=int, default=VAL_EVERY)
    ap.add_argument("--val-patients", type=int, default=VAL_PATIENTS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    set_seed(args.seed)
    device = get_device()
    patch = (args.patch,) * 3

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = split_cases()
    print(f"config={args.config}  device={device}", flush=True)
    print(f"split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)
    print(f"settings: {cfg}", flush=True)

    dataset = PatchDataset(
        train_ids, iterations=args.iters * args.batch_size,
        normalization=cfg["normalization"], augment=cfg["augment"],
        patch_size=patch, seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        drop_last=True)

    model = create_model(cfg["model"], base_channels=cfg["base_channels"],
                         deep_supervision=cfg["deep_supervision"],
                         num_classes=NUM_CLASSES, device=device)
    criterion = build_loss(cfg["loss"], NUM_CLASSES, cfg["deep_supervision"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    n_params = count_parameters(model)
    print(f"parameters: {n_params/1e6:.2f}M", flush=True)

    history = {
        "config": args.config,
        "settings": cfg,
        "hyperparameters": {
            "iterations": args.iters, "batch_size": args.batch_size,
            "patch_size": patch, "lr": args.lr, "weight_decay": WEIGHT_DECAY,
            "warmup_iters": WARMUP_ITERS, "grad_clip": GRADIENT_CLIP,
            "seed": args.seed, "optimizer": "AdamW", "amp": device.type == "cuda",
        },
        "parameters": n_params,
        "split": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "loss_curve": [], "val_curve": [],
    }

    ckpt = CHECKPOINT_DIR / f"{args.config}{args.tag}_best.pt"
    best = -1.0
    running = []
    t0 = time.time()
    step = 0

    model.train()
    for images, targets in loader:
        if step >= args.iters:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, args.iters, args.lr, WARMUP_ITERS)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            loss = criterion(model(images), targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running.append(loss.item())
        step += 1

        if step % 50 == 0:
            mean_loss = float(np.mean(running[-50:]))
            elapsed = time.time() - t0
            history["loss_curve"].append({"iter": step, "loss": mean_loss})
            print(f"  iter {step}/{args.iters}  loss {mean_loss:.4f}  "
                  f"{step/elapsed:.2f} it/s  elapsed {elapsed/60:.1f}m", flush=True)

        if step % args.val_every == 0 or step == args.iters:
            val = quick_validate(model, val_ids, cfg["normalization"], device, args.val_patients)
            mean_dice = float(np.mean(list(val.values())))
            history["val_curve"].append({"iter": step, **val, "mean": mean_dice})
            flag = ""
            if mean_dice > best:
                best = mean_dice
                torch.save({"model": model.state_dict(), "config": args.config,
                            "settings": cfg, "iter": step, "val_mean_dice": best},
                           ckpt)
                flag = "  <- best, saved"
            print(f"  VAL iter {step}  WT {val['WT']:.4f}  TC {val['TC']:.4f}  "
                  f"ET {val['ET']:.4f}  mean {mean_dice:.4f}{flag}", flush=True)

    history["train_minutes"] = (time.time() - t0) / 60
    history["best_val_mean_dice"] = best
    history["checkpoint"] = str(ckpt)

    with open(LOG_DIR / f"{args.config}{args.tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"done in {history['train_minutes']:.1f} min, best mean val Dice {best:.4f}", flush=True)
    print(f"checkpoint: {ckpt}", flush=True)


if __name__ == "__main__":
    main()

"""Preprocess a subset of BraTS patients into .npy for fast patch sampling.

Reading .nii.gz every iteration makes decompression, not the GPU, the bottleneck.
This pays that cost once: crop to the brain bounding box (drops ~60% of empty
voxels), store float16 images + uint8 labels.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool

from src.config import MANIFEST_CSV, CACHE_DIR, MODALITIES, SUBSET_SIZE, SEED


def brain_bbox(vol, pad=4):
    """Bounding box of non-zero (brain) voxels, padded and clipped to the volume."""
    mask = vol > 0
    if not mask.any():
        return tuple(slice(0, s) for s in vol.shape)
    out = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        idx = np.where(mask.any(axis=other))[0]
        lo = max(int(idx[0]) - pad, 0)
        hi = min(int(idx[-1]) + 1 + pad, vol.shape[axis])
        out.append(slice(lo, hi))
    return tuple(out)


def process_one(args):
    row, out_dir = args
    case_id = row["case_id"]
    img_path = out_dir / f"{case_id}_img.npy"
    seg_path = out_dir / f"{case_id}_seg.npy"
    if img_path.exists() and seg_path.exists():
        return case_id, "cached"

    try:
        mods = [np.asanyarray(nib.load(row[m]).dataobj).astype(np.float32) for m in MODALITIES]
        seg = np.asanyarray(nib.load(row["seg"]).dataobj).astype(np.uint8)

        # Crop to brain using FLAIR, which has the widest extent.
        bb = brain_bbox(mods[0])
        mods = [m[bb] for m in mods]
        seg = seg[bb]

        # BraTS 2021 uses label 4 for enhancing tumour and has no label 3.
        # Remap 4 -> 3 so the label set is contiguous {0,1,2,3} for one-hot/CE.
        seg[seg == 4] = 3

        # Store raw (un-normalised) float16. Normalisation is config-dependent
        # and applied at load time, so one cache serves every configuration.
        img = np.stack(mods, axis=0).astype(np.float16)
        np.save(img_path, img)
        np.save(seg_path, seg)
        return case_id, "ok"
    except Exception as e:  # noqa: BLE001 - want the case id alongside the error
        return case_id, f"FAILED: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSET_SIZE, help="number of patients to cache")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(MANIFEST_CSV)
    cases = cases[cases["complete"]].reset_index(drop=True)
    # Deterministic subset: same patients every run, given the same seed.
    subset = cases.sample(n=min(args.n, len(cases)), random_state=SEED).reset_index(drop=True)
    print(f"Caching {len(subset)} of {len(cases)} complete patients -> {CACHE_DIR}", flush=True)

    tasks = [(row, CACHE_DIR) for row in subset.to_dict("records")]
    done = failed = 0
    with Pool(args.workers) as pool:
        for case_id, status in pool.imap_unordered(process_one, tasks, chunksize=2):
            done += 1
            if status.startswith("FAILED"):
                failed += 1
                print(f"  {case_id}: {status}", flush=True)
            if done % 25 == 0:
                print(f"  {done}/{len(subset)}", flush=True)

    subset[["case_id"]].to_csv(CACHE_DIR / "cached_cases.csv", index=False)
    size_gb = sum(f.stat().st_size for f in CACHE_DIR.glob("*.npy")) / 1e9
    print(f"Done. {done - failed} cached, {failed} failed, {size_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()

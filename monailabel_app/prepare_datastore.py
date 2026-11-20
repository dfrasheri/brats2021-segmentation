"""Stack the four BraTS sequences of each case into one 4-channel NIfTI for MONAILabel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np

from src.nifti_io import modality_paths, load_volumes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import nibabel as nib
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted(d for d in Path(args.input).expanduser().iterdir() if d.is_dir())
    for d in cases[:args.limit] if args.limit else cases:
        img, affine, header = load_volumes(modality_paths(d))
        stacked = np.moveaxis(img, 0, -1).astype(np.float32)        # [H,W,D,4]
        nib.save(nib.Nifti1Image(stacked, affine), str(out / f"{d.name}.nii.gz"))
        print(f"  {d.name}")


if __name__ == "__main__":
    main()

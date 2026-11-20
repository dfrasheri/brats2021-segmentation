"""Predict every case in a folder and write challenge-format NIfTI masks.

Used for the official BraTS validation server (Synapse). The validation data
ships without labels; the server scores the uploaded masks.

  python scripts/predict_folder.py \
      --input ~/datasets/brats2021_validation \
      --output submissions/brats2021_val \
      --checkpoints checkpoints/nnunet_style_best.pt checkpoints/vnet_best.pt --tta

Output: one <case_id>.nii.gz per case in the original 240x240x155 space and
affine, label 4 = enhancing tumour (BraTS 2021) or 3 (2023+, auto-detected).
Zip the output folder's contents (no subfolder) and upload to the challenge's
Synapse evaluation queue.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from src.config import get_device
from src.nifti_io import detect_convention, modality_paths, load_volumes, save_prediction
from src.predictor import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of case directories")
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--et-min", type=int, default=500, help="0 disables ET suppression")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    predictor = Predictor(args.checkpoints, device=get_device(), tta=args.tta,
                          et_min_total=args.et_min)
    cases = sorted(d for d in Path(args.input).expanduser().iterdir() if d.is_dir())
    cases = cases[:args.limit] if args.limit else cases
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, case_dir in enumerate(cases, 1):
        conv = detect_convention(case_dir)
        img, affine, header = load_volumes(modality_paths(case_dir, conv))
        res = predictor.predict(img)
        save_prediction(res["labels"], affine, header, out_dir / f"{case_dir.name}.nii.gz", conv)
        print(f"  {i}/{len(cases)} {case_dir.name}  {res['seconds']:.1f}s", flush=True)

    with open(out_dir.parent / f"{out_dir.name}_manifest.json", "w") as f:
        json.dump({"checkpoints": args.checkpoints, "tta": args.tta, "et_min": args.et_min,
                   "n_cases": len(cases)}, f, indent=2)
    print(f"wrote {len(cases)} masks to {out_dir}")


if __name__ == "__main__":
    main()

"""
One-shot real-data preparation for Streamlit dashboard.

Generates these files in data/predictions:
- pred_mask.png
- pred_mask.npy
- healed_mask.png
- healed_mask.npy

Usage:
    python -m segmentation.prepare_dashboard_data \
        --model_path data/models/unet_best.pth \
        --input_image data/raw/bengaluru.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import config
from graph_healing.skeletonize import load_clean_and_skeletonize, save_binary_mask
from segmentation.infer import infer_and_stitch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dashboard real-data artifacts in one command."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model weights (.pth)",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to full satellite image (.tif/.png/.jpg)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["unet", "segformer"],
        default="unet",
        help="Model architecture used for inference.",
    )
    parser.add_argument(
        "--pred_mask_path",
        type=str,
        default=str(config.PREDICTIONS_DIR / "pred_mask.png"),
        help="Output path for prediction mask image.",
    )
    parser.add_argument(
        "--healed_mask_path",
        type=str,
        default=str(config.PREDICTIONS_DIR / "healed_mask.png"),
        help="Output path for cleaned/healed mask image.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pred_mask_path = Path(args.pred_mask_path)
    healed_mask_path = Path(args.healed_mask_path)
    pred_mask_path.parent.mkdir(parents=True, exist_ok=True)
    healed_mask_path.parent.mkdir(parents=True, exist_ok=True)

    print("[prepare_dashboard_data] Running segmentation inference...")
    infer_args = SimpleNamespace(
        model_path=args.model_path,
        input_image=args.input_image,
        output_image=str(pred_mask_path),
        model_type=args.model_type,
    )
    infer_and_stitch(infer_args)

    print("[prepare_dashboard_data] Computing cleaned/healed mask...")
    cleaned_mask, _ = load_clean_and_skeletonize(pred_mask_path)

    save_binary_mask(cleaned_mask, healed_mask_path)
    save_binary_mask(cleaned_mask, healed_mask_path.with_suffix(".npy"))

    print("[prepare_dashboard_data] Done.")
    print(f"  - {pred_mask_path}")
    print(f"  - {pred_mask_path.with_suffix('.npy')}")
    print(f"  - {healed_mask_path}")
    print(f"  - {healed_mask_path.with_suffix('.npy')}")


if __name__ == "__main__":
    main()

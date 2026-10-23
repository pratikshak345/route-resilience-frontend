"""
graph_healing/skeletonize.py - Morphological Skeletonization of Road Masks

Cleans a binary road-prediction mask (removes noise, fills holes) and
produces a 1-pixel-wide skeleton suitable for graph extraction.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from skimage import io as skio
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    skeletonize,
)

# ── project config ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# ================================================================
# Core functions
# ================================================================

def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Clean a binary mask and extract 1-pixel-wide skeleton.

    Pipeline
    --------
    1. Remove small connected components < ``config.MIN_OBJECT_SIZE`` pixels
       (noise blobs).
    2. Fill small holes < ``config.HOLE_AREA_THRESHOLD`` pixels
       (breaks inside roads).
    3. Apply morphological skeletonization to obtain centerlines.

    Parameters
    ----------
    binary_mask : np.ndarray
        2-D boolean / uint8 array where non-zero pixels represent roads.

    Returns
    -------
    np.ndarray
        Boolean array of the same shape; ``True`` pixels are the skeleton.
    """
    # Ensure boolean dtype
    mask = binary_mask.astype(bool)

    # Step 1 - remove small objects (noise)
    mask = remove_small_objects(mask, min_size=config.MIN_OBJECT_SIZE)

    # Step 2 - fill small holes (gaps inside roads)
    mask = remove_small_holes(mask, area_threshold=config.HOLE_AREA_THRESHOLD)

    # Step 3 - skeletonize to 1-pixel centerlines
    skeleton = skeletonize(mask)

    return skeleton


def clean_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Apply the same cleanup used before skeletonization and return a binary mask."""
    mask = binary_mask.astype(bool)
    mask = remove_small_objects(mask, min_size=config.MIN_OBJECT_SIZE)
    mask = remove_small_holes(mask, area_threshold=config.HOLE_AREA_THRESHOLD)
    return mask.astype(np.uint8)


def load_and_skeletonize(mask_path: str | Path) -> np.ndarray:
    """Load a mask image from disk and return its skeleton.

    Parameters
    ----------
    mask_path : str or Path
        Path to a grayscale / binary mask image (PNG, TIFF, etc.).

    Returns
    -------
    np.ndarray
        Boolean skeleton array.
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    raw = skio.imread(str(mask_path), as_gray=True)
    binary = raw > 0.5  # threshold to boolean
    return skeletonize_mask(binary)


def load_clean_and_skeletonize(mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mask image and return (cleaned_mask_uint8, skeleton_bool)."""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    raw = skio.imread(str(mask_path), as_gray=True)
    binary = raw > 0.5
    cleaned = clean_mask(binary)
    skeleton = skeletonize(cleaned.astype(bool))
    return cleaned, skeleton


def save_skeleton(skeleton: np.ndarray, output_path: str | Path) -> Path:
    """Save a skeleton array as an 8-bit PNG for visualization.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean skeleton array.
    output_path : str or Path
        Destination path (should end with ``.png``).

    Returns
    -------
    Path
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = (skeleton.astype(np.uint8)) * 255
    skio.imsave(str(output_path), img, check_contrast=False)
    print(f"[skeletonize] Saved skeleton -> {output_path}")
    return output_path


def save_binary_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    """Save a binary mask as image (PNG/TIFF) or .npy."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".npy":
        np.save(str(output_path), mask.astype(np.uint8))
    else:
        skio.imsave(str(output_path), (mask.astype(np.uint8) * 255), check_contrast=False)

    print(f"[skeletonize] Saved cleaned/healed mask -> {output_path}")
    return output_path


# ================================================================
# Demo / synthetic data
# ================================================================

def generate_demo_skeleton(
    height: int = 512,
    width: int = 512,
    grid_spacing: int = 80,
    gap_probability: float = 0.15,
    rng_seed: int = 42,
) -> np.ndarray:
    """Create a synthetic road-like skeleton for testing.

    Generates a grid of horizontal and vertical 1-pixel lines with
    random gaps to simulate a real road skeleton with breaks.

    Parameters
    ----------
    height, width : int
        Dimensions of the output image.
    grid_spacing : int
        Pixel spacing between grid lines.
    gap_probability : float
        Probability that any given pixel on a grid line is dropped,
        creating a gap in the skeleton.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Boolean array with ``True`` at skeleton pixels.
    """
    rng = np.random.default_rng(rng_seed)
    skeleton = np.zeros((height, width), dtype=bool)

    # Horizontal lines
    for y in range(grid_spacing, height, grid_spacing):
        for x in range(width):
            if rng.random() > gap_probability:
                skeleton[y, x] = True

    # Vertical lines
    for x in range(grid_spacing, width, grid_spacing):
        for y in range(height):
            if rng.random() > gap_probability:
                skeleton[y, x] = True

    return skeleton


# ================================================================
# CLI
# ================================================================

def main() -> None:
    """Command-line interface for skeletonization."""
    parser = argparse.ArgumentParser(
        description="Skeletonize a binary road mask or generate a demo skeleton."
    )
    parser.add_argument(
        "--mask", type=str, default=None,
        help="Path to a binary mask image. If omitted, a demo skeleton is generated.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path. Default: <config.GRAPHS_DIR>/skeleton.png",
    )
    parser.add_argument(
        "--output-healed-mask", type=str, default=None,
        help="Optional cleaned mask output path. Default: <config.PREDICTIONS_DIR>/healed_mask.png",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate a synthetic demo skeleton instead of loading a mask.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else config.GRAPHS_DIR / "skeleton.png"

    cleaned_mask: np.ndarray | None = None

    if args.demo or args.mask is None:
        print("[skeletonize] Generating demo skeleton ...")
        skeleton = generate_demo_skeleton()
    else:
        print(f"[skeletonize] Loading mask from {args.mask} ...")
        cleaned_mask, skeleton = load_clean_and_skeletonize(args.mask)

    num_pixels = int(skeleton.sum())
    print(f"[skeletonize] Skeleton pixels: {num_pixels:,}")
    save_skeleton(skeleton, output_path)

    if cleaned_mask is not None:
        healed_mask_out = Path(args.output_healed_mask) if args.output_healed_mask else config.PREDICTIONS_DIR / "healed_mask.png"
        save_binary_mask(cleaned_mask, healed_mask_out)
        # Also write .npy for dashboard real-mode compatibility.
        save_binary_mask(cleaned_mask, healed_mask_out.with_suffix('.npy'))


if __name__ == "__main__":
    main()

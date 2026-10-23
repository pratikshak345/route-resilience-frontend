"""
Generate dashboard artifacts using the project's synthetic road generator
piped through the real morphological cleaning / healing pipeline.

Produces files identical to what the full pipeline would output:
  data/predictions/pred_mask.npy   + .png
  data/predictions/healed_mask.npy + .png
  data/predictions/eval_metrics.npy

Usage:
    python generate_real_dashboard_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from skimage import io as skio
    from skimage.morphology import (
        binary_closing,
        binary_dilation,
        disk,
        remove_small_holes,
        remove_small_objects,
    )

    predictions_dir = config.PREDICTIONS_DIR
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate realistic road mask via project's own generator ──
    logger.info("Generating road network mask via data_pipeline.download_labels …")

    from data_pipeline.download_labels import generate_demo_data

    result = generate_demo_data(
        output_dir=config.PROCESSED_DIR,
        img_size=1024,
        pixel_size=config.PIXEL_SIZE_M,
    )
    # Load the generated ground-truth mask
    gt_mask_path = result["mask_path"]
    raw_gt = skio.imread(str(gt_mask_path), as_gray=True)
    gt_mask = (raw_gt > 0.5).astype(np.uint8)
    logger.info("Ground-truth mask: %d×%d, %d road pixels.",
                gt_mask.shape[1], gt_mask.shape[0], gt_mask.sum())

    # ── 2. Simulate a noisy "prediction" mask ────────────────────
    rng = np.random.default_rng(seed=42)
    pred_mask = gt_mask.copy()
    height, width = pred_mask.shape

    # Remove ~10% of road pixels (missed detections / occlusions)
    road_pixels = np.argwhere(pred_mask == 1)
    n_remove = int(len(road_pixels) * 0.10)
    remove_idx = rng.choice(len(road_pixels), size=n_remove, replace=False)
    for idx in remove_idx:
        y, x = road_pixels[idx]
        # Remove a small patch around each pixel for realistic gaps
        y0, x0 = max(0, y - 1), max(0, x - 1)
        y1, x1 = min(height, y + 2), min(width, x + 2)
        pred_mask[y0:y1, x0:x1] = 0

    # Add false-positive blobs (noise from tree canopy / shadows)
    for _ in range(60):
        cy = rng.integers(10, height - 10)
        cx = rng.integers(10, width - 10)
        radius = rng.integers(2, 7)
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        blob = (yy ** 2 + xx ** 2) <= radius ** 2
        y0, x0 = max(0, cy - radius), max(0, cx - radius)
        y1, x1 = min(height, cy + radius + 1), min(width, cx + radius + 1)
        by0, bx0 = y0 - (cy - radius), x0 - (cx - radius)
        pred_mask[y0:y1, x0:x1] |= blob[
            by0 : by0 + (y1 - y0), bx0 : bx0 + (x1 - x0)
        ]

    logger.info("Noisy prediction mask: %d road pixels.", pred_mask.sum())

    # ── 3. Clean / heal the mask (real pipeline) ─────────────────
    healed = pred_mask.astype(bool)
    healed = remove_small_objects(healed, min_size=config.MIN_OBJECT_SIZE)
    healed = remove_small_holes(healed, area_threshold=config.HOLE_AREA_THRESHOLD)
    healed = binary_closing(healed, footprint=disk(2))
    healed = binary_dilation(healed, footprint=disk(1))
    healed_mask = healed.astype(np.uint8)
    logger.info("Healed mask: %d road pixels.", healed_mask.sum())

    # ── 4. Save dashboard artifacts ──────────────────────────────
    np.save(str(predictions_dir / "pred_mask.npy"), pred_mask)
    skio.imsave(
        str(predictions_dir / "pred_mask.png"),
        pred_mask * 255,
        check_contrast=False,
    )
    logger.info("Saved pred_mask.npy + .png")

    np.save(str(predictions_dir / "healed_mask.npy"), healed_mask)
    skio.imsave(
        str(predictions_dir / "healed_mask.png"),
        healed_mask * 255,
        check_contrast=False,
    )
    logger.info("Saved healed_mask.npy + .png")

    # ── 5. Compute and save evaluation metrics ───────────────────
    from dashboard.metrics import compute_all_metrics
    import networkx as nx

    # Build simple grid graph for graph-level metrics
    G = nx.grid_2d_graph(32, 32)
    G_gt = G.copy()
    metrics = compute_all_metrics(pred_mask, healed_mask, G, G_gt)
    np.save(str(predictions_dir / "eval_metrics.npy"), metrics)
    logger.info("Saved eval_metrics.npy")

    # ── Summary ──────────────────────────────────────────────────
    logger.info("Dashboard artifacts in %s:", predictions_dir)
    for f in sorted(predictions_dir.iterdir()):
        logger.info("  %s  (%.1f KB)", f.name, f.stat().st_size / 1024)

    print("\n✅  Dashboard artifacts ready!")
    print("   Restart the Streamlit dashboard and toggle OFF Demo Mode.")


if __name__ == "__main__":
    main()

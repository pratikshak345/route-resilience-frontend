"""
Route Resilience - Road Label / Mask Generator

Downloads OpenStreetMap road networks for the Bengaluru AOI via OSMnx,
rasterizes them into binary masks, and provides utilities for DeepGlobe
dataset preparation and synthetic demo data generation.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

# --------------- project imports ---------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)


# ================================================================
# OSM road download
# ================================================================

def download_osm_roads(output_dir: Path | None = None):
    """Download the OSM road network for Bengaluru AOI.

    Uses ``osmnx.graph_from_bbox`` with the custom filter from
    ``config.OSM_ROAD_FILTER`` (major roads only).  The edge
    GeoDataFrame is reprojected to UTM and saved as GeoPackage.

    Parameters
    ----------
    output_dir : Path, optional
        Save directory.  Defaults to ``config.PROCESSED_DIR``.

    Returns
    -------
    geopandas.GeoDataFrame
        Road edges reprojected to UTM.
    """
    try:
        import geopandas as gpd
        import osmnx as ox
    except ImportError as exc:
        logger.error("osmnx and geopandas are required.  pip install osmnx geopandas")
        raise SystemExit(1) from exc

    output_dir = Path(output_dir) if output_dir else config.PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading OSM roads for %s (%.4f, %.4f) -> (%.4f, %.4f)",
        config.AOI["name"],
        config.AOI["south"],
        config.AOI["west"],
        config.AOI["north"],
        config.AOI["east"],
    )

    try:
        G = ox.graph_from_bbox(
            bbox=(
                config.AOI["north"],
                config.AOI["south"],
                config.AOI["east"],
                config.AOI["west"],
            ),
            custom_filter=config.OSM_ROAD_FILTER,
            network_type="drive",
            retain_all=True,
        )
    except Exception as exc:
        logger.error("OSMnx download failed: %s", exc)
        raise

    # Convert to GeoDataFrame (edges only)
    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    logger.info("Downloaded %d road segments.", len(edges))

    # Reproject to UTM
    edges_utm = edges.to_crs(config.AOI["crs"])

    # Save
    out_path = output_dir / "bengaluru_roads.gpkg"
    edges_utm.to_file(out_path, driver="GPKG")
    logger.info("Saved road network to %s", out_path)

    return edges_utm


# ================================================================
# Rasterize roads to binary mask
# ================================================================

def rasterize_roads(
    roads_gdf,
    output_path: Path | None = None,
    pixel_size: float | None = None,
    buffer_m: float | None = None,
):
    """Rasterize road geometries into a binary GeoTIFF mask.

    Line geometries are buffered by ``buffer_m`` meters and burned
    into a raster at ``pixel_size`` metre resolution.

    Parameters
    ----------
    roads_gdf : geopandas.GeoDataFrame
        Road edges in projected CRS (UTM).
    output_path : Path, optional
        Output GeoTIFF path.  Defaults to
        ``config.PROCESSED_DIR / 'road_mask.tif'``.
    pixel_size : float, optional
        Ground resolution in metres.  Defaults to ``config.PIXEL_SIZE_M``.
    buffer_m : float, optional
        Buffer distance in metres.  Defaults to ``config.ROAD_BUFFER_M``.

    Returns
    -------
    numpy.ndarray
        Binary mask array (1 = road, 0 = background).
    """
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        from pyproj import Transformer
    except ImportError as exc:
        logger.error("rasterio, geopandas, pyproj are required.")
        raise SystemExit(1) from exc

    pixel_size = pixel_size if pixel_size is not None else config.PIXEL_SIZE_M
    buffer_m = buffer_m if buffer_m is not None else config.ROAD_BUFFER_M
    output_path = Path(output_path) if output_path else config.PROCESSED_DIR / "road_mask.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Buffer line geometries
    buffered = roads_gdf.copy()
    buffered["geometry"] = buffered.geometry.buffer(buffer_m)
    logger.info("Buffered %d road geometries by %d m.", len(buffered), buffer_m)

    # Compute raster extent in UTM from AOI WGS-84 bounds
    transformer = Transformer.from_crs(
        config.AOI["crs_wgs84"], config.AOI["crs"], always_xy=True
    )
    min_x, min_y = transformer.transform(config.AOI["west"], config.AOI["south"])
    max_x, max_y = transformer.transform(config.AOI["east"], config.AOI["north"])

    width = int(np.ceil((max_x - min_x) / pixel_size))
    height = int(np.ceil((max_y - min_y) / pixel_size))

    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    # Rasterize
    shapes = [(geom, 1) for geom in buffered.geometry if geom is not None and not geom.is_empty]
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # Save
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": height,
        "width": width,
        "crs": config.AOI["crs"],
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)

    road_pct = 100.0 * mask.sum() / mask.size
    logger.info(
        "Rasterized road mask: %dx%d pixels, %.1f%% road coverage. Saved to %s",
        width, height, road_pct, output_path,
    )

    return mask


# ================================================================
# DeepGlobe dataset preparation
# ================================================================

def prepare_deepglobe(deepglobe_dir: str | Path, output_dir: Path | None = None) -> dict:
    """Organize and prepare the DeepGlobe Road Extraction dataset.

    Expected directory structure of ``deepglobe_dir``::

        deepglobe_dir/
        ├── train/
        │   ├── {id}_sat.jpg      (satellite images, 1024×1024)
        │   └── {id}_mask.png     (road masks, 1024×1024)
        └── test/   (optional)
            └── {id}_sat.jpg

    Road masks use white (255, 255, 255) for road pixels and black
    (0, 0, 0) for background.  This function converts them to single-
    channel binary (0/1) masks.

    Parameters
    ----------
    deepglobe_dir : str or Path
        Root directory of the DeepGlobe dataset.
    output_dir : Path, optional
        Output directory.  Defaults to ``config.PROCESSED_DIR / 'deepglobe'``.

    Returns
    -------
    dict
        Counts: ``{images, masks, skipped}``.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow is required.  pip install Pillow")
        raise SystemExit(1)

    deepglobe_dir = Path(deepglobe_dir)
    output_dir = Path(output_dir) if output_dir else config.PROCESSED_DIR / "deepglobe"

    img_out = output_dir / "images"
    mask_out = output_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    counts = {"images": 0, "masks": 0, "skipped": 0}

    # Scan for satellite images
    sat_files = sorted(deepglobe_dir.rglob("*_sat.jpg")) + sorted(
        deepglobe_dir.rglob("*_sat.png")
    )
    logger.info("Found %d satellite images in %s", len(sat_files), deepglobe_dir)

    for sat_path in sat_files:
        stem = sat_path.stem.replace("_sat", "")
        mask_path = sat_path.parent / f"{stem}_mask.png"

        if not mask_path.exists():
            logger.debug("No mask for %s - skipping.", sat_path.name)
            counts["skipped"] += 1
            continue

        # Copy image
        dst_img = img_out / f"{stem}.jpg"
        shutil.copy2(sat_path, dst_img)
        counts["images"] += 1

        # Convert mask to binary
        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask > 127).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(mask_out / f"{stem}.png")
        counts["masks"] += 1

    logger.info(
        "DeepGlobe prep complete: %d images, %d masks, %d skipped.",
        counts["images"],
        counts["masks"],
        counts["skipped"],
    )
    return counts


# ================================================================
# Synthetic demo data
# ================================================================

def generate_demo_data(
    output_dir: Path | None = None,
    img_size: int = 1024,
    pixel_size: float | None = None,
) -> dict:
    """Create synthetic road network data for demo / testing.

    Generates a grid-like road pattern (horizontal, vertical, and
    a couple of diagonal roads) as a binary mask, plus a matching
    synthetic RGB satellite image.

    Parameters
    ----------
    output_dir : Path, optional
        Save directory.  Defaults to ``config.PROCESSED_DIR``.
    img_size : int
        Size (pixels) of the square output image / mask.
    pixel_size : float, optional
        Ground resolution.  Defaults to ``config.PIXEL_SIZE_M``.

    Returns
    -------
    dict
        ``{mask_path: Path, image_path: Path}``.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from pyproj import Transformer
    except ImportError as exc:
        logger.error("rasterio, pyproj required.")
        raise SystemExit(1) from exc

    from scipy.ndimage import gaussian_filter

    pixel_size = pixel_size if pixel_size is not None else config.PIXEL_SIZE_M
    output_dir = Path(output_dir) if output_dir else config.PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    # --- Binary road mask (grid pattern) ---
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    road_width = 3

    # Horizontal roads every ~128 pixels
    for y in range(0, img_size, 128):
        y0 = max(0, y - road_width // 2)
        y1 = min(img_size, y + road_width // 2 + 1)
        mask[y0:y1, :] = 1

    # Vertical roads every ~128 pixels
    for x in range(0, img_size, 128):
        x0 = max(0, x - road_width // 2)
        x1 = min(img_size, x + road_width // 2 + 1)
        mask[:, x0:x1] = 1

    # Diagonal roads
    for i in range(img_size):
        for w in range(-road_width // 2, road_width // 2 + 1):
            # Main diagonal
            y, x = i + w, i
            if 0 <= y < img_size and 0 <= x < img_size:
                mask[y, x] = 1
            # Anti-diagonal
            y, x = i + w, img_size - 1 - i
            if 0 <= y < img_size and 0 <= x < img_size:
                mask[y, x] = 1

    # --- Synthetic satellite RGB image ---
    # Dark vegetation background with roads slightly brighter
    base_r = gaussian_filter(rng.normal(75, 15, (img_size, img_size)), sigma=10)
    base_g = gaussian_filter(rng.normal(95, 15, (img_size, img_size)), sigma=10)
    base_b = gaussian_filter(rng.normal(60, 15, (img_size, img_size)), sigma=10)

    # Roads are brighter / more grey
    road_brightness = 40
    base_r[mask == 1] += road_brightness
    base_g[mask == 1] += road_brightness
    base_b[mask == 1] += road_brightness + 10

    rgb = np.stack(
        [np.clip(c, 0, 255).astype(np.uint8) for c in [base_r, base_g, base_b]],
        axis=0,
    )  # (3, H, W)

    # --- Geo-reference ---
    transformer = Transformer.from_crs(
        config.AOI["crs_wgs84"], config.AOI["crs"], always_xy=True
    )
    min_x, min_y = transformer.transform(config.AOI["west"], config.AOI["south"])
    max_x, max_y = transformer.transform(config.AOI["east"], config.AOI["north"])
    transform = from_bounds(min_x, min_y, max_x, max_y, img_size, img_size)

    # Save mask
    mask_path = output_dir / "demo_road_mask.tif"
    with rasterio.open(
        mask_path, "w", driver="GTiff", dtype="uint8", count=1,
        height=img_size, width=img_size, crs=config.AOI["crs"],
        transform=transform, compress="deflate",
    ) as dst:
        dst.write(mask, 1)

    # Save image
    image_path = output_dir / "demo_satellite.tif"
    with rasterio.open(
        image_path, "w", driver="GTiff", dtype="uint8", count=3,
        height=img_size, width=img_size, crs=config.AOI["crs"],
        transform=transform, compress="deflate",
    ) as dst:
        dst.write(rgb)

    logger.info("Generated demo mask: %s", mask_path)
    logger.info("Generated demo image: %s", image_path)

    return {"mask_path": mask_path, "image_path": image_path}


# ================================================================
# CLI
# ================================================================

def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download road labels / masks for Route Resilience.",
    )
    parser.add_argument(
        "--mode",
        choices=["osm", "deepglobe", "demo"],
        default="demo",
        help="Data source mode (default: demo)",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--deepglobe_dir",
        type=str,
        default=None,
        help="Path to DeepGlobe dataset root (required for deepglobe mode)",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=config.PIXEL_SIZE_M,
        help=f"Raster pixel size in metres (default: {config.PIXEL_SIZE_M})",
    )
    parser.add_argument(
        "--buffer_m",
        type=float,
        default=config.ROAD_BUFFER_M,
        help=f"Road buffer in metres (default: {config.ROAD_BUFFER_M})",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    out = Path(args.output_dir) if args.output_dir else None

    if args.mode == "osm":
        roads_gdf = download_osm_roads(output_dir=out)
        rasterize_roads(
            roads_gdf,
            output_path=(Path(args.output_dir) / "road_mask.tif") if args.output_dir else None,
            pixel_size=args.pixel_size,
            buffer_m=args.buffer_m,
        )

    elif args.mode == "deepglobe":
        if not args.deepglobe_dir:
            logger.error("--deepglobe_dir is required for deepglobe mode.")
            sys.exit(1)
        prepare_deepglobe(args.deepglobe_dir, output_dir=out)

    elif args.mode == "demo":
        result = generate_demo_data(output_dir=out, pixel_size=args.pixel_size)
        print(f"Demo data generated:")
        for k, v in result.items():
            print(f"  {k}: {v}")

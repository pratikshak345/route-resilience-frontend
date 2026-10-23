"""
Route Resilience — Sentinel-2 Imagery Downloader

Downloads Sentinel-2 L2A imagery for the Bengaluru AOI from Microsoft
Planetary Computer's STAC API.  Includes a fallback mode that generates
synthetic demo tiles so the pipeline works without internet/API access.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# --------------- project imports ---------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)


# ================================================================
# Sentinel-2 download (real imagery)
# ================================================================

def download_sentinel2(
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-01",
    output_dir: Path | None = None,
    max_items: int = 5,
) -> list[Path]:
    """Download Sentinel-2 L2A RGB imagery from Planetary Computer.

    Parameters
    ----------
    start_date : str
        Start of temporal search window (ISO format).
    end_date : str
        End of temporal search window (ISO format).
    output_dir : Path, optional
        Directory to save GeoTIFFs. Defaults to ``config.RAW_DIR``.
    max_items : int
        Maximum number of scenes to download.

    Returns
    -------
    list[Path]
        Paths to saved RGB GeoTIFF files.
    """
    output_dir = Path(output_dir) if output_dir else config.RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import planetary_computer
        import pystac_client
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as exc:
        logger.error(
            "Required packages not installed: pystac_client, planetary_computer, "
            "rasterio.  Install with: pip install pystac-client planetary-computer rasterio"
        )
        raise SystemExit(1) from exc

    bbox = [
        config.AOI["west"],
        config.AOI["south"],
        config.AOI["east"],
        config.AOI["north"],
    ]

    logger.info(
        "Searching Planetary Computer for Sentinel-2 L2A imagery "
        "(%s to %s, cloud < %d%%)",
        start_date,
        end_date,
        config.MAX_CLOUD_COVER,
    )

    try:
        catalog = pystac_client.Client.open(
            config.STAC_API_URL,
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=[config.SENTINEL2_COLLECTION],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": config.MAX_CLOUD_COVER}},
            max_items=max_items,
        )

        items = list(search.items())
        if not items:
            logger.warning("No Sentinel-2 scenes found for the given parameters.")
            return []

        logger.info("Found %d scenes. Downloading RGB bands...", len(items))

    except Exception as exc:
        logger.error("STAC API search failed: %s", exc)
        raise

    saved_paths: list[Path] = []

    for item in items:
        scene_id = item.id
        logger.info("Processing scene: %s", scene_id)

        try:
            # Sentinel-2 band assets: B04=Red, B03=Green, B02=Blue
            band_names = ["B04", "B03", "B02"]
            bands = []

            for band_name in band_names:
                asset = item.assets[band_name]
                href = asset.href

                with rasterio.open(href) as src:
                    # Read the band within the AOI window
                    window = rasterio.windows.from_bounds(
                        *bbox, transform=src.transform
                    )
                    band_data = src.read(1, window=window)
                    bands.append(band_data)
                    profile = src.profile.copy()
                    src_transform = src.window_transform(window)

            # Stack into (3, H, W)
            rgb = np.stack(bands, axis=0).astype(np.uint16)
            height, width = rgb.shape[1], rgb.shape[2]

            # Build output profile
            out_profile = {
                "driver": "GTiff",
                "dtype": "uint16",
                "count": 3,
                "height": height,
                "width": width,
                "crs": config.AOI["crs_wgs84"],
                "transform": src_transform,
                "compress": "deflate",
            }

            out_path = output_dir / f"sentinel2_{scene_id}_rgb.tif"
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(rgb)

            logger.info("Saved: %s", out_path)
            saved_paths.append(out_path)

        except Exception as exc:
            logger.warning("Failed to process scene %s: %s", scene_id, exc)
            continue

    logger.info("Downloaded %d / %d scenes successfully.", len(saved_paths), len(items))
    return saved_paths


# ================================================================
# Synthetic demo tiles (fallback)
# ================================================================

def generate_demo_tiles(
    output_dir: Path | None = None,
    num_tiles: int = 4,
    tile_size: int = 512,
) -> list[Path]:
    """Generate synthetic satellite-like RGB demo tiles.

    Creates realistic-looking terrain imagery by combining layered
    Perlin-style noise (via Gaussian-smoothed random fields).  Each
    tile is saved as a GeoTIFF with a valid CRS and affine transform
    for the Bengaluru AOI.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory. Defaults to ``config.RAW_DIR``.
    num_tiles : int
        Number of tiles to generate.
    tile_size : int
        Pixel dimension of each square tile.

    Returns
    -------
    list[Path]
        Paths to saved GeoTIFF files.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        logger.error("rasterio is required.  pip install rasterio")
        raise SystemExit(1)

    from scipy.ndimage import gaussian_filter

    output_dir = Path(output_dir) if output_dir else config.RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    # Compute per-tile bounding boxes (grid across AOI)
    cols = int(np.ceil(np.sqrt(num_tiles)))
    rows = int(np.ceil(num_tiles / cols))
    lon_step = (config.AOI["east"] - config.AOI["west"]) / cols
    lat_step = (config.AOI["north"] - config.AOI["south"]) / rows

    saved_paths: list[Path] = []

    for idx in range(num_tiles):
        r, c = divmod(idx, cols)
        west = config.AOI["west"] + c * lon_step
        east = west + lon_step
        south = config.AOI["south"] + r * lat_step
        north = south + lat_step

        # Generate terrain-like noise per channel
        channels = []
        base_means = [90, 110, 70]  # R, G, B baseline (vegetation-heavy)
        for ch, base in enumerate(base_means):
            noise = rng.normal(0, 30, (tile_size, tile_size))
            # Multi-scale smoothing for realistic texture
            smooth1 = gaussian_filter(noise, sigma=20)
            smooth2 = gaussian_filter(rng.normal(0, 15, (tile_size, tile_size)), sigma=5)
            smooth3 = gaussian_filter(rng.normal(0, 10, (tile_size, tile_size)), sigma=1)
            combined = base + smooth1 + 0.5 * smooth2 + 0.3 * smooth3
            channels.append(np.clip(combined, 0, 255).astype(np.uint8))

        rgb = np.stack(channels, axis=0)  # (3, H, W)

        transform = from_bounds(west, south, east, north, tile_size, tile_size)

        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 3,
            "height": tile_size,
            "width": tile_size,
            "crs": config.AOI["crs_wgs84"],
            "transform": transform,
            "compress": "deflate",
        }

        out_path = output_dir / f"demo_tile_{idx:04d}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(rgb)

        logger.info("Generated demo tile: %s", out_path)
        saved_paths.append(out_path)

    logger.info("Generated %d synthetic demo tiles.", len(saved_paths))
    return saved_paths


# ================================================================
# CLI
# ================================================================

def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 imagery or generate demo tiles "
                    "for the Route Resilience project."
    )
    parser.add_argument("--start_date", default="2024-01-01", help="Search start date (ISO)")
    parser.add_argument("--end_date", default="2024-03-01", help="Search end date (ISO)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_items", type=int, default=5, help="Max scenes to download")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo tiles instead")
    parser.add_argument("--num_tiles", type=int, default=4, help="Number of demo tiles")
    parser.add_argument("--tile_size", type=int, default=512, help="Demo tile pixel size")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    out = Path(args.output_dir) if args.output_dir else None

    if args.demo:
        paths = generate_demo_tiles(
            output_dir=out,
            num_tiles=args.num_tiles,
            tile_size=args.tile_size,
        )
    else:
        paths = download_sentinel2(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=out,
            max_items=args.max_items,
        )

    print(f"\nSaved {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

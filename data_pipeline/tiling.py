import os
import argparse
import pandas as pd
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import TILE_SIZE, TILE_OVERLAP, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, TILES_DIR

def create_tiles_from_image(img_path, mask_path, output_dir, prefix, split_name):
    """
    Cuts a large image and corresponding mask into overlapping tiles.
    Handles both generic images (PNG/JPG) and geo-referenced (GeoTIFF) files.
    """
    img_path = Path(img_path)
    mask_path = Path(mask_path)
    
    out_img_dir = Path(output_dir) / split_name / "images"
    out_mask_dir = Path(output_dir) / split_name / "masks"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    stride = TILE_SIZE - TILE_OVERLAP
    
    records = []
    
    # Check if geo-referenced using rasterio
    is_geo = img_path.suffix.lower() in ['.tif', '.tiff']
    
    if is_geo:
        with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
            width = src_img.width
            height = src_img.height
            
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    # Adjust if we go out of bounds
                    y_start = min(y, height - TILE_SIZE)
                    x_start = min(x, width - TILE_SIZE)
                    
                    if y_start < 0 or x_start < 0:
                        continue # Image smaller than tile size
                        
                    window = Window(x_start, y_start, TILE_SIZE, TILE_SIZE)
                    
                    img_tile = src_img.read(window=window)
                    mask_tile = src_mask.read(1, window=window)
                    
                    tile_name = f"{prefix}_{y_start}_{x_start}"
                    img_out = out_img_dir / f"{tile_name}.tif"
                    mask_out = out_mask_dir / f"{tile_name}.tif"
                    
                    # Update transform for the tile
                    transform = src_img.window_transform(window)
                    
                    # Write image tile
                    out_meta = src_img.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": TILE_SIZE,
                        "width": TILE_SIZE,
                        "transform": transform
                    })
                    with rasterio.open(img_out, "w", **out_meta) as dest:
                        dest.write(img_tile)
                        
                    # Write mask tile
                    mask_meta = src_mask.meta.copy()
                    mask_meta.update({
                        "driver": "GTiff",
                        "height": TILE_SIZE,
                        "width": TILE_SIZE,
                        "transform": transform
                    })
                    with rasterio.open(mask_out, "w", **mask_meta) as dest:
                        dest.write(mask_tile, 1)
                        
                    records.append({
                        "tile_path": str(img_out),
                        "mask_path": str(mask_out),
                        "split": split_name,
                        "source_file": img_path.name
                    })
    else:
        # Standard image reading (e.g. DeepGlobe PNGs)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        height, width = img.shape[:2]
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y_start = min(y, height - TILE_SIZE)
                x_start = min(x, width - TILE_SIZE)
                
                if y_start < 0 or x_start < 0:
                    continue
                    
                img_tile = img[y_start:y_start+TILE_SIZE, x_start:x_start+TILE_SIZE]
                mask_tile = mask[y_start:y_start+TILE_SIZE, x_start:x_start+TILE_SIZE]
                
                tile_name = f"{prefix}_{y_start}_{x_start}"
                img_out = out_img_dir / f"{tile_name}.png"
                mask_out = out_mask_dir / f"{tile_name}.png"
                
                # Convert back to BGR for saving with cv2
                cv2.imwrite(str(img_out), cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_out), mask_tile)
                
                records.append({
                    "tile_path": str(img_out),
                    "mask_path": str(mask_out),
                    "split": split_name,
                    "source_file": img_path.name
                })
                
    return records

def process_dataset(images_dir, masks_dir, output_dir=TILES_DIR):
    """
    Processes all images and masks in the input directories, creates tiles,
    and generates a manifest CSV.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    img_files = sorted([f for f in images_dir.iterdir() if f.is_file()])
    
    # Train/Val/Test split based on images
    train_imgs, temp_imgs = train_test_split(img_files, train_size=TRAIN_RATIO, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, train_size=VAL_RATIO/(VAL_RATIO+TEST_RATIO), random_state=42)
    
    all_records = []
    
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }
    
    for split_name, files in splits.items():
        print(f"Processing {split_name} split...")
        for img_path in tqdm(files):
            # Try to find corresponding mask
            mask_name = img_path.stem + "_mask" + img_path.suffix # Try DeepGlobe format
            mask_path = masks_dir / mask_name
            
            if not mask_path.exists():
                mask_name = img_path.name # Try direct match
                mask_path = masks_dir / mask_name
                
            if not mask_path.exists():
                print(f"Warning: No mask found for {img_path.name}. Skipping.")
                continue
                
            records = create_tiles_from_image(img_path, mask_path, output_dir, img_path.stem, split_name)
            all_records.extend(records)
            
    # Save manifest
    manifest_df = pd.DataFrame(all_records)
    manifest_path = Path(output_dir) / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path} with {len(all_records)} tiles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile large satellite images and masks")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing raw images")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory containing raw masks")
    parser.add_argument("--output_dir", type=str, default=str(TILES_DIR), help="Output directory for tiles")
    args = parser.parse_args()
    
    process_dataset(args.images_dir, args.masks_dir, args.output_dir)

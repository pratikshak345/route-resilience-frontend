import os
import argparse
import torch
import cv2
import rasterio
import numpy as np
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from segmentation.unet import get_model as get_unet
from segmentation.transformer_seg import get_model as get_segformer
from data_pipeline.augmentation import get_val_transform
from config import TILE_SIZE, TILE_OVERLAP, PRED_THRESHOLD


def _save_dashboard_ready_mask(binary_mask: np.ndarray, output_path: Path) -> None:
    """Save a 0/1 NumPy mask alongside output for dashboard real-data mode."""
    npy_path = output_path.with_suffix('.npy')
    np.save(str(npy_path), binary_mask.astype(np.uint8))
    print(f"Saved dashboard mask to {npy_path}")

def infer_and_stitch(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    if args.model_type == 'segformer':
        model = get_segformer()
    else:
        model = get_unet()
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = get_val_transform()
    
    input_path = Path(args.input_image)
    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    is_geo = input_path.suffix.lower() in ['.tif', '.tiff']
    
    stride = TILE_SIZE - TILE_OVERLAP
    
    if is_geo:
        with rasterio.open(input_path) as src:
            width = src.width
            height = src.height
            meta = src.meta.copy()
            
            # Arrays to accumulate predictions and counts for blending
            pred_accum = np.zeros((height, width), dtype=np.float32)
            count_accum = np.zeros((height, width), dtype=np.float32)
            
            # Calculate coordinates for tiles
            y_starts = list(range(0, height, stride))
            x_starts = list(range(0, width, stride))
            
            # Ensure we cover the edges
            if y_starts[-1] + TILE_SIZE < height:
                y_starts.append(height - TILE_SIZE)
            if x_starts[-1] + TILE_SIZE < width:
                x_starts.append(width - TILE_SIZE)
                
            total_tiles = len(y_starts) * len(x_starts)
            pbar = tqdm(total=total_tiles, desc="Running Inference")
            
            for y in y_starts:
                for x in x_starts:
                    # Handle boundaries
                    y_start = max(0, min(y, height - TILE_SIZE))
                    x_start = max(0, min(x, width - TILE_SIZE))
                    
                    window = Window(x_start, y_start, TILE_SIZE, TILE_SIZE)
                    img_tile = src.read([1, 2, 3], window=window).transpose(1, 2, 0)
                    
                    # Preprocess
                    augmented = transform(image=img_tile)
                    img_tensor = augmented['image'].unsqueeze(0).to(device)
                    
                    # Predict
                    with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        logits = model(img_tensor)
                        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                        
                    # Accumulate
                    pred_accum[y_start:y_start+TILE_SIZE, x_start:x_start+TILE_SIZE] += probs
                    count_accum[y_start:y_start+TILE_SIZE, x_start:x_start+TILE_SIZE] += 1
                    
                    pbar.update(1)
            pbar.close()
            
            # Average predictions in overlapping areas
            pred_final = pred_accum / count_accum
            binary_mask = (pred_final > PRED_THRESHOLD).astype(np.uint8)

            _save_dashboard_ready_mask(binary_mask, output_path)
            
            # Save GeoTIFF
            meta.update({
                "count": 1,
                "dtype": rasterio.uint8
            })
            
            with rasterio.open(output_path, "w", **meta) as dest:
                dest.write(binary_mask, 1)
                
            # Also save as PNG for easy viewing
            cv2.imwrite(str(output_path.with_suffix('.png')), binary_mask * 255)
            print(f"Saved prediction to {output_path} and {output_path.with_suffix('.png')}")
            
    else:
        # Standard image (PNG/JPG)
        img = cv2.imread(str(input_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]
        
        pred_accum = np.zeros((height, width), dtype=np.float32)
        count_accum = np.zeros((height, width), dtype=np.float32)
        
        y_starts = list(range(0, height, stride))
        x_starts = list(range(0, width, stride))
        
        if y_starts[-1] + TILE_SIZE < height:
            y_starts.append(height - TILE_SIZE)
        if x_starts[-1] + TILE_SIZE < width:
            x_starts.append(width - TILE_SIZE)
            
        total_tiles = len(y_starts) * len(x_starts)
        pbar = tqdm(total=total_tiles, desc="Running Inference")
        
        for y in y_starts:
            for x in x_starts:
                y_start = max(0, min(y, height - TILE_SIZE))
                x_start = max(0, min(x, width - TILE_SIZE))
                
                img_tile = img[y_start:y_start+TILE_SIZE, x_start:x_start+TILE_SIZE]
                
                augmented = transform(image=img_tile)
                img_tensor = augmented['image'].unsqueeze(0).to(device)
                
                with torch.no_grad(), torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = model(img_tensor)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    
                pred_accum[y_start:y_start+TILE_SIZE, x_start:x_start+TILE_SIZE] += probs
                count_accum[y_start:y_start+TILE_SIZE, x_start:x_start+TILE_SIZE] += 1
                
                pbar.update(1)
        pbar.close()
        
        pred_final = pred_accum / count_accum
        binary_mask = (pred_final > PRED_THRESHOLD).astype(np.uint8)
        _save_dashboard_ready_mask(binary_mask, output_path)
        binary_mask_png = binary_mask * 255
        
        cv2.imwrite(str(output_path), binary_mask_png)
        print(f"Saved prediction to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference and Stitch Tiles")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--input_image", type=str, required=True, help="Path to full satellite image")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save predicted mask")
    parser.add_argument("--model_type", type=str, choices=['unet', 'segformer'], default='unet', help="Model architecture")
    args = parser.parse_args()
    
    infer_and_stitch(args)

import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_pipeline.augmentation import RoadDataset, get_train_transform, get_val_transform
from segmentation.unet import get_model as get_unet, DiceBCELoss
from segmentation.transformer_seg import get_model as get_segformer
from dashboard.metrics import iou_score, dice_score
from config import BATCH_SIZE, LEARNING_RATE, ENCODER_LR, WEIGHT_DECAY, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, MODELS_DIR

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.data_dir) / "manifest.csv"
    
    # Datasets and Loaders
    train_dataset = RoadDataset(manifest_path, split="train", transform=get_train_transform())
    val_dataset = RoadDataset(manifest_path, split="val", transform=get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    if args.model_type == 'segformer':
        model = get_segformer().to(device)
    else:
        model = get_unet().to(device)
        
    # Optimizer with differential learning rates
    # Assume encoder is named 'model.encoder'
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': ENCODER_LR},
        {'params': decoder_params, 'lr': args.lr}
    ], weight_decay=WEIGHT_DECAY)
    
    criterion = DiceBCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_iou = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                preds = model(images)
                loss = criterion(preds, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    preds = model(images)
                    loss = criterion(preds, masks)
                
                val_loss += loss.item()
                preds_binary = (torch.sigmoid(preds) > 0.5).float()
                
                val_iou += iou_score(preds_binary, masks).item()
                val_dice += dice_score(preds_binary, masks).item()
                
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice
        })
        
        # Checkpoint saving and early stopping
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f"{args.model_type}_best.pth")
            print(f"--> Saved new best model with IoU {best_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Save training history
    pd.DataFrame(history).to_csv(MODELS_DIR / f"{args.model_type}_history.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Road Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing manifest.csv")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate (decoder)")
    parser.add_argument("--model_type", type=str, choices=['unet', 'segformer'], default='unet', help="Model architecture")
    args = parser.parse_args()
    
    train(args)

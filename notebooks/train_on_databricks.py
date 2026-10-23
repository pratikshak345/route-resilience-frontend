# Databricks notebook source
# ============================================================
# Route Resilience — Databricks Training Script
# Occlusion-Robust Road Extraction & Graph-Theoretic Criticality
# ============================================================
#
# HOW TO USE ON DATABRICKS
# ========================
# 1. Create a Databricks cluster with GPU support:
#    - Runtime: Databricks ML Runtime 14+ (GPU)
#    - Instance type: Standard_NC6s_v3 (Azure) or p3.2xlarge (AWS)
#    - Single node is sufficient for this model size
#
# 2. Upload training data to DBFS:
#    - Use Databricks CLI or the DBFS UI
#    - Structure your data as:
#        /dbfs/route-resilience/data/tiles/train/images/
#        /dbfs/route-resilience/data/tiles/train/masks/
#        /dbfs/route-resilience/data/tiles/val/images/
#        /dbfs/route-resilience/data/tiles/val/masks/
#    - Images: 512×512 RGB PNGs
#    - Masks: 512×512 single-channel binary PNGs (0 = background, 255 = road)
#
# 3. Import this file as a Databricks notebook:
#    - File → Import → Upload this .py file
#    - It will auto-detect the "# Databricks notebook source" marker
#
# 4. Run all cells sequentially.
#
# 5. After training, download the model:
#    - dbutils.fs.cp(
#        "dbfs:/route-resilience/models/unet_best.pth",
#        "file:/tmp/unet_best.pth"
#      )
#    - Then download /tmp/unet_best.pth from the driver node
#
# ============================================================

# COMMAND ----------

# MAGIC %pip install segmentation-models-pytorch>=0.3.3 albumentations>=1.3.0 --quiet

# COMMAND ----------

# Restart the Python process to pick up newly-installed packages
# dbutils.library.restartPython()   # uncomment if needed

# COMMAND ----------

import os
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# COMMAND ----------

# ============================================================
# CONFIGURATION  (mirrors config.py from local project)
# ============================================================

# --- DBFS paths ---
DBFS_ROOT = Path("/dbfs/route-resilience")
DATA_DIR = DBFS_ROOT / "data"
TILES_DIR = DATA_DIR / "tiles"
MODELS_DIR = DBFS_ROOT / "models"

# Create output directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Area of Interest ---
AOI_NAME = "Bengaluru"

# --- Segmentation hyperparams (same as config.py) ---
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
NUM_CLASSES = 1
TILE_SIZE = 512

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
ENCODER_LR = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5
PRED_THRESHOLD = 0.5

# COMMAND ----------

# ============================================================
# DATASET
# ============================================================


class RoadDataset(Dataset):
    """Road-segmentation dataset reading 512×512 tiles from DBFS.

    Expects:
      - images_dir/  containing  *.png  RGB images
      - masks_dir/   containing  *.png  single-channel masks
      File names must match (same basename).
    """

    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.image_paths = sorted(glob.glob(str(self.images_dir / "*.png")))
        if not self.image_paths:
            self.image_paths = sorted(glob.glob(str(self.images_dir / "*.jpg")))
        if not self.image_paths:
            self.image_paths = sorted(glob.glob(str(self.images_dir / "*.tif")))

        print(f"  → Found {len(self.image_paths)} images in {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Try common mask extensions
        mask_path = None
        for ext in [".png", ".jpg", ".tif"]:
            candidate = self.masks_dir / f"{basename}{ext}"
            if candidate.exists():
                mask_path = str(candidate)
                break
        if mask_path is None:
            mask_path = str(self.masks_dir / f"{basename}_mask.png")

        # Read image (BGR → RGB)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask (grayscale, binarise)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask has channel dimension  (1, H, W)
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# COMMAND ----------

# ============================================================
# AUGMENTATIONS
# ============================================================

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
        border_mode=cv2.BORDER_REFLECT, p=0.5,
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(var_limit=(10.0, 50.0)),
    ], p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# COMMAND ----------

# ============================================================
# BUILD DATASETS & LOADERS
# ============================================================

train_ds = RoadDataset(
    images_dir=str(TILES_DIR / "train" / "images"),
    masks_dir=str(TILES_DIR / "train" / "masks"),
    transform=train_transform,
)

val_ds = RoadDataset(
    images_dir=str(TILES_DIR / "val" / "images"),
    masks_dir=str(TILES_DIR / "val" / "masks"),
    transform=val_transform,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} tiles  |  Val: {len(val_ds)} tiles")

# COMMAND ----------

# ============================================================
# MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = smp.Unet(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES,
    activation=None,  # raw logits — we apply sigmoid in loss / inference
)
model = model.to(device)

# COMMAND ----------

# ============================================================
# LOSS & OPTIMISER
# ============================================================

dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
bce_loss_fn = nn.BCEWithLogitsLoss()


def combined_loss(pred, target):
    """Weighted sum of Dice + BCE losses."""
    return DICE_WEIGHT * dice_loss_fn(pred, target) + BCE_WEIGHT * bce_loss_fn(pred, target)


# Differential learning rate: lower LR for pretrained encoder
optimizer = optim.AdamW([
    {"params": model.encoder.parameters(), "lr": ENCODER_LR},
    {"params": model.decoder.parameters(), "lr": LEARNING_RATE},
    {"params": model.segmentation_head.parameters(), "lr": LEARNING_RATE},
], weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True,
)

scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

# COMMAND ----------

# ============================================================
# METRICS HELPERS (inline — mirrors dashboard/metrics.py)
# ============================================================


def calc_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Batch-averaged IoU for binary segmentation."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    union = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def calc_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Batch-averaged Dice for binary segmentation."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    total = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (total + smooth)
    return dice.mean().item()


# COMMAND ----------

# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

print(f"\n{'='*60}")
print(f"  Training: {ENCODER_NAME} U-Net  |  {NUM_EPOCHS} epochs")
print(f"  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
print(f"{'='*60}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_accum = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = combined_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / max(len(train_loader), 1)

    # ---- VALIDATE ----
    model.eval()
    val_loss_accum = 0.0
    val_iou_accum = 0.0
    val_dice_accum = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = combined_loss(logits, masks)

            val_loss_accum += loss.item()
            val_iou_accum += calc_iou(logits, masks)
            val_dice_accum += calc_dice(logits, masks)

    n_val = max(len(val_loader), 1)
    avg_val_loss = val_loss_accum / n_val
    avg_val_iou = val_iou_accum / n_val
    avg_val_dice = val_dice_accum / n_val

    scheduler.step(avg_val_loss)

    # Record history
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_iou"].append(avg_val_iou)
    history["val_dice"].append(avg_val_dice)

    elapsed = time.time() - epoch_start

    print(
        f"Epoch {epoch:3d}/{NUM_EPOCHS}  |  "
        f"Train Loss: {avg_train_loss:.4f}  |  "
        f"Val Loss: {avg_val_loss:.4f}  |  "
        f"Val IoU: {avg_val_iou:.4f}  |  "
        f"Val Dice: {avg_val_dice:.4f}  |  "
        f"{elapsed:.1f}s"
    )

    # ---- CHECKPOINTING ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        save_path = str(MODELS_DIR / "unet_best.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "val_iou": avg_val_iou,
            "val_dice": avg_val_dice,
        }, save_path)
        print(f"  → Best model saved to {save_path}")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
            break

# COMMAND ----------

# ============================================================
# SAVE FINAL MODEL & TRAINING HISTORY
# ============================================================

# Save last-epoch model
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
}, str(MODELS_DIR / "unet_final.pth"))

print(f"\n✅ Training complete!")
print(f"   Best val loss : {best_val_loss:.4f}")
print(f"   Models saved  : {MODELS_DIR}")

# COMMAND ----------

# ============================================================
# DOWNLOAD INSTRUCTIONS
# ============================================================

# To download the trained model from DBFS to your local machine:
#
# Option A — Databricks CLI:
#   databricks fs cp dbfs:/route-resilience/models/unet_best.pth ./unet_best.pth
#
# Option B — From within this notebook:
#   dbutils.fs.cp(
#       "dbfs:/route-resilience/models/unet_best.pth",
#       "file:/tmp/unet_best.pth"
#   )
#   # Then download /tmp/unet_best.pth from the Databricks driver node
#
# Option C — Copy to your project's data/models/ folder:
#   Place unet_best.pth into  data/models/unet_best.pth
#   The rest of the pipeline will pick it up automatically.

print("📥 Download the model using one of the methods above.")
print(f"   DBFS path: dbfs:/route-resilience/models/unet_best.pth")

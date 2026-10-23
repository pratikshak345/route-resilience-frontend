import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import ENCODER_NAME, ENCODER_WEIGHTS, IN_CHANNELS, NUM_CLASSES, DICE_WEIGHT, BCE_WEIGHT

class RoadUNet(nn.Module):
    """
    U-Net model for road segmentation wrapping segmentation_models_pytorch.
    Uses ResNet34 encoder by default.
    """
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None # Outputs raw logits
        )
        
    def forward(self, x):
        return self.model(x)

class DiceBCELoss(nn.Module):
    """
    Combined Dice and BCE Loss for imbalanced dataset (like road segmentation).
    """
    def __init__(self, dice_weight=DICE_WEIGHT, bce_weight=BCE_WEIGHT):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        
    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

def get_model():
    """Factory function for creating the U-Net baseline."""
    return RoadUNet()

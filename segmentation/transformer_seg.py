import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import IN_CHANNELS, NUM_CLASSES

class RoadSegFormer(nn.Module):
    """
    SegFormer model for road segmentation wrapping segmentation_models_pytorch.
    Stretch goal for robust segmentation.
    Requires `timm` to be installed.
    """
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='mit_b2', # MixTransformer backbone
            encoder_weights='imagenet',
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            activation=None # Outputs raw logits
        )
        
    def forward(self, x):
        return self.model(x)

def get_model():
    """Factory function for creating the SegFormer model."""
    return RoadSegFormer()

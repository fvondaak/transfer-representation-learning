# Script to donwload and save the ResNet50 model with ImageNet weights
import os
from pathlib import Path

import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

if __name__ == "__main__":#
    # Load the pre-trained ResNet50 model with ImageNet weights (80.5% accuracy on ImageNet)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    torch.save(model.state_dict(), MODELS_DIR / "resnet50_imagenet.pth")
    ep = torch.export.export(model, args=(torch.randn(1, 3, 224, 224),))
    torch.export.save(ep, MODELS_DIR / "resnet50_imagenet.torchscript.pt2")
    print(f"ResNet50 model with ImageNet weights saved to {MODELS_DIR / 'resnet50_imagenet.pth'} and EP model saved to {MODELS_DIR / 'resnet50_imagenet.torchscript.pt2'}")
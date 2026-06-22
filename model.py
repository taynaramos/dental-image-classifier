"""ResNet-18 model definition with transfer-learning head for dental classification."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 5) -> nn.Module:
    """Load a pre-trained ResNet-18 and replace its FC layer for fine-tuning.

    The backbone is initialised with ImageNet weights; only the final
    fully-connected layer is replaced so its output size matches num_classes.

    Args:
        num_classes: Number of intraoral view categories to classify.

    Returns:
        A ResNet-18 model ready for two-phase fine-tuning.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(
    checkpoint_path: Path,
    num_classes: int = 5,
    device: torch.device | None = None,
) -> nn.Module:
    """Restore a fine-tuned model from a saved state-dict checkpoint.

    Args:
        checkpoint_path: Path to a `.pth` file produced by :func:`train.train`.
        num_classes: Must match the value used during training.
        device: Target device; defaults to CUDA when available.

    Returns:
        Model loaded with saved weights, set to evaluation mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

"""Inference utilities for the dental intraoral classifier."""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image

from dataloader import CLASSES, get_inference_transform


def predict_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
) -> str:
    """Classify a single intraoral image and return the predicted class name.

    Args:
        model: Fine-tuned model in evaluation mode.
        image_path: Path to the image file (JPEG, PNG, …).
        device: Device on which inference is performed.

    Returns:
        Predicted class name (one of :data:`dataloader.CLASSES`).
    """
    transform = get_inference_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        idx: int = logits.argmax(dim=1).item()

    return CLASSES[idx]


def predict_batch(
    model: nn.Module,
    image_paths: List[Path],
    device: torch.device,
) -> List[str]:
    """Classify a list of intraoral images.

    Images are processed one by one; for high-throughput evaluation over a
    full validation set prefer to use a DataLoader with :func:`train.eval_epoch`.

    Args:
        model: Fine-tuned model in evaluation mode.
        image_paths: Ordered list of image file paths.
        device: Device on which inference is performed.

    Returns:
        List of predicted class names in the same order as ``image_paths``.
    """
    return [predict_image(model, path, device) for path in image_paths]

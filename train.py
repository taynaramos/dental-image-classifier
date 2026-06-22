"""Two-phase fine-tuning loop for the dental intraoral classifier."""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch over the full loader.

    Args:
        model: The model to train (must be on ``device``).
        loader: DataLoader yielding (images, labels) batches.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: Parameter update rule.
        device: Device on which computation is performed.

    Returns:
        Tuple of (mean_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (outputs.detach().argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a full loader without gradient computation.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding (images, labels) batches.
        criterion: Loss function used to compute the reported loss.
        device: Device on which computation is performed.

    Returns:
        Tuple of (mean_loss, accuracy) over the loader.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: Path,
    device: torch.device,
    phase1_epochs: int = 5,
    phase2_epochs: int = 15,
) -> nn.Module:
    """Fine-tune a ResNet-18 model in two phases.

    **Phase 1** — backbone frozen, only the FC head is trained for
    ``phase1_epochs`` epochs using a higher learning rate (1e-3).

    **Phase 2** — all parameters unlocked and trained end-to-end for
    ``phase2_epochs`` epochs at a lower learning rate (1e-4). The best
    checkpoint (highest validation accuracy) is saved to ``save_path``.

    Args:
        model: ResNet-18 produced by :func:`model.build_model`, on ``device``.
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split.
        save_path: File path where the best ``state_dict`` is written.
        device: Device on which training runs.
        phase1_epochs: Number of warm-up epochs with frozen backbone.
        phase2_epochs: Number of fine-tuning epochs with all layers unlocked.

    Returns:
        The model with the best validation weights loaded.
    """
    criterion = nn.CrossEntropyLoss()

    # Phase 1: train only the FC head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    print("=== Phase 1: frozen backbone ===")
    for epoch in range(1, phase1_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

    # Phase 2: full fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    print("\n=== Phase 2: full fine-tuning ===")
    for epoch in range(1, phase2_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        tag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            tag = " <- best saved"

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}{tag}"
        )

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

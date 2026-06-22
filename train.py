"""Two-phase fine-tuning loop for the dental intraoral classifier."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score
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


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """Collect all predictions and ground-truth labels from a DataLoader.

    Args:
        model: Model in evaluation mode.
        loader: DataLoader yielding (images, labels) batches.
        device: Device on which inference is performed.

    Returns:
        Tuple of (all_predictions, all_labels) as flat Python lists.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return all_preds, all_labels


def _log_val_metrics(
    wandb_run: Any,
    all_preds: List[int],
    all_labels: List[int],
    class_names: List[str],
    va_loss: float,
    va_acc: float,
    step: int,
) -> None:
    """Compute and log per-class validation metrics to a W&B run.

    Logs cross-entropy loss, accuracy, per-class precision, per-class F1,
    and an interactive confusion matrix.

    Args:
        wandb_run: Active ``wandb.Run`` object returned by ``wandb.init()``.
        all_preds: Flat list of predicted class indices for the full val set.
        all_labels: Flat list of ground-truth class indices for the full val set.
        class_names: Ordered list of class name strings.
        va_loss: Mean cross-entropy loss on the validation set.
        va_acc: Accuracy on the validation set.
        step: Global epoch counter used as the W&B x-axis.
    """
    import wandb  # imported here so the module loads without wandb installed

    precision_per_class: List[float] = precision_score(
        all_labels, all_preds, average=None, zero_division=0
    ).tolist()
    f1_per_class: List[float] = f1_score(
        all_labels, all_preds, average=None, zero_division=0
    ).tolist()

    metrics: dict[str, Any] = {
        "val/loss": va_loss,
        "val/acc": va_acc,
    }
    for cls, prec, f1 in zip(class_names, precision_per_class, f1_per_class):
        metrics[f"val/precision/{cls}"] = prec
        metrics[f"val/f1/{cls}"] = f1

    metrics["val/confusion_matrix"] = wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=class_names,
    )

    wandb_run.log(metrics, step=step)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: Path,
    device: torch.device,
    class_names: List[str],
    phase1_epochs: int = 5,
    phase2_epochs: int = 15,
    wandb_run: Optional[Any] = None,
) -> nn.Module:
    """Fine-tune a ResNet-18 model in two phases with optional W&B tracking.

    **Phase 1** — backbone frozen, only the FC head is trained for
    ``phase1_epochs`` epochs using a higher learning rate (1e-3).

    **Phase 2** — all parameters unlocked and trained end-to-end for
    ``phase2_epochs`` epochs at a lower learning rate (1e-4). The best
    checkpoint (highest validation accuracy) is saved to ``save_path``.

    When ``wandb_run`` is provided the following metrics are logged every epoch:

    * ``train/loss`` — cross-entropy loss on the training set.
    * ``train/acc`` — accuracy on the training set.
    * ``val/loss`` — cross-entropy loss on the validation set.
    * ``val/acc`` — accuracy on the validation set.
    * ``val/precision/{class}`` — per-class precision on the validation set.
    * ``val/f1/{class}`` — per-class F1-score on the validation set.
    * ``val/confusion_matrix`` — interactive confusion matrix plot.

    Args:
        model: ResNet-18 produced by :func:`model.build_model`, on ``device``.
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split.
        save_path: File path where the best ``state_dict`` is written.
        device: Device on which training runs.
        class_names: Ordered list of class name strings (must match label indices).
        phase1_epochs: Number of warm-up epochs with frozen backbone.
        phase2_epochs: Number of fine-tuning epochs with all layers unlocked.
        wandb_run: Active ``wandb.Run`` object, or ``None`` to skip logging.

    Returns:
        The model with the best validation weights loaded.
    """
    criterion = nn.CrossEntropyLoss()
    global_epoch = 0

    # ------------------------------------------------------------------ #
    # Phase 1: train only the FC head                                      #
    # ------------------------------------------------------------------ #
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    print("=== Phase 1: frozen backbone ===")
    for epoch in range(1, phase1_epochs + 1):
        global_epoch += 1
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        if wandb_run is not None:
            all_preds, all_labels = collect_predictions(model, val_loader, device)
            wandb_run.log(
                {"train/loss": tr_loss, "train/acc": tr_acc},
                step=global_epoch,
            )
            _log_val_metrics(
                wandb_run, all_preds, all_labels, class_names,
                va_loss, va_acc, step=global_epoch,
            )

    # ------------------------------------------------------------------ #
    # Phase 2: full fine-tuning                                           #
    # ------------------------------------------------------------------ #
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    print("\n=== Phase 2: full fine-tuning ===")
    for epoch in range(1, phase2_epochs + 1):
        global_epoch += 1
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

        if wandb_run is not None:
            all_preds, all_labels = collect_predictions(model, val_loader, device)
            wandb_run.log(
                {"train/loss": tr_loss, "train/acc": tr_acc},
                step=global_epoch,
            )
            _log_val_metrics(
                wandb_run, all_preds, all_labels, class_names,
                va_loss, va_acc, step=global_epoch,
            )

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

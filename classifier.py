"""Intraoral dental image classifier — command-line entry point.

Usage examples
--------------
Train from scratch::

    python classifier.py train --dataset-root /path/to/dataset

Predict a single image::

    python classifier.py predict --image /path/to/image.jpeg --checkpoint best_model.pth

Predict a folder of images::

    python classifier.py predict --image-dir /path/to/images --checkpoint best_model.pth
"""

import argparse
from pathlib import Path

import torch

from dataloader import CLASSES, get_train_loaders
from model import build_model, load_model
from test import predict_batch, predict_image
from train import train as run_training


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cmd_train(args: argparse.Namespace) -> None:
    device = _device()
    print(f"Device: {device}")

    train_loader, val_loader = get_train_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)}"
    )

    model = build_model(num_classes=len(CLASSES)).to(device)
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=args.checkpoint,
        device=device,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
    )
    print(f"Best model saved to: {args.checkpoint}")


def _cmd_predict(args: argparse.Namespace) -> None:
    device = _device()
    model = load_model(args.checkpoint, num_classes=len(CLASSES), device=device)

    if args.image is not None:
        label = predict_image(model, args.image, device)
        print(f"{args.image.name}: {label}")
        return

    image_paths = sorted(args.image_dir.glob("*.jpeg")) + sorted(
        args.image_dir.glob("*.jpg")
    )
    if not image_paths:
        print(f"No JPEG images found in {args.image_dir}")
        return

    predictions = predict_batch(model, image_paths, device)
    for path, label in zip(image_paths, predictions):
        print(f"{path.name}: {label}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dental intraoral image classifier (ResNet-18 transfer learning)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Fine-tune the classifier.")
    train_parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder with one sub-directory per patient.",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("best_model.pth"),
        help="Output path for the best model weights (default: best_model.pth).",
    )
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--phase1-epochs", type=int, default=5)
    train_parser.add_argument("--phase2-epochs", type=int, default=15)

    # --- predict ---
    predict_parser = subparsers.add_parser(
        "predict", help="Classify one image or a folder of images."
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to a trained model checkpoint (default: best_model.pth).",
    )
    img_group = predict_parser.add_mutually_exclusive_group(required=True)
    img_group.add_argument("--image", type=Path, help="Path to a single image file.")
    img_group.add_argument(
        "--image-dir", type=Path, help="Folder of images to classify in bulk."
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "predict":
        _cmd_predict(args)


if __name__ == "__main__":
    main()

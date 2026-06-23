"""CLI command: run inference with the PCA-SVC dental classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from src.pca_svc.dataset import _STEM_TO_LABEL
from src.pca_svc.model import DentalClassifier, Prediction

_JPEG_SUFFIXES = {".jpeg", ".jpg"}


class PcaPredict:
    name: ClassVar[str] = "pca-predict"
    help: ClassVar[str] = "Classify one image or a folder with the PCA-SVC model."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--model",
            type=Path,
            required=True,
            help="Path to a model file produced by pca-train.",
        )

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--image",
            type=Path,
            help="Path to a single image file.",
        )
        group.add_argument(
            "--image-dir",
            type=Path,
            help="Folder of images to classify in bulk.",
        )

        parser.add_argument(
            "--image-size",
            type=int,
            default=128,
            help="Must match the size used during training (default: 128).",
        )

    def run(self, args: argparse.Namespace) -> None:
        print(f"[model] loading from {args.model}")
        classifier = DentalClassifier.load(args.model)

        image_size = (args.image_size, args.image_size)

        if args.image is not None:
            paths = [args.image]
        else:
            paths = sorted(
                p for p in args.image_dir.iterdir()
                if p.suffix.lower() in _JPEG_SUFFIXES
            )
            if not paths:
                print(f"No JPEG images found in {args.image_dir}")
                return

        X = np.stack([_load_luma(p, image_size) for p in paths])
        predictions = classifier.predict_proba(X)

        for path, pred in zip(paths, predictions):
            _print_prediction(path, pred)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_luma(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("YCbCr")
    luma, *_ = img.split()
    luma = luma.resize(image_size, Image.LANCZOS)
    return np.array(luma, dtype=np.float32).ravel()


def _print_prediction(path: Path, pred: Prediction) -> None:
    print(f"\n{path.name}")
    print(f"  predicted : {pred.label}")
    print("  probabilities:")
    for cls, prob in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(prob * 30)
        print(f"    {cls:<22} {prob:5.1%}  {bar}")

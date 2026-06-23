"""CLI command: train the PCA-SVC dental classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import numpy as np

from src.pca_svc.dataset import DentalDataset
from src.pca_svc.model import DentalClassifier


class PcaTrain:
    name: ClassVar[str] = "pca-train"
    help: ClassVar[str] = "Train the PCA + SVC dental view classifier."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--dataset-path",
            type=Path,
            required=True,
            help="Root directory containing one sub-folder per subject.",
        )
        parser.add_argument(
            "--model-out",
            type=Path,
            default=Path("pca_svc_model.pkl"),
            help="Output path for the serialised model (default: pca_svc_model.pkl).",
        )
        parser.add_argument(
            "--image-size",
            type=int,
            default=128,
            help="Side length (px) to resize images before flattening (default: 128).",
        )
        parser.add_argument(
            "--n-components",
            type=int,
            default=None,
            help=(
                "Number of PCA components. "
                "Omit to use the minimum that explains --variance-threshold."
            ),
        )
        parser.add_argument(
            "--variance-threshold",
            type=float,
            default=0.95,
            help="Fraction of variance to retain when --n-components is not set (default: 0.95).",
        )
        parser.add_argument(
            "--C",
            type=float,
            default=10.0,
            help="SVC regularisation parameter C (default: 10.0).",
        )
        parser.add_argument(
            "--gamma",
            type=str,
            default="scale",
            help="SVC kernel coefficient (default: scale).",
        )
        parser.add_argument(
            "--train-ratio",
            type=float,
            default=0.70,
        )
        parser.add_argument(
            "--val-ratio",
            type=float,
            default=0.15,
        )
        parser.add_argument(
            "--test-ratio",
            type=float,
            default=0.15,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
        )

    def run(self, args: argparse.Namespace) -> None:
        image_size = (args.image_size, args.image_size)

        print(f"[dataset] loading from {args.dataset_path}")
        dataset = DentalDataset(
            args.dataset_path,
            image_size=image_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        counts = dataset.subject_counts()
        print(
            f"[dataset] subjects — "
            f"train: {counts['train']}  val: {counts['val']}  test: {counts['test']}"
        )

        print("[dataset] loading train split (luma) …")
        X_train, y_train = dataset.load_split("train")

        print("[dataset] loading val split …")
        X_val, y_val = dataset.load_split("val")

        print("[dataset] loading test split …")
        X_test, y_test = dataset.load_split("test")

        print(
            f"[dataset] shapes — "
            f"train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}"
        )

        print("[train] fitting PCA + SVC …")
        classifier = DentalClassifier(
            n_components=args.n_components,
            auto_variance_threshold=args.variance_threshold,
            C=args.C,
            gamma=args.gamma,
            seed=args.seed,
        )
        classifier.fit(X_train, y_train)

        extractor = classifier.extractor
        print(
            f"[pca] components selected : {extractor.n_components_}  "
            f"(variance retained: {extractor._pca.explained_variance_ratio_.sum():.4f})"
        )
        _print_variance_table(extractor.cumulative_explained_variance())

        train_acc = classifier.score(X_train, y_train)
        val_acc = classifier.score(X_val, y_val)
        test_acc = classifier.score(X_test, y_test)

        print(f"\n[results] train accuracy : {train_acc:.4f}")
        print(f"[results] val   accuracy : {val_acc:.4f}")
        print(f"[results] test  accuracy : {test_acc:.4f}")

        classifier.save(args.model_out)
        print(f"\n[model] saved to {args.model_out}")


def _print_variance_table(cumulative: np.ndarray) -> None:
    thresholds = [0.80, 0.90, 0.95, 0.99]
    header = "  threshold | components"
    print(f"\n[pca] cumulative variance\n{header}")
    print("  " + "-" * (len(header) - 2))
    for t in thresholds:
        n = int(np.searchsorted(cumulative, t)) + 1
        print(f"      {t:.0%}   |  {n}")
    print()

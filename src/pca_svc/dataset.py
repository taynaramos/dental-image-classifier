"""Dental intraoral dataset loader — luma channel only, numpy-based."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

CLASSES: list[str] = [
    "frontal",
    "inferior",
    "superior",
    "lateral_direita",
    "lateral_esquerda",
]

_STEM_TO_LABEL: dict[str, str] = {
    "intraoral-frontal": "frontal",
    "intraoral-inferior": "inferior",
    "intraoral-superior": "superior",
    "intraoral-lateral-direita": "lateral_direita",
    "intraoral-lateral-esquerda": "lateral_esquerda",
}

Split = Literal["train", "val", "test"]


class DentalDataset:
    """Subject-level train/val/test split with luma preprocessing.

    Parameters
    ----------
    root:
        Directory containing one sub-folder per subject.
    image_size:
        (width, height) to resize each image before flattening.
    train_ratio, val_ratio, test_ratio:
        Fractions that must sum to 1.0.
    seed:
        Random seed for reproducible shuffling.
    """

    image_size: tuple[int, int]
    classes: list[str] = CLASSES

    def __init__(
        self,
        root: Path | str,
        *,
        image_size: tuple[int, int] = (128, 128),
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        self.root = Path(root)
        self.image_size = image_size
        self._split_subjects(seed, train_ratio, val_ratio, test_ratio)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_split(self, split: Split) -> tuple[np.ndarray, np.ndarray]:
        """Return *(X, y)* for the requested split.

        X shape: ``(n_samples, image_size[0] * image_size[1])`` — flattened luma.
        y shape: ``(n_samples,)`` — string class labels.
        """
        subjects = self._subjects[split]
        return self._collect_luma(subjects)

    def subject_counts(self) -> dict[str, int]:
        return {split: len(subs) for split, subs in self._subjects.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_subjects(
        self,
        seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,  # noqa: ARG002  kept for symmetry / documentation
    ) -> None:
        subject_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        rng = random.Random(seed)
        rng.shuffle(subject_dirs)

        n = len(subject_dirs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        self._subjects: dict[Split, list[Path]] = {
            "train": subject_dirs[:n_train],
            "val": subject_dirs[n_train : n_train + n_val],
            "test": subject_dirs[n_train + n_val :],
        }

    def _collect_luma(
        self, subjects: list[Path]
    ) -> tuple[np.ndarray, np.ndarray]:
        X: list[np.ndarray] = []
        y: list[str] = []
        for subject in subjects:
            for img_path in sorted(subject.glob("*.jpeg")):
                label = _STEM_TO_LABEL.get(img_path.stem)
                if label is None:
                    continue
                X.append(self._load_luma(img_path))
                y.append(label)
        return np.array(X, dtype=np.float32), np.array(y)

    def _load_luma(self, path: Path) -> np.ndarray:
        """Load one image, extract Y channel, resize and flatten."""
        img = Image.open(path).convert("YCbCr")
        luma, *_ = img.split()
        luma = luma.resize(self.image_size, Image.LANCZOS)
        return np.array(luma, dtype=np.float32).ravel()

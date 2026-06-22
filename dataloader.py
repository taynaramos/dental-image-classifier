"""Dataset loading utilities for the dental intraoral image classifier."""

import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASSES: List[str] = [
    "frontal",
    "inferior",
    "superior",
    "lateral_direita",
    "lateral_esquerda",
]

STEM_TO_IDX: dict[str, int] = {
    "intraoral-frontal": 0,
    "intraoral-inferior": 1,
    "intraoral-superior": 2,
    "intraoral-lateral-direita": 3,
    "intraoral-lateral-esquerda": 4,
}

_IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

Sample = Tuple[Path, int]


class DentalDataset(Dataset):
    """PyTorch Dataset for intraoral dental images."""

    def __init__(
        self,
        samples: List[Sample],
        transform: T.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_splits(
    dataset_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Split dataset into train and validation samples by patient folder.

    Splitting by patient (not by image) prevents the same patient from
    appearing in both train and validation at the same time.

    Args:
        dataset_root: Root directory containing one subfolder per patient.
        val_ratio: Fraction of patients reserved for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_samples, val_samples), each a list of (Path, label_index).
    """
    patient_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())

    rng = random.Random(seed)
    rng.shuffle(patient_dirs)

    n_val = int(len(patient_dirs) * val_ratio)
    val_patients = patient_dirs[:n_val]
    train_patients = patient_dirs[n_val:]

    def _collect(patients: List[Path]) -> List[Sample]:
        samples: List[Sample] = []
        for patient in patients:
            for img_file in patient.glob("*.jpeg"):
                if img_file.stem in STEM_TO_IDX:
                    samples.append((img_file, STEM_TO_IDX[img_file.stem]))
        return samples

    return _collect(train_patients), _collect(val_patients)


def get_train_loaders(
    dataset_root: Path,
    batch_size: int = 32,
    num_workers: int = 2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from a dataset root directory.

    Args:
        dataset_root: Root directory containing one subfolder per patient.
        batch_size: Number of images per batch.
        num_workers: Subprocesses for data loading.
        val_ratio: Fraction of patients reserved for validation.
        seed: Random seed for the patient split.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    val_transform = get_inference_transform()

    train_samples, val_samples = build_splits(dataset_root, val_ratio, seed)

    train_loader = DataLoader(
        DentalDataset(train_samples, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DentalDataset(val_samples, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_inference_transform() -> T.Compose:
    """Return the deterministic transform used at inference time."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

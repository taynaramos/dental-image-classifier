"""Utilitários compartilhados: dispositivo, reprodutibilidade e checkpoints."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .model import DentalCNN, ModelConfig


def get_device() -> torch.device:
    """Retorna a GPU (CUDA) se disponível, senão a CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Fixa a semente aleatória do Python, NumPy e PyTorch para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path | str,
    model: DentalCNN,
    classes: list[str],
    config: ModelConfig,
) -> None:
    """Salva o checkpoint do modelo via :func:`torch.save`.

    Grava ``model_state_dict``, a lista de *classes* (ordem dos índices de
    saída) e a *config* (hiperparâmetros necessários para reconstruir a rede
    e repetir o pré-processamento na inferência).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path | str,
    device: torch.device | None = None,
) -> tuple[DentalCNN, list[str], ModelConfig]:
    """Restaura um checkpoint salvo por :func:`save_checkpoint`.

    Retorna
    -------
    tuple
        ``(model, classes, config)`` — o modelo já em modo de avaliação (``eval``)
        e movido para *device*.
    """
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device)

    config = ModelConfig(**checkpoint["config"])
    classes = checkpoint["classes"]

    model = DentalCNN(
        num_classes=len(classes),
        in_channels=config.in_channels,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, config

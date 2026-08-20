"""Utilitários compartilhados: dispositivo, reprodutibilidade e checkpoints."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .model import DentalResNetTransfer
from .trainer import History


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
    model: DentalResNetTransfer,
    classes: list[str],
    image_size: int,
    *,
    history: History | None = None,
) -> None:
    """Salva o checkpoint do modelo via :func:`torch.save`.

    Grava ``model_state_dict``, a lista de *classes* (ordem dos índices de
    saída), o *image_size* usado no treino (necessário para repetir o mesmo
    pré-processamento na inferência) e, se fornecido, o ``history`` do
    treino — loss/acurácia de treino e validação por época, com a fase
    (``"frozen"``/``"finetune"``) de cada uma — para consulta posterior sem
    precisar re-treinar.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": image_size,
            "history": asdict(history) if history is not None else None,
        },
        path,
    )


def load_checkpoint(
    path: Path | str,
    device: torch.device | None = None,
) -> tuple[DentalResNetTransfer, list[str], int]:
    """Restaura um checkpoint salvo por :func:`save_checkpoint`.

    Retorna
    -------
    tuple
        ``(model, classes, image_size)`` — o modelo já em modo de avaliação
        (``eval``) e movido para *device*.
    """
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device)

    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]

    model = DentalResNetTransfer(num_classes=len(classes), freeze_extractor=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, image_size


def load_history(path: Path | str) -> History | None:
    """Lê apenas o ``history`` (loss/acurácia por época, por fase) de um checkpoint salvo.

    Retorna ``None`` se o checkpoint foi salvo sem ``history``.
    """
    checkpoint = torch.load(path, map_location="cpu")
    dados = checkpoint.get("history")
    return History(**dados) if dados is not None else None

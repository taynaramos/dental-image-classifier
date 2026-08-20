"""Utilitários compartilhados entre os comandos ``*-predict`` da CLI.

Evita duplicar a coleta de caminhos de imagem e a impressão legível do
resultado em cada um dos quatro comandos de predição (``predict``,
``resnet18-predict``, ``kfold-predict``, ``pca-predict``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

# Extensões aceitas para busca em diretórios (--image-dir)
EXTENSOES_JPEG = {".jpeg", ".jpg"}


class SupportsPrediction(Protocol):
    """Formato comum às três classes ``Prediction`` (pca_svc/pytorch_kfold/pytorch_resnet18_transfer).

    As três são dataclasses independentes (sem herança em comum) com o mesmo
    formato — um ``Protocol`` estrutural descreve isso sem precisar importar
    (e acoplar) as três só para checagem de tipo.
    """

    label: str
    probabilities: dict[str, float]


def coletar_caminhos(image: Path | None, image_dir: Path | None) -> list[Path]:
    """Resolve a lista de imagens a classificar a partir de ``--image``/``--image-dir``."""
    if image is not None:
        return [image]
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSOES_JPEG)


def imprimir_predicao(path: Path, pred: SupportsPrediction) -> None:
    """Imprime o resultado da predição de forma legível.

    Exibe o rótulo predito e uma barra de probabilidade para cada classe,
    ordenadas da mais para a menos provável.
    """
    print(f"\n{path.name}")
    print(f"  classe predita : {pred.label}")
    print("  probabilidades :")
    for cls, prob in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        barra = "#" * int(prob * 30)
        print(f"    {cls:<22} {prob:5.1%}  {barra}")

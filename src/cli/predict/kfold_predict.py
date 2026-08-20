"""Comando CLI: executa inferência com o classificador CNN (PyTorch) odontológico."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from src.cli.tools.io import coletar_caminhos, imprimir_predicao

if TYPE_CHECKING:
    from src.pytorch_kfold.predict import Prediction


class KfoldPredict:
    """Comando ``kfold-predict``: classifica uma imagem ou pasta com o modelo CNN (PyTorch)."""

    name: ClassVar[str] = "kfold-predict"
    help: ClassVar[str] = "Classifica uma imagem ou pasta usando o modelo CNN (PyTorch)."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--model",
            type=Path,
            required=True,
            help="Caminho para o checkpoint gerado pelo kfold-train.",
        )

        # Aceita uma imagem única OU um diretório (mutuamente exclusivos)
        grupo = parser.add_mutually_exclusive_group(required=True)
        grupo.add_argument(
            "--image",
            type=Path,
            help="Caminho para um único arquivo de imagem.",
        )
        grupo.add_argument(
            "--image-dir",
            type=Path,
            help="Pasta de imagens para classificação em lote.",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Carrega o checkpoint e executa a inferência nas imagens fornecidas."""
        caminhos = coletar_caminhos(args.image, args.image_dir)
        if not caminhos:
            print(f"Nenhuma imagem JPEG encontrada em {args.image_dir}")
            return

        predicoes = predict_kfold(args.model, caminhos)

        for caminho, pred in zip(caminhos, predicoes):
            imprimir_predicao(caminho, pred)


def predict_kfold(checkpoint: Path, caminhos: list[Path]) -> list[Prediction]:
    """Classifica uma lista de imagens com um checkpoint CNN (PyTorch) salvo.

    Reutilizada pelo comando ``predict`` unificado — retorna uma
    :class:`~src.pytorch_kfold.predict.Prediction` por caminho, na mesma
    ordem de *caminhos*.
    """
    from src.pytorch_kfold.predict import predict
    from src.pytorch_kfold.utils import get_device, load_checkpoint

    device = get_device()
    print(f"[modelo] carregando de {checkpoint}")
    model, classes, config = load_checkpoint(checkpoint, device)

    return [predict(caminho, model, classes, config, device) for caminho in caminhos]

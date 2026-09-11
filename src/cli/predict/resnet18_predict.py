"""Comando CLI: executa inferência com o classificador por transfer learning (ResNet-18)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from src.cli.tools.io import coletar_caminhos, imprimir_predicao

if TYPE_CHECKING:
    from src.pytorch_resnet18_transfer.predict import Prediction


class Resnet18Predict:
    """Comando ``resnet18-predict``: classifica uma imagem ou pasta com o modelo ResNet-18."""

    name: ClassVar[str] = "resnet18-predict"
    help: ClassVar[str] = "Classifica uma imagem ou pasta usando o modelo de transfer learning (ResNet-18)."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--model",
            type=Path,
            required=True,
            help="Caminho para o checkpoint gerado pelo resnet18-train.",
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

        predicoes = predict_resnet18(args.model, caminhos)

        for caminho, pred in zip(caminhos, predicoes):
            imprimir_predicao(caminho, pred)


def predict_resnet18(checkpoint: Path, caminhos: list[Path]) -> list[Prediction]:
    """Classifica uma lista de imagens com um checkpoint ResNet-18 (transfer learning) salvo.

    Reutilizada pelo comando ``predict`` unificado — retorna uma
    :class:`~src.pytorch_resnet18_transfer.predict.Prediction` por caminho,
    na mesma ordem de *caminhos*.
    """
    from src.pytorch_resnet18_transfer.predict import predict
    from src.pytorch_resnet18_transfer.utils import get_device, load_checkpoint

    device = get_device()
    print(f"[modelo] carregando de {checkpoint}")
    model, classes, image_size = load_checkpoint(checkpoint, device)

    return [predict(caminho, model, classes, image_size, device) for caminho in caminhos]

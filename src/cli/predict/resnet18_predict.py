"""Comando CLI: executa inferência com o classificador por transfer learning (ResNet-18)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from src.pytorch_resnet18_transfer.predict import Prediction

# Extensões aceitas para busca em diretórios
_EXTENSOES_JPEG = {".jpeg", ".jpg"}


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
        from src.pytorch_resnet18_transfer.predict import predict
        from src.pytorch_resnet18_transfer.utils import get_device, load_checkpoint

        device = get_device()
        print(f"[modelo] carregando de {args.model}")
        model, classes, image_size = load_checkpoint(args.model, device)

        if args.image is not None:
            caminhos = [args.image]
        else:
            caminhos = sorted(
                p for p in args.image_dir.iterdir()
                if p.suffix.lower() in _EXTENSOES_JPEG
            )
            if not caminhos:
                print(f"Nenhuma imagem JPEG encontrada em {args.image_dir}")
                return

        for caminho in caminhos:
            pred = predict(caminho, model, classes, image_size, device)
            _imprimir_predicao(caminho, pred)


def _imprimir_predicao(path: Path, pred: Prediction) -> None:
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

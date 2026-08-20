"""Comando CLI: executa inferência com o classificador PCA-SVC odontológico."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from src.cli.tools.io import coletar_caminhos, imprimir_predicao

if TYPE_CHECKING:
    from src.pca_svc.model import Prediction


class PcaPredict:
    """Comando ``pca-predict``: classifica uma imagem ou pasta com o modelo PCA-SVC."""

    name: ClassVar[str] = "pca-predict"
    help: ClassVar[str] = "Classifica uma imagem ou pasta usando o modelo PCA-SVC."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--model",
            type=Path,
            required=True,
            help="Caminho para o arquivo de modelo gerado pelo pca-train.",
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

        parser.add_argument(
            "--image-size",
            type=int,
            default=128,
            help="Deve corresponder ao tamanho usado no treino (padrão: 128).",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Carrega o modelo e executa a inferência nas imagens fornecidas."""
        caminhos = coletar_caminhos(args.image, args.image_dir)
        if not caminhos:
            print(f"Nenhuma imagem JPEG encontrada em {args.image_dir}")
            return

        predicoes = predict_pca(args.model, args.image_size, caminhos)

        for caminho, pred in zip(caminhos, predicoes):
            imprimir_predicao(caminho, pred)


def predict_pca(
    checkpoint: Path, image_size: int, caminhos: list[Path]
) -> list["Prediction"]:
    """Classifica uma lista de imagens com um modelo PCA-SVC salvo.

    Reutilizada pelo comando ``predict`` unificado — retorna uma
    :class:`~src.pca_svc.model.Prediction` por caminho, na mesma ordem de
    *caminhos*.
    """
    import numpy as np

    from src.pca_svc.dataset import carregar_luma
    from src.pca_svc.model import DentalClassifier

    print(f"[modelo] carregando de {checkpoint}")
    classificador = DentalClassifier.load(checkpoint)

    tamanho = (image_size, image_size)
    X = np.stack([carregar_luma(p, tamanho) for p in caminhos])
    return classificador.predict_proba(X)

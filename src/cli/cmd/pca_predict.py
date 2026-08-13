"""Comando CLI: executa inferência com o classificador PCA-SVC odontológico."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from src.pca_svc.dataset import _STEM_TO_LABEL
from src.pca_svc.model import DentalClassifier, Prediction

# Extensões aceitas para busca em diretórios
_EXTENSOES_JPEG = {".jpeg", ".jpg"}


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
        print(f"[modelo] carregando de {args.model}")
        classificador = DentalClassifier.load(args.model)

        tamanho = (args.image_size, args.image_size)

        # Coleta os caminhos de imagem a classificar
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

        # Carrega todas as imagens como luminância e empilha em uma matriz
        X = np.stack([_carregar_luma(p, tamanho) for p in caminhos])
        predicoes = classificador.predict_proba(X)

        for caminho, pred in zip(caminhos, predicoes):
            _imprimir_predicao(caminho, pred)


# ------------------------------------------------------------------
# Auxiliares
# ------------------------------------------------------------------

def _carregar_luma(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Carrega uma imagem, extrai o canal Y (luminância) e a achata.

    Parâmetros
    ----------
    path:
        Caminho para o arquivo JPEG.
    image_size:
        (largura, altura) para redimensionar a imagem.

    Retorna
    -------
    np.ndarray
        Vetor achatado de float32 com ``largura * altura`` elementos.
    """
    img = Image.open(path).convert("YCbCr")
    luma, *_ = img.split()  # descarta Cb e Cr
    luma = luma.resize(image_size, Image.LANCZOS)
    return np.array(luma, dtype=np.float32).ravel()


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

"""Comando CLI unificado: executa inferência e imprime o resultado em JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

from src.cli.tools.io import coletar_caminhos

if TYPE_CHECKING:
    from src.pca_svc.model import Prediction


class Predict:
    """Comando ``predict``: executa inferência com o modelo escolhido em ``--model``.

    Dois modos, escolhidos pelo argumento de entrada:

    * ``--image`` — **single_inference**: classifica uma única imagem e
      imprime a classe prevista e o vetor de probabilidades em JSON.
    * ``--image-dir`` — **test_case**: classifica uma pasta com as imagens
      de um caso (esperado: uma imagem por vista) e imprime, para cada
      vista, o arquivo que foi classificado com aquele rótulo — ou
      ``"Not found"`` se nenhuma imagem da pasta foi classificada com essa
      vista. Com ``--probabilities``, o vetor de probabilidades de cada
      imagem também é incluído.
    """

    name: ClassVar[str] = "predict"
    help: ClassVar[str] = "Executa inferência (imagem única ou pasta de um caso) e imprime o resultado em JSON."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--model",
            choices=["pca", "kfold", "resnet18"],
            required=True,
            help="Qual modelo usar para a inferência.",
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            required=True,
            help="Caminho para o checkpoint/modelo salvo pelo comando train (ou pelo *-train correspondente).",
        )

        # single_inference (--image) OU test_case (--image-dir), mutuamente exclusivos
        grupo = parser.add_mutually_exclusive_group(required=True)
        grupo.add_argument(
            "--image",
            type=Path,
            help="Uma única imagem — modo single_inference (classe + vetor de probabilidades).",
        )
        grupo.add_argument(
            "--image-dir",
            type=Path,
            help="Pasta com as imagens de um caso (uma por vista) — modo test_case.",
        )

        parser.add_argument(
            "--image-size",
            type=int,
            default=None,
            help="Usado apenas por --model pca — deve corresponder ao treino (padrão: 128).",
        )
        parser.add_argument(
            "--probabilities",
            action="store_true",
            help="No modo test_case, inclui também o vetor de probabilidades de cada imagem.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Além de imprimir, grava o JSON neste arquivo.",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Executa a inferência e imprime (e opcionalmente grava) o resultado em JSON."""
        caminhos = coletar_caminhos(args.image, args.image_dir)
        if not caminhos:
            print(json.dumps({"erro": f"Nenhuma imagem JPEG encontrada em {args.image_dir}"}))
            return

        predicoes = self._predict(args, caminhos)

        if args.image is not None:
            resultado = _single_inference_json(predicoes[0])
        else:
            resultado = _test_case_json(caminhos, predicoes, incluir_probabilidades=args.probabilities)

        texto = json.dumps(resultado, indent=2, ensure_ascii=False)
        print(texto)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(texto, encoding="utf-8")

    def _predict(self, args: argparse.Namespace, caminhos: list[Path]) -> list["Prediction"]:
        """Despacha para a rotina de predição do modelo escolhido em ``--model``."""
        if args.model == "pca":
            from src.cli.predict.pca_predict import predict_pca

            return predict_pca(args.checkpoint, args.image_size or 128, caminhos)
        if args.model == "kfold":
            from src.cli.predict.kfold_predict import predict_kfold

            return predict_kfold(args.checkpoint, caminhos)

        from src.cli.predict.resnet18_predict import predict_resnet18

        return predict_resnet18(args.checkpoint, caminhos)


def _single_inference_json(pred: "Prediction") -> dict:
    """Monta o JSON do modo single_inference: classe prevista + vetor de probabilidades."""
    return {"class": pred.label, "probabilities": pred.probabilities}


def _test_case_json(
    caminhos: list[Path], predicoes: list["Prediction"], *, incluir_probabilidades: bool
) -> dict:
    """Monta o JSON do modo test_case: ``{view: image_file}``, com ``"Not found"`` para vistas ausentes.

    Quando mais de uma imagem é classificada com a mesma vista, a de maior
    confiança para aquela classe é a escolhida.
    """
    melhor_por_vista: dict[str, tuple[str, float]] = {}
    for caminho, pred in zip(caminhos, predicoes):
        confianca = pred.probabilities.get(pred.label, 0.0)
        atual = melhor_por_vista.get(pred.label)
        if atual is None or confianca > atual[1]:
            melhor_por_vista[pred.label] = (caminho.name, confianca)

    # Usa a ordem de classes reportada pelo próprio checkpoint carregado —
    # não depende de nenhuma lista fixa compartilhada entre os módulos.
    vistas = list(predicoes[0].probabilities.keys()) if predicoes else []
    views = {vista: melhor_por_vista.get(vista, ("Not found", 0.0))[0] for vista in vistas}

    resultado: dict = {"views": views}
    if incluir_probabilidades:
        resultado["probabilities"] = {
            caminho.name: pred.probabilities for caminho, pred in zip(caminhos, predicoes)
        }
    return resultado

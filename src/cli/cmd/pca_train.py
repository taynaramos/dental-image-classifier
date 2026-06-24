"""Comando CLI: treina o classificador PCA-SVC odontológico."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import numpy as np

from src.pca_svc.dataset import DentalDataset
from src.pca_svc.model import DentalClassifier


class PcaTrain:
    """Comando ``pca-train``: ajusta o pipeline PCA + SVC e salva o modelo."""

    name: ClassVar[str] = "pca-train"
    help: ClassVar[str] = "Treina o classificador PCA + SVC de vistas intraorais."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--dataset-path",
            type=Path,
            required=True,
            help="Diretório raiz com uma sub-pasta por sujeito.",
        )
        parser.add_argument(
            "--model-out",
            type=Path,
            default=Path("pca_svc_model.pkl"),
            help="Caminho de saída para o modelo serializado (padrão: pca_svc_model.pkl).",
        )
        parser.add_argument(
            "--image-size",
            type=int,
            default=128,
            help="Lado (px) para redimensionar as imagens antes de achatar (padrão: 128).",
        )
        parser.add_argument(
            "--n-components",
            type=int,
            default=None,
            help=(
                "Número de componentes PCA. "
                "Omita para usar o mínimo que explique --variance-threshold."
            ),
        )
        parser.add_argument(
            "--variance-threshold",
            type=float,
            default=0.95,
            help="Fração de variância a reter quando --n-components não é definido (padrão: 0,95).",
        )
        parser.add_argument(
            "--C",
            type=float,
            default=10.0,
            help="Parâmetro de regularização C do SVC (padrão: 10,0).",
        )
        parser.add_argument(
            "--gamma",
            type=str,
            default="scale",
            help="Coeficiente do kernel do SVC (padrão: scale).",
        )
        parser.add_argument("--train-ratio", type=float, default=0.70)
        parser.add_argument("--val-ratio",   type=float, default=0.15)
        parser.add_argument("--test-ratio",  type=float, default=0.15)
        parser.add_argument("--seed",        type=int,   default=42)

    def run(self, args: argparse.Namespace) -> None:
        """Executa o fluxo completo: carrega dados, treina e salva o modelo."""
        tamanho = (args.image_size, args.image_size)

        # --- Carregamento e divisão do dataset ---
        print(f"[dataset] carregando de {args.dataset_path}")
        dataset = DentalDataset(
            args.dataset_path,
            image_size=tamanho,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        contagens = dataset.subject_counts()
        print(
            f"[dataset] sujeitos — "
            f"treino: {contagens['train']}  val: {contagens['val']}  teste: {contagens['test']}"
        )

        print("[dataset] carregando partição de treino (luminância) …")
        X_train, y_train = dataset.load_split("train")

        print("[dataset] carregando partição de validação …")
        X_val, y_val = dataset.load_split("val")

        print("[dataset] carregando partição de teste …")
        X_test, y_test = dataset.load_split("test")

        print(
            f"[dataset] formas — "
            f"treino: {X_train.shape}  val: {X_val.shape}  teste: {X_test.shape}"
        )

        # --- Treinamento: PCA + SVC ---
        print("[treino] ajustando PCA + SVC …")
        classificador = DentalClassifier(
            n_components=args.n_components,
            auto_variance_threshold=args.variance_threshold,
            C=args.C,
            gamma=args.gamma,
            seed=args.seed,
        )
        classificador.fit(X_train, y_train)

        # Exibe resumo do PCA
        extrator = classificador.extractor
        print(
            f"[pca] componentes selecionados : {extrator.n_components_}  "
            f"(variância retida: {extrator._pca.explained_variance_ratio_.sum():.4f})"
        )
        _imprimir_tabela_variancia(extrator.cumulative_explained_variance())

        # --- Avaliação ---
        acc_treino = classificador.score(X_train, y_train)
        acc_val    = classificador.score(X_val,   y_val)
        acc_teste  = classificador.score(X_test,  y_test)

        print(f"\n[resultado] acurácia treino     : {acc_treino:.4f}")
        print(f"[resultado] acurácia validação  : {acc_val:.4f}")
        print(f"[resultado] acurácia teste      : {acc_teste:.4f}")

        # --- Persistência ---
        classificador.save(args.model_out)
        print(f"\n[modelo] salvo em {args.model_out}")


def _imprimir_tabela_variancia(cumulativa: np.ndarray) -> None:
    """Exibe uma tabela com o número de componentes necessários por limiar de variância."""
    limiares = [0.80, 0.90, 0.95, 0.99]
    cabecalho = "  limiar | componentes"
    print(f"\n[pca] variância acumulada\n{cabecalho}")
    print("  " + "-" * (len(cabecalho) - 2))
    for t in limiares:
        n = int(np.searchsorted(cumulativa, t)) + 1
        print(f"   {t:.0%}   |  {n}")
    print()

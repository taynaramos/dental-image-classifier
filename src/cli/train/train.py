"""Comando CLI unificado: treina um dos modelos disponíveis e salva um checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar


class Train:
    """Comando ``train``: treina o modelo escolhido em ``--model`` e salva um checkpoint.

    Ponto de entrada único para treino, independente do modelo — expõe só os
    parâmetros comuns às três soluções. Para controle fino de hiperparâmetros
    específicos (ex.: ``--C``/``--gamma`` do SVC, ``--hidden-dim`` da CNN),
    use diretamente ``pca-train``/``kfold-train``/``resnet18-train``.
    """

    name: ClassVar[str] = "train"
    help: ClassVar[str] = "Treina um dos modelos disponíveis (pca, kfold, resnet18) e salva um checkpoint."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--model",
            choices=["pca", "kfold", "resnet18"],
            required=True,
            help="Qual modelo treinar.",
        )
        parser.add_argument(
            "--dataset-path",
            type=Path,
            required=True,
            help="Diretório raiz do dataset (uma sub-pasta por sujeito, ou já dividido em train/val/test).",
        )
        parser.add_argument(
            "--model-out",
            type=Path,
            default=None,
            help=(
                "Caminho de saída para o checkpoint. Padrão por modelo: "
                "pca_svc_model.pkl / artifacts/kfold_model.pth / artifacts/resnet18_transfer_model.pth."
            ),
        )
        parser.add_argument(
            "--image-size",
            type=int,
            default=None,
            help="Lado (px) das imagens. Padrão por modelo: 128 (pca/kfold) ou 224 (resnet18).",
        )
        parser.add_argument(
            "--epochs", type=int, default=20, help="Épocas de treino — usado apenas por --model kfold (padrão: 20)."
        )
        parser.add_argument(
            "--frozen-epochs",
            type=int,
            default=5,
            help="Épocas com o extrator congelado — usado apenas por --model resnet18 (padrão: 5).",
        )
        parser.add_argument(
            "--finetune-epochs",
            type=int,
            default=15,
            help="Épocas de fine-tune — usado apenas por --model resnet18 (padrão: 15).",
        )
        parser.add_argument("--train-ratio", type=float, default=0.70)
        parser.add_argument("--val-ratio", type=float, default=0.15)
        parser.add_argument("--test-ratio", type=float, default=0.15)
        parser.add_argument("--seed", type=int, default=42)

    def run(self, args: argparse.Namespace) -> None:
        """Despacha para a rotina de treino do modelo escolhido em ``--model``."""
        if args.model == "pca":
            self._train_pca(args)
        elif args.model == "kfold":
            self._train_kfold(args)
        else:
            self._train_resnet18(args)

    # ------------------------------------------------------------------
    # Rotinas por modelo — cada uma adapta os args comuns para o formato
    # esperado pela função de treino já existente no comando dedicado,
    # preenchendo os hiperparâmetros não expostos aqui com os padrões
    # de cada módulo.
    # ------------------------------------------------------------------

    def _train_pca(self, args: argparse.Namespace) -> None:
        from src.cli.train.pca_train import train_pca

        train_pca(
            argparse.Namespace(
                dataset_path=args.dataset_path,
                model_out=args.model_out or Path("pca_svc_model.pkl"),
                image_size=args.image_size or 128,
                n_components=None,
                variance_threshold=0.95,
                C=10.0,
                gamma="scale",
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
        )

    def _train_kfold(self, args: argparse.Namespace) -> None:
        from src.cli.train.kfold_train import train_kfold

        train_kfold(
            argparse.Namespace(
                dataset_path=args.dataset_path,
                model_out=args.model_out or Path("artifacts/kfold_model.pth"),
                epochs=args.epochs,
                patience=None,
                min_delta=0.0,
                batch_size=32,
                learning_rate=1e-3,
                image_size=args.image_size or 128,
                grayscale=True,
                hidden_dim=128,
                dropout=0.5,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                num_workers=0,
                seed=args.seed,
            )
        )

    def _train_resnet18(self, args: argparse.Namespace) -> None:
        from src.cli.train.resnet18_train import train_resnet18

        train_resnet18(
            argparse.Namespace(
                dataset_path=args.dataset_path,
                model_out=args.model_out or Path("artifacts/resnet18_transfer_model.pth"),
                frozen_epochs=args.frozen_epochs,
                finetune_epochs=args.finetune_epochs,
                head_lr=1e-3,
                finetune_lr=1e-4,
                batch_size=32,
                image_size=args.image_size or 224,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                num_workers=0,
                seed=args.seed,
            )
        )

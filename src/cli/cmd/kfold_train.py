"""Comando CLI: treina o classificador CNN (PyTorch) odontológico."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar


class KfoldTrain:
    """Comando ``kfold-train``: treina a CNN e salva o modelo."""

    name: ClassVar[str] = "kfold-train"
    help: ClassVar[str] = "Treina o classificador CNN (PyTorch) de vistas intraorais."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)

        parser.add_argument(
            "--dataset-path",
            type=Path,
            required=True,
            help="Diretório raiz do dataset (uma sub-pasta por sujeito, ou já dividido em train/val/test).",
        )
        parser.add_argument(
            "--model-out",
            type=Path,
            default=Path("artifacts/kfold_model.pth"),
            help="Caminho de saída para o checkpoint treinado (padrão: artifacts/kfold_model.pth).",
        )
        parser.add_argument("--epochs", type=int, default=20, help="Número máximo de épocas de treino (padrão: 20).")
        parser.add_argument(
            "--patience",
            type=int,
            default=None,
            help="Early stopping: interrompe o treino após N épocas sem melhora na validation loss (padrão: desativado).",
        )
        parser.add_argument(
            "--min-delta",
            type=float,
            default=0.0,
            help="Melhora mínima na validation loss para zerar a paciência do early stopping (padrão: 0,0).",
        )
        parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do lote (padrão: 32).")
        parser.add_argument(
            "--learning-rate", type=float, default=1e-3, help="Taxa de aprendizado do Adam (padrão: 0,001)."
        )
        parser.add_argument(
            "--image-size", type=int, default=128, help="Lado (px) para redimensionar as imagens (padrão: 128)."
        )
        parser.add_argument(
            "--grayscale",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Converte as imagens para escala de cinza (padrão: ativado; use --no-grayscale para manter RGB).",
        )
        parser.add_argument("--hidden-dim", type=int, default=128, help="Dimensão da camada oculta (padrão: 128).")
        parser.add_argument("--dropout", type=float, default=0.5, help="Probabilidade de dropout (padrão: 0,5).")
        parser.add_argument("--train-ratio", type=float, default=0.70)
        parser.add_argument("--val-ratio", type=float, default=0.15)
        parser.add_argument("--test-ratio", type=float, default=0.15)
        parser.add_argument("--num-workers", type=int, default=0, help="Processos auxiliares do DataLoader (padrão: 0).")
        parser.add_argument("--seed", type=int, default=42)

    def run(self, args: argparse.Namespace) -> None:
        """Executa o fluxo completo: prepara dados, treina, avalia e salva o modelo."""
        from src.pytorch_kfold.dataset import build_dataloaders, resolve_imagefolder_root
        from src.pytorch_kfold.model import DentalCNN, ModelConfig
        from src.pytorch_kfold.trainer import Trainer
        from src.pytorch_kfold.utils import get_device, save_checkpoint, set_seed

        set_seed(args.seed)
        device = get_device()
        print(f"[dispositivo] usando {device}")

        # --- Dataset: garante o layout ImageFolder e cria os DataLoaders ---
        imagefolder_root = resolve_imagefolder_root(
            args.dataset_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        train_loader, val_loader, test_loader, classes = build_dataloaders(
            imagefolder_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            grayscale=args.grayscale,
            num_workers=args.num_workers,
        )
        print(f"[dataset] classes ({len(classes)}): {classes}")
        print(
            f"[dataset] batches — treino: {len(train_loader)}  val: {len(val_loader)}  teste: {len(test_loader)}"
        )

        # --- Modelo e treinamento ---
        config = ModelConfig(
            image_size=args.image_size,
            grayscale=args.grayscale,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        model = DentalCNN(
            num_classes=len(classes),
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        trainer = Trainer(model, device, learning_rate=args.learning_rate)
        trainer.fit(train_loader, val_loader, epochs=args.epochs, patience=args.patience, min_delta=args.min_delta)

        # --- Avaliação final no conjunto de teste ---
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # --- Persistência ---
        save_checkpoint(args.model_out, model, classes, config)
        print(f"\n[modelo] salvo em {args.model_out}")

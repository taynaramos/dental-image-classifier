"""Comando CLI: treina o classificador por transfer learning (ResNet-18)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar


class Resnet18Train:
    """Comando ``resnet18-train``: treina o extrator+classificador e salva o modelo."""

    name: ClassVar[str] = "resnet18-train"
    help: ClassVar[str] = "Treina o classificador de vistas intraorais por transfer learning (ResNet-18)."

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
            default=Path("artifacts/resnet18_transfer_model.pth"),
            help="Caminho de saída para o checkpoint treinado (padrão: artifacts/resnet18_transfer_model.pth).",
        )
        parser.add_argument(
            "--frozen-epochs",
            type=int,
            default=5,
            help="Épocas da Fase 1, com o extrator ResNet-18 congelado (padrão: 5).",
        )
        parser.add_argument(
            "--finetune-epochs",
            type=int,
            default=15,
            help="Épocas da Fase 2, com fine-tune da rede inteira (padrão: 15).",
        )
        parser.add_argument("--head-lr", type=float, default=1e-3, help="Taxa de aprendizado da Fase 1 (padrão: 0,001).")
        parser.add_argument(
            "--finetune-lr", type=float, default=1e-4, help="Taxa de aprendizado da Fase 2 (padrão: 0,0001)."
        )
        parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do lote (padrão: 32).")
        parser.add_argument(
            "--image-size", type=int, default=224, help="Lado (px) da imagem após o crop (padrão: 224)."
        )
        parser.add_argument("--train-ratio", type=float, default=0.70)
        parser.add_argument("--val-ratio", type=float, default=0.15)
        parser.add_argument("--test-ratio", type=float, default=0.15)
        parser.add_argument("--num-workers", type=int, default=0, help="Processos auxiliares do DataLoader (padrão: 0).")
        parser.add_argument("--seed", type=int, default=42)

    def run(self, args: argparse.Namespace) -> None:
        """Executa o fluxo completo: prepara dados, treina em duas fases, avalia e salva o modelo."""
        from src.pytorch_resnet18_transfer.dataset import build_dataloaders, resolve_imagefolder_root
        from src.pytorch_resnet18_transfer.model import DentalResNetTransfer
        from src.pytorch_resnet18_transfer.trainer import Trainer
        from src.pytorch_resnet18_transfer.utils import get_device, save_checkpoint, set_seed

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
            num_workers=args.num_workers,
        )
        print(f"[dataset] classes ({len(classes)}): {classes}")
        print(
            f"[dataset] batches — treino: {len(train_loader)}  val: {len(val_loader)}  teste: {len(test_loader)}"
        )

        # --- Modelo: extrator ResNet-18 (congelado na Fase 1) + classificador linear ---
        model = DentalResNetTransfer(num_classes=len(classes))

        trainer = Trainer(model, device, head_lr=args.head_lr, finetune_lr=args.finetune_lr)
        trainer.fit(
            train_loader,
            val_loader,
            frozen_epochs=args.frozen_epochs,
            finetune_epochs=args.finetune_epochs,
        )

        # --- Avaliação final no conjunto de teste ---
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # --- Persistência ---
        save_checkpoint(args.model_out, model, classes, args.image_size)
        print(f"\n[modelo] salvo em {args.model_out}")

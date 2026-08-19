"""Loop de treinamento em duas fases: extrator congelado, depois fine-tune."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .model import DentalResNetTransfer


@dataclass
class History:
    """Métricas por época, na ordem em que foram registradas.

    ``phase`` marca, para cada época, se ela ocorreu com o extrator
    congelado (``"frozen"``) ou em fine-tune (``"finetune"``).
    """

    phase: list[str] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


class Trainer:
    """Treina um :class:`~src.pytorch_resnet18_transfer.model.DentalResNetTransfer` em duas fases.

    **Fase 1 — extrator congelado**: apenas o classificador
    (:attr:`~src.pytorch_resnet18_transfer.model.DentalResNetTransfer.classifier`)
    é treinado; os pesos da ResNet-18 pré-treinada permanecem fixos.

    **Fase 2 — fine-tune**: o extrator é liberado (:meth:`~src.pytorch_resnet18_transfer.feature_extractor.ResNetFeatureExtractor.unfreeze`)
    e a rede inteira continua treinando, com uma taxa de aprendizado menor.

    Parâmetros
    ----------
    model:
        Rede a treinar.
    device:
        Dispositivo de execução (CPU ou GPU).
    head_lr:
        Taxa de aprendizado da Fase 1 (só o classificador).
    finetune_lr:
        Taxa de aprendizado da Fase 2 (rede inteira).
    """

    def __init__(
        self,
        model: DentalResNetTransfer,
        device: torch.device,
        *,
        head_lr: float = 1e-3,
        finetune_lr: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.head_lr = head_lr
        self.finetune_lr = finetune_lr

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        frozen_epochs: int = 5,
        finetune_epochs: int = 15,
        frozen_patience: int = 2,
        finetune_patience: int = 3,
    ) -> History:
        """Executa as duas fases de treino, imprimindo as métricas a cada época.

        Parâmetros
        ----------
        train_loader, val_loader:
            DataLoaders de treino e validação.
        frozen_epochs:
            Número de épocas da Fase 1 (extrator congelado).
        finetune_epochs:
            Número de épocas da Fase 2 (fine-tune completo).
        frozen_patience, finetune_patience:
            Paciência do ``ReduceLROnPlateau`` em cada fase.

        Retorna
        -------
        History
            Loss e acurácia de treino/validação por época, com a fase de cada uma.
            Ao final, os pesos do modelo são os da melhor época de validação.
        """
        history = History()
        best_val_acc = 0.0
        best_state: dict[str, torch.Tensor] | None = None

        # --- Fase 1: extrator congelado, treina só o classificador ---
        self.model.extractor.freeze()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.head_lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=frozen_patience, factor=0.5)

        print("=== Fase 1: extrator congelado ===")
        for epoch in range(1, frozen_epochs + 1):
            best_val_acc, best_state = self._run_and_record(
                history, "frozen", epoch, frozen_epochs,
                train_loader, val_loader, optimizer, scheduler,
                best_val_acc, best_state,
            )

        # --- Fase 2: fine-tune completo ---
        self.model.extractor.unfreeze()
        optimizer = optim.Adam(self.model.parameters(), lr=self.finetune_lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=finetune_patience, factor=0.5)

        print("\n=== Fase 2: fine-tune completo ===")
        for epoch in range(1, finetune_epochs + 1):
            best_val_acc, best_state = self._run_and_record(
                history, "finetune", epoch, finetune_epochs,
                train_loader, val_loader, optimizer, scheduler,
                best_val_acc, best_state,
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\nMelhor acurácia de validação: {best_val_acc:.4f}")
        return history

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Retorna ``(loss, acurácia)`` do modelo em *loader*, sem atualizar pesos."""
        return self._run_epoch(loader, optimizer=None)

    # ------------------------------------------------------------------
    # Auxiliares internos
    # ------------------------------------------------------------------

    def _run_and_record(
        self,
        history: History,
        phase: str,
        epoch: int,
        total_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: ReduceLROnPlateau,
        best_val_acc: float,
        best_state: dict[str, torch.Tensor] | None,
    ) -> tuple[float, dict[str, torch.Tensor] | None]:
        """Roda uma época de treino + validação, atualiza o histórico e o melhor estado."""
        train_loss, train_acc = self._run_epoch(train_loader, optimizer=optimizer)
        val_loss, val_acc = self._run_epoch(val_loader, optimizer=None)
        scheduler.step(val_loss)

        history.phase.append(phase)
        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

        marcador = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(self.model.state_dict())
            marcador = " <- melhor até agora"

        print(
            f"Epoca {epoch:02d}/{total_epochs} | treino loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}{marcador}"
        )
        return best_val_acc, best_state

    def _run_epoch(
        self, loader: DataLoader, *, optimizer: optim.Optimizer | None
    ) -> tuple[float, float]:
        """Executa uma passada completa por *loader*, treinando (se *optimizer*) ou só avaliando."""
        train = optimizer is not None
        self.model.train(mode=train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if train:
                    optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

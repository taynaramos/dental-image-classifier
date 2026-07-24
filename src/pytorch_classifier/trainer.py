"""Loop de treinamento e avaliação do classificador CNN (PyTorch)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class History:
    """Métricas por época, na ordem em que foram registradas."""

    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


class Trainer:
    """Treina e avalia um :class:`~src.pytorch_classifier.model.DentalCNN`.

    Usa ``CrossEntropyLoss`` e o otimizador ``Adam`` com taxa de aprendizado
    configurável.

    Parâmetros
    ----------
    model:
        Rede a treinar.
    device:
        Dispositivo de execução (CPU ou GPU).
    learning_rate:
        Taxa de aprendizado do otimizador Adam.
    """

    def __init__(self, model: nn.Module, device: torch.device, *, learning_rate: float = 1e-3) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        *,
        patience: int | None = None,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> History:
        """Treina por até *epochs* épocas, imprimindo as métricas ao final de cada uma.

        Parâmetros
        ----------
        train_loader, val_loader:
            DataLoaders de treino e validação.
        epochs:
            Número máximo de épocas de treinamento.
        patience:
            Se definido, interrompe o treino quando a validation loss não
            melhorar (em pelo menos *min_delta*) por *patience* épocas
            seguidas. ``None`` desativa o early stopping.
        min_delta:
            Melhora mínima na validation loss para zerar a contagem de
            paciência.
        restore_best_weights:
            Se ``True``, restaura os pesos da época com menor validation
            loss ao final do treino (early stopped ou não).

        Retorna
        -------
        History
            Loss e acurácia de treino/validação, uma entrada por época
            efetivamente executada.
        """
        history = History()
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_sem_melhora = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)

            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_accuracy.append(val_acc)

            print(f"Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_sem_melhora = 0
                if restore_best_weights:
                    best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_sem_melhora += 1
                if patience is not None and epochs_sem_melhora >= patience:
                    print(
                        f"\nEarly stopping na época {epoch}: validation loss sem melhora "
                        f"por {patience} épocas (melhor valor: {best_val_loss:.4f})."
                    )
                    break

        if restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Retorna ``(loss, acurácia)`` do modelo em *loader*, sem atualizar pesos."""
        return self._run_epoch(loader, train=False)

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> tuple[float, float]:
        """Executa uma passada completa por *loader*, treinando ou apenas avaliando."""
        self.model.train(mode=train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

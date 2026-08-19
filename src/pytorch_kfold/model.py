"""CNN convolucional simples para classificação de vistas intraorais (PyTorch puro)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Hiperparâmetros necessários para reconstruir o modelo e o pré-processamento.

    Atributos
    ----------
    image_size:
        Lado (px) usado para redimensionar as imagens (deve casar com o treino).
    grayscale:
        Se ``True``, o modelo espera entradas com 1 canal (luminância); senão, 3 (RGB).
    hidden_dim:
        Dimensão da camada totalmente conectada oculta.
    dropout:
        Probabilidade de dropout aplicada antes da camada de saída.
    """

    image_size: int
    grayscale: bool
    hidden_dim: int = 128
    dropout: float = 0.5

    @property
    def in_channels(self) -> int:
        """Número de canais de entrada implícito por :attr:`grayscale`."""
        return 1 if self.grayscale else 3


class DentalCNN(nn.Module):
    """CNN simples treinada do zero — sem pesos pré-treinados nem transfer learning.

    Arquitetura: (Conv2d → ReLU → MaxPool) × 2 → AdaptiveAvgPool → Flatten →
    Linear → ReLU → Dropout → Linear.

    O pooling adaptativo antes do classificador torna a rede independente do
    tamanho de entrada, já que ``image_size`` é parametrizável.

    Parâmetros
    ----------
    num_classes:
        Número de classes de saída (obtido automaticamente do dataset via ``ImageFolder``).
    in_channels:
        Canais de entrada (1 para escala de cinza, 3 para RGB).
    hidden_dim:
        Dimensão da camada totalmente conectada oculta.
    dropout:
        Probabilidade de dropout antes da camada de saída.
    """

    _POOLED_SIZE = (8, 8)

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(self._POOLED_SIZE)

        pooled_features = 64 * self._POOLED_SIZE[0] * self._POOLED_SIZE[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna os logits (sem softmax) com forma ``(batch, num_classes)``."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

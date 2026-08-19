"""Classificador sobre as features extraídas pela ResNet-18 (transfer learning)."""

from __future__ import annotations

import torch
from torch import nn

from .feature_extractor import ResNetFeatureExtractor


class DentalResNetTransfer(nn.Module):
    """Extrator ResNet-18 + classificador linear.

    Ao contrário de uma CNN treinada do zero, aqui a extração de features
    (:attr:`extractor`) e a classificação (:attr:`classifier`) são componentes
    independentes: o extrator pode ser usado sozinho (``self.extractor(x)``)
    para obter apenas o vetor de features de 512 dimensões, sem passar pelo
    classificador. O treino ainda passa pelos dois em sequência via
    :meth:`forward`, mas o extrator normalmente permanece **congelado**
    durante a primeira fase de treino (ver :class:`~src.pytorch_resnet18_transfer.trainer.Trainer`).

    Parâmetros
    ----------
    num_classes:
        Número de classes de saída.
    freeze_extractor:
        Se ``True`` (padrão), o extrator começa congelado.
    """

    def __init__(self, num_classes: int, *, freeze_extractor: bool = True) -> None:
        super().__init__()
        self.extractor = ResNetFeatureExtractor(freeze=freeze_extractor)
        self.classifier = nn.Linear(self.extractor.out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna os logits (sem softmax) com forma ``(batch, num_classes)``."""
        features = self.extractor(x)
        return self.classifier(features)

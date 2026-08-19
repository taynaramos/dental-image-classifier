"""Extrator de features via ResNet-18 pré-treinada na ImageNet."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """Backbone ResNet-18 pré-treinada, usada como extrator de features.

    Remove a camada totalmente conectada original (``fc``) e expõe diretamente
    o vetor de features (dimensão 512) produzido pelo global average pooling
    da rede. Por padrão os pesos ficam congelados (``requires_grad=False``);
    chame :meth:`unfreeze` para liberar o fine-tuning de toda a rede.

    Parâmetros
    ----------
    freeze:
        Se ``True`` (padrão), congela os pesos do backbone logo na construção.
    """

    out_features: int = 512

    def __init__(self, *, freeze: bool = True) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove a camada de classificação original — mantém só o extrator de features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if freeze:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna o vetor de features com forma ``(batch, 512)`` para o lote *x*."""
        return self.backbone(x)

    def freeze(self) -> None:
        """Congela todos os pesos do backbone — nenhum gradiente é calculado para eles."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Libera todos os pesos do backbone para fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

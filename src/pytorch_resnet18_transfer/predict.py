"""Inferência com o classificador ResNet-18 (transfer learning) treinado."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_eval_transform
from .model import DentalResNetTransfer


@dataclass
class Prediction:
    """Resultado da predição para uma única imagem.

    Atributos
    ----------
    label:
        Rótulo da classe predita (mais provável).
    probabilities:
        Dicionário mapeando cada classe à sua probabilidade (softmax).
    """

    label: str
    probabilities: dict[str, float]


def predict(
    image_path: Path | str,
    model: DentalResNetTransfer,
    classes: list[str],
    image_size: int,
    device: torch.device,
) -> Prediction:
    """Classifica uma única imagem.

    Aplica o mesmo pré-processamento de avaliação usado no treino (resize +
    center crop + normalização ImageNet) e roda a rede em modo de avaliação,
    sem calcular gradientes.

    Parâmetros
    ----------
    image_path:
        Caminho para o arquivo de imagem.
    model:
        Modelo treinado (tipicamente obtido de ``utils.load_checkpoint``).
    classes:
        Lista de rótulos na mesma ordem dos índices de saída do modelo.
    image_size:
        Tamanho usado no treino (deve corresponder).
    device:
        Dispositivo onde a inferência será executada.

    Retorna
    -------
    Prediction
        Classe prevista e a probabilidade de cada classe.
    """
    transform = build_eval_transform(image_size)
    imagem = Image.open(image_path).convert("RGB")
    tensor = transform(imagem).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probabilidades = F.softmax(logits, dim=1).squeeze(0)

    indice_previsto = int(probabilidades.argmax().item())
    return Prediction(
        label=classes[indice_previsto],
        probabilities={cls: float(p) for cls, p in zip(classes, probabilidades.tolist())},
    )

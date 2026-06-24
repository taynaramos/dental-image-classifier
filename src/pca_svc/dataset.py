"""Carregador do dataset odontológico — apenas canal de luminância, baseado em numpy."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

CLASSES: list[str] = [
    "frontal",
    "inferior",
    "superior",
    "lateral_direita",
    "lateral_esquerda",
]

# Mapeamento do nome do arquivo (stem) para o rótulo da classe
_STEM_TO_LABEL: dict[str, str] = {
    "intraoral-frontal": "frontal",
    "intraoral-inferior": "inferior",
    "intraoral-superior": "superior",
    "intraoral-lateral-direita": "lateral_direita",
    "intraoral-lateral-esquerda": "lateral_esquerda",
}

Split = Literal["train", "val", "test"]


class DentalDataset:
    """Divisão treino/val/teste no nível do sujeito com pré-processamento de luminância.

    A divisão é feita **por sujeito**: todas as imagens de um mesmo sujeito
    ficam no mesmo conjunto, evitando vazamento de dados entre treino e avaliação.

    Parâmetros
    ----------
    root:
        Diretório raiz com uma sub-pasta por sujeito.
    image_size:
        (largura, altura) para redimensionar cada imagem antes de achatar.
    train_ratio, val_ratio, test_ratio:
        Proporções que devem somar 1,0.
    seed:
        Semente aleatória para embaralhamento reprodutível.
    """

    image_size: tuple[int, int]
    classes: list[str] = CLASSES

    def __init__(
        self,
        root: Path | str,
        *,
        image_size: tuple[int, int] = (128, 128),
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio deve ser igual a 1,0"
            )

        self.root = Path(root)
        self.image_size = image_size
        self._dividir_sujeitos(seed, train_ratio, val_ratio, test_ratio)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def load_split(self, split: Split) -> tuple[np.ndarray, np.ndarray]:
        """Retorna *(X, y)* para a partição solicitada.

        Parâmetros
        ----------
        split:
            ``"train"``, ``"val"`` ou ``"test"``.

        Retorna
        -------
        X : np.ndarray
            Forma ``(n_amostras, largura * altura)`` — luminância achatada (float32).
        y : np.ndarray
            Forma ``(n_amostras,)`` — rótulos de classe como strings.
        """
        sujeitos = self._sujeitos[split]
        return self._coletar_luma(sujeitos)

    def subject_counts(self) -> dict[str, int]:
        """Retorna a quantidade de sujeitos em cada partição."""
        return {split: len(subs) for split, subs in self._sujeitos.items()}

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _dividir_sujeitos(
        self,
        seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,  # mantido por simetria / documentação
    ) -> None:
        """Embaralha os sujeitos e os distribui nas partições."""
        pastas = sorted(p for p in self.root.iterdir() if p.is_dir())
        rng = random.Random(seed)
        rng.shuffle(pastas)

        n = len(pastas)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # teste recebe o restante para que nenhum sujeito seja perdido por arredondamento

        self._sujeitos: dict[Split, list[Path]] = {
            "train": pastas[:n_train],
            "val": pastas[n_train : n_train + n_val],
            "test": pastas[n_train + n_val :],
        }

    def _coletar_luma(
        self, sujeitos: list[Path]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Carrega o canal Y de todas as imagens dos sujeitos fornecidos."""
        X: list[np.ndarray] = []
        y: list[str] = []
        for sujeito in sujeitos:
            for caminho in sorted(sujeito.glob("*.jpeg")):
                rotulo = _STEM_TO_LABEL.get(caminho.stem)
                if rotulo is None:
                    continue  # ignora arquivos com nome desconhecido
                X.append(self._carregar_luma(caminho))
                y.append(rotulo)
        return np.array(X, dtype=np.float32), np.array(y)

    def _carregar_luma(self, path: Path) -> np.ndarray:
        """Carrega uma imagem, extrai o canal Y (YCbCr), redimensiona e achata."""
        # Converte para YCbCr e descarta os canais de crominância (Cb, Cr)
        img = Image.open(path).convert("YCbCr")
        luma, *_ = img.split()
        luma = luma.resize(self.image_size, Image.LANCZOS)
        return np.array(luma, dtype=np.float32).ravel()

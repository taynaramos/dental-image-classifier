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
_SPLITS: tuple[Split, ...] = ("train", "val", "test")


def _layout_ja_dividido(root: Path) -> bool:
    """Verifica se *root* já está no layout ``ImageFolder`` (train/val/test com classes populadas)."""
    for split in _SPLITS:
        split_dir = root / split
        if not split_dir.is_dir() or not any(split_dir.glob("*/*")):
            return False
    return True


class DentalDataset:
    """Divisão treino/val/teste no nível do sujeito com pré-processamento de luminância.

    Aceita dois layouts de entrada:

    * **Bruto** — um diretório com uma sub-pasta por sujeito (uma imagem por
      vista dentro de cada uma). A divisão é feita **por sujeito**: todas as
      imagens de um mesmo sujeito ficam no mesmo conjunto, evitando
      vazamento de dados entre treino e avaliação.
    * **Já dividido** — um diretório já organizado em ``train``/``val``/
      ``test``, cada um com uma sub-pasta por classe (layout ``ImageFolder``,
      o mesmo produzido por ``resolve_imagefolder_root`` nos módulos
      ``pytorch_kfold``/``pytorch_resnet18_transfer``). Nesse caso o layout é
      usado diretamente, sem re-dividir nada — útil para reaproveitar
      exatamente a mesma partição usada pelos outros modelos.

    Parâmetros
    ----------
    root:
        Diretório raiz do dataset bruto (uma sub-pasta por sujeito) ou de um
        dataset já organizado em ``train``/``val``/``test``.
    image_size:
        (largura, altura) para redimensionar cada imagem antes de achatar.
    train_ratio, val_ratio, test_ratio:
        Proporções que devem somar 1,0. Ignoradas quando *root* já está no
        layout dividido.
    seed:
        Semente aleatória para embaralhamento reprodutível. Ignorada quando
        *root* já está no layout dividido.
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
        self.root = Path(root)
        self.image_size = image_size

        if _layout_ja_dividido(self.root):
            print(f"[dataset] usando layout já dividido em {self.root}")
            self.layout: Literal["subjects", "prepared"] = "prepared"
            self._sujeitos = None
            return

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio deve ser igual a 1,0"
            )

        self.layout = "subjects"
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
        if self.layout == "prepared":
            return self._coletar_luma_pasta(self.root / split)
        sujeitos = self._sujeitos[split]
        return self._coletar_luma(sujeitos)

    def subject_counts(self) -> dict[str, int]:
        """Retorna a quantidade de sujeitos (ou de imagens, no layout já dividido) em cada partição."""
        if self.layout == "prepared":
            return {split: sum(1 for _ in (self.root / split).glob("*/*")) for split in _SPLITS}
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
                X.append(carregar_luma(caminho, self.image_size))
                y.append(rotulo)
        return np.array(X, dtype=np.float32), np.array(y)

    def _coletar_luma_pasta(self, split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """Carrega o canal Y de todas as imagens de um split no layout ``ImageFolder``.

        Diferente de :meth:`_coletar_luma`, o rótulo vem do nome da
        sub-pasta (a classe), não do mapeamento por nome de arquivo — é
        assim que ``resolve_imagefolder_root`` organiza o layout já dividido.
        """
        X: list[np.ndarray] = []
        y: list[str] = []
        for classe_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for caminho in sorted(classe_dir.glob("*.jpeg")):
                X.append(carregar_luma(caminho, self.image_size))
                y.append(classe_dir.name)
        return np.array(X, dtype=np.float32), np.array(y)


def carregar_luma(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Carrega uma imagem, extrai o canal Y (YCbCr), redimensiona e achata.

    Função de pré-processamento compartilhada entre o carregamento do
    dataset de treino (:class:`DentalDataset`) e a inferência
    (``pca-predict``), para que as duas execuções apliquem exatamente o
    mesmo tratamento à imagem.
    """
    # Converte para YCbCr e descarta os canais de crominância (Cb, Cr)
    img = Image.open(path).convert("YCbCr")
    luma, *_ = img.split()
    luma = luma.resize(image_size, Image.LANCZOS)
    return np.array(luma, dtype=np.float32).ravel()

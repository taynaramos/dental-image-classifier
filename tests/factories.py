"""Fábricas de dados sintéticos compartilhadas entre os testes deste pacote.

Qualquer teste novo adicionado a ``tests/`` deve reaproveitar estas funções
em vez de duplicar a criação de imagens/datasets fake.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Mesmo mapeamento nome-de-arquivo -> vista usado pelos três módulos de
# treino (pca_svc, pytorch_kfold, pytorch_resnet18_transfer).
VIEW_TO_STEM: dict[str, str] = {
    "frontal": "intraoral-frontal",
    "inferior": "intraoral-inferior",
    "superior": "intraoral-superior",
    "lateral_direita": "intraoral-lateral-direita",
    "lateral_esquerda": "intraoral-lateral-esquerda",
}
CLASSES: frozenset[str] = frozenset(VIEW_TO_STEM)

# Uma cor sólida e distinta por vista, só para as imagens sintéticas não
# ficarem todas idênticas entre classes.
_VIEW_COLORS: dict[str, tuple[int, int, int]] = {
    "frontal": (200, 40, 40),
    "inferior": (40, 200, 40),
    "superior": (40, 40, 200),
    "lateral_direita": (200, 200, 40),
    "lateral_esquerda": (200, 40, 200),
}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return True


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


# Vários testes precisam de PyTorch/torchvision (módulos pytorch_kfold e
# pytorch_resnet18_transfer) ou scikit-learn (módulo pca_svc). Nem todo
# ambiente tem as três instaladas ao mesmo tempo — cada módulo pode ser
# instalado isoladamente (ver README.md na raiz) — então os testes que
# dependem delas usam estas flags para se auto-pular em vez de quebrar a
# coleta de todo o pacote.
TORCH_AVAILABLE = _torch_available()
SKLEARN_AVAILABLE = _sklearn_available()


def make_rgb_image(
    size: tuple[int, int] = (32, 32), color: tuple[int, int, int] = (128, 64, 32)
) -> Image.Image:
    """Cria uma imagem RGB sólida (cor única) — suficiente para testar forma/canais/normalização."""
    return Image.new("RGB", size, color=color)


def make_grayscale_image(size: tuple[int, int] = (32, 32), value: int = 128) -> Image.Image:
    """Cria uma imagem em escala de cinza (modo ``L``) — usada para testar tolerância a entradas não-RGB."""
    return Image.new("L", size, color=value)


def save_jpeg(image: Image.Image, path: Path) -> Path:
    """Salva *image* como JPEG em *path*, criando os diretórios pais se necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG")
    return path


def build_subject(
    subject_dir: Path,
    *,
    image_size: tuple[int, int] = (32, 32),
    views: dict[str, str] = VIEW_TO_STEM,
) -> None:
    """Cria as imagens de vista de um único sujeito em *subject_dir* (uma por entrada de *views*)."""
    for vista, stem in views.items():
        cor = _VIEW_COLORS.get(vista, (128, 128, 128))
        save_jpeg(make_rgb_image(image_size, cor), subject_dir / f"{stem}.jpeg")


def build_raw_subject_dataset(
    root: Path,
    n_subjects: int,
    *,
    image_size: tuple[int, int] = (32, 32),
) -> list[Path]:
    """Cria um dataset bruto (uma pasta por sujeito, 5 imagens por vista cada) em *root*.

    Retorna a lista das pastas de sujeito criadas.
    """
    sujeitos = []
    for i in range(n_subjects):
        subject_dir = root / f"SUBJ{i:03d}"
        build_subject(subject_dir, image_size=image_size)
        sujeitos.append(subject_dir)
    return sujeitos


def build_imagefolder_dataset(
    root: Path,
    *,
    per_class: dict[str, int] | int = 3,
    image_size: tuple[int, int] = (32, 32),
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Cria um dataset já dividido no layout ``ImageFolder`` (``<split>/<classe>/img.jpeg``)."""
    contagem = per_class if isinstance(per_class, dict) else {classe: per_class for classe in CLASSES}
    for split in splits:
        for classe in CLASSES:
            cor = _VIEW_COLORS.get(classe, (128, 128, 128))
            for i in range(contagem[classe]):
                save_jpeg(
                    make_rgb_image(image_size, cor),
                    root / split / classe / f"img_{i:02d}.jpeg",
                )

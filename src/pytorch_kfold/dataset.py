"""Preparo do dataset e DataLoaders para o pipeline CNN (PyTorch)."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

# Mapeamento do nome do arquivo (stem) para o rótulo da classe.
# O dataset bruto tem uma pasta por sujeito, com uma imagem por vista dentro de cada pasta.
_STEM_TO_LABEL: dict[str, str] = {
    "intraoral-frontal": "frontal",
    "intraoral-inferior": "inferior",
    "intraoral-superior": "superior",
    "intraoral-lateral-direita": "lateral_direita",
    "intraoral-lateral-esquerda": "lateral_esquerda",
}

_SPLITS = ("train", "val", "test")


def resolve_imagefolder_root(
    dataset_path: Path | str,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """Garante um diretório com layout ``<split>/<classe>/imagem``, para o ``ImageFolder``.

    Se *dataset_path* já contiver as sub-pastas ``train``, ``val`` e ``test``,
    é usado diretamente (dataset já preparado). Caso contrário, é tratado como
    um dataset organizado por sujeito — uma pasta por paciente, com uma imagem
    por vista — e materializado em ``<dataset_path>_imagefolder``, dividindo os
    **sujeitos** (não as imagens) entre treino/validação/teste para não vazar
    dados do mesmo paciente entre conjuntos. Execuções seguintes reaproveitam
    o diretório já preparado.

    Parâmetros
    ----------
    dataset_path:
        Raiz do dataset bruto (uma sub-pasta por sujeito) ou de um dataset já
        organizado em ``train``/``val``/``test``.
    train_ratio, val_ratio, test_ratio:
        Proporções de sujeitos por partição. Devem somar 1,0.
    seed:
        Semente aleatória para o embaralhamento dos sujeitos.

    Retorna
    -------
    Path
        Raiz do dataset pronta para ``torchvision.datasets.ImageFolder``.
    """
    dataset_path = Path(dataset_path)
    if all((dataset_path / split).is_dir() for split in _SPLITS) and _layout_completo(dataset_path):
        print(f"[dataset] usando layout ImageFolder existente em {dataset_path}")
        return dataset_path

    prepared_root = dataset_path.parent / f"{dataset_path.name}_imagefolder"
    if all((prepared_root / split).is_dir() for split in _SPLITS) and _layout_completo(prepared_root):
        print(f"[dataset] reaproveitando layout preparado em {prepared_root}")
        return prepared_root

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio deve ser igual a 1,0")

    print(f"[dataset] preparando layout ImageFolder a partir de {dataset_path} …")
    splits = _dividir_sujeitos(dataset_path, train_ratio, val_ratio, test_ratio, seed)

    # Cria de antemão todas as pastas de classe em todos os splits — cada ImageFolder
    # descobre suas classes de forma independente, então um split sem alguma classe
    # (ex.: paciente sem uma das vistas) desalinharia os índices de rótulo entre eles.
    for split in splits:
        for rotulo in set(_STEM_TO_LABEL.values()):
            (prepared_root / split / rotulo).mkdir(parents=True, exist_ok=True)

    for split, sujeitos in splits.items():
        for sujeito in sujeitos:
            for img_path in sorted(sujeito.glob("*.jpeg")):
                rotulo = _STEM_TO_LABEL.get(img_path.stem)
                if rotulo is None:
                    continue
                destino = prepared_root / split / rotulo
                destino.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, destino / f"{sujeito.name}_{img_path.name}")

    print(
        f"[dataset] layout pronto em {prepared_root} — sujeitos: "
        f"treino {len(splits['train'])}, val {len(splits['val'])}, teste {len(splits['test'])}"
    )
    return prepared_root


def _layout_completo(root: Path) -> bool:
    """Verifica se todo split tem ao menos uma imagem em ao menos uma classe.

    Um layout ``train``/``val``/``test`` pode existir apenas como diretórios
    vazios (ex.: preparo anterior interrompido no meio da cópia) — nesse caso
    não deve ser reaproveitado como se estivesse pronto.
    """
    for split in _SPLITS:
        split_dir = root / split
        if not any(split_dir.glob("*/*")):
            return False
    return True


def build_transform(image_size: int, grayscale: bool) -> T.Compose:
    """Monta o pipeline de pré-processamento: Resize → Grayscale (opcional) → ToTensor → Normalize.

    Parâmetros
    ----------
    image_size:
        Lado (px) para redimensionar a imagem (imagem final é quadrada).
    grayscale:
        Se ``True``, converte para 1 canal (luminância); caso contrário mantém RGB.
    """
    passos = [T.Resize((image_size, image_size))]
    if grayscale:
        passos.append(T.Grayscale(num_output_channels=1))
        passos += [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    else:
        passos += [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    return T.Compose(passos)


def build_dataloaders(
    imagefolder_root: Path | str,
    *,
    image_size: int,
    batch_size: int,
    grayscale: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Cria os DataLoaders de treino/validação/teste a partir de um diretório ImageFolder.

    Parâmetros
    ----------
    imagefolder_root:
        Raiz com as sub-pastas ``train``, ``val`` e ``test`` (ver :func:`resolve_imagefolder_root`).
    image_size:
        Lado (px) usado no pré-processamento.
    batch_size:
        Tamanho do lote para os três DataLoaders.
    grayscale:
        Repassado a :func:`build_transform`.
    num_workers:
        Processos auxiliares de carregamento (``0`` desativa o multiprocessing).

    Retorna
    -------
    tuple
        ``(train_loader, val_loader, test_loader, classes)``, onde *classes* é a
        lista de rótulos na ordem usada pelos índices do modelo.
    """
    imagefolder_root = Path(imagefolder_root)
    transform = build_transform(image_size, grayscale)

    train_set = ImageFolder(imagefolder_root / "train", transform=transform)
    val_set = ImageFolder(imagefolder_root / "val", transform=transform)
    test_set = ImageFolder(imagefolder_root / "test", transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_set.classes


def _dividir_sujeitos(
    dataset_path: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,  # mantido por simetria / documentação
    seed: int,
) -> dict[str, list[Path]]:
    """Embaralha os sujeitos de *dataset_path* e os distribui nas partições."""
    sujeitos = sorted(p for p in dataset_path.iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(sujeitos)

    n = len(sujeitos)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # teste recebe o restante para nenhum sujeito ser perdido por arredondamento

    return {
        "train": sujeitos[:n_train],
        "val": sujeitos[n_train : n_train + n_val],
        "test": sujeitos[n_train + n_val :],
    }

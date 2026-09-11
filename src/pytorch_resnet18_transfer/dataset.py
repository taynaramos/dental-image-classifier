"""Preparo do dataset e DataLoaders para o pipeline de transfer learning (ResNet-18)."""

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

# Estatísticas de normalização da ImageNet — necessárias porque o backbone
# ResNet-18 foi pré-treinado com imagens normalizadas dessa forma.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_imagefolder_root(
    dataset_path: Path | str,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """Garante um diretório com layout ``<split>/<classe>/imagem``, para o ``ImageFolder``.

    Mesma lógica usada pelo módulo ``pytorch_kfold``: se *dataset_path* já
    contiver as sub-pastas ``train``/``val``/``test``, é usado diretamente.
    Caso contrário, é tratado como um dataset organizado por sujeito — uma
    pasta por paciente, com uma imagem por vista — e materializado em
    ``<dataset_path>_imagefolder``, dividindo os **sujeitos** (não as
    imagens) entre treino/validação/teste para não vazar dados do mesmo
    paciente entre conjuntos.

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


def build_train_transform(image_size: int = 224) -> T.Compose:
    """Monta o pipeline de pré-processamento de treino (com aumento de dados).

    ``Resize`` para um lado maior, ``RandomCrop`` para o tamanho final e
    ``ColorJitter`` como aumento de dados, seguidos de normalização com as
    estatísticas da ImageNet (necessárias para o backbone pré-treinado).
    """
    return T.Compose(
        [
            T.Resize(int(image_size * 256 / 224)),
            T.RandomCrop(image_size),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> T.Compose:
    """Monta o pipeline de pré-processamento de validação/teste/inferência.

    Igual ao de treino, mas com ``CenterCrop`` (determinístico) no lugar de
    ``RandomCrop`` e sem aumento de dados.
    """
    return T.Compose(
        [
            T.Resize(int(image_size * 256 / 224)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    imagefolder_root: Path | str,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Cria os DataLoaders de treino/validação/teste a partir de um diretório ImageFolder.

    Parâmetros
    ----------
    imagefolder_root:
        Raiz com as sub-pastas ``train``, ``val`` e ``test`` (ver :func:`resolve_imagefolder_root`).
    image_size:
        Lado (px) da imagem final, após o crop (padrão: 224, o esperado pela ResNet-18).
    batch_size:
        Tamanho do lote para os três DataLoaders.
    num_workers:
        Processos auxiliares de carregamento (``0`` desativa o multiprocessing).

    Retorna
    -------
    tuple
        ``(train_loader, val_loader, test_loader, classes)``, onde *classes* é a
        lista de rótulos na ordem usada pelos índices do modelo.
    """
    imagefolder_root = Path(imagefolder_root)

    train_set = ImageFolder(imagefolder_root / "train", transform=build_train_transform(image_size))
    val_set = ImageFolder(imagefolder_root / "val", transform=build_eval_transform(image_size))
    test_set = ImageFolder(imagefolder_root / "test", transform=build_eval_transform(image_size))

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

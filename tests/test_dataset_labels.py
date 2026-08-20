"""Todas as 5 vistas devem aparecer nas partições de treino, validação e teste.

Esta é a checagem que teria detectado, antes do treino, o problema real que
motivou este módulo de testes: um preparo de dataset interrompido no meio
(ex.: cópia para um Drive montado que caiu no meio do caminho) deixa
partições com diretórios de classe vazios, e o treino só descobre isso na
hora do erro do ``ImageFolder``/``PCA`` — bem depois de já ter gastado tempo
de GPU.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pca_svc.dataset import DentalDataset

from .factories import CLASSES, TORCH_AVAILABLE, build_imagefolder_dataset, build_raw_subject_dataset

if TORCH_AVAILABLE:
    from src.pytorch_kfold.dataset import resolve_imagefolder_root as kfold_resolve
    from src.pytorch_resnet18_transfer.dataset import resolve_imagefolder_root as resnet18_resolve

# 70/15/15 de 20 sujeitos -> 14/3/3: nenhuma partição fica vazia, e como todo
# sujeito sintético tem as 5 vistas, qualquer partição não-vazia tem as 5 classes.
_N_SUBJECTS = 20


def _classes_com_conteudo(split_dir: Path) -> set[str]:
    return {p.name for p in split_dir.iterdir() if p.is_dir() and any(p.glob("*"))}


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class KfoldLabelCoverageTests(unittest.TestCase):
    def test_all_classes_present_in_every_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "dataset"
            build_raw_subject_dataset(raw_root, _N_SUBJECTS)

            prepared_root = kfold_resolve(raw_root, seed=0)

            for split in ("train", "val", "test"):
                self.assertEqual(_classes_com_conteudo(prepared_root / split), set(CLASSES), msg=f"split={split}")


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class Resnet18LabelCoverageTests(unittest.TestCase):
    def test_all_classes_present_in_every_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "dataset"
            build_raw_subject_dataset(raw_root, _N_SUBJECTS)

            prepared_root = resnet18_resolve(raw_root, seed=0)

            for split in ("train", "val", "test"):
                self.assertEqual(_classes_com_conteudo(prepared_root / split), set(CLASSES), msg=f"split={split}")


class PcaSvcLabelCoverageTests(unittest.TestCase):
    def test_all_classes_present_in_every_split_from_raw_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "dataset"
            build_raw_subject_dataset(raw_root, _N_SUBJECTS)

            dataset = DentalDataset(raw_root, seed=0)
            for split in ("train", "val", "test"):
                _, y = dataset.load_split(split)
                self.assertEqual(set(y.tolist()), set(CLASSES), msg=f"split={split}")

    def test_all_classes_present_in_every_split_from_prepared_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared_root = Path(tmp) / "dataset_imagefolder"
            build_imagefolder_dataset(prepared_root, per_class=3)

            dataset = DentalDataset(prepared_root)
            self.assertEqual(dataset.layout, "prepared")
            for split in ("train", "val", "test"):
                _, y = dataset.load_split(split)
                self.assertEqual(set(y.tolist()), set(CLASSES), msg=f"split={split}")


if __name__ == "__main__":
    unittest.main()

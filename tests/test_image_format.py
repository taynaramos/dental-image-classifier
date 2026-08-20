"""Validação do formato de entrada: imagem colorida RGB e "adequação" para treino.

Cobre:
* o pré-processamento compartilhado entre treino e inferência trata a
  imagem de entrada como colorida (RGB) antes de qualquer outra conversão;
* o carregamento do dataset só considera "própria para treino" uma imagem
  ``.jpeg`` cujo nome (stem) corresponda a uma das 5 vistas conhecidas —
  qualquer outro arquivo dentro da pasta do sujeito é ignorado silenciosamente,
  nunca quebra o carregamento.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pca_svc.dataset import DentalDataset, carregar_luma

from .factories import TORCH_AVAILABLE, make_grayscale_image, make_rgb_image, save_jpeg


class CarregarLumaColorFormatTests(unittest.TestCase):
    """``carregar_luma`` (usado no treino e na inferência do PCA) deve aceitar imagem colorida RGB."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_rgb_image_produces_flat_luma_vector(self) -> None:
        caminho = save_jpeg(make_rgb_image((16, 16), (30, 200, 90)), self.root / "img.jpeg")
        vetor = carregar_luma(caminho, (16, 16))
        self.assertEqual(vetor.shape, (16 * 16,))
        self.assertEqual(vetor.dtype.name, "float32")

    def test_non_rgb_input_is_still_accepted_via_color_conversion(self) -> None:
        # A imagem bruta pode não vir em RGB (ex.: escala de cinza) — o
        # carregamento converte para YCbCr internamente, então não deve falhar.
        caminho = save_jpeg(make_grayscale_image((16, 16), 100), self.root / "gray.jpeg")
        vetor = carregar_luma(caminho, (16, 16))
        self.assertEqual(vetor.shape, (16 * 16,))


class PcaDatasetInputFiltersTests(unittest.TestCase):
    """O dataset PCA só deve considerar imagens com nome de vista reconhecido e extensão ``.jpeg``."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_unknown_stem_and_wrong_extension_are_ignored(self) -> None:
        subject = self.root / "SUBJ000"
        save_jpeg(make_rgb_image((8, 8), (10, 20, 30)), subject / "intraoral-frontal.jpeg")
        # Nome desconhecido — não está em _STEM_TO_LABEL, deve ser ignorado.
        save_jpeg(make_rgb_image((8, 8), (40, 50, 60)), subject / "anotacao-do-dentista.jpeg")
        # Extensão errada — o carregamento só busca "*.jpeg", deve ser ignorada.
        make_rgb_image((8, 8), (70, 80, 90)).save(subject / "intraoral-inferior.png", format="PNG")

        # Força todos os sujeitos para o split "test", já que com um único
        # sujeito a divisão 70/15/15 padrão o descartaria de treino/val.
        dataset = DentalDataset(self.root, image_size=(8, 8), train_ratio=0.0, val_ratio=0.0, test_ratio=1.0)
        X, y = dataset.load_split("test")

        self.assertEqual(len(y), 1)
        self.assertEqual(y[0], "frontal")
        self.assertEqual(X.shape, (1, 8 * 8))


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class PytorchDatasetInputFiltersTests(unittest.TestCase):
    """``resolve_imagefolder_root`` (kfold e resnet18) aplica o mesmo filtro de "propriedade para treino"."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _build_subject_with_noise(self) -> Path:
        raw_root = self.root / "dataset"
        subject = raw_root / "SUBJ000"
        save_jpeg(make_rgb_image((8, 8), (10, 20, 30)), subject / "intraoral-frontal.jpeg")
        save_jpeg(make_rgb_image((8, 8), (40, 50, 60)), subject / "anotacao-do-dentista.jpeg")
        make_rgb_image((8, 8), (70, 80, 90)).save(subject / "intraoral-inferior.png", format="PNG")
        return raw_root

    def test_kfold_only_copies_recognized_jpeg_views(self) -> None:
        from src.pytorch_kfold.dataset import resolve_imagefolder_root

        raw_root = self._build_subject_with_noise()
        prepared_root = resolve_imagefolder_root(raw_root, train_ratio=0.0, val_ratio=0.0, test_ratio=1.0, seed=0)

        self.assertEqual(len(list((prepared_root / "test" / "frontal").glob("*"))), 1)
        self.assertEqual(len(list((prepared_root / "test" / "inferior").glob("*"))), 0)

    def test_resnet18_only_copies_recognized_jpeg_views(self) -> None:
        from src.pytorch_resnet18_transfer.dataset import resolve_imagefolder_root

        raw_root = self._build_subject_with_noise()
        prepared_root = resolve_imagefolder_root(raw_root, train_ratio=0.0, val_ratio=0.0, test_ratio=1.0, seed=0)

        self.assertEqual(len(list((prepared_root / "test" / "frontal").glob("*"))), 1)
        self.assertEqual(len(list((prepared_root / "test" / "inferior").glob("*"))), 0)


if __name__ == "__main__":
    unittest.main()

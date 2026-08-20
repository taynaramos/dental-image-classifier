"""Validação do pré-processamento usado para a inferência dos módulos PyTorch.

Verifica forma, tipo e a fórmula de normalização exata dos pipelines de
``build_train_transform``/``build_eval_transform`` (ResNet-18) e
``build_transform`` (CNN k-fold) — as mesmas funções que ``predict()`` usa em
cada módulo (ver ``test_inference_color_space.py`` para a checagem cruzada
com ``predict()``).
"""

from __future__ import annotations

import unittest

from .factories import TORCH_AVAILABLE, make_rgb_image

if TORCH_AVAILABLE:
    import torch

    from src.pytorch_kfold.dataset import build_transform
    from src.pytorch_resnet18_transfer.dataset import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_eval_transform,
        build_train_transform,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class ResNet18PreprocessingTests(unittest.TestCase):
    IMAGE_SIZE = 64

    def setUp(self) -> None:
        # Cor sólida: resize/crop não alteram o valor do pixel, o que permite
        # verificar a fórmula exata de normalização abaixo.
        self.image = make_rgb_image((300, 300), (128, 64, 32))

    def test_eval_transform_shape_and_dtype(self) -> None:
        tensor = build_eval_transform(self.IMAGE_SIZE)(self.image)
        self.assertEqual(tuple(tensor.shape), (3, self.IMAGE_SIZE, self.IMAGE_SIZE))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_eval_transform_applies_imagenet_normalization(self) -> None:
        tensor = build_eval_transform(self.IMAGE_SIZE)(self.image)
        pixel_0_255 = torch.tensor([128, 64, 32], dtype=torch.float32)
        esperado = (pixel_0_255 / 255.0 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
        obtido = tensor[:, self.IMAGE_SIZE // 2, self.IMAGE_SIZE // 2]
        torch.testing.assert_close(obtido, esperado, atol=1e-3, rtol=0)

    def test_train_transform_keeps_three_rgb_channels(self) -> None:
        # A fase de treino tem RandomCrop/ColorJitter (não-determinístico),
        # mas o formato (canais/tamanho) deve continuar fixo.
        tensor = build_train_transform(self.IMAGE_SIZE)(self.image)
        self.assertEqual(tuple(tensor.shape), (3, self.IMAGE_SIZE, self.IMAGE_SIZE))


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class KfoldPreprocessingTests(unittest.TestCase):
    IMAGE_SIZE = 40

    def setUp(self) -> None:
        self.image = make_rgb_image((100, 100), (90, 180, 30))

    def test_grayscale_transform_reduces_to_single_channel(self) -> None:
        tensor = build_transform(self.IMAGE_SIZE, grayscale=True)(self.image)
        self.assertEqual(tuple(tensor.shape), (1, self.IMAGE_SIZE, self.IMAGE_SIZE))

    def test_rgb_transform_keeps_three_channels(self) -> None:
        tensor = build_transform(self.IMAGE_SIZE, grayscale=False)(self.image)
        self.assertEqual(tuple(tensor.shape), (3, self.IMAGE_SIZE, self.IMAGE_SIZE))

    def test_grayscale_transform_applies_luma_and_normalization_formula(self) -> None:
        r, g, b = 90, 180, 30
        # Mesma fórmula da conversão "L" do PIL (ITU-R BT.601), delegada
        # internamente por torchvision.transforms.Grayscale para imagens PIL.
        luma_0_255 = round(r * 299 / 1000 + g * 587 / 1000 + b * 114 / 1000)
        esperado = (luma_0_255 / 255.0 - 0.5) / 0.5

        tensor = build_transform(self.IMAGE_SIZE, grayscale=True)(self.image)
        obtido = tensor[0, self.IMAGE_SIZE // 2, self.IMAGE_SIZE // 2].item()
        self.assertAlmostEqual(obtido, esperado, delta=0.02)


if __name__ == "__main__":
    unittest.main()

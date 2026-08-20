"""Consistência entre o espaço de cor usado no treino e na inferência.

Garante que ``predict()`` de cada módulo PyTorch aplica exatamente o mesmo
pré-processamento (canais de cor, normalização) usado para montar o
``DataLoader`` de avaliação no treino — se divergissem, o modelo receberia
uma distribuição de entrada diferente da que foi treinada, mesmo sem erro
explícito (silenciosamente pior acurácia).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .factories import TORCH_AVAILABLE, make_rgb_image, save_jpeg

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from src.pytorch_kfold.dataset import build_transform as kfold_build_transform
    from src.pytorch_kfold.model import DentalCNN, ModelConfig
    from src.pytorch_kfold.predict import predict as kfold_predict
    from src.pytorch_resnet18_transfer.dataset import build_eval_transform
    from src.pytorch_resnet18_transfer.model import DentalResNetTransfer
    from src.pytorch_resnet18_transfer.predict import predict as resnet18_predict

    # Referência capturada ANTES de qualquer patch — usada dentro do
    # side_effect abaixo. Resolver "torchvision.models.resnet18" de novo ali
    # dentro pegaria a própria função mockada (mesmo objeto de módulo) e
    # causaria recursão infinita.
    from torchvision.models import resnet18 as _real_resnet18


def _resnet18_sem_download(*args, **kwargs):
    """Substitui o download dos pesos pré-treinados da ImageNet por inicialização
    aleatória — evita depender de rede/cache do torch hub durante os testes.
    A propriedade testada (mesmo espaço de cor entre treino e inferência) não
    depende dos pesos serem os pré-treinados."""
    kwargs["weights"] = None
    return _real_resnet18(*args, **kwargs)


_CLASSES = ["frontal", "inferior", "superior", "lateral_direita", "lateral_esquerda"]


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class Resnet18InferenceColorSpaceTests(unittest.TestCase):
    IMAGE_SIZE = 64

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.image_path = save_jpeg(
            make_rgb_image((300, 300), (128, 64, 32)), Path(self._tmp.name) / "foto.jpeg"
        )

    def test_predict_uses_same_transform_as_eval_dataloader(self) -> None:
        from unittest import mock

        with mock.patch(
            "src.pytorch_resnet18_transfer.feature_extractor.models.resnet18",
            side_effect=_resnet18_sem_download,
        ):
            model = DentalResNetTransfer(num_classes=len(_CLASSES))
            device = torch.device("cpu")

            pred = resnet18_predict(self.image_path, model, _CLASSES, self.IMAGE_SIZE, device)

            # Reconstrói manualmente o tensor com o transform de avaliação do
            # treino (o mesmo usado para val/test) e roda o modelo à parte —
            # se predict() usasse um espaço de cor diferente, os logits não bateriam.
            transform = build_eval_transform(self.IMAGE_SIZE)
            tensor = transform(Image.open(self.image_path).convert("RGB")).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                esperado = F.softmax(model(tensor), dim=1).squeeze(0)

            obtido = torch.tensor([pred.probabilities[c] for c in _CLASSES])
            torch.testing.assert_close(obtido, esperado, atol=1e-6, rtol=0)


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class KfoldInferenceColorSpaceTests(unittest.TestCase):
    IMAGE_SIZE = 32

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.image_path = save_jpeg(
            make_rgb_image((80, 80), (90, 180, 30)), Path(self._tmp.name) / "foto.jpeg"
        )

    def test_predict_respects_grayscale_config_from_train(self) -> None:
        # Não deve levantar erro de shape/canal — se predict() ignorasse
        # config.grayscale e usasse RGB, a primeira camada convolucional
        # (esperando 1 canal) receberia um tensor de 3 canais e quebraria.
        config = ModelConfig(image_size=self.IMAGE_SIZE, grayscale=True)
        model = DentalCNN(num_classes=len(_CLASSES), in_channels=config.in_channels)
        device = torch.device("cpu")

        pred = kfold_predict(self.image_path, model, _CLASSES, config, device)
        self.assertEqual(len(pred.probabilities), len(_CLASSES))

    def test_predict_tensor_matches_manual_transform(self) -> None:
        config = ModelConfig(image_size=self.IMAGE_SIZE, grayscale=True)
        model = DentalCNN(num_classes=len(_CLASSES), in_channels=config.in_channels)
        device = torch.device("cpu")

        pred = kfold_predict(self.image_path, model, _CLASSES, config, device)

        transform = kfold_build_transform(config.image_size, config.grayscale)
        tensor = transform(Image.open(self.image_path).convert("RGB")).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            esperado = F.softmax(model(tensor), dim=1).squeeze(0)

        obtido = torch.tensor([pred.probabilities[c] for c in _CLASSES])
        torch.testing.assert_close(obtido, esperado, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()

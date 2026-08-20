"""O vetor de probabilidades retornado por ``predict()``/``predict_proba()`` deve somar 1."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pca_svc.dataset import carregar_luma

from .factories import SKLEARN_AVAILABLE, TORCH_AVAILABLE, make_rgb_image, save_jpeg

if SKLEARN_AVAILABLE:
    from src.pca_svc.model import DentalClassifier

if TORCH_AVAILABLE:
    import torch

    from src.pytorch_kfold.model import DentalCNN, ModelConfig
    from src.pytorch_kfold.predict import predict as kfold_predict
    from src.pytorch_resnet18_transfer.model import DentalResNetTransfer
    from src.pytorch_resnet18_transfer.predict import predict as resnet18_predict

    # Referência capturada ANTES de qualquer patch — ver test_inference_color_space.py
    # para a justificativa completa (evita recursão infinita no side_effect abaixo).
    from torchvision.models import resnet18 as _real_resnet18

_CLASSES = ["frontal", "inferior", "superior", "lateral_direita", "lateral_esquerda"]


def _resnet18_sem_download(*args, **kwargs):
    """Evita baixar os pesos pré-treinados da ImageNet durante os testes."""
    kwargs["weights"] = None
    return _real_resnet18(*args, **kwargs)


@unittest.skipUnless(SKLEARN_AVAILABLE, "requer scikit-learn (ver requirements-pca-svc.txt)")
class PcaSvcPredictProbaSumTests(unittest.TestCase):
    def test_predict_proba_sums_to_one(self) -> None:
        image_size = (8, 8)
        n_features = image_size[0] * image_size[1]
        rng = np.random.default_rng(0)

        # Conjunto de treino sintético: várias amostras por classe, com um
        # pouco de ruído — o SVC (probability=True) precisa de pelo menos 5
        # amostras/classe para ajustar a calibração de Platt internamente.
        X_treino, y_treino = [], []
        for i, classe in enumerate(_CLASSES):
            base = np.full(n_features, fill_value=float(i * 40), dtype=np.float32)
            for _ in range(8):
                X_treino.append(base + rng.normal(0, 2, n_features).astype(np.float32))
                y_treino.append(classe)

        classificador = DentalClassifier(n_components=4, C=1.0, seed=0)
        classificador.fit(np.array(X_treino, dtype=np.float32), np.array(y_treino))

        with tempfile.TemporaryDirectory() as tmp:
            caminho = save_jpeg(make_rgb_image(image_size, (80, 80, 80)), Path(tmp) / "foto.jpeg")
            vetor = carregar_luma(caminho, image_size)

        [predicao] = classificador.predict_proba(np.array([vetor]))
        self.assertAlmostEqual(sum(predicao.probabilities.values()), 1.0, places=5)


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class Resnet18PredictProbaSumTests(unittest.TestCase):
    def test_predict_sums_to_one(self) -> None:
        from unittest import mock

        with mock.patch(
            "src.pytorch_resnet18_transfer.feature_extractor.models.resnet18",
            side_effect=_resnet18_sem_download,
        ):
            model = DentalResNetTransfer(num_classes=len(_CLASSES))
            device = torch.device("cpu")
            with tempfile.TemporaryDirectory() as tmp:
                caminho = save_jpeg(make_rgb_image((64, 64), (128, 64, 32)), Path(tmp) / "foto.jpeg")
                pred = resnet18_predict(caminho, model, _CLASSES, 64, device)

        self.assertAlmostEqual(sum(pred.probabilities.values()), 1.0, places=5)


@unittest.skipUnless(TORCH_AVAILABLE, "requer torch/torchvision (ver requirements-pytorch-*.txt)")
class KfoldPredictProbaSumTests(unittest.TestCase):
    def test_predict_sums_to_one_grayscale(self) -> None:
        config = ModelConfig(image_size=32, grayscale=True)
        model = DentalCNN(num_classes=len(_CLASSES), in_channels=config.in_channels)
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            caminho = save_jpeg(make_rgb_image((32, 32), (90, 180, 30)), Path(tmp) / "foto.jpeg")
            pred = kfold_predict(caminho, model, _CLASSES, config, device)
        self.assertAlmostEqual(sum(pred.probabilities.values()), 1.0, places=5)

    def test_predict_sums_to_one_rgb(self) -> None:
        config = ModelConfig(image_size=32, grayscale=False)
        model = DentalCNN(num_classes=len(_CLASSES), in_channels=config.in_channels)
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            caminho = save_jpeg(make_rgb_image((32, 32), (10, 200, 240)), Path(tmp) / "foto.jpeg")
            pred = kfold_predict(caminho, model, _CLASSES, config, device)
        self.assertAlmostEqual(sum(pred.probabilities.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

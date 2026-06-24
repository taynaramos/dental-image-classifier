"""Classificador SVC que opera sobre features reduzidas pelo PCA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from .feature_extractor import FeatureExtractor


@dataclass
class Prediction:
    """Resultado da predição para uma única imagem.

    Atributos
    ----------
    label:
        Rótulo da classe predita (mais provável).
    probabilities:
        Dicionário mapeando cada classe à sua probabilidade (escala de Platt).
    """

    label: str
    probabilities: dict[str, float]


class DentalClassifier:
    """Classificador SVC construído sobre o :class:`FeatureExtractor`.

    O pipeline completo é: pixels de luminância → StandardScaler → PCA → SVC (RBF).

    Parâmetros
    ----------
    n_components:
        Repassado ao :class:`FeatureExtractor`.  ``None`` aciona a seleção
        automática pelo limiar de variância de 95 %.
    auto_variance_threshold:
        Usado quando *n_components* é ``None``.
    C:
        Parâmetro de regularização do SVC.
    gamma:
        Coeficiente do kernel RBF (``"scale"`` ou ``"auto"`` ou float).
    seed:
        Semente compartilhada entre PCA e SVC.
    """

    def __init__(
        self,
        *,
        n_components: int | None = None,
        auto_variance_threshold: float = 0.95,
        C: float = 10.0,
        gamma: str = "scale",
        seed: int = 42,
    ) -> None:
        self.extractor = FeatureExtractor(
            n_components=n_components,
            auto_variance_threshold=auto_variance_threshold,
            seed=seed,
        )
        # probability=True habilita predict_proba via escala de Platt
        self._svc = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            random_state=seed,
            class_weight="balanced",
            probability=True,
        )
        self.classes_: list[str] | None = None

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DentalClassifier":
        """Ajusta o pipeline completo nos dados de treino.

        Parâmetros
        ----------
        X:
            Features brutas de luminância com forma ``(n_amostras, n_pixels)``.
        y:
            Rótulos de classe com forma ``(n_amostras,)``.

        Retorna
        -------
        self
        """
        # Ajusta scaler + PCA e projeta os dados de treino
        X_pca = self.extractor.fit_transform(X)
        self._svc.fit(X_pca, y)
        self.classes_ = list(self._svc.classes_)
        return self

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna o rótulo predito para cada linha de *X*.

        Parâmetros
        ----------
        X:
            Features brutas com forma ``(n_amostras, n_pixels)``.
        """
        return self._svc.predict(self.extractor.transform(X))

    def predict_proba(self, X: np.ndarray) -> list[Prediction]:
        """Retorna uma :class:`Prediction` por linha de *X*.

        Cada :class:`Prediction` contém o rótulo mais provável e a
        probabilidade associada a cada classe.

        Parâmetros
        ----------
        X:
            Features brutas com forma ``(n_amostras, n_pixels)``.

        Retorna
        -------
        list[Prediction]
            Um elemento por amostra.
        """
        self._verificar_ajuste()
        X_pca = self.extractor.transform(X)
        # predict_proba retorna probabilidades por classe (escala de Platt)
        matriz_proba = self._svc.predict_proba(X_pca)
        rotulos = self._svc.predict(X_pca)

        return [
            Prediction(
                label=str(rotulo),
                probabilities={
                    cls: float(prob)
                    for cls, prob in zip(self.classes_, linha)  # type: ignore[arg-type]
                },
            )
            for rotulo, linha in zip(rotulos, matriz_proba)
        ]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Retorna a acurácia do classificador em *(X, y)*."""
        return self._svc.score(self.extractor.transform(X), y)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serializa o classificador (extrator + SVC) em *path* via :mod:`joblib`."""
        import joblib

        joblib.dump(
            {
                "extractor": self.extractor,
                "svc": self._svc,
                "classes": self.classes_,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "DentalClassifier":
        """Restaura um classificador serializado a partir de *path*."""
        import joblib

        dados = joblib.load(path)
        obj = cls.__new__(cls)
        obj.extractor = dados["extractor"]
        obj._svc = dados["svc"]
        obj.classes_ = dados["classes"]
        return obj

    # ------------------------------------------------------------------
    # Auxiliares internos
    # ------------------------------------------------------------------

    def _verificar_ajuste(self) -> None:
        """Lança erro caso o classificador ainda não tenha sido treinado."""
        if self.classes_ is None:
            raise RuntimeError(
                "DentalClassifier ainda não foi treinado. Chame fit() primeiro."
            )

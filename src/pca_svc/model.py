"""SVC classifier that operates on PCA-reduced luma features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from .feature_extractor import FeatureExtractor


@dataclass
class Prediction:
    """Result for a single image.

    Attributes
    ----------
    label:
        Predicted class label.
    probabilities:
        Mapping from each class label to its Platt-scaled probability.
    """

    label: str
    probabilities: dict[str, float]


class DentalClassifier:
    """SVC classifier built on top of :class:`FeatureExtractor`.

    The full pipeline is: luma pixels → StandardScaler → PCA → SVC (RBF).

    Parameters
    ----------
    n_components:
        Passed directly to :class:`FeatureExtractor`.  ``None`` triggers the
        automatic 95%-variance selection.
    auto_variance_threshold:
        Used when *n_components* is ``None``.
    C, gamma:
        SVC regularisation and kernel coefficient.
    seed:
        Shared seed for PCA and SVC.
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
        # probability=True enables predict_proba via Platt scaling
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
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DentalClassifier":
        """Fit the full pipeline on training data."""
        X_pca = self.extractor.fit_transform(X)
        self._svc.fit(X_pca, y)
        self.classes_ = list(self._svc.classes_)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class label for each row in *X*."""
        return self._svc.predict(self.extractor.transform(X))

    def predict_proba(self, X: np.ndarray) -> list[Prediction]:
        """Return a :class:`Prediction` per row in *X*.

        Each :class:`Prediction` contains the most likely label and a
        probability for every class.
        """
        self._require_fitted()
        X_pca = self.extractor.transform(X)
        proba_matrix = self._svc.predict_proba(X_pca)
        labels = self._svc.predict(X_pca)

        return [
            Prediction(
                label=str(label),
                probabilities={
                    cls: float(prob)
                    for cls, prob in zip(self.classes_, row)  # type: ignore[arg-type]
                },
            )
            for label, row in zip(labels, proba_matrix)
        ]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._svc.score(self.extractor.transform(X), y)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serialise the classifier (extractor + SVC) to *path*."""
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
        """Restore a serialised classifier from *path*."""
        import joblib

        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.extractor = data["extractor"]
        obj._svc = data["svc"]
        obj.classes_ = data["classes"]
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self.classes_ is None:
            raise RuntimeError("DentalClassifier is not fitted yet. Call fit() first.")

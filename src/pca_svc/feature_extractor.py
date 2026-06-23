"""PCA feature extractor with StandardScaler normalisation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    pass


class FeatureExtractor:
    """StandardScaler → PCA pipeline fitted on training data only.

    Parameters
    ----------
    n_components:
        Number of PCA components to keep.  When ``None`` the value is chosen
        automatically as the minimum number of components that explain at
        least ``auto_variance_threshold`` of the total variance (default 95%).
    auto_variance_threshold:
        Fraction of variance to retain when *n_components* is ``None``.
    seed:
        Random seed passed to :class:`~sklearn.decomposition.PCA`.

    Attributes
    ----------
    mean_ : np.ndarray
        Per-feature mean from the fitted StandardScaler (shape ``(n_features,)``).
    std_ : np.ndarray
        Per-feature standard deviation from the fitted StandardScaler.
    n_components_ : int
        Actual number of components selected after fitting.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        auto_variance_threshold: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.auto_variance_threshold = auto_variance_threshold
        self.seed = seed

        self._scaler: StandardScaler = StandardScaler()
        # Full PCA kept for inspection (eigenfaces, variance plots)
        self._pca_full: PCA | None = None
        # Reduced PCA used for actual feature transformation
        self._pca: PCA | None = None

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_components_: int | None = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FeatureExtractor":
        """Fit scaler and PCA on *X* (training data only)."""
        X_scaled = self._scaler.fit_transform(X)
        self.mean_ = self._scaler.mean_
        self.std_ = self._scaler.scale_

        # Full PCA — used exclusively for analysis / inspection
        self._pca_full = PCA(random_state=self.seed)
        self._pca_full.fit(X_scaled)

        # Choose n_components
        n = self.n_components or self._auto_n_components()
        self.n_components_ = n

        self._pca = PCA(n_components=n, random_state=self.seed)
        self._pca.fit(X_scaled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale with training statistics then project onto PCA components."""
        self._require_fitted()
        return self._pca.transform(self._scaler.transform(X))  # type: ignore[union-attr]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Variance analysis
    # ------------------------------------------------------------------

    def explained_variance_ratio(self) -> np.ndarray:
        """Per-component explained variance ratio from the *full* PCA."""
        self._require_fitted()
        return self._pca_full.explained_variance_ratio_  # type: ignore[union-attr]

    def cumulative_explained_variance(self) -> np.ndarray:
        """Cumulative sum of :meth:`explained_variance_ratio`."""
        return np.cumsum(self.explained_variance_ratio())

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """Return the minimum number of components that explain *threshold* variance."""
        cumulative = self.cumulative_explained_variance()
        return int(np.searchsorted(cumulative, threshold)) + 1

    # ------------------------------------------------------------------
    # Eigenfaces
    # ------------------------------------------------------------------

    def eigenfaces(
        self,
        n: int | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return the top *n* principal components.

        Parameters
        ----------
        n:
            Number of components to return (all by default).
        image_size:
            If provided as *(height, width)* each component is reshaped to a
            2-D array; otherwise the components are returned flat.

        Returns
        -------
        np.ndarray
            Shape ``(n, n_features)`` when *image_size* is ``None``,
            or ``(n, height, width)`` when *image_size* is given.
        """
        self._require_fitted()
        components = self._pca_full.components_  # type: ignore[union-attr]
        if n is not None:
            components = components[:n]
        if image_size is not None:
            components = components.reshape(-1, *image_size)
        return components

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serialise the extractor to *path* using :mod:`joblib`."""
        import joblib

        joblib.dump(
            {
                "scaler": self._scaler,
                "pca_full": self._pca_full,
                "pca": self._pca,
                "n_components": self.n_components,
                "n_components_": self.n_components_,
                "auto_variance_threshold": self.auto_variance_threshold,
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "FeatureExtractor":
        """Restore a previously saved extractor from *path*."""
        import joblib

        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._scaler = data["scaler"]
        obj._pca_full = data["pca_full"]
        obj._pca = data["pca"]
        obj.n_components = data["n_components"]
        obj.n_components_ = data["n_components_"]
        obj.auto_variance_threshold = data["auto_variance_threshold"]
        obj.seed = data["seed"]
        obj.mean_ = obj._scaler.mean_
        obj.std_ = obj._scaler.scale_
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_n_components(self) -> int:
        cumulative = np.cumsum(self._pca_full.explained_variance_ratio_)  # type: ignore[union-attr]
        return int(np.searchsorted(cumulative, self.auto_variance_threshold)) + 1

    def _require_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("FeatureExtractor is not fitted yet. Call fit() first.")

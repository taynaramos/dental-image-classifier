"""Extrator de features via PCA com normalização por StandardScaler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    pass


class FeatureExtractor:
    """Pipeline StandardScaler → PCA ajustado exclusivamente nos dados de treino.

    O fluxo é:
    1. ``StandardScaler`` — subtrai a média e divide pelo desvio padrão do treino.
    2. **PCA completo** — retido apenas para inspeção (variância, eigenfaces).
    3. **PCA reduzido** — usado para transformar os dados de fato.

    Parâmetros
    ----------
    n_components:
        Número de componentes PCA a manter.  Quando ``None``, o valor é escolhido
        automaticamente como o mínimo de componentes que explica pelo menos
        ``auto_variance_threshold`` da variância total (padrão: 95 %).
    auto_variance_threshold:
        Fração de variância a reter quando *n_components* é ``None``.
    seed:
        Semente aleatória repassada ao :class:`~sklearn.decomposition.PCA`.

    Atributos
    ----------
    mean_ : np.ndarray
        Média por feature do StandardScaler ajustado (forma ``(n_features,)``).
    std_ : np.ndarray
        Desvio padrão por feature do StandardScaler ajustado.
    n_components_ : int
        Número efetivo de componentes selecionados após o ajuste.
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
        # PCA completo — mantido para análise (eigenfaces, gráficos de variância)
        self._pca_full: PCA | None = None
        # PCA reduzido — usado na transformação real das features
        self._pca: PCA | None = None

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_components_: int | None = None

    # ------------------------------------------------------------------
    # Ajuste e transformação
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FeatureExtractor":
        """Ajusta o scaler e o PCA nos dados de treino *X*.

        Parâmetros
        ----------
        X:
            Matriz de features de treino com forma ``(n_amostras, n_features)``.

        Retorna
        -------
        self
        """
        # 1. Escalonamento: salva média e desvio padrão do treino
        X_scaled = self._scaler.fit_transform(X)
        self.mean_ = self._scaler.mean_
        self.std_ = self._scaler.scale_

        # 2. PCA completo — usado apenas para inspeção do espectro de variância
        self._pca_full = PCA(random_state=self.seed)
        self._pca_full.fit(X_scaled)

        # 3. Seleciona o número de componentes e ajusta o PCA reduzido
        n = self.n_components or self._selecionar_n_componentes()
        self.n_components_ = n
        self._pca = PCA(n_components=n, random_state=self.seed)
        self._pca.fit(X_scaled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Escala com as estatísticas do treino e projeta nos componentes PCA.

        Parâmetros
        ----------
        X:
            Dados a transformar com forma ``(n_amostras, n_features)``.

        Retorna
        -------
        np.ndarray
            Forma ``(n_amostras, n_components_)``.
        """
        self._verificar_ajuste()
        return self._pca.transform(self._scaler.transform(X))  # type: ignore[union-attr]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta e transforma em uma única chamada. Equivale a ``fit(X).transform(X)``."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Análise de variância
    # ------------------------------------------------------------------

    def explained_variance_ratio(self) -> np.ndarray:
        """Razão de variância explicada por componente, obtida do PCA **completo**.

        Retorna
        -------
        np.ndarray
            Forma ``(n_total_components,)``.
        """
        self._verificar_ajuste()
        return self._pca_full.explained_variance_ratio_  # type: ignore[union-attr]

    def cumulative_explained_variance(self) -> np.ndarray:
        """Soma acumulada de :meth:`explained_variance_ratio`.

        Útil para determinar visualmente quantos componentes são necessários.
        """
        return np.cumsum(self.explained_variance_ratio())

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """Retorna o número mínimo de componentes que explicam *threshold* da variância.

        Parâmetros
        ----------
        threshold:
            Fração de variância desejada, p. ex. ``0.95`` para 95 %.
        """
        cumulativa = self.cumulative_explained_variance()
        return int(np.searchsorted(cumulativa, threshold)) + 1

    # ------------------------------------------------------------------
    # Eigenfaces
    # ------------------------------------------------------------------

    def eigenfaces(
        self,
        n: int | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Retorna os *n* primeiros componentes principais (eigenfaces).

        Parâmetros
        ----------
        n:
            Quantidade de componentes a retornar (todos por padrão).
        image_size:
            Se informado como *(altura, largura)*, cada componente é
            remodelado para um array 2D; caso contrário, retorna achatado.

        Retorna
        -------
        np.ndarray
            Forma ``(n, n_features)`` quando *image_size* é ``None``,
            ou ``(n, altura, largura)`` quando *image_size* é fornecido.
        """
        self._verificar_ajuste()
        componentes = self._pca_full.components_  # type: ignore[union-attr]
        if n is not None:
            componentes = componentes[:n]
        if image_size is not None:
            componentes = componentes.reshape(-1, *image_size)
        return componentes

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serializa o extrator em *path* usando :mod:`joblib`.

        Salva o scaler (média e desvio padrão do treino), o PCA completo
        e o PCA reduzido para que a normalização seja idêntica na inferência.
        """
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
        """Restaura um extrator previamente salvo em *path*."""
        import joblib

        dados = joblib.load(path)
        obj = cls.__new__(cls)
        obj._scaler = dados["scaler"]
        obj._pca_full = dados["pca_full"]
        obj._pca = dados["pca"]
        obj.n_components = dados["n_components"]
        obj.n_components_ = dados["n_components_"]
        obj.auto_variance_threshold = dados["auto_variance_threshold"]
        obj.seed = dados["seed"]
        # Expõe média e std diretamente para conveniência
        obj.mean_ = obj._scaler.mean_
        obj.std_ = obj._scaler.scale_
        return obj

    # ------------------------------------------------------------------
    # Auxiliares internos
    # ------------------------------------------------------------------

    def _selecionar_n_componentes(self) -> int:
        """Seleciona automaticamente o menor n que atinge o limiar de variância."""
        cumulativa = np.cumsum(self._pca_full.explained_variance_ratio_)  # type: ignore[union-attr]
        return int(np.searchsorted(cumulativa, self.auto_variance_threshold)) + 1

    def _verificar_ajuste(self) -> None:
        """Lança erro caso o extrator ainda não tenha sido ajustado."""
        if self._pca is None:
            raise RuntimeError(
                "FeatureExtractor ainda não foi ajustado. Chame fit() primeiro."
            )

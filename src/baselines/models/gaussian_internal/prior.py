from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky


class Wishart:
    """
    Data structure for the normal-inverse-Wishart prior shared by Gaussian families.
    """

    def __init__(
        self,
        mu: np.ndarray,
        embedding_size: int,
        kappa: float = 0.1,
        scale_sigma: float | None = None,
    ) -> None:
        scale = 0.1 if scale_sigma is None else float(scale_sigma)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Wishart scale_sigma must be finite and > 0.")
        self.kappa = kappa
        self.mu = np.asarray(mu, dtype=np.float64)
        self.nu = embedding_size
        self.scale_sigma = scale
        self.sigma = np.eye(embedding_size, dtype=np.float64) * scale
        self.chol_sigma = cholesky(self.sigma)

    @classmethod
    def from_word_vectors(
        cls,
        word_vecs: np.ndarray,
        kappa: float = 0.1,
        scale_sigma: float | None = None,
    ) -> "Wishart":
        word_vecs = np.asarray(word_vecs, dtype=np.float64)
        return cls(
            mu=np.mean(word_vecs, axis=0),
            embedding_size=word_vecs.shape[1],
            kappa=kappa,
            scale_sigma=scale_sigma,
        )

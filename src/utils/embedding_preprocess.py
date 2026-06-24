from __future__ import annotations

from collections.abc import Iterable

import numpy as np

_VALID_MODES = {"none", "mean_center", "whitening"}


class EmbeddingPreprocessor:
    """Pre-normalization transform for sentence embeddings."""

    def __init__(self, mode: str = "none", whitening_eps: float = 1e-5) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        if whitening_eps <= 0.0:
            raise ValueError("whitening_eps must be > 0.")

        self.mode = normalized_mode
        self.whitening_eps = float(whitening_eps)
        self.mean_: np.ndarray | None = None
        self.whitening_matrix_: np.ndarray | None = None
        self._fitted = normalized_mode == "none"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_batches(self, batches: Iterable[np.ndarray], embedding_dim: int) -> None:
        """Fit transform statistics from raw embedding batches."""
        if self.mode == "none":
            self._fitted = True
            return

        d = int(embedding_dim)
        if d <= 0:
            raise ValueError("embedding_dim must be > 0.")

        count = 0
        sum_vec = np.zeros(d, dtype=np.float64)
        sum_outer = np.zeros((d, d), dtype=np.float64)

        for batch in batches:
            arr = np.asarray(batch, dtype=np.float64)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != d:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {d}, got {arr.shape[1]}."
                )

            count += int(arr.shape[0])
            sum_vec += arr.sum(axis=0)
            sum_outer += arr.T @ arr

        if count <= 0:
            raise ValueError(
                "Cannot fit embedding preprocessor: no sentence embeddings were found."
            )

        mean = sum_vec / float(count)
        self.mean_ = mean

        if self.mode == "whitening":
            cov = sum_outer / float(count) - np.outer(mean, mean)
            cov = 0.5 * (cov + cov.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 0.0, None)
            inv_sqrt = 1.0 / np.sqrt(eigvals + self.whitening_eps)
            self.whitening_matrix_ = (eigvecs * inv_sqrt) @ eigvecs.T

        self._fitted = True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply fitted transform to embeddings."""
        arr = np.asarray(embeddings, dtype=np.float64)
        was_vector = arr.ndim == 1
        if was_vector:
            arr = arr.reshape(1, -1)

        if arr.size == 0 or self.mode == "none":
            return arr.reshape(-1) if was_vector else arr
        if not self._fitted:
            raise RuntimeError("EmbeddingPreprocessor must be fitted before transform.")

        out = arr
        if self.mean_ is not None:
            out = out - self.mean_
        if self.mode == "whitening":
            if self.whitening_matrix_ is None:
                raise RuntimeError("Whitening matrix is not available.")
            out = out @ self.whitening_matrix_

        return out.reshape(-1) if was_vector else out

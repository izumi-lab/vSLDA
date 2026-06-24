from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.core.progress import ProgressReporter, TqdmProgressReporter
from src.utils.embedding_preprocess import EmbeddingPreprocessor

if TYPE_CHECKING:
    from src.utils.encoder import SentenceEncoder


def normalize_pre_normalize_transform(name: str) -> str:
    normalized_transform = str(name).strip().lower().replace("-", "_")
    if normalized_transform == "meancenter":
        normalized_transform = "mean_center"
    if normalized_transform == "whiten":
        normalized_transform = "whitening"
    return normalized_transform


class VMFDocumentEncoder:
    """Encode, transform, and normalize documents for vMF training/inference."""

    STORAGE_DTYPE = np.float32

    def __init__(
        self,
        *,
        encoder: SentenceEncoder,
        embedding_size: int,
        pre_normalize_transform: str,
        whitening_eps: float,
        log,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.encoder = encoder
        self.embedding_size = int(embedding_size)
        self.pre_normalize_transform = normalize_pre_normalize_transform(
            pre_normalize_transform
        )
        self.whitening_eps = float(whitening_eps)
        self.log = log
        self.progress = progress or TqdmProgressReporter()
        self.embedding_preprocessor = EmbeddingPreprocessor(
            mode=self.pre_normalize_transform,
            whitening_eps=self.whitening_eps,
        )

    def iter_raw_encoded_document_batches(
        self, corpus: Sequence[Sequence[str]]
    ) -> Iterable[np.ndarray]:
        pbar = self.progress.wrap(
            corpus,
            desc=f"Fitting embedding transform ({self.pre_normalize_transform})",
        )
        for doc in pbar:
            enc = self.encoder.encode(doc)
            arr = np.asarray(enc, dtype=np.float64)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            yield arr

    def fit_on_corpus(self, corpus: Sequence[Sequence[str]]) -> None:
        if self.pre_normalize_transform == "none":
            return

        self.log.info(
            "Fitting embedding transform: mode=%s",
            self.pre_normalize_transform,
        )
        self.embedding_preprocessor.fit_batches(
            self.iter_raw_encoded_document_batches(corpus),
            embedding_dim=self.embedding_size,
        )
        self.log.info("Embedding transform fitting complete")

    def apply_pre_normalize_transform(self, enc: np.ndarray) -> np.ndarray:
        return self.embedding_preprocessor.transform(enc)

    def encode_and_normalize(self, doc: Sequence[str]) -> np.ndarray:
        enc = self.encoder.encode(doc)
        arr = np.asarray(enc, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, self.embedding_size), dtype=self.STORAGE_DTYPE)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        arr = self.apply_pre_normalize_transform(arr)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return np.asarray(arr / norms, dtype=self.STORAGE_DTYPE)

    def encode_corpus(
        self,
        corpus: Sequence[Sequence[str]],
        *,
        desc: str | None = None,
    ) -> list[np.ndarray]:
        encoded_docs: list[np.ndarray] = []
        iterator: Iterable[Sequence[str]]
        if desc is None:
            iterator = corpus
        else:
            iterator = self.progress.wrap(corpus, desc=desc)
        for doc in iterator:
            encoded_docs.append(self.encode_and_normalize(doc))
        return encoded_docs

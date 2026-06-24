from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _load_vectors(word2vec: object, *, wikientvec_cache_dir: str | None = None):
    from src.baselines.models.gaussian_helpers import load_word_vectors

    return load_word_vectors(word2vec, wikientvec_cache_dir=wikientvec_cache_dir)


@dataclass(frozen=True)
class UsifState:
    word_probabilities: dict[str, float]
    alpha: float
    alpha_hat: float | None
    average_sentence_length: float
    word_probability_threshold: float | None
    common_components: np.ndarray
    singular_values: np.ndarray
    component_weights: np.ndarray
    embedding_dim: int
    fingerprint: str


class UsifSentenceEncoder:
    """uSIF sentence embeddings over pretrained word vectors.

    The word vectors are fixed. Fitting estimates token probabilities and common
    discourse components from the train split.
    """

    def __init__(
        self,
        *,
        word2vec: object = "glove-wiki-gigaword-100",
        wikientvec_cache_dir: str | None = None,
        alpha: float | None = None,
        n_components: int | None = None,
        component_policy: str = "auto",
        word_probability_source: str = "train",
        min_count: int = 1,
        normalize_embeddings: bool = False,
    ) -> None:
        self.word2vec = word2vec
        self.wikientvec_cache_dir = wikientvec_cache_dir
        self.vectors = _load_vectors(
            word2vec, wikientvec_cache_dir=wikientvec_cache_dir
        )
        self.alpha = None if alpha is None else float(alpha)
        if self.alpha is not None and self.alpha <= 0.0:
            raise ValueError("uSIF alpha must be > 0.")
        self.n_components = None if n_components is None else int(n_components)
        if self.n_components is not None and self.n_components < 0:
            raise ValueError("uSIF n_components must be >= 0.")
        self.component_policy = str(component_policy or "auto").strip().lower()
        if self.component_policy not in {"auto", "fixed", "none"}:
            raise ValueError("uSIF component_policy must be one of auto, fixed, none.")
        self.word_probability_source = (
            str(word_probability_source or "train").strip().lower()
        )
        if self.word_probability_source != "train":
            raise ValueError("Only uSIF word_probability_source='train' is supported.")
        self.min_count = int(min_count)
        if self.min_count <= 0:
            raise ValueError("uSIF min_count must be > 0.")
        self.normalize_embeddings = bool(normalize_embeddings)
        self.state: UsifState | None = None

    @property
    def embedding_dimension(self) -> int:
        return int(self.vectors.vector_size)

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dimension

    @property
    def is_fitted(self) -> bool:
        return self.state is not None

    def fit_tokenized(self, tokenized_sentences: Sequence[Sequence[str]]) -> None:
        sentences = [
            tuple(str(token) for token in sent) for sent in tokenized_sentences
        ]
        counts: Counter[str] = Counter(
            token
            for sent in sentences
            for token in sent
            if token in self.vectors.key_to_index
        )
        counts = Counter(
            {token: count for token, count in counts.items() if count >= self.min_count}
        )
        total = int(sum(counts.values()))
        nonempty_fitted_lengths = [len(sent) for sent in sentences if len(sent) > 0]
        average_sentence_length = (
            float(np.mean(nonempty_fitted_lengths)) if nonempty_fitted_lengths else 0.0
        )
        if total <= 0:
            alpha, alpha_hat, threshold = self._resolve_alpha(
                word_probabilities={},
                average_sentence_length=0.0,
            )
            self.state = UsifState(
                word_probabilities={},
                alpha=alpha,
                alpha_hat=alpha_hat,
                average_sentence_length=0.0,
                word_probability_threshold=threshold,
                common_components=np.zeros(
                    (0, self.embedding_dimension), dtype=np.float64
                ),
                singular_values=np.zeros((0,), dtype=np.float64),
                component_weights=np.zeros((0,), dtype=np.float64),
                embedding_dim=self.embedding_dimension,
                fingerprint=self._fingerprint(
                    {},
                    alpha=alpha,
                    alpha_hat=alpha_hat,
                    average_sentence_length=0.0,
                    word_probability_threshold=threshold,
                    components=np.zeros((0, self.embedding_dimension)),
                ),
            )
            return

        probabilities = {
            token: float(count) / float(total)
            for token, count in sorted(counts.items())
        }
        alpha, alpha_hat, threshold = self._resolve_alpha(
            word_probabilities=probabilities,
            average_sentence_length=average_sentence_length,
        )
        raw_embeddings = self._weighted_average_batch(
            sentences,
            word_probabilities=probabilities,
            alpha=alpha,
        )
        components, singular_values, component_weights = self._fit_components(
            raw_embeddings
        )
        self.state = UsifState(
            word_probabilities=probabilities,
            alpha=alpha,
            alpha_hat=alpha_hat,
            average_sentence_length=average_sentence_length,
            word_probability_threshold=threshold,
            common_components=components,
            singular_values=singular_values,
            component_weights=component_weights,
            embedding_dim=self.embedding_dimension,
            fingerprint=self._fingerprint(
                probabilities,
                alpha=alpha,
                alpha_hat=alpha_hat,
                average_sentence_length=average_sentence_length,
                word_probability_threshold=threshold,
                components=components,
            ),
        )

    def encode_tokenized(
        self,
        tokenized_sentences: Sequence[Sequence[str]],
        **encode_kwargs: Any,
    ) -> np.ndarray:
        encode_kwargs.pop("show_progress_bar", None)
        encode_kwargs.pop("batch_size", None)
        if encode_kwargs:
            unsupported = ", ".join(sorted(encode_kwargs))
            raise TypeError(f"Unsupported uSIF encode kwargs: {unsupported}")
        if self.state is None:
            raise ValueError("uSIF encoder must be fitted before encode().")
        sentences = [
            tuple(str(token) for token in sent) for sent in tokenized_sentences
        ]
        raw = self._weighted_average_batch(
            sentences,
            word_probabilities=self.state.word_probabilities,
            alpha=self.state.alpha,
        )
        embeddings = self._remove_common_components(raw)
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms <= 0.0] = 1.0
            embeddings = embeddings / norms
        return embeddings.astype(np.float32, copy=False)

    def encode(self, sentences: Sequence[str], **encode_kwargs: Any) -> np.ndarray:
        tokenized = [str(sentence).split() for sentence in sentences]
        return self.encode_tokenized(tokenized, **encode_kwargs)

    def _resolve_alpha(
        self,
        *,
        word_probabilities: Mapping[str, float],
        average_sentence_length: float,
    ) -> tuple[float, float | None, float | None]:
        if self.alpha is not None:
            return self.alpha, None, None
        vocab_size = len(word_probabilities)
        if vocab_size <= 0 or average_sentence_length <= 0.0:
            return 1e-3, None, None

        threshold = 1.0 - (1.0 - (1.0 / float(vocab_size))) ** float(
            average_sentence_length
        )
        alpha_hat = sum(
            1 for probability in word_probabilities.values() if probability > threshold
        ) / float(vocab_size)

        eps = np.finfo(np.float64).eps
        alpha_hat = float(np.clip(alpha_hat, eps, 1.0 - eps))
        normalization = float(vocab_size) / 2.0
        alpha = (1.0 - alpha_hat) / (alpha_hat * normalization)
        return max(float(alpha), 1e-12), alpha_hat, float(threshold)

    def _word_vector(self, token: str) -> np.ndarray | None:
        if token not in self.vectors.key_to_index:
            return None
        vec = np.asarray(self.vectors[token], dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0 and np.isfinite(norm):
            vec = vec / norm
        return vec

    def _weighted_average_batch(
        self,
        tokenized_sentences: Sequence[Sequence[str]],
        *,
        word_probabilities: Mapping[str, float],
        alpha: float,
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        for sentence in tokenized_sentences:
            weighted: list[np.ndarray] = []
            for token in sentence:
                vec = self._word_vector(str(token))
                if vec is None:
                    continue
                prob = float(word_probabilities.get(str(token), 0.0))
                weight = alpha / (prob + (alpha / 2.0))
                weighted.append(weight * vec)
            if weighted:
                rows.append(np.mean(np.vstack(weighted), axis=0))
            else:
                rows.append(np.zeros((self.embedding_dimension,), dtype=np.float64))
        if not rows:
            return np.zeros((0, self.embedding_dimension), dtype=np.float64)
        return np.vstack(rows).astype(np.float64, copy=False)

    def _fit_components(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.component_policy == "none"
            or embeddings.size == 0
            or embeddings.shape[0] < 2
        ):
            return (
                np.zeros((0, self.embedding_dimension), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        max_components = min(vt.shape[0], self.embedding_dimension)
        if self.component_policy == "fixed":
            n_components = 5 if self.n_components is None else self.n_components
        elif self.n_components is not None:
            n_components = self.n_components
        else:
            n_components = min(5, max_components)
        n_components = max(0, min(int(n_components), max_components))
        if n_components == 0:
            return (
                np.zeros((0, self.embedding_dimension), dtype=np.float64),
                singular_values,
                np.zeros((0,), dtype=np.float64),
            )
        components = np.asarray(vt[:n_components], dtype=np.float64)
        selected = singular_values[:n_components].astype(np.float64, copy=False)
        denom = float(np.sum(selected**2))
        if denom <= 0.0 or not np.isfinite(denom):
            weights = np.zeros((n_components,), dtype=np.float64)
        else:
            weights = (selected**2) / denom
        return components, singular_values, weights

    def _remove_common_components(self, embeddings: np.ndarray) -> np.ndarray:
        if self.state is None or self.state.common_components.size == 0:
            return np.asarray(embeddings, dtype=np.float64)
        out = np.asarray(embeddings, dtype=np.float64).copy()
        for component, weight in zip(
            self.state.common_components,
            self.state.component_weights,
        ):
            out -= float(weight) * np.outer(out @ component, component)
        return out

    def _fingerprint(
        self,
        probabilities: Mapping[str, float],
        *,
        alpha: float,
        alpha_hat: float | None,
        average_sentence_length: float,
        word_probability_threshold: float | None,
        components: np.ndarray,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(self.word2vec).encode("utf-8"))
        hasher.update(str(self.embedding_dimension).encode("utf-8"))
        hasher.update(repr(float(alpha)).encode("utf-8"))
        alpha_hat_payload = None if alpha_hat is None else float(alpha_hat)
        hasher.update(repr(alpha_hat_payload).encode("utf-8"))
        hasher.update(repr(float(average_sentence_length)).encode("utf-8"))
        threshold_payload = (
            None
            if word_probability_threshold is None
            else float(word_probability_threshold)
        )
        hasher.update(repr(threshold_payload).encode("utf-8"))
        for token, probability in sorted(probabilities.items()):
            hasher.update(token.encode("utf-8"))
            hasher.update(repr(float(probability)).encode("utf-8"))
        hasher.update(np.asarray(components, dtype=np.float64).tobytes())
        return hasher.hexdigest()[:12]

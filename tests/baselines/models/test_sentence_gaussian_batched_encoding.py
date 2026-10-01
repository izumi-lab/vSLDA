"""Batched corpus encoding for Sentence Gaussian LDA.

The trainer, the training soft posteriors and test inference used to call the encoder
once per document. They now encode a corpus in one call and split the rows back per
document; these tests pin that the split is exact and that the reused embeddings give
the same posteriors as re-encoding.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.models.sentence_gaussianlda import _sentence_topic_soft
from src.utils.encoder_inputs import encode_documents_batched

DIM = 4


class CountingEncoder:
    """Deterministic per-sentence encoder that records how often it is called."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[int] = []

    def _row(self, sentence: str) -> np.ndarray:
        seed = sum(ord(ch) * (index + 1) for index, ch in enumerate(sentence))
        return np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)

    def encode(self, sentences, **_kwargs) -> np.ndarray:
        sentences = list(sentences)
        self.calls.append(len(sentences))
        if not sentences:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._row(sentence) for sentence in sentences])

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


class QuadraticModel:
    """Stand-in for SentenceGaussianLdaModel: fixed per-topic log densities."""

    def __init__(self, encoder: CountingEncoder, num_topics: int = 3) -> None:
        self.encoder = encoder
        self.centers = np.random.default_rng(7).normal(size=(num_topics, DIM))

    def log_multivariate_tdensity_tables(self, x: np.ndarray) -> np.ndarray:
        diff = self.centers - np.asarray(x, dtype=np.float64)[None, :]
        return -0.5 * np.sum(diff * diff, axis=1)


CORPUS = [
    ["the first sentence", "another one here"],
    [],
    ["a single sentence"],
    ["x", "y", "z", "longer sentence with more words"],
]


def test_batched_encoding_splits_rows_per_document_in_one_call() -> None:
    encoder = CountingEncoder()

    batched = encode_documents_batched(encoder, CORPUS)

    assert encoder.calls == [sum(len(doc) for doc in CORPUS)]
    assert [part.shape for part in batched] == [(len(doc), DIM) for doc in CORPUS]
    for doc, part in zip(CORPUS, batched):
        assert part.dtype == np.float64
        if doc:
            expected = np.asarray(CountingEncoder().encode(doc), dtype=np.float64)
            np.testing.assert_array_equal(part, expected)


def test_batched_encoding_of_an_all_empty_corpus_keeps_the_dimension() -> None:
    encoder = CountingEncoder()

    batched = encode_documents_batched(encoder, [[], []])

    assert encoder.calls == []
    assert [part.shape for part in batched] == [(0, DIM), (0, DIM)]


def test_batched_encoding_rejects_a_row_count_mismatch() -> None:
    class ShortEncoder(CountingEncoder):
        def encode(self, sentences, **kwargs) -> np.ndarray:
            return super().encode(list(sentences)[:-1], **kwargs)

    with pytest.raises(ValueError, match="for 3 sentences"):
        encode_documents_batched(ShortEncoder(), [["a", "b"], ["c"]])


def test_soft_posteriors_from_reused_embeddings_match_re_encoding() -> None:
    kwargs = dict(
        corpus=CORPUS,
        num_topics=3,
        batch_size=8,
        soft_temperature=1.0,
        show_progress_bar=False,
    )
    re_encoding = CountingEncoder()
    expected = _sentence_topic_soft(model=QuadraticModel(re_encoding), **kwargs)

    reusing = CountingEncoder()
    encoded = encode_documents_batched(CountingEncoder(), CORPUS)
    actual = _sentence_topic_soft(
        model=QuadraticModel(reusing), encoded_corpus=encoded, **kwargs
    )

    assert reusing.calls == []
    assert len(re_encoding.calls) == len(CORPUS)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


def test_soft_posteriors_reject_misaligned_embeddings() -> None:
    encoder = CountingEncoder()
    with pytest.raises(ValueError, match="encoded_corpus has 1 documents"):
        _sentence_topic_soft(
            corpus=CORPUS,
            model=QuadraticModel(encoder),
            num_topics=3,
            batch_size=8,
            soft_temperature=1.0,
            show_progress_bar=False,
            encoded_corpus=[np.zeros((2, DIM))],
        )

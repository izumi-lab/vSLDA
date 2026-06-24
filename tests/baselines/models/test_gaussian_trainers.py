from __future__ import annotations

import numpy as np
import pytest

from src.baselines.models.gaussian_trainer import (
    GaussianLDATrainer as WordGaussianTrainer,
)
from src.baselines.models.sentence_gaussian_trainer import (
    GaussianLDATrainer as SentenceGaussianTrainer,
)

pytest.importorskip("choldate")


class CountingEncoder:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in mapping.items()
        }
        self.call_count = 0

    def encode(self, sentences, **_kwargs) -> np.ndarray:
        self.call_count += 1
        batch = list(sentences)
        if not batch:
            return np.zeros(
                (0, self.get_sentence_embedding_dimension()), dtype=np.float64
            )
        return np.vstack([self.mapping[str(sentence)] for sentence in batch]).astype(
            np.float64
        )

    def get_sentence_embedding_dimension(self) -> int:
        return next(iter(self.mapping.values())).shape[0]


def test_gaussian_trainer_is_deterministic_with_fixed_seed() -> None:
    corpus = [[0, 1, 2], [2, 1, 0]]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    vocab = ["a", "b", "c"]

    np.random.seed(11)
    trainer_a = WordGaussianTrainer(
        corpus,
        embeddings,
        vocab,
        num_tables=2,
        alpha=0.1,
    )
    trainer_a.sample(2)

    np.random.seed(11)
    trainer_b = WordGaussianTrainer(
        corpus,
        embeddings,
        vocab,
        num_tables=2,
        alpha=0.1,
    )
    trainer_b.sample(2)

    assert np.array_equal(trainer_a.table_counts, trainer_b.table_counts)
    assert np.array_equal(
        trainer_a.table_counts_per_doc, trainer_b.table_counts_per_doc
    )
    assert np.array_equal(
        np.asarray(trainer_a.table_assignments, dtype=np.int64),
        np.asarray(trainer_b.table_assignments, dtype=np.int64),
    )
    assert np.allclose(trainer_a.average_ll, trainer_b.average_ll)
    assert len(trainer_a.iteration_diagnostics) == 2
    assert trainer_a.iteration_diagnostics[0].sampling_sec >= 0.0
    assert trainer_a.iteration_diagnostics[0].avg_log_likelihood_sec >= 0.0
    assert trainer_a.iteration_diagnostics[0].iteration_elapsed_sec >= 0.0


def test_sentence_gaussian_trainer_preencoding_preserves_results_and_reduces_encode_calls() -> (
    None
):
    corpus = [["s1", "s2"], ["s3", "s4"]]
    mapping = {
        "s1": np.asarray([1.0, 0.0]),
        "s2": np.asarray([0.8, 0.2]),
        "s3": np.asarray([0.0, 1.0]),
        "s4": np.asarray([0.2, 0.8]),
    }

    encoder_pre = CountingEncoder(mapping)
    np.random.seed(23)
    trainer_pre = SentenceGaussianTrainer(
        corpus,
        encoder_pre,
        num_tables=2,
        alpha=0.1,
        kappa=0.1,
        preencode_corpus=True,
    )
    trainer_pre.sample(2)

    encoder_raw = CountingEncoder(mapping)
    np.random.seed(23)
    trainer_raw = SentenceGaussianTrainer(
        corpus,
        encoder_raw,
        num_tables=2,
        alpha=0.1,
        kappa=0.1,
        preencode_corpus=False,
    )
    trainer_raw.sample(2)

    assert np.array_equal(trainer_pre.table_counts, trainer_raw.table_counts)
    assert np.array_equal(
        trainer_pre.table_counts_per_doc, trainer_raw.table_counts_per_doc
    )
    assert np.array_equal(
        np.asarray(trainer_pre.table_assignments, dtype=np.int64),
        np.asarray(trainer_raw.table_assignments, dtype=np.int64),
    )
    assert np.allclose(trainer_pre.average_ll, trainer_raw.average_ll)
    assert trainer_pre.training_corpus_preencoded is True
    assert trainer_raw.training_corpus_preencoded is False
    assert encoder_pre.call_count < encoder_raw.call_count
    assert len(trainer_pre.iteration_diagnostics) == 2
    assert trainer_pre.iteration_diagnostics[1].sampling_sec >= 0.0

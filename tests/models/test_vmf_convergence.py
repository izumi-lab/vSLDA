from __future__ import annotations

import logging

import numpy as np
import pytest

from src.core.progress import NullProgressReporter
from src.models.vmf_sentence_lda import VMFLDATrainer
from src.utils.random import set_global_seed


class _DeterministicEncoder:
    def __init__(self, mapping: dict[str, np.ndarray], dim: int) -> None:
        self._mapping = mapping
        self._dim = dim

    def encode(self, sentences) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self._dim), dtype=np.float64)
        return np.vstack([self._mapping[s] for s in sentences]).astype(np.float64)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def _build_two_cluster_corpus(
    rng: np.random.Generator,
    *,
    docs_per_cluster: int = 5,
    sentences_per_doc: int = 4,
    dim: int = 8,
) -> tuple[list[list[str]], dict[str, np.ndarray]]:
    cluster_a_center = np.zeros(dim)
    cluster_a_center[0] = 1.0
    cluster_b_center = np.zeros(dim)
    cluster_b_center[1] = 1.0

    corpus: list[list[str]] = []
    embedding_map: dict[str, np.ndarray] = {}
    sentence_id = 0
    for cluster_idx, center in enumerate([cluster_a_center, cluster_b_center]):
        for _ in range(docs_per_cluster):
            doc = []
            for _ in range(sentences_per_doc):
                vec = center + rng.normal(0, 0.1, size=dim)
                vec = vec / np.linalg.norm(vec)
                key = f"s{sentence_id}_c{cluster_idx}"
                embedding_map[key] = vec
                doc.append(key)
                sentence_id += 1
            corpus.append(doc)
    return corpus, embedding_map


def _best_binary_topic_accuracy(
    corpus: list[list[str]],
    topic_assignments: list[np.ndarray],
) -> float:
    predicted: list[int] = []
    expected: list[int] = []
    for doc, assignments in zip(corpus, topic_assignments, strict=True):
        predicted.extend(int(topic) for topic in assignments)
        expected.extend(
            int(sentence_id.rsplit("_c", maxsplit=1)[1]) for sentence_id in doc
        )

    predicted_arr = np.asarray(predicted, dtype=np.int32)
    expected_arr = np.asarray(expected, dtype=np.int32)
    direct = float(np.mean(predicted_arr == expected_arr))
    swapped = float(np.mean((1 - predicted_arr) == expected_arr))
    return max(direct, swapped)


@pytest.mark.slow
def test_vmf_trainer_average_log_likelihood_does_not_collapse() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(seed=42)
    corpus, embedding_map = _build_two_cluster_corpus(rng)

    trainer = VMFLDATrainer(
        corpus=corpus,
        encoder=_DeterministicEncoder(embedding_map, dim=8),
        num_topics=2,
        alpha=1.0,
        kappa=20.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-converge"),
    )
    trainer.sample(num_iterations=10, num_sweeps=1, num_samples=1, estimate_alpha=False)

    initial_ll = trainer.average_ll[0]
    final_ll = trainer.average_ll[-1]
    assert final_ll >= initial_ll - 0.5


@pytest.mark.slow
def test_vmf_trainer_topic_means_separate_two_well_separated_clusters() -> None:
    set_global_seed(7)
    rng = np.random.default_rng(seed=7)
    corpus, embedding_map = _build_two_cluster_corpus(rng, docs_per_cluster=10)

    trainer = VMFLDATrainer(
        corpus=corpus,
        encoder=_DeterministicEncoder(embedding_map, dim=8),
        num_topics=2,
        alpha=1.0,
        kappa=20.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-cluster"),
    )
    trainer.sample(num_iterations=20, num_sweeps=2, num_samples=1, estimate_alpha=False)

    means = trainer.topic_means
    assert means.shape == (2, 8)
    cos_sim = float(
        np.dot(means[0], means[1])
        / (np.linalg.norm(means[0]) * np.linalg.norm(means[1]) + 1e-12)
    )
    assert cos_sim < 0.5


@pytest.mark.slow
def test_vmf_trainer_recovers_two_synthetic_clusters_above_threshold() -> None:
    set_global_seed(7)
    rng = np.random.default_rng(seed=7)
    corpus, embedding_map = _build_two_cluster_corpus(rng, docs_per_cluster=10)

    trainer = VMFLDATrainer(
        corpus=corpus,
        encoder=_DeterministicEncoder(embedding_map, dim=8),
        num_topics=2,
        alpha=1.0,
        kappa=20.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-cluster-recovery"),
        progress=NullProgressReporter(),
    )
    trainer.sample(num_iterations=20, num_sweeps=2, num_samples=1, estimate_alpha=False)

    best_accuracy = _best_binary_topic_accuracy(corpus, trainer.topic_assignments)

    assert np.count_nonzero(trainer.topic_counts) == 2
    assert best_accuracy >= 0.80


@pytest.mark.slow
def test_vmf_trainer_is_reproducible_with_fixed_seed() -> None:
    def _run_once() -> list[np.ndarray]:
        set_global_seed(123)
        rng = np.random.default_rng(seed=123)
        corpus, embedding_map = _build_two_cluster_corpus(rng)
        trainer = VMFLDATrainer(
            corpus=corpus,
            encoder=_DeterministicEncoder(embedding_map, dim=8),
            num_topics=2,
            alpha=1.0,
            kappa=10.0,
            num_components=1,
            pre_normalize_transform="none",
            log=logging.getLogger("test-vmf-repro"),
        )
        trainer.sample(
            num_iterations=5, num_sweeps=1, num_samples=1, estimate_alpha=False
        )
        return [doc.copy() for doc in trainer.topic_assignments]

    first = _run_once()
    second = _run_once()
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("transform", ["none", "mean_center", "whitening"])
def test_vmf_trainer_runs_with_each_pre_normalize_transform(transform: str) -> None:
    set_global_seed(0)
    rng = np.random.default_rng(seed=0)
    corpus, embedding_map = _build_two_cluster_corpus(rng, docs_per_cluster=5)

    trainer = VMFLDATrainer(
        corpus=corpus,
        encoder=_DeterministicEncoder(embedding_map, dim=8),
        num_topics=2,
        alpha=1.0,
        kappa=10.0,
        num_components=1,
        pre_normalize_transform=transform,
        log=logging.getLogger("test-vmf-transform"),
    )
    trainer.sample(num_iterations=2, num_sweeps=1, num_samples=1, estimate_alpha=False)
    assert trainer.assert_valid_state().is_valid is True

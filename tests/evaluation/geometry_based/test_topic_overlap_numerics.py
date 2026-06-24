from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.geometry_based.metrics import (
    aggregate_metrics,
    compute_cosine_similarity,
    compute_overlap_metrics,
)


def test_cosine_similarity_of_orthogonal_unit_vectors_is_zero() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = compute_cosine_similarity(vectors)

    assert sim.shape == (2, 2)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[1, 1] == pytest.approx(1.0)
    assert sim[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert sim[1, 0] == pytest.approx(0.0, abs=1e-9)


def test_cosine_similarity_of_parallel_vectors_is_one() -> None:
    vectors = np.array([[2.0, 0.0], [3.0, 0.0]])
    sim = compute_cosine_similarity(vectors)

    assert sim[0, 1] == pytest.approx(1.0)


def test_cosine_similarity_of_anti_parallel_vectors_is_minus_one() -> None:
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0]])
    sim = compute_cosine_similarity(vectors)

    assert sim[0, 1] == pytest.approx(-1.0)


def test_compute_overlap_metrics_returns_zero_for_orthogonal_topics() -> None:
    cosine = compute_cosine_similarity(np.eye(3))

    metrics = compute_overlap_metrics(cosine, dup_threshold=0.9)

    assert metrics["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["max_pairwise_cosine"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["diversity_score"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["num_pairs_above_threshold"] == pytest.approx(0.0)


def test_aggregate_metrics_averages_per_iteration_values() -> None:
    per_iter = [
        {
            "mean_pairwise_cosine": 0.2,
            "diversity_score": 0.8,
            "max_pairwise_cosine": 0.3,
            "num_pairs_above_threshold": 1.0,
        },
        {
            "mean_pairwise_cosine": 0.4,
            "diversity_score": 0.6,
            "max_pairwise_cosine": 0.5,
            "num_pairs_above_threshold": 3.0,
        },
    ]

    agg = aggregate_metrics(per_iter)

    assert agg["mean_pairwise_cosine"]["mean"] == pytest.approx(0.3)
    assert agg["diversity_score"]["mean"] == pytest.approx(0.7)
    assert agg["max_pairwise_cosine"]["mean"] == pytest.approx(0.4)
    assert agg["num_pairs_above_threshold"]["mean"] == pytest.approx(2.0)
    assert agg["diversity_score"]["std"] == pytest.approx(np.sqrt(0.02))

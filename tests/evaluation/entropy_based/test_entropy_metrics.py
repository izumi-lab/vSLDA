from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.evaluation.entropy_based.entropy_metrics import (
    SUMMARY_METRIC_KEYS,
    aggregate_metrics,
    compute_entropy_metrics,
    document_topic_entropy,
    shannon_entropy,
    topic_document_distribution,
    topic_document_entropy,
    topic_flags,
    topic_rank1_doc_fraction,
)

THETA_EXAMPLE = np.asarray(
    [
        [0.97, 0.01, 0.01, 0.01],
        [0.50, 0.50, 0.00, 0.00],
        [0.25, 0.25, 0.25, 0.25],
    ]
)


def test_shannon_entropy_treats_zero_as_zero_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = shannon_entropy(np.asarray([1.0, 0.0, 0.0]))
    assert value == pytest.approx(0.0)
    assert shannon_entropy(np.full(4, 0.25)) == pytest.approx(np.log(4))


def test_document_topic_entropy_uniform_and_one_hot() -> None:
    uniform = np.full((4, 5), 0.2)
    doc = document_topic_entropy(uniform)
    assert np.allclose(doc["doc_topic_entropy"], np.log(5))
    assert np.allclose(doc["doc_topic_entropy_normalized"], 1.0)
    assert doc["valid"].all()

    one_hot = np.eye(3)
    doc = document_topic_entropy(one_hot)
    assert np.allclose(doc["doc_topic_entropy"], 0.0)
    assert np.allclose(doc["doc_topic_entropy_normalized"], 0.0)


def test_document_topic_entropy_matches_hand_computed_example() -> None:
    doc = document_topic_entropy(THETA_EXAMPLE)
    expected_a = -(0.97 * np.log(0.97) + 3 * 0.01 * np.log(0.01)) / np.log(4)
    assert doc["doc_topic_entropy_normalized"][0] == pytest.approx(expected_a)
    assert doc["doc_topic_entropy_normalized"][1] == pytest.approx(0.5)
    assert doc["doc_topic_entropy_normalized"][2] == pytest.approx(1.0)


def test_document_topic_entropy_single_topic_is_nan_normalized() -> None:
    doc = document_topic_entropy(np.ones((3, 1)))
    assert np.allclose(doc["doc_topic_entropy"], 0.0)
    assert np.isnan(doc["doc_topic_entropy_normalized"]).all()


def test_topic_document_distribution_columns_sum_to_one_and_flag_empty() -> None:
    theta = np.asarray([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    distribution, empty = topic_document_distribution(theta)
    assert distribution.shape == (3, 2)
    assert np.allclose(distribution[0].sum(), 1.0)
    assert np.allclose(distribution[1], [1.0, 0.0])
    assert empty.tolist() == [False, False, True]
    assert np.isnan(distribution[2]).all()


def test_topic_document_entropy_uniform_spread_is_one_and_single_doc_is_zero() -> None:
    uniform = np.full((4, 5), 0.2)
    topic = topic_document_entropy(uniform)
    assert np.allclose(topic["topic_doc_entropy"], np.log(4))
    assert np.allclose(topic["topic_doc_entropy_normalized"], 1.0)

    concentrated = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    topic = topic_document_entropy(concentrated)
    # topic 1 lives in one document -> zero entropy; AlSumait background KL = ln D.
    assert topic["topic_doc_entropy"][1] == pytest.approx(0.0)
    assert np.log(3) - topic["topic_doc_entropy"][1] == pytest.approx(np.log(3))


def test_topic_document_entropy_hand_computed_example() -> None:
    topic = topic_document_entropy(THETA_EXAMPLE)
    column = THETA_EXAMPLE[:, 3]
    p = column / column.sum()
    expected = float(shannon_entropy(p)) / np.log(3)
    assert topic["topic_doc_entropy_normalized"][3] == pytest.approx(expected)
    # hand-computed: p = (0.0385, 0, 0.9615) -> H = 0.1631 nats
    assert topic["topic_doc_entropy"][3] == pytest.approx(0.1631, abs=1e-3)


def test_topic_document_entropy_single_document_is_nan_normalized() -> None:
    topic = topic_document_entropy(np.asarray([[0.4, 0.6]]))
    assert np.allclose(topic["topic_doc_entropy"], 0.0)
    assert np.isnan(topic["topic_doc_entropy_normalized"]).all()


def test_rank1_fraction_sums_to_one_and_breaks_ties_on_first_topic() -> None:
    rank1 = topic_rank1_doc_fraction(THETA_EXAMPLE)
    assert rank1.sum() == pytest.approx(1.0)
    # docs B and C tie; argmax picks the first topic.
    assert rank1.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])

    spread = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
    assert topic_rank1_doc_fraction(spread).tolist() == pytest.approx([0.5, 0.5])


def test_compute_entropy_metrics_handles_zero_mass_docs_and_empty_topics() -> None:
    theta = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    result = compute_entropy_metrics(theta)
    assert result.num_documents == 3
    assert result.num_topics == 3
    assert result.num_zero_mass_docs == 1
    assert result.num_empty_topics == 1
    assert not result.doc["valid"][1]
    assert np.isnan(result.doc["doc_topic_entropy"][1])
    # zero-mass doc is excluded from the document summary.
    expected_mean = np.mean([0.0, np.log(2) / np.log(3)])
    assert result.summary["doc_topic_entropy_normalized_mean"] == pytest.approx(
        expected_mean
    )
    # rank-1 fractions are computed over the two valid docs only.
    assert result.topic["topic_rank1_doc_fraction"].tolist() == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert result.topic["empty"].tolist() == [False, False, True]
    assert not np.isnan(result.summary["topic_doc_entropy_normalized_max"])
    for key in SUMMARY_METRIC_KEYS:
        assert key in result.summary


def test_topic_flags_use_strict_comparisons_and_handle_nan() -> None:
    topic = {
        "topic_doc_entropy_normalized": np.asarray([0.96, 0.95, 0.10, np.nan]),
        "topic_rank1_doc_fraction": np.asarray([0.00, 0.01, 0.50, 0.005]),
    }
    flags = topic_flags(topic)
    # strict > for diffuse: exactly at the threshold does not count; NaN never counts.
    assert flags["is_diffuse"].tolist() == [True, False, False, False]
    # strict < for dead: exactly at the threshold does not count.
    assert flags["is_dead"].tolist() == [True, False, False, True]


def test_topic_flags_honour_custom_thresholds() -> None:
    topic = {
        "topic_doc_entropy_normalized": np.asarray([0.90, 0.50]),
        "topic_rank1_doc_fraction": np.asarray([0.03, 0.20]),
    }
    flags = topic_flags(topic, diffuse_entropy_threshold=0.8, dead_rank1_threshold=0.05)
    assert flags["is_diffuse"].tolist() == [True, False]
    assert flags["is_dead"].tolist() == [True, False]


def test_diffuse_and_dead_fractions_denominator_is_all_topics() -> None:
    # topic 0 dominates every document; topic 1 is spread; topic 2 has no mass.
    theta = np.asarray(
        [
            [0.90, 0.10, 0.0],
            [0.90, 0.10, 0.0],
            [0.90, 0.10, 0.0],
            [0.90, 0.10, 0.0],
        ]
    )
    result = compute_entropy_metrics(theta)
    # both non-empty topics are perfectly uniform over documents -> normalized 1.0
    assert result.topic["is_diffuse"].tolist() == [True, True, False]
    # topic 0 wins every document; topics 1 and 2 never do.
    assert result.topic["is_dead"].tolist() == [False, True, True]
    assert result.summary["topic_diffuse_fraction"] == pytest.approx(2 / 3)
    assert result.summary["topic_dead_fraction"] == pytest.approx(2 / 3)
    assert result.diffuse_entropy_threshold == pytest.approx(0.95)
    assert result.dead_rank1_threshold == pytest.approx(0.01)


def test_empty_topic_is_dead_but_not_diffuse() -> None:
    theta = np.asarray([[0.6, 0.4, 0.0], [0.3, 0.7, 0.0]])
    result = compute_entropy_metrics(theta)
    assert result.num_empty_topics == 1
    assert not result.topic["is_diffuse"][2]
    assert result.topic["is_dead"][2]


def test_fractions_respond_to_custom_thresholds() -> None:
    # topic 0 is the argmax of every document -> rank-1 = [1.0, 0.0]
    theta = np.asarray([[0.7, 0.3], [0.6, 0.4], [0.55, 0.45]])
    strict = compute_entropy_metrics(theta, diffuse_entropy_threshold=0.999)
    loose = compute_entropy_metrics(theta, diffuse_entropy_threshold=0.5)
    assert strict.summary["topic_diffuse_fraction"] == pytest.approx(0.0)
    assert loose.summary["topic_diffuse_fraction"] == pytest.approx(1.0)

    default_dead = compute_entropy_metrics(theta)
    assert default_dead.summary["topic_dead_fraction"] == pytest.approx(0.5)
    # `is_dead` uses a strict `<`, so a topic that wins every document can never be
    # dead, whatever the threshold. Only the never-winning topic 1 is counted.
    for threshold in (0.99, 1.0):
        result = compute_entropy_metrics(theta, dead_rank1_threshold=threshold)
        assert result.summary["topic_dead_fraction"] == pytest.approx(0.5)
    # a lower threshold can only remove dead topics, never add them
    assert compute_entropy_metrics(theta, dead_rank1_threshold=0.0).summary[
        "topic_dead_fraction"
    ] == pytest.approx(0.0)


def test_compute_entropy_metrics_rejects_negative_or_non_2d_input() -> None:
    with pytest.raises(ValueError):
        compute_entropy_metrics(np.asarray([[0.5, -0.5]]))
    with pytest.raises(ValueError):
        compute_entropy_metrics(np.asarray([0.5, 0.5]))


def test_aggregate_metrics_uses_sample_std_and_ignores_nan() -> None:
    aggregated = aggregate_metrics(
        [{"a": 1.0, "b": float("nan")}, {"a": 3.0, "b": 2.0}, {"a": 2.0, "b": 4.0}],
        keys=["a", "b", "missing"],
    )
    assert aggregated["a"]["mean"] == pytest.approx(2.0)
    assert aggregated["a"]["std"] == pytest.approx(np.std([1.0, 3.0, 2.0], ddof=1))
    assert aggregated["b"]["mean"] == pytest.approx(3.0)
    assert aggregated["b"]["std"] == pytest.approx(np.std([2.0, 4.0], ddof=1))
    assert np.isnan(aggregated["missing"]["mean"])

    single = aggregate_metrics([{"a": 0.7}])
    assert single["a"] == {"mean": pytest.approx(0.7), "std": 0.0}

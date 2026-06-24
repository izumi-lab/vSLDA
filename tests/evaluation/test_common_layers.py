from __future__ import annotations

import numpy as np

from src.core.contracts import TopicModelOutput
from src.evaluation.features.doc_topic import get_doc_topic_features
from src.evaluation.features.sentence_topic import flatten_sentence_topic_features
from src.evaluation.features.topic_word_proxy import (
    build_topic_word_from_doc_topic,
    build_topic_word_from_sentence_topic,
)
from src.evaluation.metrics.diversity import topic_diversity
from src.evaluation.reports.latex_tables import format_mean_std


def test_get_doc_topic_features_row_normalizes_zero_rows() -> None:
    features = get_doc_topic_features(
        TopicModelOutput(doc_topic=np.array([[2.0, 2.0], [0.0, 0.0]]))
    )

    assert np.allclose(features, np.array([[0.5, 0.5], [0.0, 0.0]]))


def test_topic_word_proxy_builders_row_normalize() -> None:
    doc_topic = np.array([[1.0, 0.0], [0.0, 1.0]])
    bow = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]])
    sentence_topic = np.array([[1.0, 0.0], [0.5, 0.5]])

    doc_proxy = build_topic_word_from_doc_topic(doc_topic, bow)
    sent_proxy = build_topic_word_from_sentence_topic(sentence_topic, bow)

    assert np.allclose(doc_proxy.sum(axis=1), np.ones(2))
    assert np.allclose(sent_proxy.sum(axis=1), np.ones(2))


def test_flatten_sentence_topic_features_handles_list_inputs() -> None:
    flat = flatten_sentence_topic_features(
        [np.array([[1.0, 0.0]]), np.array([[0.25, 0.75], [0.5, 0.5]])]
    )

    assert flat.shape == (3, 2)


def test_topic_diversity_and_report_formatting() -> None:
    topic_word = np.array([[0.9, 0.8, 0.1], [0.7, 0.1, 0.6]])

    assert topic_diversity(topic_word, top_n=2) == 0.75
    assert format_mean_std(1.23456, 0.1, digits=2) == "1.23 +/- 0.10"

from __future__ import annotations

import numpy as np


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return matrix / row_sums


def build_topic_word_from_doc_topic(
    doc_topic: np.ndarray,
    doc_bow: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Build a topic-word proxy from document-topic features and document BOW."""

    topic_word = np.asarray(doc_topic).T @ np.asarray(doc_bow)
    if normalize:
        topic_word = _row_normalize(topic_word)
    return topic_word


def build_topic_word_from_sentence_topic(
    sentence_topic: np.ndarray,
    sentence_bow: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Build a topic-word proxy from sentence-topic features and sentence BOW."""

    topic_word = np.asarray(sentence_topic).T @ np.asarray(sentence_bow)
    if normalize:
        topic_word = _row_normalize(topic_word)
    return topic_word

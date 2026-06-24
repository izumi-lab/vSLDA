from __future__ import annotations

import numpy as np

from src.core.contracts import TopicModelOutput


def get_doc_topic_features(output: TopicModelOutput) -> np.ndarray:
    """Return row-normalized document-topic features."""

    doc_topic = np.asarray(output.doc_topic, dtype=np.float64)
    row_sums = doc_topic.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return doc_topic / row_sums

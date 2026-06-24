from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def flatten_sentence_topic_features(
    sentence_topic: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
    """Flatten per-document sentence-topic arrays into one 2D matrix."""

    if isinstance(sentence_topic, np.ndarray) and sentence_topic.dtype != object:
        arr = np.asarray(sentence_topic)
        if arr.ndim != 2:
            raise ValueError(f"sentence_topic must be 2D, got shape {arr.shape}")
        return arr

    arrays = [np.asarray(item) for item in sentence_topic]
    if not arrays:
        return np.zeros((0, 0), dtype=np.float64)
    num_topics = arrays[0].shape[1] if arrays[0].ndim == 2 else 0
    non_empty = [arr for arr in arrays if arr.size > 0]
    if not non_empty:
        return np.zeros((0, num_topics), dtype=np.float64)
    return np.vstack(non_empty)

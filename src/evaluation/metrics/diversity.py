from __future__ import annotations

import numpy as np


def topic_diversity(topic_word: np.ndarray, top_n: int = 10) -> float:
    """Compute topic diversity from top-N words per topic."""

    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    topic_word = np.asarray(topic_word)
    if topic_word.ndim != 2:
        raise ValueError(f"topic_word must be 2D, got shape {topic_word.shape}")
    if topic_word.size == 0:
        return 0.0
    top_n = min(int(top_n), topic_word.shape[1])
    top_words = np.argsort(-topic_word, axis=1)[:, :top_n]
    unique_words = np.unique(top_words)
    return float(len(unique_words) / top_words.size)

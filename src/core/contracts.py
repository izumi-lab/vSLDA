from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TopicModelOutput:
    """Common output format for topic models."""

    doc_topic: np.ndarray
    sentence_topic: np.ndarray | list[np.ndarray] | None = None
    topic_word: np.ndarray | None = None
    topic_embeddings: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    """Settings that uniquely identify one model run."""

    dataset_name: str
    model_name: str
    num_topics: int
    seed: int
    category: str | None = None
    iteration: int | None = None
    embedding_model: str | None = None

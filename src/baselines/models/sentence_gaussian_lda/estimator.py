from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.contracts import TopicModelOutput


@dataclass
class SentenceGaussianLDA:
    """Common-output adapter for sentence Gaussian LDA results."""

    doc_topic: np.ndarray
    sentence_topic: list[np.ndarray] | np.ndarray | None = None
    topic_embeddings: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_output(self) -> TopicModelOutput:
        return TopicModelOutput(
            doc_topic=self.doc_topic,
            sentence_topic=self.sentence_topic,
            topic_embeddings=self.topic_embeddings,
            metadata={
                "model_name": "sentence_gaussian_lda",
                **self.metadata,
            },
        )

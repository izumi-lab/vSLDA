from __future__ import annotations

import logging

import numpy as np

from src.core.progress import NullProgressReporter
from src.models.vmf_sentence_lda import VMFLDATrainer

FLOAT_ATOL = 1e-12
FLOAT_RTOL = 1e-12


class FixedEncoder:
    def __init__(self) -> None:
        raw = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.9, 0.1, 0.0],
            "c": [0.0, 1.0, 0.0],
            "d": [0.0, 0.9, 0.1],
            "e": [0.0, 0.0, 1.0],
            "f": [0.1, 0.0, 0.9],
        }
        self.vectors = {
            key: self._normalize(np.asarray(value, dtype=np.float64))
            for key, value in raw.items()
        }

    @staticmethod
    def _normalize(value: np.ndarray) -> np.ndarray:
        return value / np.linalg.norm(value)

    def encode(self, sentences) -> np.ndarray:
        if not sentences:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack([self.vectors[text] for text in sentences])

    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_vmf_sentence_lda_golden_output() -> None:
    np.random.seed(0)
    model = VMFLDATrainer(
        corpus=[["a", "b"], ["c", "d"], ["e", "f"]],
        encoder=FixedEncoder(),
        num_topics=3,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-golden"),
        progress=NullProgressReporter(),
    )

    model.sample(
        num_iterations=1,
        num_sweeps=2,
        num_samples=1,
        estimate_alpha=False,
    )

    assert [item.tolist() for item in model.topic_assignments] == [
        [1, 2],
        [0, 0],
        [0, 2],
    ]
    assert np.array_equal(
        model.topic_counts_per_doc.T,
        np.array([[0, 1, 1], [2, 0, 0], [1, 0, 1]], dtype=np.int32),
    )
    assert np.array_equal(model.topic_counts, np.array([3, 1, 2], dtype=np.int32))
    assert np.allclose(
        model.get_document_topic_distribution(),
        np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.5, 0.0, 0.5]]),
        atol=FLOAT_ATOL,
        rtol=FLOAT_RTOL,
    )
    assert np.allclose(
        model.topic_means,
        np.array(
            [
                [0.0, 0.87365115, 0.48655283],
                [1.0, 0.0, 0.0],
                [0.7412493, 0.07412493, 0.6671244],
            ],
            dtype=np.float32,
        ),
        atol=FLOAT_ATOL,
        rtol=FLOAT_RTOL,
    )
    assert np.allclose(
        model.kappa_per_topic,
        np.array(
            [4.372491095802893, 1.0, 4.091855419462651],
            dtype=np.float64,
        ),
        atol=FLOAT_ATOL,
        rtol=FLOAT_RTOL,
    )

    output = model.to_output()
    assert output.doc_topic.shape == (3, 3)
    assert output.topic_embeddings is model.topic_means
    assert output.metadata["model_name"] == "vmf_sentence_lda"

from __future__ import annotations

import numpy as np

from src.baselines.models.sentence_gaussian_lda import SentenceGaussianLDA


def test_sentence_gaussian_lda_subpackage_exports_common_output_adapter() -> None:
    estimator = SentenceGaussianLDA(
        doc_topic=np.eye(2),
        sentence_topic=[np.eye(2)],
        topic_embeddings=np.ones((2, 3)),
        metadata={"num_topics": 2},
    )

    output = estimator.to_output()

    assert output.doc_topic.shape == (2, 2)
    assert output.sentence_topic is not None
    assert output.topic_embeddings is not None
    assert output.metadata["model_name"] == "sentence_gaussian_lda"

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.contracts import RunSpec, TopicModelOutput
from src.core.io import load_topic_model_output, save_topic_model_output
from src.core.paths import ResultPathBuilder


def test_save_and_load_topic_model_output_with_doc_topic_only(tmp_path: Path) -> None:
    spec = RunSpec(
        dataset_name="dummy",
        model_name="vmf_sentence_lda",
        num_topics=2,
        seed=0,
    )
    builder = ResultPathBuilder(tmp_path)
    output = TopicModelOutput(
        doc_topic=np.array([[0.25, 0.75]], dtype=np.float64),
        metadata={"model_name": "vmf_sentence_lda"},
    )

    save_topic_model_output(output, spec, builder)
    loaded = load_topic_model_output(spec, builder)

    assert np.array_equal(loaded.doc_topic, output.doc_topic)
    assert loaded.topic_word is None
    assert loaded.sentence_topic is None
    assert builder.metadata_path(spec).exists()
    assert loaded.metadata["has_topic_word"] is False


def test_save_and_load_topic_model_output_with_optional_arrays(
    tmp_path: Path,
) -> None:
    spec = RunSpec(
        dataset_name="dummy",
        model_name="baseline",
        num_topics=2,
        seed=7,
        category="all",
        iteration=3,
    )
    builder = ResultPathBuilder(tmp_path)
    output = TopicModelOutput(
        doc_topic=np.eye(2),
        sentence_topic=np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float64),
        topic_embeddings=np.ones((2, 3), dtype=np.float64),
    )

    save_topic_model_output(output, spec, builder)
    loaded = load_topic_model_output(spec, builder)

    assert loaded.doc_topic.shape == (2, 2)
    assert loaded.sentence_topic is not None
    assert loaded.sentence_topic.shape == (2, 2)
    assert loaded.topic_embeddings is not None
    assert loaded.topic_embeddings.shape == (2, 3)
    assert loaded.topic_word is None

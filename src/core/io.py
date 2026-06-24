from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.core.contracts import RunSpec, TopicModelOutput
from src.core.paths import ResultPathBuilder


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _save_array(path: Path, value: np.ndarray | list[np.ndarray]) -> None:
    if isinstance(value, list):
        np.save(path, np.asarray(value, dtype=object), allow_pickle=True)
    else:
        np.save(path, value)


def _load_optional_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def save_topic_model_output(
    output: TopicModelOutput,
    spec: RunSpec,
    path_builder: ResultPathBuilder,
) -> None:
    """Save a topic model output in the common format."""

    run_dir = path_builder.run_dir(spec)
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(path_builder.doc_topic_path(spec), output.doc_topic)

    if output.sentence_topic is not None:
        _save_array(path_builder.sentence_topic_path(spec), output.sentence_topic)

    if output.topic_word is not None:
        np.save(path_builder.topic_word_path(spec), output.topic_word)

    if output.topic_embeddings is not None:
        np.save(path_builder.topic_embeddings_path(spec), output.topic_embeddings)

    metadata = {
        **output.metadata,
        "run_spec": asdict(spec),
        "has_sentence_topic": output.sentence_topic is not None,
        "has_topic_word": output.topic_word is not None,
        "has_topic_embeddings": output.topic_embeddings is not None,
    }
    save_json(metadata, path_builder.metadata_path(spec))


def load_topic_model_output(
    spec: RunSpec,
    path_builder: ResultPathBuilder,
) -> TopicModelOutput:
    """Load a topic model output saved in the common format."""

    metadata_path = path_builder.metadata_path(spec)
    metadata = load_json(metadata_path) if metadata_path.exists() else {}

    return TopicModelOutput(
        doc_topic=np.load(path_builder.doc_topic_path(spec)),
        sentence_topic=_load_optional_array(path_builder.sentence_topic_path(spec)),
        topic_word=_load_optional_array(path_builder.topic_word_path(spec)),
        topic_embeddings=_load_optional_array(path_builder.topic_embeddings_path(spec)),
        metadata=metadata,
    )

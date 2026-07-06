from __future__ import annotations

from typing import Sequence

import numpy as np

from src.data.preprocessing import PreprocessedDocument, SelectedCorpus


def require_selected_corpus(
    selection: SelectedCorpus | None,
    *,
    model_name: str,
    split: str,
) -> SelectedCorpus:
    if selection is None:
        raise ValueError(
            f"{model_name} {split} preprocessing selection is required for artifact "
            "persistence."
        )
    return selection


def validate_selected_artifact_alignment(
    *,
    model_name: str,
    split: str,
    doc_topic: np.ndarray,
    preprocessed: Sequence[PreprocessedDocument],
    selection: SelectedCorpus,
    sentence_topic_soft: Sequence[np.ndarray] | None = None,
) -> None:
    doc_topic_array = np.asarray(doc_topic)
    if doc_topic_array.ndim < 1:
        raise ValueError(f"{model_name} {split} doc-topic artifact must be an array.")
    if doc_topic_array.shape[0] != len(preprocessed):
        raise ValueError(
            f"{model_name} {split} doc-topic rows ({doc_topic_array.shape[0]}) do "
            f"not match preprocessed documents ({len(preprocessed)})."
        )
    if len(selection.documents) != len(preprocessed):
        raise ValueError(
            f"{model_name} {split} selected documents ({len(selection.documents)}) "
            f"do not match preprocessed documents ({len(preprocessed)})."
        )
    if len(selection.raw_doc_indices) != len(preprocessed):
        raise ValueError(
            f"{model_name} {split} raw_doc_indices ({len(selection.raw_doc_indices)}) "
            f"do not match preprocessed documents ({len(preprocessed)})."
        )
    if len(selection.sentence_indices_by_doc) != len(preprocessed):
        raise ValueError(
            f"{model_name} {split} sentence index rows "
            f"({len(selection.sentence_indices_by_doc)}) do not match preprocessed "
            f"documents ({len(preprocessed)})."
        )
    try:
        [int(index) for index in selection.raw_doc_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model_name} {split} raw_doc_indices must contain integer-compatible "
            "values."
        ) from exc
    if sentence_topic_soft is not None and len(sentence_topic_soft) != len(
        preprocessed
    ):
        raise ValueError(
            f"{model_name} {split} sentence-topic rows ({len(sentence_topic_soft)}) "
            f"do not match preprocessed documents ({len(preprocessed)})."
        )


def validate_no_additional_document_drop(
    *,
    model_name: str,
    split: str,
    selected_count: int,
    model_input_count: int,
) -> None:
    if model_input_count != selected_count:
        raise ValueError(
            f"{model_name} {split} model input filtering changed selected document "
            f"count from {selected_count} to {model_input_count}."
        )

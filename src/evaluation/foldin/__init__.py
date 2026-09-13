"""Collapsed fold-in document-topic distributions for vMF Sentence LDA runs."""

from src.evaluation.foldin.theta import (
    DOC_TOPIC_FILENAME,
    FOLDIN_ASSIGNMENT,
    META_FILENAME,
    SENTENCE_TOPIC_FILENAME,
    FoldInRunResult,
    compute_foldin_for_run,
    foldin_theta_from_posterior,
    write_foldin_artifacts,
)

__all__ = [
    "DOC_TOPIC_FILENAME",
    "FOLDIN_ASSIGNMENT",
    "META_FILENAME",
    "SENTENCE_TOPIC_FILENAME",
    "FoldInRunResult",
    "compute_foldin_for_run",
    "foldin_theta_from_posterior",
    "write_foldin_artifacts",
]

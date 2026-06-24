from __future__ import annotations

from .catalog import DATASET_TARGETS, get_dataset_targets, resolve_dataset_dir
from .corpus import load_corpus, load_preprocessed_corpus, split_document
from .datasets import DatasetRun, resolve_dataset_categories, resolve_dataset_runs
from .newsgroups import prepare_20newsgroups
from .newsgroups_subset import create_20newsgroups_subset
from .nyt import prepare_nyt
from .preprocessing import (
    PreprocessedCorpus,
    PreprocessedDocument,
    preprocess_document,
    preprocess_documents,
)
from .segmentation import segment_text
from .splits import (
    load_dataset_split,
    load_filtered_split_labels,
    load_filtered_split_texts,
)

__all__ = [
    "DATASET_TARGETS",
    "DatasetRun",
    "create_20newsgroups_subset",
    "get_dataset_targets",
    "load_corpus",
    "load_dataset_split",
    "load_filtered_split_labels",
    "load_filtered_split_texts",
    "load_preprocessed_corpus",
    "PreprocessedCorpus",
    "PreprocessedDocument",
    "prepare_20newsgroups",
    "prepare_nyt",
    "preprocess_document",
    "preprocess_documents",
    "resolve_dataset_categories",
    "resolve_dataset_dir",
    "resolve_dataset_runs",
    "segment_text",
    "split_document",
]

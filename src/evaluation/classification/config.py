from __future__ import annotations

"""Shared configuration for classification evaluation modules."""

from pathlib import Path
from typing import Dict, List, Optional

from src.core.paths import (
    BASELINE_RESULTS_ROOT,
    CLASSIFICATION_RESULTS_ROOT,
)
from src.data.catalog import (
    DATASET_TARGETS,
)
from src.data.catalog import get_dataset_targets as _get_dataset_targets
from src.data.catalog import resolve_dataset_dir as _resolve_dataset_dir
from src.data.catalog import resolve_dataset_name as _resolve_dataset_name

BASELINE_ROOT = BASELINE_RESULTS_ROOT
RESULT_ROOT = CLASSIFICATION_RESULTS_ROOT
TARGETS: Dict[str, Dict[str, List[str]]] = DATASET_TARGETS

MODEL_NAMES = [
    "Blei LDA",
    "sentLDA",
    "Gaussian k-means",
    "Spherical k-means",
    "Gaussian mixture",
    "movMF",
    "Gaussian LDA",
    "MvTM",
    "ETM",
    "Contextual TM",
    "SenClu",
    "BERTopic (UMAP + k-means)",
    "Sentence LDA",
    "vMF Sentence LDA",
]

MODEL_TABLE_LABELS = {
    "Blei LDA": "LDA",
    "sentLDA": "SLDA",
    "Gaussian k-means": "GCLU",
    "Spherical k-means": "SCLU",
    "Gaussian mixture": "MGCLU",
    "movMF": "MSCLU",
    "Gaussian LDA": "GLDA",
    "MvTM": "vLDA",
    "ETM": "ETM",
    "Contextual TM": "CTM",
    "SenClu": "SenClu",
    "BERTopic (UMAP + k-means)": "BERTopic",
    "Sentence LDA": "GSLDA",
    "vMF Sentence LDA": "vSLDA",
}

ALIGNMENT_MODES = ("intersection", "strict_skip")
DEFAULT_ALIGNMENT_MODE = "intersection"
FEATURE_RESOLVE_MODES = ("all", "strict")
DEFAULT_FEATURE_RESOLVE_MODE = "all"


def model_table_label(model_name: str) -> str:
    base_name = str(model_name).split(" [", 1)[0]
    return MODEL_TABLE_LABELS.get(base_name, str(model_name))


def normalize_model_selector(value: str) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def model_matches_selector(
    model_name: str,
    selector: str,
    *,
    model_key: str | None = None,
) -> bool:
    normalized_selector = normalize_model_selector(selector)
    if not normalized_selector:
        return False
    base_name = str(model_name).split(" [", 1)[0]
    candidates = {
        str(model_name),
        base_name,
        model_table_label(model_name),
    }
    if model_key is not None:
        candidates.add(str(model_key))
    return normalized_selector in {
        normalize_model_selector(candidate) for candidate in candidates
    }


def get_dataset_dir(name: str) -> Optional[Path]:
    return _resolve_dataset_dir(name)


def resolve_dataset_name(name: str) -> Optional[str]:
    return _resolve_dataset_name(name)


def get_dataset_targets(
    dataset: str,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> Optional[Dict[str, List[str]]]:
    return _get_dataset_targets(
        dataset,
        target_column=target_column,
        label_schema=label_schema,
    )

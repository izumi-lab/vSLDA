from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from src.core.artifacts import (
    METADATA_FILENAME,
    load_artifact_json,
    load_artifact_pickle,
)
from src.core.errors import MissingArtifactError
from src.core.paths import (
    RESULTS_ROOT,
    build_baseline_doc_topic_path,
    build_vmf_doc_topic_path,
    resolve_baseline_condition_dir,
    resolve_vmf_experiment_dir,
)
from src.evaluation.model_provenance import load_model_provenance

ModelType = Literal[
    "vmf",
    "gaussian",
    "sentence_gaussianlda",
    "sentlda",
    "bertopic_kmeans",
    "bleilda",
    "gaussianlda",
    "etm",
    "mvtm",
    "ctm",
    "senclu",
    "spherical_kmeans",
    "gaussian_kmeans",
    "movmf",
    "gaussian_mixture",
]

MODEL_ALIASES = {"gaussian": "sentence_gaussianlda"}
MODEL_CHOICES = [
    "vmf",
    "sentlda",
    "sentence_gaussianlda",
]
ANALYSIS_ROOT = RESULTS_ROOT / "topic_analysis"
DEFAULT_OUT_ROOT = ANALYSIS_ROOT / "coherence"
DEFAULT_EMBEDDING_VARIANT = "mpnet"
EMBEDDING_VARIANT_MODELS = {
    "vmf",
    "sentence_gaussianlda",
}


def effective_embedding_variant(
    model: str,
    embedding_variant: str | None,
) -> str | None:
    if model not in EMBEDDING_VARIANT_MODELS:
        return None
    if embedding_variant is None:
        return None
    variant = str(embedding_variant).strip()
    if not variant:
        return None
    if model == "sentence_gaussianlda" and not variant.endswith(("_raw", "_norm")):
        return f"{variant}_raw"
    return variant


def normalize_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def build_result_dir(
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> Path:
    if model == "vmf":
        return resolve_vmf_experiment_dir(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            run_name=data_run,
            embedding_variant=effective_embedding_variant(model, embedding_variant),
        )
    raise ValueError(f"Unsupported model in vmf result path resolver: '{model}'.")


def build_baseline_param_dir(
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> Path:
    if model not in {
        "bleilda",
        "ctm",
        "gaussianlda",
        "etm",
        "mvtm",
        "sentence_gaussianlda",
        "sentlda",
        "bertopic_kmeans",
        "senclu",
        "spherical_kmeans",
        "gaussian_kmeans",
        "movmf",
        "gaussian_mixture",
    }:
        raise ValueError(f"Unsupported model in baseline path resolver: '{model}'.")
    return (
        resolve_baseline_condition_dir(
            model=model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            embedding_variant=effective_embedding_variant(model, embedding_variant),
        )
        / "params"
    )


def resolve_model_provenance(
    *,
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> dict[str, object]:
    if model == "vmf":
        return load_model_provenance(
            build_result_dir(
                model=model,
                dataset=dataset,
                iteration=iteration,
                num_topics=num_topics,
                category=category,
                data_run=data_run,
                embedding_variant=embedding_variant,
            ),
            model_key="vmf_sentence_lda",
        )
    return load_model_provenance(
        build_baseline_param_dir(
            model=model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            embedding_variant=embedding_variant,
        ),
        model_key=model,
    )


def _find_metadata_path(result_dir: Path) -> Path:
    for candidate in [result_dir, *result_dir.parents[:3]]:
        metadata_path = candidate / METADATA_FILENAME
        if metadata_path.exists():
            return metadata_path
    return result_dir / METADATA_FILENAME


def _load_result_metadata(
    *,
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    embedding_variant: str | None = None,
) -> dict[str, object]:
    if model == "vmf":
        result_dir = build_result_dir(
            model,
            dataset,
            iteration,
            num_topics,
            category,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
    else:
        try:
            result_dir = build_baseline_param_dir(
                model,
                dataset,
                iteration,
                num_topics,
                category,
                data_run=data_run,
                embedding_variant=embedding_variant,
            )
        except MissingArtifactError:
            return {}
    metadata_path = _find_metadata_path(result_dir)
    if not metadata_path.exists():
        return {}
    payload = load_artifact_json(metadata_path)
    return payload if isinstance(payload, dict) else {}


def resolve_split_csvs_and_target_column(
    *,
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
) -> tuple[tuple[str, ...] | None, str]:
    payload = _load_result_metadata(
        model=model,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        embedding_variant=embedding_variant,
    )
    key = "train_csvs" if split == "train" else "test_csvs"
    raw_paths = payload.get(key)
    target_column = str(payload.get("target_column") or "target_str")
    if not isinstance(raw_paths, (list, tuple)):
        return None, target_column
    csv_paths = tuple(str(path) for path in raw_paths if str(path).strip())
    return (csv_paths or None), target_column


def load_doc_topics(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    prefer_soft: bool = False,
    embedding_variant: str | None = None,
) -> np.ndarray:
    if model == "vmf":
        result_dir = build_result_dir(
            model,
            dataset,
            iteration,
            num_topics,
            category,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
        soft_path = build_vmf_doc_topic_path(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            assignment="soft",
            condition_id=result_dir.name,
            run_name=data_run,
        )
        hard_path = build_vmf_doc_topic_path(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            assignment="hard",
            condition_id=result_dir.name,
            run_name=data_run,
        )
        path = soft_path if prefer_soft and soft_path.exists() else hard_path
    else:
        if split != "test":
            raise ValueError(
                f"Model '{model}' only provides doc-topic distributions on the test split."
            )
        baseline_param_dir = build_baseline_param_dir(
            model,
            dataset,
            iteration,
            num_topics,
            category,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
        soft_path = build_baseline_doc_topic_path(
            model=model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            prefer_soft=True,
            condition_id=baseline_param_dir.parent.name,
            data_run=data_run,
        )
        hard_path = build_baseline_doc_topic_path(
            model=model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            prefer_soft=False,
            condition_id=baseline_param_dir.parent.name,
            data_run=data_run,
        )
        assert soft_path is not None
        assert hard_path is not None
        path = soft_path if prefer_soft and soft_path.exists() else hard_path

    arr = np.asarray(load_artifact_pickle(path), dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D doc-topic array, got shape {arr.shape}")
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return arr / row_sums


def aggregate_doc_topics_from_sentence_topics(
    sentence_topics_by_doc: list[np.ndarray],
    num_topics: int,
) -> np.ndarray:
    doc_topics = np.zeros((len(sentence_topics_by_doc), num_topics), dtype=np.float64)
    for d, sent_probs in enumerate(sentence_topics_by_doc):
        probs = np.asarray(sent_probs, dtype=np.float64)
        if probs.size == 0:
            continue
        if probs.ndim != 2 or probs.shape[1] != num_topics:
            raise ValueError(
                f"Invalid sentence-topic shape at doc {d}: {probs.shape}, expected (*, {num_topics})"
            )
        topic_sum = probs.sum(axis=0)
        s = topic_sum.sum()
        if s > 0.0:
            topic_sum /= s
        doc_topics[d] = topic_sum
    return doc_topics


def load_doc_topics_proxy_soft_preferred(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
) -> np.ndarray:
    try:
        return load_doc_topics(
            model=model,
            dataset=dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            prefer_soft=True,
            embedding_variant=embedding_variant,
        )
    except FileNotFoundError:
        sentence_topics_by_doc = load_sentence_topics(
            model=model,
            dataset=dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            embedding_variant=embedding_variant,
        )
        return aggregate_doc_topics_from_sentence_topics(
            sentence_topics_by_doc=sentence_topics_by_doc,
            num_topics=num_topics,
        )


def resolve_sentence_topics_path(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
) -> Path:
    if model == "vmf":
        result_dir = build_result_dir(
            model,
            dataset,
            iteration,
            num_topics,
            category,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
        return result_dir / f"sentence_topic_{split}_soft.pkl"
    condition_dir = resolve_baseline_condition_dir(
        model=model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=effective_embedding_variant(model, embedding_variant),
    )
    split_dir = "infer" if split == "test" else "params"
    return condition_dir / split_dir / f"{category}_sentence_topic_soft.pkl"


def resolve_preprocessed_corpus_path(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
) -> Path:
    if model == "vmf":
        raise ValueError("vmf does not persist baseline preprocessed corpora.")
    condition_dir = resolve_baseline_condition_dir(
        model=model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=effective_embedding_variant(model, embedding_variant),
    )
    split_dir = "infer" if split == "test" else "params"
    return condition_dir / split_dir / "preprocessed_corpus.pkl"


def load_sentence_topics(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
) -> list[np.ndarray]:
    path = resolve_sentence_topics_path(
        model=model,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        embedding_variant=embedding_variant,
    )
    raw = load_artifact_pickle(path)
    if isinstance(raw, np.ndarray):
        if raw.ndim != 3:
            raise ValueError(
                f"Expected 3D ndarray for sentence topics, got shape {raw.shape} at {path}"
            )
        seq: list[object] = [raw[i] for i in range(raw.shape[0])]
    elif isinstance(raw, (list, tuple)):
        seq = list(raw)
    else:
        raise ValueError(
            "Unsupported sentence-topic format. Expected list/tuple of 2D arrays "
            f"or a 3D ndarray, got {type(raw)} at {path}."
        )

    normalized: list[np.ndarray] = []
    for doc_idx, item in enumerate(seq):
        item_arr = np.asarray(item)
        if item_arr.size == 0:
            normalized.append(np.zeros((0, num_topics), dtype=np.float64))
            continue
        if item_arr.ndim == 1 and np.issubdtype(item_arr.dtype, np.integer):
            topic_ids = item_arr.astype(np.int64)
            if np.any(topic_ids < 0) or np.any(topic_ids >= num_topics):
                raise ValueError(
                    f"Invalid topic id in sentence topics at doc {doc_idx}: {topic_ids}"
                )
            probs = np.zeros((topic_ids.shape[0], num_topics), dtype=np.float64)
            probs[np.arange(topic_ids.shape[0]), topic_ids] = 1.0
            normalized.append(probs)
            continue
        probs = np.asarray(item, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        if probs.ndim != 2:
            raise ValueError(
                f"Expected 2D sentence-topic matrix for doc {doc_idx}, got {probs.shape}"
            )
        if probs.shape[1] != num_topics:
            raise ValueError(
                f"Sentence-topic width {probs.shape[1]} != num_topics {num_topics} "
                f"(doc {doc_idx}, file={path})"
            )
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        normalized.append(probs / row_sums)
    return normalized

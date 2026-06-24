from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.core.artifacts import (
    VMF_METRICS_FILENAME,
    VMF_PARAMS_FILENAME,
    save_json,
    save_pickles,
)
from src.utils.embedding_preprocess import EmbeddingPreprocessor


@dataclass(frozen=True)
class VMFModelArtifactPayload:
    params: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class VMFRunOutputPayload:
    artifacts: Mapping[str, Any]
    metrics: Mapping[str, Any]


def build_vmf_model_artifact_payload(
    *,
    average_ll: list[float],
    iteration_diagnostics: list[dict[str, Any]],
    embedding_cache: Mapping[str, Any],
    alpha: np.ndarray,
    num_topics: int,
    kappa_default: float,
    num_components: int,
    pre_normalize_transform: str,
    whitening_eps: float,
    algorithm_variant: str | None,
    topic_counts: np.ndarray,
    topic_counts_per_doc: np.ndarray,
    topic_means: np.ndarray,
    sum_topic_vectors: np.ndarray,
    kappa_per_topic: np.ndarray,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    embedding_preprocessor: EmbeddingPreprocessor,
) -> VMFModelArtifactPayload:
    arrays: dict[str, np.ndarray] = {
        "topic_counts": topic_counts,
        "topic_counts_per_doc": topic_counts_per_doc.T,
        "topic_means": topic_means,
        "sum_topic_vectors": sum_topic_vectors,
        "kappa_per_topic": kappa_per_topic,
        "mixture_weights": mixture_weights,
        "component_means": component_means,
    }
    if embedding_preprocessor.mean_ is not None:
        arrays["embedding_transform_mean"] = embedding_preprocessor.mean_
    if embedding_preprocessor.whitening_matrix_ is not None:
        arrays["embedding_transform_whitening_matrix"] = (
            embedding_preprocessor.whitening_matrix_
        )

    return VMFModelArtifactPayload(
        params={
            "average_ll": list(average_ll),
            "iteration_diagnostics": [dict(item) for item in iteration_diagnostics],
            "embedding_cache": dict(embedding_cache),
            "alpha": np.asarray(alpha, dtype=np.float64).tolist(),
            "num_topics": int(num_topics),
            "kappa_default": float(kappa_default),
            "num_components": int(num_components),
            "pre_normalize_transform": pre_normalize_transform,
            "whitening_eps": float(whitening_eps),
            "algorithm_variant": algorithm_variant,
        },
        arrays=arrays,
    )


def save_vmf_model_artifacts(
    payload: VMFModelArtifactPayload, output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / VMF_PARAMS_FILENAME
    save_json(dict(payload.params), params_path)
    saved_arrays = save_pickles(payload.arrays, output_dir)
    return {
        "params": params_path,
        **saved_arrays,
    }


def build_vmf_run_output_payload(
    *,
    theta_train: np.ndarray,
    theta_test: np.ndarray,
    theta_train_soft: np.ndarray,
    theta_test_soft: np.ndarray,
    sentence_topic_train_soft: list[np.ndarray],
    sentence_topic_test_soft: list[np.ndarray],
    train_preprocessed: Any | None = None,
    test_preprocessed: Any | None = None,
    counts_train: np.ndarray | None,
    metrics: Mapping[str, Any],
    embedding_cache: Mapping[str, Any] | None = None,
) -> VMFRunOutputPayload:
    artifacts: dict[str, Any] = {
        "doc_topic_train": theta_train,
        "doc_topic_test": theta_test,
        "doc_topic_train_soft": theta_train_soft,
        "doc_topic_test_soft": theta_test_soft,
        "sentence_topic_train_soft": sentence_topic_train_soft,
        "sentence_topic_test_soft": sentence_topic_test_soft,
    }
    if train_preprocessed is not None:
        artifacts["train_preprocessed"] = train_preprocessed
    if test_preprocessed is not None:
        artifacts["test_preprocessed"] = test_preprocessed
    if counts_train is not None:
        artifacts["table_counts_per_doc"] = counts_train
    metric_payload = dict(metrics)
    if embedding_cache is not None:
        metric_payload["embedding_cache"] = dict(embedding_cache)
    return VMFRunOutputPayload(artifacts=artifacts, metrics=metric_payload)


def save_vmf_run_outputs(
    payload: VMFRunOutputPayload, output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_artifacts = save_pickles(payload.artifacts, output_dir)
    metrics_path = output_dir / VMF_METRICS_FILENAME
    save_json(dict(payload.metrics), metrics_path)
    return {
        **saved_artifacts,
        "metrics_path": metrics_path,
    }

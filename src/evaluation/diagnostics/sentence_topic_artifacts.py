from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.artifacts import load_artifact_pickle, load_json
from src.core.paths import (
    RESULTS_ROOT,
    resolve_baseline_condition_dir,
)
from src.core.paths import resolve_vmf_experiment_dir as resolve_vmf_condition_dir


def resolve_vmf_experiment_dir(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    return resolve_vmf_condition_dir(
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        run_name=data_run,
        condition_id=condition_id,
        num_components=num_components,
        embedding_variant=embedding_variant,
        dataset_root=results_root / "experiments" / dataset,
    )


def resolve_sentence_gaussian_dir(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    return (
        resolve_baseline_condition_dir(
            model="sentence_gaussianlda",
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            condition_id=condition_id,
            num_components=num_components,
            embedding_variant=embedding_variant,
            baseline_root=results_root / "baselines",
        )
        / "params"
    )


def load_topic_means(exp_dir: Path) -> np.ndarray:
    return np.asarray(load_artifact_pickle(exp_dir / "topic_means.pkl"), dtype=float)


def load_doc_topics(exp_dir: Path, *, split: str = "train") -> np.ndarray:
    return np.asarray(
        load_artifact_pickle(exp_dir / f"doc_topic_{split}.pkl"), dtype=float
    )


def load_average_ll(exp_dir: Path) -> list[float]:
    params_path = exp_dir / "params.json"
    if not params_path.exists():
        return []
    params = load_json(params_path)
    average_ll = params.get("average_ll", [])
    return [float(value) for value in average_ll]


def load_vmf_params(
    exp_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | float]:
    kappa_per_topic = load_artifact_pickle(exp_dir / "kappa_per_topic.pkl")
    mixture_weights = load_artifact_pickle(exp_dir / "mixture_weights.pkl")
    component_means = load_artifact_pickle(exp_dir / "component_means.pkl")
    topic_counts = load_artifact_pickle(exp_dir / "topic_counts.pkl")
    params = load_json(exp_dir / "params.json")

    alpha_raw = params.get("alpha", 0.0)
    if isinstance(alpha_raw, list):
        alpha: np.ndarray | float = np.asarray(alpha_raw, dtype=float)
    else:
        alpha = float(alpha_raw)

    return (
        np.asarray(kappa_per_topic, dtype=float),
        np.asarray(mixture_weights, dtype=float),
        np.asarray(component_means, dtype=float),
        np.asarray(topic_counts, dtype=float),
        alpha,
    )


def load_gaussian_params(
    gaussian_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray(
        load_artifact_pickle(gaussian_dir / "table_means.pkl"), dtype=float
    )
    cholesky = np.asarray(
        load_artifact_pickle(gaussian_dir / "table_cholesky_ltriangular_mat.pkl"),
        dtype=float,
    )
    log_determinants = np.asarray(
        load_artifact_pickle(gaussian_dir / "log_determinants.pkl"),
        dtype=float,
    )
    return means, cholesky, log_determinants

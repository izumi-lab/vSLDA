from __future__ import annotations

from src.core.runner_contracts import MODEL_KIND_CLUSTERING, MODEL_KIND_TOPIC_MODEL

CLUSTERING_RUNNER_KEYS = frozenset(
    {
        "bertopic_kmeans",
        "spherical_kmeans",
        "gaussian_kmeans",
        "movmf",
        "gaussian_mixture",
    }
)


def baseline_method_kind(runner_key: str) -> str:
    normalized = str(runner_key).strip().lower()
    if normalized in CLUSTERING_RUNNER_KEYS:
        return MODEL_KIND_CLUSTERING
    return MODEL_KIND_TOPIC_MODEL


def is_clustering_runner(runner_key: str) -> bool:
    return baseline_method_kind(runner_key) == MODEL_KIND_CLUSTERING

from __future__ import annotations

from pathlib import Path

from src.core.result_identity import build_condition_id
from src.evaluation.reporting import write_evaluation_json
from src.evaluation.word_based.topic_word_metrics import (
    EPSILON_SMOOTHED_COHERENCES,
    PALMETTO_CV_IMPLEMENTATION,
    PMI_SMOOTHING_EPSILON,
)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_output_condition_id(
    *,
    model: str,
    dataset: str,
    data_run: str,
    category: str,
    iterations: list[int],
    num_topics: int | list[int],
    coherence: str,
    coherences: list[str] | None = None,
    coherence_topn: int,
    coherence_window_size: int | dict[str, int | None] | None,
    coherence_implementation: str | dict[str, str] | None,
    coherence_min_window_count: int | dict[str, int | None] | None,
    coherence_reference: str,
    coherence_reference_path: str | None,
    coherence_reference_format: str | None,
    coherence_reference_max_docs: int | None,
    coherence_reference_min_doc_tokens: int,
    coherence_reference_streaming: bool,
    diversity_topn: int,
    coherence_split: str,
    topic_word_source: str,
    proxy_npmi_mode: str,
    proxy_word_score_mode: str,
    embedding_variant: str | None,
    metric_names: list[str],
) -> tuple[str, str]:
    return build_condition_id(
        iteration=int(min(iterations)),
        num_topics=int(num_topics),
        fingerprint_payload={
            "task": "word_based_metrics",
            "model": model,
            "dataset": dataset,
            "data_run": data_run,
            "category": category,
            "iterations": [int(value) for value in iterations],
            "num_topics": int(num_topics),
            "coherence": coherence,
            "coherences": list(coherences) if coherences is not None else None,
            "coherence_implementation": coherence_implementation,
            "coherence_topn": int(coherence_topn),
            "coherence_window_size": coherence_window_size,
            "coherence_min_window_count": coherence_min_window_count,
            "coherence_pmi_smoothing_epsilon": (
                PMI_SMOOTHING_EPSILON
                if coherence in EPSILON_SMOOTHED_COHERENCES or coherence == "c_v"
                else None
            ),
            "coherence_reference": coherence_reference,
            "coherence_reference_path": coherence_reference_path,
            "coherence_reference_format": coherence_reference_format,
            "coherence_reference_max_docs": coherence_reference_max_docs,
            "coherence_reference_min_doc_tokens": int(
                coherence_reference_min_doc_tokens
            ),
            "coherence_reference_streaming": bool(coherence_reference_streaming),
            "diversity_topn": int(diversity_topn),
            "coherence_split": coherence_split,
            "topic_word_source": topic_word_source,
            "proxy_npmi_mode": proxy_npmi_mode,
            "proxy_word_score_mode": proxy_word_score_mode,
            "embedding_variant": embedding_variant,
            "metric_names": list(metric_names),
        },
        extra_labels=[
            model,
            *(
                ["palmetto-cv"]
                if coherence_implementation == PALMETTO_CV_IMPLEMENTATION
                else []
            ),
            *([embedding_variant] if embedding_variant else []),
        ],
        include_fingerprint=False,
    )


def round_sigfigs(value: float, sig: int = 4) -> float:
    import numpy as np

    if np.isnan(value) or np.isinf(value):
        return float(value)
    return float(f"{value:.{sig}g}")


def write_summary_outputs(
    *,
    out_root: Path,
    summary_rows: list[dict[str, str | float]],
    dataset: str,
    data_runs: list[str],
    num_topics: int,
    iterations: list[int],
    coherence_metric: str,
    metric_names: list[str],
    summary_provenance: list[dict[str, object]],
) -> Path:
    """Deprecated no-op for root-level word-based summary files.

    The root-level summary_metrics.{csv,json} files were batch-local snapshots,
    so partial reruns and skip-only runs could make them misleading. The
    canonical outputs are the per-condition archive artifacts and latest
    CURRENT.json pointers.
    """
    _ = (
        summary_rows,
        dataset,
        data_runs,
        num_topics,
        iterations,
        coherence_metric,
        metric_names,
        summary_provenance,
    )
    return out_root

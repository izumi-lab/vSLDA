from __future__ import annotations

import json
from pathlib import Path

from src.baselines.params import (
    format_covariance_variant,
    format_prior_scale_variant,
    normalize_covariance_type,
)
from src.core.result_identity import build_condition_id
from src.core.vmf_variant import is_vmf_parameter_variant
from src.evaluation.reporting import read_evaluation_json
from src.evaluation.word_based.resumability import atomic_save_json
from src.evaluation.word_based.topic_word_metrics import (
    EPSILON_SMOOTHED_COHERENCES,
    PALMETTO_CV_IMPLEMENTATION,
    PMI_SMOOTHING_EPSILON,
)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


DEFAULT_DICT_NO_ABOVE = 0.7


def _evaluation_vocabulary_restriction(
    *,
    dict_no_above: float | None,
    reference_min_df: int,
    reference_max_df_ratio: float,
) -> dict[str, object] | None:
    """Describe the opt-in V_eval restriction, or None when nothing restricts."""

    restricts_reference = (
        int(reference_min_df) > 0 or float(reference_max_df_ratio) < 1.0
    )
    relaxes_no_above = (
        dict_no_above is not None and float(dict_no_above) != DEFAULT_DICT_NO_ABOVE
    )
    if not restricts_reference and not relaxes_no_above:
        return None
    return {
        "reference_min_df": int(reference_min_df),
        "reference_max_df_ratio": float(reference_max_df_ratio),
        "dict_no_above": (None if dict_no_above is None else float(dict_no_above)),
    }


def _vocabulary_restriction_label(restriction: dict[str, object]) -> str:
    minimum = int(restriction["reference_min_df"])  # type: ignore[arg-type]
    ratio = float(restriction["reference_max_df_ratio"])  # type: ignore[arg-type]
    return f"refdf{minimum}-{int(round(ratio * 100))}"


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
    embedding_variant: str | None,
    metric_names: list[str],
    dict_exclude_tokens: frozenset[str] = frozenset(),
    posterior_settings: dict[str, object] | None = None,
    topic_word_score_mode: str | None = None,
    topic_word_ranking_schema_version: int | None = None,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    source_condition_id: str | None = None,
    source_condition_fingerprint: str | None = None,
    parameter_variant: str | None = None,
    dict_no_above: float | None = None,
    reference_min_df: int = 0,
    reference_max_df_ratio: float = 1.0,
) -> tuple[str, str]:
    # The reference-frequency band is opt-in. When it is inactive the payload is
    # left untouched so condition ids stay byte-identical to historical runs.
    vocabulary_restriction = _evaluation_vocabulary_restriction(
        dict_no_above=dict_no_above,
        reference_min_df=reference_min_df,
        reference_max_df_ratio=reference_max_df_ratio,
    )
    fingerprint_payload: dict[str, object] = {
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
        "coherence_reference_min_doc_tokens": int(coherence_reference_min_doc_tokens),
        "coherence_reference_streaming": bool(coherence_reference_streaming),
        "diversity_topn": int(diversity_topn),
        "coherence_split": coherence_split,
        "topic_word_source": topic_word_source,
        "topic_word_score_mode": topic_word_score_mode,
        "topic_word_ranking_schema_version": topic_word_ranking_schema_version,
        "embedding_variant": embedding_variant,
        "metric_names": list(metric_names),
        "dict_exclude_tokens": sorted(dict_exclude_tokens),
        "posterior_settings": posterior_settings,
        "prior_scale": prior_scale,
        "source_condition_id": source_condition_id,
        "source_condition_fingerprint": source_condition_fingerprint,
        "parameter_variant": parameter_variant,
    }
    # Only a reduced covariance type enters the payload, keeping historical ids intact.
    covariance_label = format_covariance_variant(covariance_type)
    if covariance_label is not None:
        fingerprint_payload["covariance_type"] = normalize_covariance_type(
            covariance_type
        )
    if vocabulary_restriction is not None:
        fingerprint_payload["evaluation_vocabulary_restriction"] = (
            vocabulary_restriction
        )
    return build_condition_id(
        iteration=int(min(iterations)),
        num_topics=int(num_topics),
        fingerprint_payload=fingerprint_payload,
        extra_labels=[
            model,
            *(
                ["palmetto-cv"]
                if coherence_implementation == PALMETTO_CV_IMPLEMENTATION
                else []
            ),
            *([embedding_variant] if embedding_variant else []),
            *(
                [format_prior_scale_variant(prior_scale)]
                if prior_scale is not None
                else []
            ),
            *([covariance_label] if covariance_label is not None else []),
            # vMF hyperparameter-sweep runs carry their label (kappa0-100, ...) so the
            # evaluation directory names them; the default run adds nothing.
            *(
                [str(parameter_variant)]
                if model == "vmf"
                and is_vmf_parameter_variant(str(parameter_variant or ""))
                else []
            ),
            *(
                [_vocabulary_restriction_label(vocabulary_restriction)]
                if vocabulary_restriction is not None
                else []
            ),
        ],
        # The fingerprint suffix keeps outputs from different settings (for
        # example posterior configurations) in distinct directories so
        # --skip-existing never reuses results computed under other settings.
        include_fingerprint=True,
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
    failure_records: list[dict[str, object]] | None = None,
    failure_checkpoint_root: Path | None = None,
) -> Path:
    """Rebuild a durable condition index from atomically completed outputs."""
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
    completed: list[dict[str, object]] = []
    for completion_path in out_root.rglob("COMPLETE.json"):
        relative_parts = completion_path.relative_to(out_root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        metrics_path = completion_path.parent / "metrics_agg.json"
        if not metrics_path.exists():
            continue
        try:
            meta, results = read_evaluation_json(metrics_path)
        except (OSError, ValueError, TypeError):
            continue
        completed.append(
            {
                "dataset": meta.get("dataset"),
                "data_run": meta.get("data_run"),
                "model": meta.get("model"),
                "category": meta.get("category"),
                "num_topics": meta.get("num_topics"),
                "iterations": meta.get("iterations"),
                "condition_id": meta.get("condition_id"),
                "condition_fingerprint": meta.get("condition_fingerprint"),
                "metrics_path": str(metrics_path),
                "aggregate": (
                    results.get("aggregate") if isinstance(results, dict) else None
                ),
            }
        )
    completed.sort(
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("data_run")),
            str(row.get("model")),
            str(row.get("category")),
            str(row.get("num_topics")),
            str(row.get("iterations")),
        )
    )
    failures_by_key: dict[str, dict[str, object]] = {}
    failure_root = (
        (
            failure_checkpoint_root
            if failure_checkpoint_root is not None
            else out_root / ".checkpoints"
        )
        / "failures"
        / "v1"
    )
    if failure_root.exists():
        for path in sorted(failure_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(payload, dict):
                normalized = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"schema", "schema_version", "identity"}
                }
                failures_by_key[
                    json.dumps(normalized, ensure_ascii=False, sort_keys=True)
                ] = normalized
    for failure in failure_records or []:
        normalized = dict(failure)
        failures_by_key[json.dumps(normalized, ensure_ascii=False, sort_keys=True)] = (
            normalized
        )
    failures = list(failures_by_key.values())
    atomic_save_json(
        {
            "schema": "word_based_condition_index",
            "schema_version": 1,
            "completed_count": len(completed),
            "failed_count": len(failures),
            "completed": completed,
        },
        out_root / "condition_index.json",
    )
    atomic_save_json(failures, out_root / "failed_conditions.json")
    return out_root

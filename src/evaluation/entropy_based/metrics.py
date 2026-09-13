"""Entropy-based evaluation of document-topic distributions across models.

Runner conventions mirror :mod:`src.evaluation.geometry_based.metrics`: one
condition per (model, num_topics, category) aggregating all iterations, written
under ``results/topic_analysis/entropy_based/`` with the archive/latest layout.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.baselines.params import (
    format_covariance_variant,
    format_prior_scale_variant,
    normalize_covariance_type,
)
from src.core.artifacts import load_json, save_json
from src.core.errors import MissingArtifactError
from src.core.paths import (
    RESULTS_ROOT,
    build_archive_result_dir,
    build_latest_result_dir,
    resolve_project_path,
    write_latest_result_pointer,
)
from src.core.result_identity import build_condition_id, build_execution_id
from src.core.vmf_variant import normalize_vmf_parameter_variant
from src.evaluation.entropy_based.entropy_metrics import (
    DEFAULT_DEAD_RANK1_THRESHOLD,
    DEFAULT_DIFFUSE_ENTROPY_THRESHOLD,
    DOC_METRIC_COLUMNS,
    SUMMARY_METRIC_KEYS,
    TOPIC_FLAG_COLUMNS,
    TOPIC_METRIC_COLUMNS,
    aggregate_metrics,
    compute_entropy_metrics,
)
from src.evaluation.entropy_based.inputs import (
    DEFAULT_WORD_EMBEDDING_VARIANT,
    DOC_TOPIC_SOURCES,
    SUPPORTED_MODELS,
    effective_embedding_variant,
    load_doc_topic_matrix,
    normalize_model_names,
    parameter_variant_for,
    provenance_for,
    resolve_doc_topic_source,
)
from src.evaluation.reporting import (
    read_evaluation_json,
    write_csv_rows,
    write_evaluation_json,
    write_tabular_report_json,
)
from src.evaluation.schema import build_evaluation_meta
from src.utils.encoder_profiles import embedding_variant_base, encoder_model_alias
from src.utils.logging import get_logger

TASK_NAME = "entropy_based_metrics"
METRIC_SCHEMA_VERSION = 1
ANALYSIS_ROOT = RESULTS_ROOT / "topic_analysis"
DEFAULT_OUT_ROOT = ANALYSIS_ROOT / "entropy_based"
METRICS_FILENAME = "entropy_metrics_agg.json"
METADATA_FILENAME = "metadata.json"
FAILED_CONDITIONS_FILENAME = "failed_conditions.json"
CONDITION_FAILURE_POLICIES: tuple[str, ...] = ("fail-fast", "isolate")
logger = get_logger(__name__)

SUMMARY_BASE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "data_run",
    "num_topics",
    "category",
    "model",
    "embedding_variant",
    "prior_scale",
    "split",
    "doc_topic_source",
    "num_iterations",
    "num_documents_mean",
)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _uses_default_output_layout(out_root: Path) -> bool:
    return resolve_project_path(out_root) == DEFAULT_OUT_ROOT


def round_sigfigs(value: float, sig: int = 4) -> float:
    if value is None or np.isnan(value) or np.isinf(value):
        return float("nan") if value is None else float(value)
    return float(f"{value:.{sig}g}")


def _resolve_requested_embedding_variant(
    *,
    embedding_variant: str | None,
    encoder_model: str | None,
) -> str | None:
    requested_variant = (
        None if embedding_variant in {None, ""} else str(embedding_variant)
    )
    if encoder_model in {None, ""}:
        return requested_variant

    encoder_variant = encoder_model_alias(str(encoder_model))
    if requested_variant is not None:
        requested_base = embedding_variant_base(requested_variant)
        encoder_base = embedding_variant_base(encoder_variant)
        if requested_base != encoder_base:
            raise ValueError(
                "encoder_model and embedding_variant mismatch: "
                f"encoder_model='{encoder_model}' resolves to '{encoder_base}', "
                f"but embedding_variant='{requested_variant}' resolves to "
                f"'{requested_base}'."
            )
        return requested_variant
    return encoder_variant


def _build_output_condition_id(
    *,
    model: str,
    dataset: str,
    data_run: str,
    category: str,
    iterations: Sequence[int],
    num_topics: int,
    split: str,
    doc_topic_source: str,
    embedding_variant: str | None,
    parameter_variant: str | None,
    prior_scale: float | None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
    diffuse_entropy_threshold: float,
    dead_rank1_threshold: float,
) -> tuple[str, str]:
    extra_labels: list[str] = [model]
    if embedding_variant not in {None, ""}:
        extra_labels.append(str(embedding_variant))
    if parameter_variant not in {None, ""}:
        extra_labels.append(str(parameter_variant))
    return build_condition_id(
        iteration=int(min(iterations)),
        num_topics=int(num_topics),
        fingerprint_payload={
            "task": TASK_NAME,
            "model": model,
            "dataset": dataset,
            "data_run": data_run,
            "category": category,
            "iterations": [int(value) for value in iterations],
            "num_topics": int(num_topics),
            "split": split,
            "doc_topic_source": doc_topic_source,
            "embedding_variant": embedding_variant,
            "prior_scale": None if prior_scale is None else float(prior_scale),
            # Only a reduced covariance type enters the payload (historical ids intact).
            **(
                {"covariance_type": normalize_covariance_type(covariance_type)}
                if format_covariance_variant(covariance_type) is not None
                else {}
            ),
            # Likewise only a vMF hyperparameter variant enters the payload.
            **({"vmf_variant": str(vmf_variant)} if vmf_variant else {}),
            "diffuse_entropy_threshold": float(diffuse_entropy_threshold),
            "dead_rank1_threshold": float(dead_rank1_threshold),
            "metric_schema_version": METRIC_SCHEMA_VERSION,
        },
        extra_labels=extra_labels,
    )


def _normalize_topics(num_topics: int | Sequence[int]) -> list[int]:
    if isinstance(num_topics, (int, np.integer)):
        return [int(num_topics)]
    values: list[int] = []
    for value in num_topics:
        value_int = int(value)
        if value_int not in values:
            values.append(value_int)
    if not values:
        raise ValueError("num_topics must contain at least one value.")
    return values


def _write_iteration_artifacts(
    *,
    iter_out_dir: Path,
    result,
    raw_doc_indices: list[int] | None,
) -> dict[str, str]:
    ensure_directory(iter_out_dir)
    doc_rows = []
    doc = result.doc
    for doc_index in range(result.num_documents):
        row: dict[str, Any] = {
            "doc_index": doc_index,
            "raw_doc_index": (
                raw_doc_indices[doc_index]
                if raw_doc_indices is not None and doc_index < len(raw_doc_indices)
                else ""
            ),
        }
        for column in DOC_METRIC_COLUMNS:
            row[column] = float(doc[column][doc_index])
        row["valid"] = bool(doc["valid"][doc_index])
        doc_rows.append(row)
    doc_fieldnames = ["doc_index", "raw_doc_index", *DOC_METRIC_COLUMNS, "valid"]
    doc_path = iter_out_dir / "doc_metrics.csv"
    write_csv_rows(fieldnames=doc_fieldnames, rows=doc_rows, path=doc_path)

    topic_rows = []
    topic = result.topic
    for topic_index in range(result.num_topics):
        row = {"topic": topic_index}
        for column in TOPIC_METRIC_COLUMNS:
            row[column] = float(topic[column][topic_index])
        for column in TOPIC_FLAG_COLUMNS:
            row[column] = bool(topic[column][topic_index])
        topic_rows.append(row)
    topic_fieldnames = ["topic", *TOPIC_METRIC_COLUMNS, *TOPIC_FLAG_COLUMNS]
    topic_path = iter_out_dir / "topic_metrics.csv"
    write_csv_rows(fieldnames=topic_fieldnames, rows=topic_rows, path=topic_path)
    return {
        "doc_metrics_csv": f"{iter_out_dir.name}/{doc_path.name}",
        "topic_metrics_csv": f"{iter_out_dir.name}/{topic_path.name}",
    }


def _summary_row(
    *,
    meta: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    aggregate = results.get("aggregate", {})
    per_iteration = results.get("per_iteration", [])
    num_docs = [float(entry.get("num_documents", np.nan)) for entry in per_iteration]
    row: dict[str, Any] = {
        "dataset": meta.get("dataset"),
        "data_run": meta.get("data_run"),
        "num_topics": meta.get("num_topics"),
        "category": meta.get("category"),
        "model": meta.get("model"),
        "embedding_variant": meta.get("effective_embedding_variant"),
        "prior_scale": meta.get("prior_scale"),
        "split": meta.get("split"),
        "doc_topic_source": meta.get("doc_topic_source"),
        "num_iterations": len(per_iteration),
        "num_documents_mean": (
            round_sigfigs(float(np.nanmean(num_docs))) if num_docs else float("nan")
        ),
    }
    for key in SUMMARY_METRIC_KEYS:
        stats = aggregate.get(key, {})
        row[f"{key}_mean"] = round_sigfigs(float(stats.get("mean", np.nan)))
        row[f"{key}_std"] = round_sigfigs(float(stats.get("std", np.nan)))
    return row


def summary_fieldnames() -> list[str]:
    fieldnames = list(SUMMARY_BASE_COLUMNS)
    for key in SUMMARY_METRIC_KEYS:
        fieldnames.append(f"{key}_mean")
        fieldnames.append(f"{key}_std")
    return fieldnames


def _existing_output(
    *,
    out_root: Path,
    uses_default_output_layout: bool,
    dataset: str,
    data_run: str,
    category: str,
    condition_id: str,
) -> Path | None:
    """Return the existing aggregated metrics path for a condition, if any."""
    if uses_default_output_layout:
        latest_dir = build_latest_result_dir(
            base_root=out_root,
            dataset=dataset,
            data_run=data_run,
            category=category,
            display_key=condition_id,
        )
        pointer_path = latest_dir / "CURRENT.json"
        if not pointer_path.exists():
            return None
        payload = load_json(pointer_path)
        if not isinstance(payload, dict) or not payload.get("archive_dir"):
            return None
        candidate = resolve_project_path(str(payload["archive_dir"])) / METRICS_FILENAME
        return candidate if candidate.exists() else None
    candidate = (
        out_root / dataset / data_run / category / condition_id / METRICS_FILENAME
    )
    return candidate if candidate.exists() else None


def run_entropy_based_metrics(
    *,
    models: Sequence[str],
    dataset: str,
    iterations: Sequence[int],
    num_topics: int | Sequence[int],
    categories: Sequence[str],
    data_runs: Sequence[str] = ("default",),
    split: str = "test",
    doc_topic_source: str = "auto",
    diffuse_entropy_threshold: float = DEFAULT_DIFFUSE_ENTROPY_THRESHOLD,
    dead_rank1_threshold: float = DEFAULT_DEAD_RANK1_THRESHOLD,
    out_root: Path = DEFAULT_OUT_ROOT,
    save_per_iter_artifacts: bool = True,
    embedding_variant: str | None = None,
    encoder_model: str | None = None,
    word_embedding_variant: str | None = DEFAULT_WORD_EMBEDDING_VARIANT,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
    skip_existing: bool = False,
    condition_failure_policy: str = "fail-fast",
) -> Path:
    if doc_topic_source not in DOC_TOPIC_SOURCES:
        raise ValueError(
            f"Unsupported doc_topic_source '{doc_topic_source}'. Use one of {DOC_TOPIC_SOURCES}."
        )
    if condition_failure_policy not in CONDITION_FAILURE_POLICIES:
        raise ValueError(
            f"Unsupported condition_failure_policy '{condition_failure_policy}'. "
            f"Use one of {CONDITION_FAILURE_POLICIES}."
        )
    if not 0.0 <= float(diffuse_entropy_threshold) <= 1.0:
        raise ValueError("diffuse_entropy_threshold must be in [0, 1].")
    if not 0.0 <= float(dead_rank1_threshold) <= 1.0:
        raise ValueError("dead_rank1_threshold must be in [0, 1].")
    if prior_scale is not None:
        format_prior_scale_variant(prior_scale)
    vmf_variant = normalize_vmf_parameter_variant(vmf_variant)
    resolved_models = normalize_model_names(models)
    topic_values = _normalize_topics(num_topics)
    iteration_values = [int(value) for value in iterations]
    if not iteration_values:
        raise ValueError("iterations must contain at least one value.")
    out_root = Path(out_root)
    uses_default_output_layout = _uses_default_output_layout(out_root)
    requested_embedding_variant = _resolve_requested_embedding_variant(
        embedding_variant=embedding_variant,
        encoder_model=encoder_model,
    )

    summary_rows: list[dict[str, Any]] = []
    summary_provenance: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for data_run in data_runs:
        for model in resolved_models:
            # ``auto`` becomes the model's concrete source here, so the condition id,
            # the recorded protocol and the loader all see the same value.
            model_doc_topic_source = resolve_doc_topic_source(model, doc_topic_source)
            model_variant = effective_embedding_variant(
                model,
                requested_embedding_variant,
                word_embedding_variant=word_embedding_variant,
            )
            parameter_variant = (
                vmf_variant
                if model == "vmf"
                else parameter_variant_for(model, prior_scale, covariance_type)
            )
            for topic_count in topic_values:
                for category in categories:
                    condition_id, condition_fingerprint = _build_output_condition_id(
                        model=model,
                        dataset=dataset,
                        data_run=data_run,
                        category=category,
                        iterations=iteration_values,
                        num_topics=topic_count,
                        split=split,
                        doc_topic_source=model_doc_topic_source,
                        embedding_variant=model_variant,
                        parameter_variant=parameter_variant,
                        prior_scale=prior_scale,
                        covariance_type=covariance_type,
                        vmf_variant=vmf_variant if model == "vmf" else None,
                        diffuse_entropy_threshold=diffuse_entropy_threshold,
                        dead_rank1_threshold=dead_rank1_threshold,
                    )
                    display_key = condition_id

                    if skip_existing:
                        existing = _existing_output(
                            out_root=out_root,
                            uses_default_output_layout=uses_default_output_layout,
                            dataset=dataset,
                            data_run=data_run,
                            category=category,
                            condition_id=condition_id,
                        )
                        if existing is not None:
                            existing_meta, existing_results = read_evaluation_json(
                                existing
                            )
                            summary_rows.append(
                                _summary_row(
                                    meta=existing_meta, results=existing_results
                                )
                            )
                            logger.info(
                                "[%s] skip existing condition %s (%s)",
                                model,
                                condition_id,
                                existing,
                            )
                            continue

                    per_iteration: list[dict[str, Any]] = []
                    loaded_results = []
                    resolved_sources: dict[str, str] = {}
                    resolved_paths: dict[str, str] = {}
                    first_condition_dir: Path | None = None
                    try:
                        for iteration in iteration_values:
                            load = load_doc_topic_matrix(
                                model=model,
                                dataset=dataset,
                                data_run=data_run,
                                iteration=iteration,
                                num_topics=topic_count,
                                category=category,
                                split=split,
                                doc_topic_source=model_doc_topic_source,
                                embedding_variant=requested_embedding_variant,
                                word_embedding_variant=word_embedding_variant,
                                prior_scale=prior_scale,
                                covariance_type=covariance_type,
                                vmf_variant=vmf_variant if model == "vmf" else None,
                            )
                            if first_condition_dir is None:
                                first_condition_dir = load.condition_dir
                            result = compute_entropy_metrics(
                                load.theta,
                                diffuse_entropy_threshold=diffuse_entropy_threshold,
                                dead_rank1_threshold=dead_rank1_threshold,
                            )
                            entry: dict[str, Any] = {
                                "iteration": int(iteration),
                                "num_documents": result.num_documents,
                                "num_topics": result.num_topics,
                                "num_zero_mass_docs": result.num_zero_mass_docs,
                                "num_empty_topics": result.num_empty_topics,
                                "doc_topic_source_resolved": load.source,
                            }
                            entry.update(result.summary)
                            entry["per_topic"] = {
                                column: [float(v) for v in result.topic[column]]
                                for column in TOPIC_METRIC_COLUMNS
                            }
                            per_iteration.append(entry)
                            loaded_results.append((iteration, result, load))
                            resolved_sources[str(iteration)] = load.source
                            resolved_paths[str(iteration)] = str(load.path)
                    # ValueError covers the loader's artifact-shape rejections
                    # (doc-topic width, negative theta, unsupported payload);
                    # isolating them keeps one stale condition from aborting a
                    # whole sweep, as the topic-pair runner already does.
                    except (MissingArtifactError, FileNotFoundError, ValueError) as exc:
                        if condition_failure_policy == "fail-fast":
                            raise
                        failure = {
                            "model": model,
                            "dataset": dataset,
                            "data_run": data_run,
                            "category": category,
                            "num_topics": topic_count,
                            "iterations": iteration_values,
                            "embedding_variant": model_variant,
                            "prior_scale": prior_scale,
                            "condition_id": condition_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        failures.append(failure)
                        logger.warning(
                            "[%s] skipping condition %s after %s: %s",
                            model,
                            condition_id,
                            type(exc).__name__,
                            exc,
                        )
                        continue

                    aggregate = aggregate_metrics(
                        [entry for entry in per_iteration],
                        keys=[
                            key
                            for key in per_iteration[0]
                            if key
                            not in {
                                "iteration",
                                "doc_topic_source_resolved",
                                "per_topic",
                            }
                        ],
                    )
                    started_at = datetime.now(UTC).isoformat()
                    execution_id = build_execution_id(
                        prefix="exec", started_at=started_at
                    )

                    if uses_default_output_layout:
                        archive_out_dir = build_archive_result_dir(
                            base_root=out_root,
                            dataset=dataset,
                            data_run=data_run,
                            category=category,
                            display_key=display_key,
                            started_at=started_at,
                            execution_id=execution_id,
                        )
                        latest_out_dir = build_latest_result_dir(
                            base_root=out_root,
                            dataset=dataset,
                            data_run=data_run,
                            category=category,
                            display_key=display_key,
                        )
                        out_dir = archive_out_dir
                    else:
                        archive_out_dir = None
                        latest_out_dir = None
                        out_dir = (
                            out_root / dataset / data_run / category / condition_id
                        )
                    ensure_directory(out_dir)

                    artifacts: dict[str, str] = {
                        "metrics": METRICS_FILENAME,
                        "metadata": METADATA_FILENAME,
                    }
                    if save_per_iter_artifacts:
                        for iteration, result, load in loaded_results:
                            written = _write_iteration_artifacts(
                                iter_out_dir=out_dir / f"iter{iteration}",
                                result=result,
                                raw_doc_indices=load.raw_doc_indices,
                            )
                            artifacts[f"doc_metrics_csv_iter{iteration}"] = written[
                                "doc_metrics_csv"
                            ]
                            artifacts[f"topic_metrics_csv_iter{iteration}"] = written[
                                "topic_metrics_csv"
                            ]

                    assert first_condition_dir is not None
                    provenance = provenance_for(first_condition_dir, model=model)

                    meta = build_evaluation_meta(
                        task=TASK_NAME,
                        model=model,
                        dataset=dataset,
                        data_run=data_run,
                        num_topics=topic_count,
                        category=category,
                        condition_id=condition_id,
                        display_key=display_key,
                        condition_fingerprint=condition_fingerprint,
                        iterations=iteration_values,
                        started_at=started_at,
                        execution_id=execution_id,
                        archive_dir=str(out_dir),
                        latest_dir=(
                            None if latest_out_dir is None else str(latest_out_dir)
                        ),
                        split=split,
                        doc_topic_source=model_doc_topic_source,
                        diffuse_entropy_threshold=float(diffuse_entropy_threshold),
                        dead_rank1_threshold=float(dead_rank1_threshold),
                        doc_topic_source_resolved=resolved_sources,
                        doc_topic_paths=resolved_paths,
                        embedding_variant=requested_embedding_variant,
                        effective_embedding_variant=model_variant,
                        encoder_model=encoder_model,
                        word_embedding_variant=word_embedding_variant,
                        prior_scale=None if prior_scale is None else float(prior_scale),
                        parameter_variant=parameter_variant,
                        log_base="e",
                        metric_schema_version=METRIC_SCHEMA_VERSION,
                        model_provenance=provenance,
                    )
                    results = {
                        "aggregate": aggregate,
                        "per_iteration": per_iteration,
                    }
                    out_path = out_dir / METRICS_FILENAME
                    write_evaluation_json(meta=meta, results=results, path=out_path)
                    metadata_path = out_dir / METADATA_FILENAME
                    save_json(meta, metadata_path)
                    logger.info("[%s] entropy metrics saved to %s", model, out_path)

                    if uses_default_output_layout and archive_out_dir is not None:
                        pointer_path = write_latest_result_pointer(
                            base_root=out_root,
                            task=TASK_NAME,
                            dataset=dataset,
                            data_run=data_run,
                            category=category,
                            display_key=display_key,
                            archive_dir=archive_out_dir,
                            started_at=started_at,
                            execution_id=execution_id,
                            condition_fingerprint=condition_fingerprint,
                            artifacts=artifacts,
                        )
                        logger.info(
                            "[%s] updated latest pointer at %s", model, pointer_path
                        )

                    summary_rows.append(_summary_row(meta=meta, results=results))
                    summary_provenance.append(
                        {
                            "model": model,
                            "data_run": data_run,
                            "category": category,
                            "num_topics": topic_count,
                            "embedding_variant": model_variant,
                            "prior_scale": prior_scale,
                            "model_provenance": provenance,
                        }
                    )

    ensure_directory(out_root)
    summary_path = out_root / "summary.csv"
    if summary_rows:
        fieldnames = summary_fieldnames()
        write_csv_rows(fieldnames=fieldnames, rows=summary_rows, path=summary_path)
        write_tabular_report_json(
            meta={
                "task": f"{TASK_NAME}_summary",
                "dataset": dataset,
                "data_runs": list(data_runs),
                "split": split,
                "doc_topic_source": doc_topic_source,
                "diffuse_entropy_threshold": float(diffuse_entropy_threshold),
                "dead_rank1_threshold": float(dead_rank1_threshold),
                "embedding_variant": requested_embedding_variant,
                "encoder_model": encoder_model,
                "word_embedding_variant": word_embedding_variant,
                "prior_scale": prior_scale,
                "out_root": str(out_root),
                "model_provenance": summary_provenance,
                "failed_conditions": failures,
            },
            columns=fieldnames,
            rows=summary_rows,
            path=out_root / "summary.json",
        )
        logger.info("summary CSV written to %s", summary_path)
    if failures:
        failures_path = out_root / FAILED_CONDITIONS_FILENAME
        save_json({"task": TASK_NAME, "failures": failures}, failures_path)
        logger.warning(
            "%d condition(s) failed; details written to %s",
            len(failures),
            failures_path,
        )
    return summary_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entropy-based metrics on document-topic distributions "
            "(MALLET document_entropy / rank_1_docs and H(theta_d))."
        )
    )
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_run", nargs="+", default=["default"])
    parser.add_argument("--iteration", type=int, nargs="+", required=True)
    parser.add_argument("--num_topics", type=int, nargs="+", required=True)
    parser.add_argument("--category", nargs="+", default=["all"])
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--doc_topic_source",
        "--doc-topic-source",
        dest="doc_topic_source",
        choices=list(DOC_TOPIC_SOURCES),
        default="auto",
    )
    parser.add_argument(
        "--diffuse_entropy_threshold",
        "--diffuse-entropy-threshold",
        dest="diffuse_entropy_threshold",
        type=float,
        default=DEFAULT_DIFFUSE_ENTROPY_THRESHOLD,
    )
    parser.add_argument(
        "--dead_rank1_threshold",
        "--dead-rank1-threshold",
        dest="dead_rank1_threshold",
        type=float,
        default=DEFAULT_DEAD_RANK1_THRESHOLD,
    )
    parser.add_argument("--embedding_variant", "--embedding-variant", default=None)
    parser.add_argument("--encoder_model", "--encoder-model", default=None)
    parser.add_argument(
        "--word_embedding_variant",
        "--word-embedding-variant",
        default=DEFAULT_WORD_EMBEDDING_VARIANT,
    )
    parser.add_argument("--prior-scale", "--prior_scale", type=float, default=None)
    parser.add_argument("--out_root", "--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true")
    parser.add_argument(
        "--no_per_iter_artifacts",
        "--no-per-iter-artifacts",
        dest="save_per_iter_artifacts",
        action="store_false",
    )
    parser.add_argument(
        "--condition-failure-policy",
        "--condition_failure_policy",
        dest="condition_failure_policy",
        choices=list(CONDITION_FAILURE_POLICIES),
        default="fail-fast",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_entropy_based_metrics(
        models=args.model,
        dataset=args.dataset,
        data_runs=args.data_run,
        iterations=args.iteration,
        num_topics=args.num_topics,
        categories=args.category,
        split=args.split,
        doc_topic_source=args.doc_topic_source,
        diffuse_entropy_threshold=args.diffuse_entropy_threshold,
        dead_rank1_threshold=args.dead_rank1_threshold,
        out_root=args.out_root,
        save_per_iter_artifacts=args.save_per_iter_artifacts,
        embedding_variant=args.embedding_variant,
        encoder_model=args.encoder_model,
        word_embedding_variant=args.word_embedding_variant,
        prior_scale=args.prior_scale,
        skip_existing=args.skip_existing,
        condition_failure_policy=args.condition_failure_policy,
    )


__all__ = [
    "DEFAULT_OUT_ROOT",
    "SUPPORTED_MODELS",
    "run_entropy_based_metrics",
    "summary_fieldnames",
    "main",
]


if __name__ == "__main__":
    main()

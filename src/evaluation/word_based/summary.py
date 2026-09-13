"""Summarize word-based (topic coherence / diversity) evaluation runs.

This is the word-based counterpart of
:mod:`src.evaluation.classification.summary`. Both write the same three kinds of
artifact next to each other:

``<stem>.scores.json``
    The raw per-run values plus the provenance of every cell. This is the
    interface downstream consumers (the paper repository) read; they do their
    own aggregation from it, so no rounded or ranked number crosses the
    boundary.
``<stem>.tex`` / ``<stem>.runs.json`` / ``<stem>.runs.csv``
    Review artifacts for reading the results here. They are rendered with the
    shared helpers in :mod:`src.evaluation.reports.latex_tables`, so mean,
    standard deviation and the best/second-best marking are defined once.

One summary group is a (dataset, data run, topic count, encoder variant) tuple;
one cell is a (category, model) pair within it, holding one value per run.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.baselines.params import normalize_covariance_type
from src.core.paths import REPO_ROOT
from src.core.vmf_variant import normalize_vmf_parameter_variant, vmf_variant_matches
from src.evaluation.reports.latex_tables import (
    format_pm,
    latex_escape_text,
    mean_std,
    rank_and_mark,
)
from src.evaluation.word_based.model_inputs import (
    DEFAULT_OUT_ROOT,
    EMBEDDING_VARIANT_MODELS,
    MODEL_CHOICES,
)

DEFAULT_COHERENCE_ROOT = DEFAULT_OUT_ROOT
DEFAULT_GROUP_BY = (
    "dataset",
    "data_run",
    "category",
    "num_topics",
    "encoder_model",
)

RUN_COVERAGE_FIELDS = [
    "metric",
    "dataset",
    "data_run",
    "topics",
    "coherence_reference",
    "coherence_topn",
    "coherence_split",
    "selector",
    "model",
    "encoder_variant",
    "encoder_model",
    "category",
    "run_count",
    "expected_runs",
    "missing_runs",
    "status",
    "source_path",
]

MODEL_ORDER = [
    "bleilda",
    "sam",
    "sam_tf",
    "sentlda",
    "gaussianlda",
    "mvtm",
    "etm",
    "ctm",
    "sentence_gaussianlda",
    "vmf",
]

# Table column labels. Kept in step with MODEL_TABLE_LABELS in
# src/evaluation/classification/summary.py, which maps the classification
# display names ("sentLDA", "Contextual TM") onto these same labels.
MODEL_TABLE_LABELS = {
    "bleilda": "LDA",
    "sam": "SAM (tf-idf)",
    "sam_tf": "SAM",
    "sentlda": "SentLDA",
    "gaussianlda": "GLDA",
    "mvtm": "vLDA",
    "etm": "ETM",
    "ctm": "ConTM",
    "sentence_gaussianlda": "GSLDA",
    "vmf": "vSLDA",
}

# Models whose result is independent of the sentence encoder. They are run once,
# recorded under whichever embedding variant that invocation happened to use,
# and therefore belong in every encoder group of the same dataset/topic count.
ENCODER_INDEPENDENT_MODELS = frozenset(MODEL_CHOICES) - frozenset(
    EMBEDDING_VARIANT_MODELS
)

# Evaluation-protocol fields that must agree across the runs of one cell; they
# describe how the metric was computed, so mixing them would average
# incomparable numbers. A missing value counts as "not recorded", not a conflict.
# Derived per-run quantities (coherence_reference_vocab_size, for instance,
# depends on the topic words of that run) legitimately differ and are excluded.
PROTOCOL_FIELDS = (
    "coherence_reference",
    "coherence_reference_num_docs",
    "coherence_topn",
    "diversity_topn",
    "coherence_split",
    "topic_word_source",
    "topic_word_score_mode",
    "word2vec",
    "prior_scale",
    "covariance_type",
    "vmf_variant",
    "vmf_hyperparameters",
)

METRIC_ORDER = ("coherence_c_v", "coherence_c_npmi", "coherence_c_uci", "diversity")

# Gaussian prior scale of a run that predates the flag or used the default (see
# BaselineParams.prior_scale in src/baselines/params.py). Runs of the prior-scale
# sweep record their own value and are a different condition, not another run of
# this one, so they are excluded unless explicitly requested.
DEFAULT_PRIOR_SCALE = 0.1

# Models that have a Gaussian prior scale at all; for every other model the
# field is simply absent (see _effective_prior_scale_for_model in metrics.py).
PRIOR_SCALE_MODELS = frozenset({"gaussianlda", "sentence_gaussianlda", "gaussian"})


class SummaryError(Exception):
    """Raised when the recorded runs cannot be summarized as they stand."""


@dataclass(frozen=True)
class SummarySource:
    metrics_path: Path
    pointer_path: Path | None = None
    archive_dir: Path | None = None


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _csv_cell(value: Any) -> Any:
    # List-valued columns (per-iteration values) are ';'-joined in CSV.
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = collect_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _csv_cell(value) for key, value in row.items()} for row in rows
        )


def _write_csv_with_fields(
    *,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _as_project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or "unknown"


def _resolve_existing_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_relative = (Path.cwd() / path).resolve()
    if cwd_relative.exists():
        return cwd_relative
    return (base_dir / path).resolve()


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def discover_latest_sources(coherence_root: Path) -> list[SummarySource]:
    """One source per ``latest/**/CURRENT.json`` pointer (the default)."""
    latest_root = coherence_root / "latest"
    sources: list[SummarySource] = []
    for pointer_path in sorted(latest_root.rglob("CURRENT.json")):
        pointer = _read_json(pointer_path)
        if not isinstance(pointer, dict):
            continue
        archive_dir_value = pointer.get("archive_dir")
        artifacts = pointer.get("artifacts")
        metrics_name = (
            artifacts.get("metrics", "metrics_agg.json")
            if isinstance(artifacts, dict)
            else "metrics_agg.json"
        )
        if archive_dir_value:
            archive_dir = _resolve_existing_path(
                str(archive_dir_value), base_dir=coherence_root
            )
        else:
            archive_dir = pointer_path.parent
        sources.append(
            SummarySource(
                metrics_path=archive_dir / str(metrics_name),
                pointer_path=pointer_path,
                archive_dir=archive_dir,
            )
        )
    return sources


def discover_archive_sources(coherence_root: Path) -> list[SummarySource]:
    archive_root = coherence_root / "archive"
    return [
        SummarySource(metrics_path=metrics_path, archive_dir=metrics_path.parent)
        for metrics_path in sorted(archive_root.rglob("metrics_agg.json"))
    ]


def discover_indexed_sources(coherence_root: Path) -> list[SummarySource]:
    """Fallback for when ``latest/`` is empty: read the condition index.

    The index accumulates re-runs, so it can list several executions of the same
    condition; the pointers under ``latest/`` cannot.
    """
    index_path = coherence_root / "condition_index.json"
    if not index_path.exists():
        return []
    payload = _read_json(index_path)
    if not isinstance(payload, dict):
        return []
    completed = payload.get("completed", [])
    if not isinstance(completed, list):
        return []
    sources: list[SummarySource] = []
    for condition in completed:
        if not isinstance(condition, dict):
            continue
        metrics_path_value = condition.get("metrics_path")
        if not metrics_path_value:
            continue
        metrics_path = _resolve_existing_path(
            str(metrics_path_value), base_dir=coherence_root
        )
        sources.append(
            SummarySource(metrics_path=metrics_path, archive_dir=metrics_path.parent)
        )
    return sources


# ---------------------------------------------------------------------------
# One flat row per recorded run condition
# ---------------------------------------------------------------------------


def _stringify_list(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _iteration_list(value: Any) -> list[int]:
    """``_meta.iterations`` as integers; unparsable entries are dropped."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    iterations: list[int] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        try:
            iterations.append(int(text))
        except ValueError:
            continue
    return iterations


def _round_metric(value: Any, digits: int = 6) -> Any:
    if not isinstance(value, (int, float)):
        return value
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return value
    return round(value, digits)


COVERAGE_METRICS = ("num_active_topics", "num_empty_topics", "topic_utilization")


def _coverage_sort_key(key: Any) -> tuple[int, Any]:
    """Order ``coverage_by_iteration`` keys numerically when they are numbers."""
    try:
        return (0, int(key))
    except (TypeError, ValueError):
        return (1, str(key))


def _metric_name_from_column(column: str) -> str:
    for suffix in ("_mean", "_std"):
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def _metric_sort_key(column: str) -> tuple[int, str]:
    metric = _metric_name_from_column(column)
    if metric in METRIC_ORDER:
        return (METRIC_ORDER.index(metric), column)
    if metric.startswith("coherence_"):
        return (len(METRIC_ORDER), column)
    return (len(METRIC_ORDER) + 1, column)


def available_metrics(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Metric names that carry per-run values in ``rows``, in reporting order."""
    names = {
        column[: -len("_values")]
        for row in rows
        for column in row
        if column.endswith("_values") and "_active_only_" not in column
    }
    return sorted(names, key=_metric_sort_key)


def _vocabulary_arm(coherence: dict[str, Any]) -> str:
    """Label the evaluation-vocabulary arm a row belongs to.

    ``full`` is the historical vocabulary; anything restricted by the
    reference-frequency band gets a compact, comparable label.
    """

    minimum = int(coherence.get("reference_min_df", 0) or 0)
    ratio_value = coherence.get("reference_max_df_ratio", 1.0)
    ratio = 1.0 if ratio_value in (None, "") else float(ratio_value)
    if minimum <= 0 and ratio >= 1.0:
        return "full"
    return f"refdf{minimum}-{int(round(ratio * 100))}"


def build_summary_row(source: SummarySource) -> dict[str, Any]:
    """Flatten one ``metrics_agg.json`` into a single row."""
    payload = _read_json(source.metrics_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {source.metrics_path}")

    meta = payload.get("_meta", {})
    results = payload.get("results", {})
    if not isinstance(meta, dict) or not isinstance(results, dict):
        raise ValueError(f"Expected evaluation payload: {source.metrics_path}")

    topic_words = meta.get("topic_words", {})
    coherence = meta.get("coherence", {})
    model_provenance = meta.get("model_provenance", {})
    topic_words = topic_words if isinstance(topic_words, dict) else {}
    coherence = coherence if isinstance(coherence, dict) else {}
    model_provenance = model_provenance if isinstance(model_provenance, dict) else {}

    embedding_variant = meta.get("embedding_variant", "")
    effective_embedding_variant = meta.get("effective_embedding_variant", "")
    encoder_model = effective_embedding_variant or embedding_variant
    baseline_params = model_provenance.get("baseline_params", {})
    baseline_params = baseline_params if isinstance(baseline_params, dict) else {}

    row: dict[str, Any] = {
        "dataset": meta.get("dataset", ""),
        "data_run": meta.get("data_run", ""),
        "category": meta.get("category", ""),
        "model": meta.get("model", ""),
        "num_topics": meta.get("num_topics", ""),
        "iterations": _iteration_list(meta.get("iterations")),
        "condition_id": meta.get("condition_id", ""),
        "display_key": meta.get("display_key", ""),
        "condition_fingerprint": meta.get("condition_fingerprint", ""),
        "execution_id": meta.get("execution_id", ""),
        "started_at": meta.get("started_at", ""),
        "embedding_variant": embedding_variant,
        "effective_embedding_variant": effective_embedding_variant,
        "encoder_model": encoder_model,
        "runner_family": model_provenance.get("runner_family", ""),
        "method_kind": model_provenance.get("method_kind", ""),
        # Variant provenance for word-embedding / Gaussian models. Downstream
        # table builders filter on these; empty when the model has no such knob.
        "word2vec": baseline_params.get("word2vec") or "",
        "prior_scale": baseline_params.get("prior_scale"),
        "covariance_type": baseline_params.get("covariance_type") or "",
        # vMF hyperparameter-sweep provenance (src/core/vmf_variant.py): the label of a
        # sweep run ("" for the default run) and the hyperparameters it trained with,
        # serialized so the CSV and the per-cell protocol check can carry them.
        "vmf_variant": (
            (model_provenance.get("parameter_variant") or "")
            if meta.get("model") == "vmf"
            else ""
        ),
        "vmf_hyperparameters": (
            json.dumps(model_provenance["vmf_hyperparameters"], sort_keys=True)
            if meta.get("model") == "vmf"
            and isinstance(model_provenance.get("vmf_hyperparameters"), dict)
            else ""
        ),
        "topic_word_source": topic_words.get("source", ""),
        "topic_word_score_mode": topic_words.get("score_mode", ""),
        "coherence_metrics": _stringify_list(coherence.get("metrics")),
        "primary_coherence_metric": coherence.get("primary_metric", ""),
        "coherence_reference": coherence.get("coherence_reference", ""),
        "coherence_reference_num_docs": coherence.get(
            "coherence_reference_num_docs", ""
        ),
        "coherence_reference_vocab_size": coherence.get(
            "coherence_reference_vocab_size", ""
        ),
        "coherence_topn": coherence.get("topn", topic_words.get("coherence_topn", "")),
        "coherence_split": coherence.get("split", ""),
        "diversity_topn": topic_words.get("diversity_topn", ""),
        # Evaluation-vocabulary arm. Empty/default values mark the historical
        # arm; a filtered arm carries the reference-frequency band it used.
        "dict_no_above": coherence.get("dict_no_above", ""),
        "reference_min_df": coherence.get("reference_min_df", 0),
        "reference_max_df_ratio": coherence.get("reference_max_df_ratio", 1.0),
        "vocabulary_arm": _vocabulary_arm(coherence),
        "metrics_path": _as_project_relative(source.metrics_path),
        "archive_dir": (
            _as_project_relative(source.archive_dir)
            if source.archive_dir is not None
            else _as_project_relative(source.metrics_path.parent)
        ),
        "pointer_path": (
            _as_project_relative(source.pointer_path)
            if source.pointer_path is not None
            else ""
        ),
    }

    aggregate = results.get("aggregate", {})
    if isinstance(aggregate, dict):
        for metric_name, stats in sorted(aggregate.items()):
            if not isinstance(stats, dict):
                continue
            for stat_name in ("mean", "std"):
                if stat_name in stats:
                    row[f"{metric_name}_{stat_name}"] = _round_metric(stats[stat_name])

    per_iteration = results.get("per_iteration", [])
    per_iteration = per_iteration if isinstance(per_iteration, list) else []
    row["iteration_count"] = len(per_iteration)
    # Unrounded per-iteration values. Downstream aggregation (mean/std) is done
    # from these; the *_mean/*_std above stay as a cross-check for readers.
    for metric_name in sorted(
        {
            key
            for item in per_iteration
            if isinstance(item, dict)
            for key in item
            if key != "num_topics"
        }
    ):
        row[f"{metric_name}_values"] = [
            float(item[metric_name])
            for item in per_iteration
            if isinstance(item, dict)
            and isinstance(item.get(metric_name), (int, float))
        ]

    # Topic-utilization coverage is recorded per iteration under
    # ``coverage_by_iteration`` for every model. Only the MvTM protocol folds it
    # into ``aggregate``/``per_iteration``, so without this the columns stay
    # empty for vMF/LDA/sentLDA/ETM even though the numbers were computed.
    # Rows that already carry the metric (MvTM) are left untouched.
    coverage_by_iteration = results.get("coverage_by_iteration", {})
    if isinstance(coverage_by_iteration, dict):
        entries = [
            coverage_by_iteration[key]
            for key in sorted(coverage_by_iteration, key=_coverage_sort_key)
            if isinstance(coverage_by_iteration[key], dict)
        ]
        for metric_name in COVERAGE_METRICS:
            if f"{metric_name}_values" in row:
                continue
            values = [
                float(entry[metric_name])
                for entry in entries
                if isinstance(entry.get(metric_name), (int, float))
            ]
            if not values:
                continue
            row[f"{metric_name}_values"] = values
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            row.setdefault(f"{metric_name}_mean", _round_metric(mean))
            row.setdefault(f"{metric_name}_std", _round_metric(variance**0.5))
    return row


# ---------------------------------------------------------------------------
# Ranking and ordering of the flat rows (summary.csv / summary.json)
# ---------------------------------------------------------------------------


def choose_default_rank_metric(rows: Sequence[Mapping[str, Any]]) -> str | None:
    columns = set().union(*(row.keys() for row in rows)) if rows else set()
    if "coherence_c_v_mean" in columns:
        return "coherence_c_v_mean"
    coherence_columns = sorted(
        column
        for column in columns
        if column.startswith("coherence_") and column.endswith("_mean")
    )
    if coherence_columns:
        return coherence_columns[0]
    if "diversity_mean" in columns:
        return "diversity_mean"
    return None


def add_rank_columns(
    rows: Sequence[Mapping[str, Any]],
    *,
    rank_metric: str | None,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
) -> list[dict[str, Any]]:
    ranked_rows = [dict(row) for row in rows]
    if not rank_metric:
        return ranked_rows
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        grouped[tuple(_hashable(row.get(key, "")) for key in group_by)].append(row)
    for group_rows in grouped.values():
        sortable = [
            row
            for row in group_rows
            if isinstance(row.get(rank_metric), (int, float))
            and not math.isnan(float(row[rank_metric]))
        ]
        sortable.sort(key=lambda row: float(row[rank_metric]), reverse=True)
        for index, row in enumerate(sortable, start=1):
            row["rank_metric"] = rank_metric
            row["rank"] = index
            row["is_best"] = index == 1
        for row in group_rows:
            row.setdefault("rank_metric", rank_metric)
            row.setdefault("rank", "")
            row.setdefault("is_best", "")
    return ranked_rows


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def collect_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "dataset",
        "data_run",
        "category",
        "num_topics",
        "model",
        "rank_metric",
        "rank",
        "is_best",
        "coherence_c_v_mean",
        "coherence_c_v_std",
        "coherence_c_npmi_mean",
        "coherence_c_npmi_std",
        "coherence_c_uci_mean",
        "coherence_c_uci_std",
        "diversity_mean",
        "diversity_std",
        "iterations",
        "iteration_count",
        "condition_id",
        "display_key",
        "execution_id",
        "started_at",
        "embedding_variant",
        "effective_embedding_variant",
        "encoder_model",
        "runner_family",
        "method_kind",
        "topic_word_source",
        "topic_word_score_mode",
        "coherence_metrics",
        "primary_coherence_metric",
        "coherence_reference",
        "coherence_reference_num_docs",
        "coherence_reference_vocab_size",
        "coherence_topn",
        "coherence_split",
        "diversity_topn",
        "word2vec",
        "prior_scale",
        "condition_fingerprint",
        "pointer_path",
        "metrics_path",
        "archive_dir",
    ]
    all_fields = set().union(*(row.keys() for row in rows)) if rows else set()
    dynamic_metrics = sorted(
        (
            field_name
            for field_name in all_fields
            if field_name.endswith("_mean") or field_name.endswith("_std")
        ),
        key=_metric_sort_key,
    )
    ordered: list[str] = []
    for field_name in [*preferred, *dynamic_metrics]:
        if field_name in all_fields and field_name not in ordered:
            ordered.append(field_name)
    ordered.extend(sorted(all_fields - set(ordered)))
    return ordered


def sort_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("data_run", "")),
            str(row.get("category", "")),
            int(row.get("num_topics", 0) or 0),
            int(row.get("rank", 999999) or 999999),
            str(row.get("model", "")),
            str(row.get("display_key", "")),
        ),
    )


def _model_sort_key(model: str) -> tuple[int, str]:
    try:
        index = MODEL_ORDER.index(model)
    except ValueError:
        index = len(MODEL_ORDER)
    return (index, model)


def model_table_label(model: str) -> str:
    return MODEL_TABLE_LABELS.get(model, model)


def _encoder_model_value(row: Mapping[str, Any]) -> str:
    return str(
        row.get("encoder_model")
        or row.get("effective_embedding_variant")
        or row.get("embedding_variant")
        or ""
    )


def encoder_variant_of(row: Mapping[str, Any]) -> str:
    """Short encoder name of a row (``minilm_raw`` -> ``minilm``)."""
    value = _encoder_model_value(row).strip()
    return value.split("_", 1)[0] if value else ""


def _selector_key(row: Mapping[str, Any]) -> str:
    model = str(row.get("model", ""))
    encoder_model = _encoder_model_value(row)
    return f"{model}::{encoder_model}" if encoder_model else model


def _category_label(category: str) -> str:
    return str(category).replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Grouping the flat rows into (dataset, data_run, topics, encoder) tables
# ---------------------------------------------------------------------------


@dataclass
class SummaryCell:
    """The runs of one (category, model) pair within a summary group."""

    category: str
    model: str
    runs: dict[int, dict[str, Any]] = field(default_factory=dict)

    def values(self, metric: str) -> list[float]:
        """Per-run values of ``metric``, ordered by iteration."""
        collected: list[float] = []
        for iteration in sorted(self.runs):
            row = self.runs[iteration]
            metric_values = row.get(f"{metric}_values")
            if isinstance(metric_values, list) and metric_values:
                collected.append(float(metric_values[0]))
        return collected

    def iterations(self, metric: str) -> list[int]:
        return [
            iteration
            for iteration in sorted(self.runs)
            if self.runs[iteration].get(f"{metric}_values")
        ]


@dataclass
class SummaryGroup:
    dataset: str
    data_run: str
    topics: int
    encoder_variant: str
    expected_iterations: list[int]
    cells: dict[tuple[str, str], SummaryCell] = field(default_factory=dict)

    @property
    def categories(self) -> list[str]:
        return sorted({category for category, _ in self.cells})

    @property
    def models(self) -> list[str]:
        return sorted({model for _, model in self.cells}, key=_model_sort_key)

    def cell(self, category: str, model: str) -> SummaryCell | None:
        return self.cells.get((category, model))


def _protocol_value(cell: SummaryCell, field_name: str) -> Any:
    """The single recorded value of ``field_name`` across a cell's runs.

    Runs that did not record the field are ignored; two different recorded
    values mean the runs used different protocols and cannot be pooled.
    """
    recorded: list[tuple[int, Any]] = []
    for iteration, row in sorted(cell.runs.items()):
        value = row.get(field_name)
        if value is None or value == "":
            continue
        recorded.append((iteration, value))
    if not recorded:
        return None
    distinct = {str(value) for _, value in recorded}
    if len(distinct) > 1:
        conditions = ", ".join(
            f"it{iteration}={cell.runs[iteration].get('condition_id') or '?'}"
            f" ({value})"
            for iteration, value in recorded
        )
        raise SummaryError(
            f"{cell.category}/{cell.model}: runs disagree on {field_name}: {conditions}"
        )
    return recorded[0][1]


def cell_provenance(cell: SummaryCell) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "condition_ids": [
            str(cell.runs[iteration].get("condition_id") or "")
            for iteration in sorted(cell.runs)
        ],
        "iterations": sorted(cell.runs),
        "encoder_model": _protocol_value(cell, "encoder_model") or "",
    }
    for field_name in PROTOCOL_FIELDS:
        provenance[field_name] = _protocol_value(cell, field_name)
    return provenance


def _row_prior_scale(row: Mapping[str, Any]) -> float | None:
    """Recorded Gaussian prior scale of a run, or None when it used the default."""
    value = row.get("prior_scale")
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str] | None = None,
    word2vec: str | None = None,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
    coherence_reference_num_docs: int | None = None,
    warnings: list[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Keep the runs of one condition; sweep variants are separate conditions."""
    warn = warnings if warnings is not None else []
    dataset_filter = {str(name) for name in datasets} if datasets else None
    wanted_prior = DEFAULT_PRIOR_SCALE if prior_scale is None else float(prior_scale)
    wanted_covariance = normalize_covariance_type(covariance_type)
    wanted_vmf_variant = normalize_vmf_parameter_variant(vmf_variant)
    selected: list[Mapping[str, Any]] = []
    dropped_prior: set[str] = set()
    dropped_covariance: set[str] = set()
    dropped_vmf: set[str] = set()
    for row in rows:
        if (
            dataset_filter is not None
            and str(row.get("dataset", "")) not in dataset_filter
        ):
            continue
        if str(row.get("model", "")) in PRIOR_SCALE_MODELS:
            # An unrecorded scale means the run predates the flag and used the
            # default, so it answers to that value and to no other.
            recorded_prior = _row_prior_scale(row)
            effective_prior = (
                DEFAULT_PRIOR_SCALE if recorded_prior is None else recorded_prior
            )
            if not math.isclose(effective_prior, wanted_prior):
                dropped_prior.add(str(effective_prior))
                continue
        if str(row.get("model", "")) == "sentence_gaussianlda":
            # An unrecorded covariance type means the run predates the reduced
            # variants and is full-covariance.
            recorded_covariance = normalize_covariance_type(
                row.get("covariance_type") or None
            )
            if recorded_covariance != wanted_covariance:
                dropped_covariance.add(recorded_covariance)
                continue
        if str(row.get("model", "")) == "vmf":
            # An unrecorded variant means the run predates the hyperparameter sweep and
            # is a default run; a sweep variant is only summarized when requested.
            recorded_vmf = row.get("vmf_variant") or None
            if not vmf_variant_matches(recorded_vmf, wanted_vmf_variant):
                dropped_vmf.add(str(recorded_vmf or "default"))
                continue
        if word2vec is not None and str(row.get("word2vec", "")) not in {"", word2vec}:
            continue
        if coherence_reference_num_docs is not None:
            recorded_docs = row.get("coherence_reference_num_docs")
            if recorded_docs not in ("", None) and int(recorded_docs) != int(
                coherence_reference_num_docs
            ):
                continue
        selected.append(row)
    if dropped_prior:
        warn.append(
            "excluded runs from the Gaussian prior-scale sweep "
            f"(prior_scale in {sorted(dropped_prior)}); summarizing prior_scale="
            f"{wanted_prior} (unrecorded values count as this default)"
        )
    if dropped_covariance:
        warn.append(
            "excluded sentence Gaussian LDA runs of other covariance types "
            f"({sorted(dropped_covariance)}); summarizing covariance_type="
            f"{wanted_covariance} (unrecorded values count as full)"
        )
    if dropped_vmf:
        warn.append(
            "excluded vMF Sentence LDA runs of other hyperparameter variants "
            f"({sorted(dropped_vmf)}); summarizing vmf_variant="
            f"{wanted_vmf_variant or 'default'}"
        )
    return selected


def group_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: Sequence[int] | None = None,
    strict: bool = False,
    warnings: list[str] | None = None,
) -> list[SummaryGroup]:
    """Bucket flat rows into one group per (dataset, data run, K, encoder).

    Encoder-independent models are recorded under whichever embedding variant
    their run happened to use, so they are collected separately and then placed
    in every encoder group of the same dataset and topic count.
    """
    warn = warnings if warnings is not None else []

    # (dataset, data_run, topics) -> encoder -> (category, model) -> [rows]
    encoder_aware: dict[tuple[str, str, int], dict[str, list[Mapping[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    shared: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    shared_variants: dict[tuple[str, str, int], set[str]] = defaultdict(set)

    for row in rows:
        dataset = str(row.get("dataset", ""))
        model = str(row.get("model", ""))
        try:
            topics = int(row.get("num_topics", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not dataset or not model or not topics:
            continue
        key = (dataset, str(row.get("data_run", "")), topics)
        if model in ENCODER_INDEPENDENT_MODELS:
            shared[key].append(row)
            shared_variants[key].add(encoder_variant_of(row))
        else:
            encoder_aware[key][encoder_variant_of(row)].append(row)

    groups: list[SummaryGroup] = []
    for key in sorted(set(encoder_aware) | set(shared)):
        dataset, data_run, topics = key
        variants = sorted(v for v in encoder_aware.get(key, {}) if v)
        if not variants:
            # Nothing encoder-aware here; fall back to the variants the
            # encoder-independent runs were recorded under.
            variants = sorted(v for v in shared_variants.get(key, set()) if v) or [""]
        for variant in variants:
            group_rows = [*encoder_aware.get(key, {}).get(variant, []), *shared[key]]
            if not group_rows:
                continue
            group = SummaryGroup(
                dataset=dataset,
                data_run=data_run,
                topics=topics,
                encoder_variant=variant,
                expected_iterations=[],
            )
            for row in group_rows:
                _add_row_to_group(group, row, strict=strict, warnings=warn)
            if iterations is not None:
                group.expected_iterations = sorted(int(it) for it in iterations)
            else:
                group.expected_iterations = sorted(
                    {it for cell in group.cells.values() for it in cell.runs}
                )
            groups.append(group)
    return groups


def _row_for_single_iteration(
    row: Mapping[str, Any], iteration: int, offset: int
) -> dict[str, Any]:
    """One iteration of a multi-iteration condition, shaped like a single run.

    Every ``<metric>_values`` list is indexed at ``offset``; the matching
    ``_mean`` becomes that value and ``_std`` becomes zero, so the row carries
    exactly what a one-iteration condition would have recorded.
    """
    single = dict(row)
    single["iterations"] = [int(iteration)]
    single["iteration_count"] = 1
    for column in list(row):
        if not column.endswith("_values"):
            continue
        values = row.get(column)
        if not isinstance(values, list) or offset >= len(values):
            continue
        metric_name = column[: -len("_values")]
        value = values[offset]
        single[column] = [value]
        single[f"{metric_name}_mean"] = value
        single[f"{metric_name}_std"] = 0.0
    return single


def _add_row_to_group(
    group: SummaryGroup,
    row: Mapping[str, Any],
    *,
    strict: bool,
    warnings: list[str],
) -> None:
    category = str(row.get("category", ""))
    model = str(row.get("model", ""))
    cell = group.cells.setdefault(
        (category, model), SummaryCell(category=category, model=model)
    )
    row_iterations = row.get("iterations") or []
    if not isinstance(row_iterations, list):
        row_iterations = _iteration_list(row_iterations)
    if not row_iterations:
        message = (
            f"{group.dataset}/K={group.topics}/{category}/{model}: expected one "
            f"iteration per condition, found none "
            f"({row.get('condition_id') or row.get('metrics_path')})"
        )
        if strict:
            raise SummaryError(message)
        warnings.append(message)
        return
    if len(row_iterations) > 1:
        # A condition evaluated with several iterations in one invocation
        # (``--iteration 0 1 2 ...``) holds one value per iteration. Split it so
        # every iteration becomes its own run and the cell pools all of them;
        # otherwise only the first iteration would be read.
        for offset, iteration in enumerate(row_iterations):
            _add_row_to_group(
                group,
                _row_for_single_iteration(row, iteration, offset),
                strict=strict,
                warnings=warnings,
            )
        return
    iteration = int(row_iterations[0])
    existing = cell.runs.get(iteration)
    if existing is None:
        cell.runs[iteration] = dict(row)
        return
    # GSLDA の raw / norm が同じ反復に並ぶことがある（未正規化の旧 run と
    # L2 正規化した新 run）。本稿が報告するのは norm なので、タイムスタンプ順
    # ではなく変種で明示的に選ぶ。混在したまま集計に進むと _protocol_value が
    # encoder_model の不一致で SummaryError を投げる。
    existing_variant = str(existing.get("encoder_model", ""))
    row_variant = str(row.get("encoder_model", ""))
    if existing_variant != row_variant and {existing_variant, row_variant} <= {
        f"{existing_variant.rsplit('_', 1)[0]}_raw",
        f"{existing_variant.rsplit('_', 1)[0]}_norm",
    }:
        if row_variant.endswith("_norm"):
            cell.runs[iteration] = dict(row)
        return
    message = (
        f"{group.dataset}/K={group.topics}/{category}/{model}: iteration "
        f"{iteration} recorded twice "
        f"({existing.get('condition_id')} and {row.get('condition_id')})"
    )
    if strict:
        raise SummaryError(message)
    warnings.append(message + "; keeping the newer execution")
    if str(row.get("started_at", "")) > str(existing.get("started_at", "")):
        cell.runs[iteration] = dict(row)


# ---------------------------------------------------------------------------
# Group outputs: scores.json, LaTeX review table, run coverage
# ---------------------------------------------------------------------------


def build_scores_payload(
    group: SummaryGroup, *, metrics: Sequence[str]
) -> dict[str, Any]:
    """The per-run values and provenance of a group (the downstream interface)."""
    categories = group.categories
    models = group.models
    scores: dict[str, dict[str, dict[str, list[float]]]] = {}
    for metric in metrics:
        by_category: dict[str, dict[str, list[float]]] = {}
        for category in categories:
            by_model = {}
            for model in models:
                cell = group.cell(category, model)
                if cell is None:
                    continue
                by_model[model] = cell.values(metric)
            by_category[category] = by_model
        scores[metric] = by_category

    run_iterations: dict[str, dict[str, list[int]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for category in categories:
        run_iterations[category] = {}
        provenance[category] = {}
        for model in models:
            cell = group.cell(category, model)
            if cell is None:
                continue
            run_iterations[category][model] = sorted(cell.runs)
            provenance[category][model] = cell_provenance(cell)

    return {
        "task": "word_based_summary",
        "dataset": group.dataset,
        "data_run": group.data_run,
        "topics": group.topics,
        "encoder_variant": group.encoder_variant,
        "iterations": list(group.expected_iterations),
        "metrics": list(metrics),
        "models": models,
        "categories": categories,
        "scores": scores,
        "run_iterations": run_iterations,
        "provenance": provenance,
    }


def build_latex_table(
    group: SummaryGroup,
    *,
    metric: str,
    table_digits: int = 4,
) -> str:
    """Review table for one metric of one group (not the manuscript table)."""
    models = group.models
    categories = group.categories
    sample_cell = next(iter(group.cells.values()), None)
    sample = next(iter(sample_cell.runs.values()), {}) if sample_cell else {}
    caption_parts = [
        "Coherence summary",
        group.dataset,
        f"k={group.topics}",
        metric.removeprefix("coherence_"),
    ]
    if group.data_run:
        caption_parts.append(group.data_run)
    if sample.get("coherence_reference"):
        caption_parts.append(f"ref={sample['coherence_reference']}")
    if sample.get("coherence_topn") != "":
        caption_parts.append(f"topn={sample.get('coherence_topn')}")
    if group.encoder_variant:
        caption_parts.append(f"encoder={group.encoder_variant}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        f"\\caption{{{latex_escape_text(', '.join(caption_parts))}}}",
        f"\\begin{{tabular}}{{{'l' + 'c' * len(models)}}}",
        r"\hline",
        "Category & "
        + " & ".join(latex_escape_text(model_table_label(model)) for model in models)
        + r" \\",
        r"\hline",
    ]
    for category in categories:
        cells: dict[str, str] = {}
        means: dict[str, float] = {}
        for model in models:
            cell = group.cell(category, model)
            values = cell.values(metric) if cell is not None else []
            if not values:
                cells[model] = "-"
                continue
            mean, std = mean_std(values)
            cells[model] = format_pm(mean, std, digits=table_digits)
            means[model] = mean
        marked = rank_and_mark(cells, means)
        lines.append(
            latex_escape_text(_category_label(category))
            + " & "
            + " & ".join(marked.get(model, "-") for model in models)
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_run_coverage(group: SummaryGroup, *, metric: str) -> list[dict[str, Any]]:
    expected_runs = len(group.expected_iterations)
    coverage_rows: list[dict[str, Any]] = []
    for category in group.categories:
        for model in group.models:
            cell = group.cell(category, model)
            if cell is None:
                continue
            run_iterations = cell.iterations(metric)
            run_count = len(run_iterations)
            sample = cell.runs[min(cell.runs)] if cell.runs else {}
            coverage_rows.append(
                {
                    "metric": metric.removeprefix("coherence_"),
                    "dataset": group.dataset,
                    "data_run": group.data_run,
                    "topics": group.topics,
                    "coherence_reference": sample.get("coherence_reference", ""),
                    "coherence_topn": sample.get("coherence_topn", ""),
                    "coherence_split": sample.get("coherence_split", ""),
                    "selector": _selector_key(sample),
                    "model": model_table_label(model),
                    "encoder_variant": group.encoder_variant,
                    "encoder_model": _encoder_model_value(sample),
                    "category": category,
                    "run_count": run_count,
                    "expected_runs": expected_runs,
                    "missing_runs": max(expected_runs - run_count, 0),
                    "status": (
                        "missing"
                        if run_count == 0
                        else ("complete" if run_count >= expected_runs else "partial")
                    ),
                    "source_path": sample.get("metrics_path", ""),
                }
            )
    return coverage_rows


# ---------------------------------------------------------------------------
# Writing the summary tree
# ---------------------------------------------------------------------------


def _group_dir(summary_root: Path, group: SummaryGroup) -> Path:
    return (
        summary_root
        / group.dataset
        / group.data_run
        / (_slug(group.encoder_variant) if group.encoder_variant else "no_encoder")
    )


def _group_stem(
    group: SummaryGroup, *, metric: str | None = None, stem_suffix: str = ""
) -> str:
    parts = [
        "coherence",
        _slug(group.dataset),
        _slug(group.data_run),
        *([_slug(metric.removeprefix("coherence_"))] if metric else []),
        _slug(group.encoder_variant) if group.encoder_variant else "no_encoder",
        f"{group.topics}topic",
    ]
    # A vMF hyperparameter-sweep summary is written beside the default one under its
    # label (``..._20topic_kappa0-100``), mirroring the classification sidecars.
    return "_".join(parts) + (f"_{stem_suffix}" if stem_suffix else "")


def write_group_outputs(
    group: SummaryGroup,
    *,
    summary_root: Path,
    metrics: Sequence[str],
    table_digits: int = 4,
    stem_suffix: str = "",
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Write one group's scores sidecar and its per-metric review artifacts."""
    directory = _group_dir(summary_root, group)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        build_scores_payload(group, metrics=metrics),
        directory / f"{_group_stem(group, stem_suffix=stem_suffix)}.scores.json",
    )

    written_tables: list[Path] = []
    coverage_index: list[dict[str, Any]] = []
    for metric in metrics:
        stem = _group_stem(group, metric=metric, stem_suffix=stem_suffix)
        table_path = directory / f"{stem}.tex"
        table_path.write_text(
            build_latex_table(group, metric=metric, table_digits=table_digits) + "\n",
            encoding="utf-8",
        )
        written_tables.append(table_path)

        coverage_rows = build_run_coverage(group, metric=metric)
        _write_json(
            {
                "expected_runs": len(group.expected_iterations),
                "iterations": list(group.expected_iterations),
                "rows": coverage_rows,
                "incomplete": [
                    row for row in coverage_rows if row.get("status") != "complete"
                ],
            },
            directory / f"{stem}.runs.json",
        )
        _write_csv_with_fields(
            rows=coverage_rows,
            fieldnames=RUN_COVERAGE_FIELDS,
            path=directory / f"{stem}.runs.csv",
        )
        coverage_index.extend(
            {"summary_path": _as_project_relative(table_path), **row}
            for row in coverage_rows
        )
    return written_tables, coverage_index


ARM_COMPARISON_KEYS = (
    "dataset",
    "data_run",
    "category",
    "num_topics",
    "model",
    "encoder_model",
)

ARM_COMPARISON_METRICS = (
    "coherence_c_v_mean",
    "coherence_c_npmi_mean",
    "coherence_c_uci_mean",
    "diversity_mean",
)


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _rank_within(rows: Sequence[Mapping[str, Any]], *, metric: str) -> dict[str, int]:
    """Rank models (best first) inside one comparison group."""

    scored = [
        (str(row["model"]), _as_float(row.get(metric)))
        for row in rows
        if _as_float(row.get(metric)) is not None
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))  # type: ignore[operator]
    return {model: index + 1 for index, (model, _) in enumerate(scored)}


def build_arm_comparison(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair the historical vocabulary arm with each filtered arm.

    Rows whose counterpart is missing are still emitted, with the absent side
    left blank, so a partially-run filtered arm is visible rather than silently
    dropped.
    """

    by_arm: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    for row in rows:
        arm = str(row.get("vocabulary_arm") or "full")
        key = tuple(row.get(field, "") for field in ARM_COMPARISON_KEYS)
        by_arm.setdefault(arm, {})[key] = row

    full_rows = by_arm.get("full", {})
    comparisons: list[dict[str, Any]] = []
    for arm, arm_rows in sorted(by_arm.items()):
        if arm == "full":
            continue
        group_keys = {
            tuple(row.get(field, "") for field in DEFAULT_GROUP_BY)
            for row in list(arm_rows.values()) + list(full_rows.values())
        }
        for metric in ARM_COMPARISON_METRICS:
            for group in sorted(group_keys, key=lambda item: tuple(map(str, item))):

                def _in_group(source: dict[tuple[Any, ...], Mapping[str, Any]]):
                    return [
                        row
                        for row in source.values()
                        if tuple(row.get(field, "") for field in DEFAULT_GROUP_BY)
                        == group
                    ]

                full_ranks = _rank_within(_in_group(full_rows), metric=metric)
                arm_ranks = _rank_within(_in_group(arm_rows), metric=metric)
                for key in sorted(
                    set(arm_rows) | set(full_rows),
                    key=lambda item: tuple(map(str, item)),
                ):
                    arm_row = arm_rows.get(key)
                    full_row = full_rows.get(key)
                    reference = arm_row or full_row
                    if reference is None:
                        continue
                    if (
                        tuple(reference.get(field, "") for field in DEFAULT_GROUP_BY)
                        != group
                    ):
                        continue
                    model = str(reference.get("model", ""))
                    score_full = _as_float((full_row or {}).get(metric))
                    score_arm = _as_float((arm_row or {}).get(metric))
                    rank_full = full_ranks.get(model)
                    rank_arm = arm_ranks.get(model)
                    comparisons.append(
                        {
                            **{
                                field: reference.get(field, "")
                                for field in ARM_COMPARISON_KEYS
                            },
                            "arm": arm,
                            "metric": metric,
                            "score_full": (
                                _round_metric(score_full)
                                if score_full is not None
                                else ""
                            ),
                            "score_filtered": (
                                _round_metric(score_arm)
                                if score_arm is not None
                                else ""
                            ),
                            "delta": (
                                _round_metric(score_arm - score_full)
                                if score_full is not None and score_arm is not None
                                else ""
                            ),
                            "rank_full": rank_full if rank_full is not None else "",
                            "rank_filtered": rank_arm if rank_arm is not None else "",
                            "rank_delta": (
                                rank_arm - rank_full
                                if rank_full is not None and rank_arm is not None
                                else ""
                            ),
                            "reference_min_df": (arm_row or {}).get(
                                "reference_min_df", ""
                            ),
                            "reference_max_df_ratio": (arm_row or {}).get(
                                "reference_max_df_ratio", ""
                            ),
                        }
                    )
    return comparisons


def write_concatenated_tables(
    *, summary_root: Path, table_paths: Sequence[Path]
) -> None:
    lines = [
        "% Concatenated coherence summary tables.",
        f"% Source directory: {_as_project_relative(summary_root)}",
        f"% Number of source files: {len(table_paths)}",
        "",
    ]
    for index, path in enumerate(sorted(table_paths), start=1):
        lines.append(f"% --- {index:02d}: {_as_project_relative(path)} ---")
        lines.append(path.read_text(encoding="utf-8").rstrip())
        lines.append("")
    (summary_root / "all_summaries.tex").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def collect_summary_rows(
    *,
    coherence_root: Path,
    source_mode: str = "latest",
    strict: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read every recorded run under ``coherence_root`` into flat rows."""
    if source_mode == "latest":
        sources = discover_latest_sources(coherence_root)
        if not sources:
            sources = discover_indexed_sources(coherence_root)
    elif source_mode == "archive":
        sources = discover_archive_sources(coherence_root)
    elif source_mode == "all":
        sources = [
            *discover_latest_sources(coherence_root),
            *discover_archive_sources(coherence_root),
            *discover_indexed_sources(coherence_root),
        ]
    else:
        raise ValueError(f"Unknown source mode: {source_mode}")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[Path] = set()
    for source in sources:
        metrics_path = source.metrics_path.resolve()
        if metrics_path in seen:
            continue
        seen.add(metrics_path)
        if not metrics_path.exists():
            message = f"Missing metrics file: {source.metrics_path}"
            if strict:
                raise FileNotFoundError(message)
            warnings.append(message)
            continue
        try:
            rows.append(build_summary_row(source))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            message = f"Could not read {source.metrics_path}: {exc}"
            if strict:
                raise
            warnings.append(message)
    return rows, warnings


def build_report_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    coherence_root: Path,
    source_mode: str,
    rank_metric: str | None,
    group_by: Sequence[str],
) -> dict[str, Any]:
    columns = collect_fieldnames(rows)
    best_rows = [
        dict(row)
        for row in rows
        if row.get("is_best") is True or str(row.get("is_best")).lower() == "true"
    ]
    return {
        "_meta": {
            "task": "word_based_summary",
            "schema": "coherence_summary",
            "schema_version": 1,
            "coherence_root": _as_project_relative(coherence_root),
            "source_mode": source_mode,
            "rank_metric": rank_metric,
            "group_by": list(group_by),
            "row_count": len(rows),
            "best_row_count": len(best_rows),
            "warning_count": len(warnings),
            "columns": columns,
        },
        "warnings": list(warnings),
        "results": {
            "columns": columns,
            "rows": [dict(row) for row in rows],
            "best_rows": best_rows,
        },
    }


def run_word_based_summary(
    *,
    coherence_root: Path = DEFAULT_COHERENCE_ROOT,
    source_mode: str = "latest",
    iterations: Sequence[int] | None = None,
    summary_root: Path | None = None,
    datasets: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    table_digits: int = 4,
    include_all_category: bool = False,
    strict: bool = False,
    word2vec: str | None = None,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
    coherence_reference_num_docs: int | None = None,
    write_flat_summary: bool = True,
) -> Path:
    """Summarize the recorded coherence/diversity runs into ``summary_root``.

    ``vmf_variant`` selects the vMF hyperparameter-sweep runs of that label (None = the
    default runs); its sidecars are written beside the default ones with the label as a
    stem suffix, and the flat ``summary.csv`` is left to the default summary.
    """
    vmf_variant = normalize_vmf_parameter_variant(vmf_variant)
    coherence_root = Path(coherence_root)
    summary_root = Path(summary_root) if summary_root else coherence_root / "summaries"

    rows, warnings = collect_summary_rows(
        coherence_root=coherence_root, source_mode=source_mode, strict=strict
    )
    if not rows:
        raise SummaryError(f"No word-based results found under {coherence_root}")

    selected_metrics = list(metrics) if metrics else available_metrics(rows)
    if not selected_metrics:
        raise SummaryError("No per-run metric values found in the recorded results")

    selected = select_rows(
        rows,
        datasets=datasets,
        word2vec=word2vec,
        prior_scale=prior_scale,
        covariance_type=covariance_type,
        vmf_variant=vmf_variant,
        coherence_reference_num_docs=coherence_reference_num_docs,
        warnings=warnings,
    )
    if not selected:
        raise SummaryError("No runs match the requested datasets and condition")
    if vmf_variant is not None:
        # The sweep summary concerns vSLDA alone; the other models' cells would repeat the
        # default summary under a sweep stem.
        selected = [row for row in selected if str(row.get("model", "")) == "vmf"]
        if not selected:
            raise SummaryError(
                f"No vMF Sentence LDA runs recorded for vmf_variant={vmf_variant}"
            )

    groups = group_summary_rows(
        selected,
        iterations=iterations,
        strict=strict,
        warnings=warnings,
    )
    if not include_all_category:
        for group in groups:
            for key, cell in list(group.cells.items()):
                if cell.category == "all":
                    del group.cells[key]

    all_tables: list[Path] = []
    coverage_index: list[dict[str, Any]] = []
    for group in groups:
        if not group.cells:
            continue
        tables, coverage = write_group_outputs(
            group,
            summary_root=summary_root,
            metrics=selected_metrics,
            table_digits=table_digits,
            stem_suffix=vmf_variant or "",
        )
        all_tables.extend(tables)
        coverage_index.extend(coverage)

    summary_root.mkdir(parents=True, exist_ok=True)
    write_concatenated_tables(summary_root=summary_root, table_paths=all_tables)
    indexed_fields = ["summary_path", *RUN_COVERAGE_FIELDS]
    _write_csv_with_fields(
        rows=coverage_index,
        fieldnames=indexed_fields,
        path=summary_root / "run_coverage.csv",
    )
    _write_csv_with_fields(
        rows=[row for row in coverage_index if row.get("status") != "complete"],
        fieldnames=indexed_fields,
        path=summary_root / "run_coverage_incomplete.csv",
    )

    if write_flat_summary and vmf_variant is None:
        rank_metric = choose_default_rank_metric(rows)
        ranked = sort_rows(
            add_rank_columns(rows, rank_metric=rank_metric, group_by=DEFAULT_GROUP_BY)
        )
        payload = build_report_payload(
            rows=ranked,
            warnings=warnings,
            coherence_root=coherence_root,
            source_mode=source_mode,
            rank_metric=rank_metric,
            group_by=DEFAULT_GROUP_BY,
        )
        _write_json(payload, coherence_root / "summary.json")
        _write_csv(ranked, coherence_root / "summary.csv")
        comparison = build_arm_comparison(ranked)
        if comparison:
            _write_csv(comparison, coherence_root / "arm_comparison.csv")

    for message in warnings:
        print(f"[word-based-summary] warning: {message}")
    return summary_root

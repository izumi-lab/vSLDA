"""Aggregate the topic-count (K) sweep from artifacts that already exist.

The classification and word-based pipelines each answer "how good is this model
at K?", but they answer it in different files, one file per K, keyed by
different spellings of the model name. Nothing here recomputes a metric: this
module reads

``results/classification/summaries/<ds>/<run>/<clf>/<enc>/*_<K>topic.scores.json``
    per-seed classification scores, laid out as ``category -> model -> [value]``
``results/topic_analysis/coherence/summary.csv``
    the tidy word-based summary, which already carries ``num_topics``

and flattens both onto one long table whose unit is a single
(dataset, category, model, K, seed, metric) observation. Everything downstream
-- the figures in :mod:`src.evaluation.reports.topic_sweep_plot` and the LaTeX
tables written here -- reads that table and nothing else.

Two details make the join possible and are handled here rather than by callers:

*model identity*
    ``canonical_model_key`` folds ``"Blei LDA [SVM]"`` and ``"bleilda"``, and
    ``"vMF Sentence LDA [c1_minilm] [SVM]"`` and ``"vmf"``, onto one key.
*corpus size*
    ``all`` holds 19 (20 Newsgroups) or 24 (NYT) labels while a coarse category
    holds 2-9, so a raw K is not comparable across them. Every row carries
    ``num_labels`` and ``topics_per_label = K / num_labels``.

Conditions that were asked for but are absent are not an error: they are
recorded in the ``missing`` list of the sidecar JSON, so a partially finished
sweep still produces a table and says exactly what it is waiting for.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.paths import CLASSIFICATION_RESULTS_ROOT, REPO_ROOT, RESULTS_ROOT
from src.core.vmf_assignment import DEFAULT_VMF_ASSIGNMENT as DEFAULT_ESTIMATOR
from src.core.vmf_assignment import VMF_FAMILY_MODELS
from src.data.catalog import get_dataset_targets
from src.evaluation.reporting import write_csv_rows, write_tabular_report_json
from src.evaluation.reports.latex_tables import (
    format_pm,
    latex_escape_text,
    mean_std,
    rank_and_mark,
)
from src.evaluation.reports.model_style import (
    MODEL_ORDER,
    canonical_model_key,
    display_model_name,
    label_for_key,
)

__all__ = [
    "DEFAULT_CLASSIFICATION_METRICS",
    "DEFAULT_CLASSIFICATION_ROOT",
    "DEFAULT_COHERENCE_SUMMARY",
    "DEFAULT_MODELS",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_WORD_BASED_METRICS",
    "LOWER_IS_BETTER_METRICS",
    "SWEEP_FIELDS",
    "SweepCollection",
    "build_topic_sweep_latex",
    "collect_topic_sweep_rows",
    "resolve_categories",
    "run_topic_sweep_summary",
]

DEFAULT_OUT_ROOT = RESULTS_ROOT / "analysis" / "topic_sweep"
DEFAULT_CLASSIFICATION_ROOT = CLASSIFICATION_RESULTS_ROOT / "summaries"
DEFAULT_COHERENCE_SUMMARY = (
    RESULTS_ROOT / "topic_analysis" / "coherence" / "summary.csv"
)

DEFAULT_MODELS = ("vmf_sentence_lda", "bleilda", "sentlda", "etm")
DEFAULT_CLASSIFICATION_METRICS = ("acc", "f1mac")
DEFAULT_WORD_BASED_METRICS = (
    "coherence_c_npmi",
    "coherence_c_v",
    "diversity",
    "num_active_topics",
    "num_empty_topics",
    "topic_utilization",
)
# Metrics where a smaller value is the better one; the LaTeX tables leave these
# unmarked rather than bolding the largest cell.
LOWER_IS_BETTER_METRICS = frozenset({"num_empty_topics"})

# Composite reported alongside the raw word-based metrics. C_V is used rather
# than NPMI because NPMI turns negative once K outgrows the corpus, which makes
# the product reward a redundant model over a specialised one. C_V stays in
# [0, 1], so multiplying by diversity penalises duplicated topics as intended.
TOPIC_QUALITY_METRIC = "topic_quality"
TOPIC_QUALITY_FACTORS = ("coherence_c_v", "diversity")

CLASSIFICATION_SOURCE = "classification"
# The vSLDA document-topic estimator whose classification sidecars are read.
DEFAULT_VMF_ASSIGNMENT = DEFAULT_ESTIMATOR
VMF_MODEL_KEY = "vmf_sentence_lda"
WORD_BASED_SOURCE = "word_based"

SWEEP_FIELDS = [
    "dataset",
    "data_run",
    "category",
    "model",
    "model_label",
    "num_topics",
    "num_labels",
    "topics_per_label",
    "iteration",
    "metric",
    "value",
    "source",
    "embedding_variant",
    "classifier",
    "condition_id",
]
SOURCE_FIELDS = ["source", "path", "rows"]
MISSING_FIELDS = ["source", "dataset", "category", "model", "num_topics"]

# ``acc_20newsgroup_default_svm_minilm_hard_30topic.scores.json``: the dataset
# and the data run can both contain underscores, so only the tail is parsed and
# everything else is read from the payload.
_SCORES_TAIL_RE = re.compile(
    r"_(?P<assignment>[a-z]+)_(?P<topics>\d+)topic\.scores\.json$"
)
# Models whose artifacts are tied to a sentence encoder. For every other model
# the encoder is meaningless and an encoder filter must not exclude it.
_ENCODER_DEPENDENT_MODELS = frozenset(
    {"vmf_sentence_lda", "sentence_gaussianlda", "ctm", "bertopic_kmeans"}
)


class TopicSweepError(Exception):
    """Raised when the sweep cannot be assembled at all."""


@dataclass
class SweepCollection:
    """The long table plus what it was read from and what was missing."""

    rows: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    missing: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_project_relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _split_values(raw: Any) -> list[str]:
    """Split a summary.csv list cell, which joins with ``;`` (or ``,``)."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw]
    text = str(raw).strip()
    if not text:
        return []
    separator = ";" if ";" in text else ","
    return [item.strip() for item in text.split(separator) if item.strip()]


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


def _label_counts(dataset: str) -> dict[str, int]:
    targets = get_dataset_targets(dataset) or {}
    return {category: len(labels) for category, labels in targets.items()}


def resolve_categories(
    dataset: str,
    *,
    categories: Sequence[str] | None,
    include_all_category: bool,
) -> list[str]:
    """Categories to aggregate, with ``all`` opted in explicitly.

    ``all`` is the axis this analysis is about, but it is also the one category
    the classification summaries omit by default, so it is never inferred: it is
    included when asked for and reported as missing when its artifacts are not
    there yet.
    """
    if categories:
        resolved = [
            str(category).strip() for category in categories if str(category).strip()
        ]
    else:
        known = _label_counts(dataset)
        resolved = [category for category in known if category != "all"]
    if include_all_category and "all" not in resolved:
        resolved.append("all")
    if not include_all_category:
        resolved = [category for category in resolved if category != "all"]
    return resolved


def _variant_matches(model_key: str, variant: str, requested: str | None) -> bool:
    if not requested:
        return True
    if model_key not in _ENCODER_DEPENDENT_MODELS:
        return True
    return _short_variant(variant) == _short_variant(requested)


def _short_variant(variant: Any) -> str:
    """``"minilm_raw"`` -> ``"minilm"``; ``"c1_minilm"`` -> ``"minilm"``."""
    text = str(variant or "").strip()
    if not text:
        return ""
    if text.startswith("c") and "_" in text and text.split("_", 1)[0][1:].isdigit():
        text = text.split("_", 1)[1]
    return text.split("_", 1)[0]


def _row_key(row: Mapping[str, Any]) -> tuple:
    return (
        row["dataset"],
        row["data_run"],
        row["category"],
        row["model"],
        row["num_topics"],
        row["iteration"],
        row["metric"],
        row["source"],
        row["classifier"],
    )


# ---------------------------------------------------------------------------
# Classification side
# ---------------------------------------------------------------------------


def _scores_files(
    root: Path,
    *,
    dataset: str,
    data_run: str,
    classifier: str,
    embedding_variant: str,
    topics: Sequence[int],
    vmf_assignment: str = DEFAULT_VMF_ASSIGNMENT,
) -> list[Path]:
    """The summary sidecars of one (dataset, run, classifier, encoder, assignment).

    The vSLDA document-topic estimator (``hard``, ``soft``, ``foldin``) is the
    ``<assignment>`` field of the file name; sidecars of another estimator share
    the directory and must not be pooled with the requested one.
    """
    directory = root / dataset / data_run / classifier / embedding_variant
    if not directory.is_dir():
        return []
    wanted = {int(topic) for topic in topics}
    files: list[Path] = []
    for path in sorted(directory.glob("*.scores.json")):
        match = _SCORES_TAIL_RE.search(path.name)
        if match is None or int(match.group("topics")) not in wanted:
            continue
        if match.group("assignment") != str(vmf_assignment):
            continue
        files.append(path)
    return files


def _classification_rows(
    path: Path,
    *,
    metrics: Sequence[str],
    models: Sequence[str],
    categories: Sequence[str],
    embedding_variant: str,
    label_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    metric = str(payload.get("metric") or "")
    if metric not in set(metrics):
        return []
    num_topics = int(payload.get("topics") or 0)
    iterations = [int(value) for value in payload.get("iterations") or []]
    classifiers = payload.get("classifiers") or []
    classifier = str(classifiers[0]) if classifiers else ""
    dataset = str(payload.get("dataset") or "")
    data_run = str(payload.get("data_run") or "")
    scores = payload.get("scores") or {}
    provenance = payload.get("provenance") or {}
    wanted_models = set(models)
    wanted_categories = set(categories)

    rows: list[dict[str, Any]] = []
    for category, per_model in scores.items():
        if category not in wanted_categories:
            continue
        num_labels = label_counts.get(category)
        for raw_model, values in (per_model or {}).items():
            model_key = canonical_model_key(raw_model)
            if model_key not in wanted_models:
                continue
            cell_provenance = (provenance.get(category) or {}).get(raw_model) or {}
            variant = cell_provenance.get("embedding_variant") or ""
            if not _variant_matches(model_key, variant, embedding_variant):
                continue
            for index, value in enumerate(values or []):
                number = _as_float(value)
                if number is None:
                    continue
                iteration = iterations[index] if index < len(iterations) else index
                rows.append(
                    {
                        "dataset": dataset,
                        "data_run": data_run,
                        "category": category,
                        "model": model_key,
                        "model_label": label_for_key(model_key)
                        or display_model_name(raw_model),
                        "num_topics": num_topics,
                        "num_labels": num_labels,
                        "topics_per_label": (
                            round(num_topics / num_labels, 6) if num_labels else None
                        ),
                        "iteration": iteration,
                        "metric": metric,
                        "value": number,
                        "source": CLASSIFICATION_SOURCE,
                        "embedding_variant": variant or embedding_variant,
                        "classifier": classifier,
                        "condition_id": cell_provenance.get("source_condition_id")
                        or "",
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Word-based side
# ---------------------------------------------------------------------------


def _coherence_rows(
    summary_path: Path,
    *,
    datasets: Sequence[str],
    data_run: str,
    metrics: Sequence[str],
    models: Sequence[str],
    categories_by_dataset: Mapping[str, Sequence[str]],
    topics: Sequence[int],
    embedding_variant: str,
    label_counts_by_dataset: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    wanted_models = set(models)
    wanted_topics = {int(topic) for topic in topics}
    wanted_metrics = list(metrics)
    rows: list[dict[str, Any]] = []
    # (condition, metric, seed) -> row; ties are broken by ``started_at`` because
    # the same condition can be recomputed on another host under a new
    # execution id (the reference corpus path is part of the fingerprint).
    newest: dict[tuple, tuple[str, dict[str, Any]]] = {}

    with summary_path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            dataset = str(record.get("dataset") or "")
            if dataset not in set(datasets):
                continue
            if str(record.get("data_run") or "") != data_run:
                continue
            category = str(record.get("category") or "")
            if category not in set(categories_by_dataset.get(dataset, ())):
                continue
            num_topics = int(_as_float(record.get("num_topics")) or 0)
            if num_topics not in wanted_topics:
                continue
            model_key = canonical_model_key(
                record.get("model") or record.get("runner_family") or ""
            )
            if model_key not in wanted_models:
                continue
            variant = (
                record.get("effective_embedding_variant")
                or record.get("embedding_variant")
                or record.get("encoder_model")
                or ""
            )
            if not _variant_matches(model_key, variant, embedding_variant):
                continue

            iterations = [
                int(_as_float(item) or 0)
                for item in _split_values(record.get("iterations"))
            ]
            num_labels = (label_counts_by_dataset.get(dataset) or {}).get(category)
            started_at = str(record.get("started_at") or "")
            for metric in wanted_metrics:
                values = _split_values(record.get(f"{metric}_values"))
                if values and len(values) == len(iterations):
                    pairs = list(zip(iterations, values))
                else:
                    # No per-seed breakdown: fall back to the aggregate, which for
                    # a single-iteration row is the seed value itself.
                    mean = record.get(f"{metric}_mean")
                    if mean in (None, ""):
                        continue
                    iteration = iterations[0] if len(iterations) == 1 else None
                    pairs = [(iteration, mean)]
                for iteration, raw_value in pairs:
                    number = _as_float(raw_value)
                    if number is None:
                        continue
                    row = {
                        "dataset": dataset,
                        "data_run": data_run,
                        "category": category,
                        "model": model_key,
                        "model_label": label_for_key(model_key),
                        "num_topics": num_topics,
                        "num_labels": num_labels,
                        "topics_per_label": (
                            round(num_topics / num_labels, 6) if num_labels else None
                        ),
                        "iteration": iteration,
                        "metric": metric,
                        "value": number,
                        "source": WORD_BASED_SOURCE,
                        "embedding_variant": _short_variant(variant),
                        "classifier": "",
                        "condition_id": str(record.get("condition_id") or ""),
                    }
                    key = _row_key(row)
                    previous = newest.get(key)
                    if previous is None or previous[0] <= started_at:
                        newest[key] = (started_at, row)

    rows.extend(row for _, row in newest.values())
    return rows


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def derive_topic_quality_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Emit ``topic_quality`` = C_V coherence x topic diversity per condition.

    Both factors come from the same word-based row set, so a condition only
    yields a composite when both were collected for the same seed.
    """
    factor_a, factor_b = TOPIC_QUALITY_FACTORS
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("source") != WORD_BASED_SOURCE:
            continue
        metric = row.get("metric")
        if metric not in TOPIC_QUALITY_FACTORS:
            continue
        key = (
            row.get("dataset"),
            row.get("data_run"),
            row.get("category"),
            row.get("model"),
            row.get("num_topics"),
            row.get("iteration"),
        )
        entry = grouped.setdefault(key, {"template": row})
        entry[metric] = row.get("value")

    derived: list[dict[str, Any]] = []
    for entry in grouped.values():
        left = _as_float(entry.get(factor_a))
        right = _as_float(entry.get(factor_b))
        if left is None or right is None:
            continue
        row = dict(entry["template"])
        row["metric"] = TOPIC_QUALITY_METRIC
        row["value"] = round(left * right, 6)
        derived.append(row)
    return derived


def collect_topic_sweep_rows(
    *,
    datasets: Sequence[str],
    topics: Sequence[int],
    models: Sequence[str] = DEFAULT_MODELS,
    categories: Sequence[str] | None = None,
    include_all_category: bool = True,
    data_run: str = "default",
    classifier: str = "svm",
    embedding_variant: str = "minilm",
    classification_metrics: Sequence[str] = DEFAULT_CLASSIFICATION_METRICS,
    word_based_metrics: Sequence[str] = DEFAULT_WORD_BASED_METRICS,
    classification_root: Path = DEFAULT_CLASSIFICATION_ROOT,
    coherence_summary: Path = DEFAULT_COHERENCE_SUMMARY,
    vmf_assignment: str = DEFAULT_VMF_ASSIGNMENT,
    baseline_vmf_assignment: str | None = None,
) -> SweepCollection:
    """Read both pipelines into one long table.

    Missing artifacts never raise; the returned ``missing`` list names every
    requested (source, dataset, category, model, K) that produced no row.

    ``baseline_vmf_assignment`` reads the other baselines from the sidecars of
    another estimator: the fold-in sidecars (``_foldin_``, ``_foldincounts_``)
    carry the vMF family alone (vSLDA and MvTM / vLDA, whose document-topic
    features depend on the estimator), so a sweep under a fold-in estimator
    takes those two from the fold-in sidecars and every other model from the
    ``hard`` sidecars. ``None`` reads everything from ``vmf_assignment``.
    """
    model_keys = [canonical_model_key(model) for model in models]
    if baseline_vmf_assignment is None or baseline_vmf_assignment == vmf_assignment:
        file_sets: list[tuple[str, list[str]]] = [(vmf_assignment, model_keys)]
    else:
        vmf_keys = [key for key in model_keys if key in VMF_FAMILY_MODELS]
        other_keys = [key for key in model_keys if key not in VMF_FAMILY_MODELS]
        file_sets = [(vmf_assignment, vmf_keys), (baseline_vmf_assignment, other_keys)]
    topic_list = sorted({int(topic) for topic in topics})
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    categories_by_dataset: dict[str, list[str]] = {}
    label_counts_by_dataset: dict[str, dict[str, int]] = {}
    for dataset in datasets:
        label_counts_by_dataset[dataset] = _label_counts(dataset)
        categories_by_dataset[dataset] = resolve_categories(
            dataset,
            categories=categories,
            include_all_category=include_all_category,
        )

    for dataset in datasets:
        for assignment, wanted_keys in file_sets:
            if not wanted_keys:
                continue
            for path in _scores_files(
                Path(classification_root),
                dataset=dataset,
                data_run=data_run,
                classifier=classifier,
                embedding_variant=embedding_variant,
                topics=topic_list,
                vmf_assignment=assignment,
            ):
                file_rows = _classification_rows(
                    path,
                    metrics=classification_metrics,
                    models=wanted_keys,
                    categories=categories_by_dataset[dataset],
                    embedding_variant=embedding_variant,
                    label_counts=label_counts_by_dataset[dataset],
                )
                rows.extend(file_rows)
                sources.append(
                    {
                        "source": CLASSIFICATION_SOURCE,
                        "path": _as_project_relative(path),
                        "rows": len(file_rows),
                    }
                )

    coherence_path = Path(coherence_summary)
    if coherence_path.is_file():
        coherence_rows = _coherence_rows(
            coherence_path,
            datasets=datasets,
            data_run=data_run,
            metrics=word_based_metrics,
            models=model_keys,
            categories_by_dataset=categories_by_dataset,
            topics=topic_list,
            embedding_variant=embedding_variant,
            label_counts_by_dataset=label_counts_by_dataset,
        )
        rows.extend(coherence_rows)
        sources.append(
            {
                "source": WORD_BASED_SOURCE,
                "path": _as_project_relative(coherence_path),
                "rows": len(coherence_rows),
            }
        )

    rows.extend(derive_topic_quality_rows(rows))

    missing = _missing_conditions(
        rows,
        datasets=datasets,
        categories_by_dataset=categories_by_dataset,
        models=model_keys,
        topics=topic_list,
        classification_metrics=classification_metrics,
        word_based_metrics=word_based_metrics,
    )
    category_rank = {
        (dataset, category): index
        for dataset, names in categories_by_dataset.items()
        for index, category in enumerate(names)
    }
    rows.sort(
        key=lambda row: (
            row["dataset"],
            category_rank.get((row["dataset"], row["category"]), len(category_rank)),
            row["metric"],
            _model_order(row["model"]),
            row["num_topics"],
            row["iteration"] if row["iteration"] is not None else -1,
        )
    )
    return SweepCollection(rows=rows, sources=sources, missing=missing)


def _model_order(model_key: str) -> tuple[int, str]:
    label = label_for_key(model_key)
    try:
        return (MODEL_ORDER.index(label), label)
    except ValueError:
        return (len(MODEL_ORDER), label)


def _missing_conditions(
    rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str],
    categories_by_dataset: Mapping[str, Sequence[str]],
    models: Sequence[str],
    topics: Sequence[int],
    classification_metrics: Sequence[str],
    word_based_metrics: Sequence[str],
) -> list[dict[str, Any]]:
    present: set[tuple] = {
        (
            row["source"],
            row["dataset"],
            row["category"],
            row["model"],
            row["num_topics"],
        )
        for row in rows
    }
    wanted_sources = []
    if classification_metrics:
        wanted_sources.append(CLASSIFICATION_SOURCE)
    if word_based_metrics:
        wanted_sources.append(WORD_BASED_SOURCE)

    missing: list[dict[str, Any]] = []
    for source in wanted_sources:
        for dataset in datasets:
            for category in categories_by_dataset.get(dataset, ()):
                for model in models:
                    for topic in topics:
                        key = (source, dataset, category, model, int(topic))
                        if key in present:
                            continue
                        missing.append(
                            {
                                "source": source,
                                "dataset": dataset,
                                "category": category,
                                "model": model,
                                "num_topics": int(topic),
                            }
                        )
    return missing


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def build_topic_sweep_latex(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    metric: str,
    topics: Sequence[int],
    digits: int = 3,
) -> str:
    """One ``model x K`` tabular per category, best cell per K column marked."""
    topic_list = sorted({int(topic) for topic in topics})
    values: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        if row["dataset"] != dataset or row["metric"] != metric:
            continue
        values[row["category"]][row["model"]][int(row["num_topics"])].append(
            float(row["value"])
        )
    if not values:
        return ""

    blocks: list[str] = []
    for category in sorted(values, key=lambda name: (name != "all", name)):
        per_model = values[category]
        models = sorted(per_model, key=_model_order)
        cells: dict[int, dict[str, str]] = {}
        means: dict[int, dict[str, float]] = {}
        for topic in topic_list:
            column_cells: dict[str, str] = {}
            column_means: dict[str, float] = {}
            for model in models:
                observations = per_model[model].get(topic) or []
                if not observations:
                    continue
                mean, std = mean_std(observations)
                column_cells[model] = format_pm(mean, std, digits=digits)
                column_means[model] = mean
            if metric not in LOWER_IS_BETTER_METRICS:
                column_cells = rank_and_mark(column_cells, column_means)
            cells[topic] = column_cells
            means[topic] = column_means

        header = " & ".join(["Model"] + [f"$K={topic}$" for topic in topic_list])
        body_lines = []
        for model in models:
            row_cells = [latex_escape_text(label_for_key(model))]
            row_cells.extend(cells[topic].get(model, "--") for topic in topic_list)
            body_lines.append(" & ".join(row_cells) + r" \\")
        blocks.append(
            "\n".join(
                [
                    r"\begin{tabular}{l" + "r" * len(topic_list) + "}",
                    r"\hline",
                    f"\\multicolumn{{{len(topic_list) + 1}}}{{l}}"
                    f"{{{latex_escape_text(dataset)} / {latex_escape_text(category)}"
                    f" -- {latex_escape_text(metric)}}} \\\\",
                    r"\hline",
                    header + r" \\",
                    r"\hline",
                    *body_lines,
                    r"\hline",
                    r"\end{tabular}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_topic_sweep_summary(
    *,
    datasets: Sequence[str],
    topics: Sequence[int],
    models: Sequence[str] = DEFAULT_MODELS,
    categories: Sequence[str] | None = None,
    include_all_category: bool = True,
    data_run: str = "default",
    classifier: str = "svm",
    embedding_variant: str = "minilm",
    classification_metrics: Sequence[str] = DEFAULT_CLASSIFICATION_METRICS,
    word_based_metrics: Sequence[str] = DEFAULT_WORD_BASED_METRICS,
    classification_root: Path = DEFAULT_CLASSIFICATION_ROOT,
    coherence_summary: Path = DEFAULT_COHERENCE_SUMMARY,
    out_root: Path = DEFAULT_OUT_ROOT,
    write_latex: bool = True,
    plot: bool = False,
    paper: bool = False,
    vmf_assignment: str = DEFAULT_VMF_ASSIGNMENT,
    baseline_vmf_assignment: str | None = None,
) -> Path:
    """Write ``topic_sweep.csv``, its provenance sidecar and the LaTeX tables.

    Returns the output root; one subdirectory per dataset is written under it.
    """
    if not datasets:
        raise TopicSweepError("At least one dataset is required.")
    if not topics:
        raise TopicSweepError("At least one topic count is required.")

    collection = collect_topic_sweep_rows(
        datasets=datasets,
        topics=topics,
        models=models,
        categories=categories,
        include_all_category=include_all_category,
        data_run=data_run,
        classifier=classifier,
        embedding_variant=embedding_variant,
        classification_metrics=classification_metrics,
        word_based_metrics=word_based_metrics,
        classification_root=classification_root,
        coherence_summary=coherence_summary,
        vmf_assignment=vmf_assignment,
        baseline_vmf_assignment=baseline_vmf_assignment,
    )

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_root = Path(out_root)
    topic_list = sorted({int(topic) for topic in topics})

    for dataset in datasets:
        dataset_rows = [row for row in collection.rows if row["dataset"] == dataset]
        dataset_dir = out_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        write_csv_rows(
            fieldnames=SWEEP_FIELDS,
            rows=dataset_rows,
            path=dataset_dir / "topic_sweep.csv",
        )
        write_tabular_report_json(
            meta={
                "task": "topic_sweep_summary",
                "started_at": started_at,
                "dataset": dataset,
                "data_run": data_run,
                "topics": topic_list,
                "models": [canonical_model_key(model) for model in models],
                "categories": resolve_categories(
                    dataset,
                    categories=categories,
                    include_all_category=include_all_category,
                ),
                "classifier": classifier,
                "embedding_variant": embedding_variant,
                "vmf_assignment": vmf_assignment,
                "baseline_vmf_assignment": (
                    vmf_assignment
                    if baseline_vmf_assignment is None
                    else baseline_vmf_assignment
                ),
                "classification_metrics": list(classification_metrics),
                "word_based_metrics": list(word_based_metrics),
                "row_count": len(dataset_rows),
                "missing": [
                    entry for entry in collection.missing if entry["dataset"] == dataset
                ],
            },
            columns=SOURCE_FIELDS,
            rows=collection.sources,
            path=dataset_dir / "topic_sweep.json",
        )

        if write_latex:
            table_metrics = [
                *classification_metrics,
                *word_based_metrics,
                TOPIC_QUALITY_METRIC,
            ]
            for metric in table_metrics:
                table = build_topic_sweep_latex(
                    dataset_rows,
                    dataset=dataset,
                    metric=metric,
                    topics=topic_list,
                )
                if not table:
                    continue
                (dataset_dir / f"topic_sweep_{metric}.tex").write_text(table)

        print(f"[write] {dataset_dir / 'topic_sweep.csv'} ({len(dataset_rows)} rows)")

        if plot:
            # Imported here so that aggregating never requires a plotting backend.
            from src.evaluation.reports.topic_sweep_plot import run_topic_sweep_plots

            run_topic_sweep_plots(
                dataset=dataset,
                sweep_root=out_root,
                topics=topic_list,
                primary_metric=(
                    classification_metrics[0] if classification_metrics else "acc"
                ),
                paper=paper,
            )

    return out_root

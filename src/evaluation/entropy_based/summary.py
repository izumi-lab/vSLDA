"""Cross-condition summary of entropy-based metrics for reporting.

The per-invocation ``summary.csv`` written by :mod:`.metrics` is overwritten on
every run, so the paper tables are rebuilt from the durable
``latest/<dataset>/<data_run>/<category>/<condition_id>/CURRENT.json`` pointers.

Two kinds of output leave this module:

* review artifacts (``entropy_summary_wide.csv`` etc., the ``.tex`` fragments
  and the per-document histograms), which carry means and pooled standard
  deviations for reading here; and
* ``*.scores.json`` sidecars, one per (dataset, data run, sentence encoder, K),
  which carry the raw per-run values, the per-topic arrays and the provenance
  of every cell and nothing aggregated. That is the interface downstream
  consumers (the paper repository) read; they do their own aggregation from
  it, so no rounded or ranked number crosses the boundary. The layout mirrors
  the coherence sidecars of :mod:`src.evaluation.word_based.summary`.

``--paper`` additionally draws the manuscript's box-plot figure of the
per-topic and per-document distributions at the page geometry of
:mod:`src.evaluation.reports.grid_layout`.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import matplotlib.ticker
import numpy as np

from src.core.artifacts import load_json
from src.core.paths import resolve_project_path
from src.evaluation.entropy_based.entropy_metrics import SUMMARY_METRIC_KEYS
from src.evaluation.entropy_based.inputs import (
    SENTENCE_ENCODER_MODELS,
    normalize_model_name,
)
from src.evaluation.entropy_based.metrics import DEFAULT_OUT_ROOT, METRICS_FILENAME
from src.evaluation.reporting import (
    read_evaluation_json,
    write_csv_rows,
    write_tabular_report_json,
)
from src.evaluation.reports.grid_layout import (
    DEFAULT_FORMATS,
    DPI,
    GRID_MARGIN_TOP,
    PAPER_RC,
    grid_geometry,
    panel_frame,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

SUMMARY_TASK_NAME = "entropy_based_summary"
DEFAULT_EXCLUDE_CATEGORIES: tuple[str, ...] = ("all",)
GROUP_COLUMNS: tuple[str, ...] = (
    "dataset",
    "data_run",
    "model",
    "embedding_variant",
    "parameter_variant",
    "num_topics",
    "split",
    "doc_topic_source",
)
MODEL_LABELS: dict[str, str] = {
    "vmf": "vMF Sentence LDA",
    "bleilda": "LDA",
    "sam": "SAM (tf-idf)",
    "sam_tf": "SAM",
    "sentlda": "sentLDA",
    "sentence_gaussianlda": "Sentence Gaussian LDA",
    "gaussianlda": "Gaussian LDA",
    "mvtm": "MvTM",
    "etm": "ETM",
    "ctm": "CTM",
    "senclu": "SenClu",
}
# Scalar per-run values the scores sidecar carries, in this order. They are the
# summary keys plus the document count of the run, which a consumer needs to
# turn a normalized entropy back into a number of documents.
SCORES_METRIC_KEYS: tuple[str, ...] = (*SUMMARY_METRIC_KEYS, "num_documents")
# Per-topic arrays the sidecar carries (one list per run), so that a consumer
# can apply its own diffuse/dead cut-offs instead of the recorded ones.
SCORES_PER_TOPIC_KEYS: tuple[str, ...] = (
    "topic_doc_entropy_normalized",
    "topic_rank1_doc_fraction",
)
# Protocol fields every condition in one sidecar must agree on; they are
# recorded in each cell's provenance so a consumer can check what it claims.
SCORES_PROTOCOL_FIELDS: tuple[str, ...] = (
    "split",
    "doc_topic_source",
    "diffuse_entropy_threshold",
    "dead_rank1_threshold",
    "log_base",
    "metric_schema_version",
)
# Group of a sidecar when no sentence-encoder model contributes a variant.
NO_ENCODER_GROUP = "no_encoder"
# 本稿の GSLDA は NIW 事前の Ψ₀=0.1 で報告する（Ψ₀ 掃引は別節の結果）。
MANUSCRIPT_PRIOR_SCALE = 0.1
# Column order of the sidecar's ``models`` list (the paper's table order).
SCORES_MODEL_ORDER: tuple[str, ...] = (
    "bleilda",
    "sentlda",
    "gaussianlda",
    "sam",
    "sam_tf",
    "mvtm",
    "etm",
    "ctm",
    "sentence_gaussianlda",
    "vmf",
    "senclu",
)
# The manuscript figure: the topic count and encoder of the main-text tables.
PAPER_NUM_TOPICS = 20
PAPER_ENCODER = "minilm"
# The manuscript reports SAM on raw term frequencies (runner ``sam_tf``, label
# "SAM"); the tf-idf runner ``sam`` stays in the summaries and sidecars for the
# record but is not drawn, so the figure shows the nine models of the tables.
PAPER_EXCLUDED_MODELS: frozenset[str] = frozenset({"sam"})
PAPER_DATASET_LABELS: dict[str, str] = {"20newsgroup": "20 Newsgroups", "nyt": "NYT"}
# Rotated tick labels below every row and no shared x label: the gutters and
# the bottom margin are wider than the reference grid's.
PAPER_HSPACE = 0.62
PAPER_MARGIN_BOTTOM = 0.55
PAPER_BOX_ALPHA = 0.35
METRIC_LABELS: dict[str, str] = {
    "doc_topic_entropy_normalized_mean": r"$\bar{H}(\theta_d)/\ln K$",
    "doc_topic_entropy_normalized_median": r"$\mathrm{med}\,H(\theta_d)/\ln K$",
    "topic_doc_entropy_normalized_mean": r"$\bar{H}(P(d|k))/\ln D$",
    "topic_doc_entropy_normalized_max": r"$\max_k H(P(d|k))/\ln D$",
    "topic_rank1_doc_fraction_min": r"$\min_k$ rank-1",
    "topic_rank1_doc_fraction_std": r"$\mathrm{sd}_k$ rank-1",
    "topic_diffuse_fraction": "diffuse topics",
    "topic_dead_fraction": "dead topics",
}


def _matches(value: Any, allowed: Sequence[Any] | None) -> bool:
    if allowed is None:
        return True
    return str(value) in {str(item) for item in allowed}


def collect_entropy_conditions(
    *,
    out_root: Path = DEFAULT_OUT_ROOT,
    datasets: Sequence[str] | None = None,
    data_runs: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    topics: Sequence[int] | None = None,
    split: str | None = None,
    doc_topic_source: str | None = None,
    exclude_categories: Sequence[str] = DEFAULT_EXCLUDE_CATEGORIES,
) -> list[dict[str, Any]]:
    """Read every condition reachable from ``latest/**/CURRENT.json``."""
    latest_root = Path(out_root) / "latest"
    excluded = {str(value) for value in exclude_categories}
    records: list[dict[str, Any]] = []
    if not latest_root.exists():
        return records
    for pointer_path in sorted(latest_root.glob("*/*/*/*/CURRENT.json")):
        category = pointer_path.parents[1].name
        if category in excluded:
            continue
        payload = load_json(pointer_path)
        if not isinstance(payload, dict) or not payload.get("archive_dir"):
            continue
        archive_dir = resolve_project_path(str(payload["archive_dir"]))
        metrics_path = archive_dir / METRICS_FILENAME
        if not metrics_path.exists():
            logger.warning("missing metrics file for pointer %s", pointer_path)
            continue
        meta, results = read_evaluation_json(metrics_path)
        if not isinstance(results, dict):
            continue
        if not _matches(meta.get("dataset"), datasets):
            continue
        if not _matches(meta.get("data_run"), data_runs):
            continue
        if not _matches(meta.get("model"), models):
            continue
        if not _matches(meta.get("num_topics"), topics):
            continue
        if split is not None and str(meta.get("split")) != str(split):
            continue
        if doc_topic_source is not None and str(meta.get("doc_topic_source")) != str(
            doc_topic_source
        ):
            continue
        records.append(
            {
                "meta": meta,
                "results": results,
                "archive_dir": archive_dir,
                "pointer_path": pointer_path,
                "category": str(meta.get("category", category)),
            }
        )
    return records


def _group_key(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(meta.get("dataset")),
        str(meta.get("data_run")),
        str(meta.get("model")),
        meta.get("effective_embedding_variant"),
        meta.get("parameter_variant"),
        int(meta.get("num_topics")),
        str(meta.get("split")),
        str(meta.get("doc_topic_source")),
    )


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def build_entropy_summary_table(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = SUMMARY_METRIC_KEYS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate conditions to (dataset, model, K, ...) rows.

    ``<metric>_mean`` is the mean over categories of each condition's iteration
    mean; ``<metric>_std`` is the pooled standard deviation (ddof=1) over all
    category x iteration values.
    """
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record["meta"])].append(record)

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        group = grouped[key]
        base = dict(zip(GROUP_COLUMNS, key))
        categories = sorted({str(record["category"]) for record in group})
        num_iterations = sum(
            len(record["results"].get("per_iteration", [])) for record in group
        )
        row: dict[str, Any] = dict(base)
        row["n_categories"] = len(categories)
        row["n_iterations"] = num_iterations
        row["categories"] = ";".join(categories)
        for metric in metric_keys:
            condition_means = _finite(
                record["results"]
                .get("aggregate", {})
                .get(metric, {})
                .get("mean", np.nan)
                for record in group
            )
            per_iteration_values = _finite(
                entry.get(metric, np.nan)
                for record in group
                for entry in record["results"].get("per_iteration", [])
            )
            mean = (
                float(condition_means.mean()) if condition_means.size else float("nan")
            )
            if per_iteration_values.size > 1:
                std = float(per_iteration_values.std(ddof=1))
            elif per_iteration_values.size == 1:
                std = 0.0
            else:
                std = float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            long_rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "n_categories": len(categories),
                    "n_iterations": int(per_iteration_values.size),
                }
            )
        wide_rows.append(row)
    return wide_rows, long_rows


def _model_label(row: Mapping[str, Any]) -> str:
    label = MODEL_LABELS.get(str(row["model"]), str(row["model"]))
    variant = row.get("embedding_variant")
    if variant not in {None, "", "None"}:
        label = f"{label} ({variant})"
    return label


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def build_latex_tables(
    wide_rows: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = SUMMARY_METRIC_KEYS,
    digits: int = 3,
) -> dict[tuple[str, int], str]:
    tables: dict[tuple[str, int], str] = {}
    by_table: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in wide_rows:
        by_table[(str(row["dataset"]), int(row["num_topics"]))].append(row)
    for table_key in sorted(by_table):
        rows = sorted(by_table[table_key], key=_model_label)
        header = " & ".join(
            [
                "Model",
                *[
                    METRIC_LABELS.get(metric, _latex_escape(metric))
                    for metric in metric_keys
                ],
            ]
        )
        lines = [
            r"\begin{tabular}{l" + "r" * len(metric_keys) + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]
        for row in rows:
            cells = [_latex_escape(_model_label(row))]
            for metric in metric_keys:
                mean = row.get(f"{metric}_mean", float("nan"))
                std = row.get(f"{metric}_std", float("nan"))
                if mean is None or not np.isfinite(mean):
                    cells.append("--")
                else:
                    cells.append(f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}")
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        tables[table_key] = "\n".join(lines) + "\n"
    return tables


def _read_doc_entropy_values(archive_dir: Path) -> list[float]:
    values: list[float] = []
    for csv_path in sorted(archive_dir.glob("iter*/doc_metrics.csv")):
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("valid", "True")).lower() not in {"true", "1"}:
                    continue
                try:
                    value = float(row["doc_topic_entropy_normalized"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
    return values


def build_entropy_histograms(
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    bins: int = 40,
) -> list[Path]:
    """Overlay per-document normalized entropy histograms per (dataset, K, category)."""
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # noqa: PLC0415 - import after backend selection

    by_figure: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        meta = record["meta"]
        by_figure[
            (
                str(meta.get("dataset")),
                int(meta.get("num_topics")),
                str(record["category"]),
            )
        ].append(record)

    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, num_topics, category), group in sorted(by_figure.items()):
        series: list[tuple[str, list[float]]] = []
        for record in sorted(group, key=lambda item: _model_label(item["meta"])):
            values = _read_doc_entropy_values(Path(record["archive_dir"]))
            if values:
                series.append((_model_label(record["meta"]), values))
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, values in series:
            ax.hist(
                values,
                bins=bins,
                range=(0.0, 1.0),
                density=True,
                histtype="step",
                linewidth=1.5,
                label=label,
            )
        ax.set_xlabel(r"$H(\theta_d)/\ln K$")
        ax.set_ylabel("density")
        ax.set_title(f"{dataset} / {category} / K={num_topics}")
        ax.set_xlim(0.0, 1.0)
        ax.legend(fontsize=7)
        fig.tight_layout()
        out_path = output_dir / f"entropy_hist_{dataset}_k{num_topics}_{category}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written


class EntropySummaryError(RuntimeError):
    """Conditions that cannot be pooled into one sidecar."""


def _json_number(value: Any) -> float | int | None:
    """A JSON-safe scalar: non-finite floats become ``null``."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _model_order_index(model: str) -> tuple[int, str]:
    try:
        return (SCORES_MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(SCORES_MODEL_ORDER), model)


def encoder_group(meta: Mapping[str, Any]) -> str | None:
    """Sentence-encoder group of a condition, or None when it reads none.

    Sentence-encoder models are grouped by their variant with the ``_raw`` or
    ``_norm`` preprocessing suffix removed (``minilm_norm`` -> ``minilm``);
    models that do not read sentence embeddings return None and are repeated in every group,
    as in the coherence sidecars, so that each file compares all models under
    one encoder.
    """
    if str(meta.get("model")) not in SENTENCE_ENCODER_MODELS:
        return None
    variant = meta.get("effective_embedding_variant") or meta.get("embedding_variant")
    if variant in {None, "", "None"}:
        return None
    return str(variant).removesuffix("_raw").removesuffix("_norm")


def _record_iterations(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = list(record["results"].get("per_iteration", []))
    return sorted(entries, key=lambda entry: int(entry.get("iteration", 0)))


def _cell_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    meta = record["meta"]
    entries = _record_iterations(record)
    iterations = [int(entry.get("iteration", 0)) for entry in entries]
    model_provenance = meta.get("model_provenance") or {}
    resolved = meta.get("doc_topic_source_resolved")
    if isinstance(resolved, Mapping):
        resolved = {str(key): value for key, value in resolved.items()}
    provenance: dict[str, Any] = {
        "condition_ids": [str(meta.get("condition_id") or "")],
        "condition_fingerprint": meta.get("condition_fingerprint"),
        "iterations": iterations,
        # The variant string the runs were read from (``minilm``, ``minilm_raw``,
        # ``googlenews300`` or null); consumers compare it with their column.
        "encoder_model": meta.get("effective_embedding_variant"),
        "embedding_variant": meta.get("effective_embedding_variant"),
        "encoder_model_name": model_provenance.get("encoder_model"),
        "word_embedding_variant": meta.get("word_embedding_variant"),
        "prior_scale": meta.get("prior_scale"),
        "parameter_variant": meta.get("parameter_variant"),
        "doc_topic_source_resolved": resolved,
        "num_documents": [
            _json_number(entry.get("num_documents")) for entry in entries
        ],
        "archive_dir": str(meta.get("archive_dir") or record.get("archive_dir") or ""),
    }
    for field_name in SCORES_PROTOCOL_FIELDS:
        provenance[field_name] = meta.get(field_name)
    return provenance


def _check_protocol(records: Sequence[Mapping[str, Any]], where: str) -> None:
    for field_name in SCORES_PROTOCOL_FIELDS:
        distinct: dict[str, str] = {}
        for record in records:
            value = record["meta"].get(field_name)
            if value is None or value == "":
                continue
            distinct.setdefault(str(value), str(record["meta"].get("condition_id")))
        if len(distinct) > 1:
            listed = ", ".join(f"{value} ({cid})" for value, cid in distinct.items())
            raise EntropySummaryError(
                f"{where}: conditions disagree on {field_name}: {listed}; "
                "filter with --split/--doc-topic-source or re-run the metrics"
            )


def build_entropy_scores_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    data_run: str,
    num_topics: int,
    encoder_variant: str | None,
    metric_keys: Sequence[str] = SCORES_METRIC_KEYS,
    per_topic_keys: Sequence[str] = SCORES_PER_TOPIC_KEYS,
) -> dict[str, Any]:
    """The per-run values and provenance of one sidecar (the downstream interface).

    ``records`` are the conditions of one (dataset, data run, encoder group, K);
    each condition holds every run of one category x model cell.
    """
    where = f"{dataset}/{data_run}/{encoder_variant or NO_ENCODER_GROUP}/K={num_topics}"
    _check_protocol(records, where)
    cells: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (str(record["category"]), str(record["meta"].get("model")))
        if key in cells:
            # GSLDA は未正規化(_raw)と L2 正規化(_norm)の両方の run が残るので、
            # encoder_group が同じグループに畳んだ結果ここで衝突する。本稿が
            # 報告するのは norm なので、変種で明示的に選ぶ。
            def _variant(rec: Mapping[str, Any]) -> str:
                meta = rec["meta"]
                return str(
                    meta.get("effective_embedding_variant")
                    or meta.get("embedding_variant")
                    or ""
                )

            old_variant, new_variant = _variant(cells[key]), _variant(record)
            if old_variant != new_variant and {old_variant, new_variant} <= {
                f"{old_variant.rsplit('_', 1)[0]}_raw",
                f"{old_variant.rsplit('_', 1)[0]}_norm",
            }:
                if new_variant.endswith("_norm"):
                    cells[key] = record
                continue

            # Ψ₀ 掃引の run は本稿が報告する Ψ₀=0.1 と同じセルに落ちる。
            # 掃引は別節の結果なので、ここでは本稿の値を選ぶ。
            def _scale(rec: Mapping[str, Any]) -> float | None:
                value = rec["meta"].get("prior_scale")
                return None if value is None else float(value)

            old_scale, new_scale = _scale(cells[key]), _scale(record)
            if old_scale != new_scale and MANUSCRIPT_PRIOR_SCALE in {
                old_scale,
                new_scale,
            }:
                if new_scale == MANUSCRIPT_PRIOR_SCALE:
                    cells[key] = record
                continue
            raise EntropySummaryError(
                f"{where}: two conditions for {key[0]}/{key[1]} "
                f"({cells[key]['meta'].get('condition_id')} and "
                f"{record['meta'].get('condition_id')}); narrow the selection with "
                "--model/--data-run or a parameter variant"
            )
        cells[key] = record
    categories = sorted({category for category, _ in cells})
    models = sorted({model for _, model in cells}, key=_model_order_index)
    iterations = sorted(
        {
            int(entry.get("iteration", 0))
            for record in records
            for entry in _record_iterations(record)
        }
    )

    scores: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for metric in metric_keys:
        by_category: dict[str, dict[str, list[Any]]] = {}
        for category in categories:
            by_model: dict[str, list[Any]] = {}
            for model in models:
                record = cells.get((category, model))
                if record is None:
                    continue
                by_model[model] = [
                    _json_number(entry.get(metric))
                    for entry in _record_iterations(record)
                ]
            by_category[category] = by_model
        scores[metric] = by_category

    per_topic: dict[str, dict[str, dict[str, list[list[Any]]]]] = {}
    run_iterations: dict[str, dict[str, list[int]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for category in categories:
        per_topic[category] = {}
        run_iterations[category] = {}
        provenance[category] = {}
        for model in models:
            record = cells.get((category, model))
            if record is None:
                continue
            entries = _record_iterations(record)
            per_topic[category][model] = {
                key: [
                    [
                        _json_number(v)
                        for v in (entry.get("per_topic") or {}).get(key, [])
                    ]
                    for entry in entries
                ]
                for key in per_topic_keys
            }
            run_iterations[category][model] = [
                int(entry.get("iteration", 0)) for entry in entries
            ]
            provenance[category][model] = _cell_provenance(record)

    protocol = {
        field_name: next(
            (
                record["meta"].get(field_name)
                for record in records
                if record["meta"].get(field_name) not in {None, ""}
            ),
            None,
        )
        for field_name in SCORES_PROTOCOL_FIELDS
    }
    return {
        "task": SUMMARY_TASK_NAME,
        "dataset": dataset,
        "data_run": data_run,
        "topics": int(num_topics),
        "encoder_variant": encoder_variant,
        "iterations": iterations,
        "metrics": list(metric_keys),
        "per_topic_metrics": list(per_topic_keys),
        "protocol": protocol,
        "models": models,
        "categories": categories,
        "scores": scores,
        "per_topic": per_topic,
        "run_iterations": run_iterations,
        "provenance": provenance,
    }


def group_records_for_scores(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int], list[Mapping[str, Any]]]:
    """(dataset, data_run, encoder group, K) -> its conditions.

    Encoder-independent conditions are added to every encoder group of their
    (dataset, data_run, K); when no sentence-encoder model contributed, they
    form a ``no_encoder`` group of their own.
    """
    by_scope: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        meta = record["meta"]
        by_scope[
            (
                str(meta.get("dataset")),
                str(meta.get("data_run")),
                int(meta.get("num_topics")),
            )
        ].append(record)
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for (dataset, data_run, num_topics), scoped in sorted(by_scope.items()):
        encoders = sorted(
            {group for group in (encoder_group(r["meta"]) for r in scoped) if group}
        )
        if not encoders:
            encoders = [NO_ENCODER_GROUP]
        for encoder in encoders:
            members = [
                record
                for record in scoped
                if encoder_group(record["meta"]) in {None, encoder}
            ]
            grouped[(dataset, data_run, encoder, num_topics)] = members
    return grouped


def _slug(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def scores_sidecar_path(
    output_dir: Path, *, dataset: str, data_run: str, encoder: str, num_topics: int
) -> Path:
    return (
        Path(output_dir)
        / _slug(dataset)
        / _slug(data_run)
        / _slug(encoder)
        / f"entropy_{_slug(dataset)}_{_slug(data_run)}_{_slug(encoder)}_{num_topics}topic.scores.json"
    )


def write_entropy_scores(
    records: Sequence[Mapping[str, Any]], *, output_dir: Path
) -> list[Path]:
    """Write one ``*.scores.json`` per (dataset, data run, encoder, K)."""
    written: list[Path] = []
    for (dataset, data_run, encoder, num_topics), members in group_records_for_scores(
        records
    ).items():
        payload = build_entropy_scores_payload(
            members,
            dataset=dataset,
            data_run=data_run,
            num_topics=num_topics,
            encoder_variant=None if encoder == NO_ENCODER_GROUP else encoder,
        )
        path = scores_sidecar_path(
            output_dir,
            dataset=dataset,
            data_run=data_run,
            encoder=encoder,
            num_topics=num_topics,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# The manuscript figure
# ---------------------------------------------------------------------------

PAPER_PANELS: tuple[dict[str, Any], ...] = (
    dict(
        key="topic_doc_entropy_normalized",
        source="per_topic",
        ylabel="topic spread over documents",
        ylim=(0.0, 1.02),
        showfliers=True,
    ),
    dict(
        key="topic_rank1_doc_fraction",
        source="per_topic",
        ylabel="share as main topic",
        ylim=None,
        showfliers=True,
    ),
    dict(
        key="doc_topic_entropy_normalized",
        source="per_document",
        ylabel="document spread over topics",
        ylim=(0.0, 1.02),
        showfliers=False,
    ),
)


def select_paper_records(
    records: Sequence[Mapping[str, Any]],
    *,
    num_topics: int = PAPER_NUM_TOPICS,
    encoder_variant: str = PAPER_ENCODER,
) -> list[Mapping[str, Any]]:
    """The conditions of one topic count under one sentence encoder.

    ``encoder_group`` folds ``_raw`` and ``_norm`` into one group, and the Psi_0 sweep of the
    appendix writes conditions under the same (model, K, encoder). Without the two filters
    below the GSLDA box would pool the unnormalized runs and every swept Psi_0 with the runs
    the manuscript reports. The sidecar builder applies the same preference; see the
    ``_raw``/``_norm`` and ``MANUSCRIPT_PRIOR_SCALE`` branches of ``build_scores_payload``.
    """

    def _keep(record: Mapping[str, Any]) -> bool:
        meta = record["meta"]
        if str(meta.get("model")) in PAPER_EXCLUDED_MODELS:
            return False
        if int(meta.get("num_topics")) != int(num_topics):
            return False
        if encoder_group(meta) not in {None, encoder_variant}:
            return False
        variant = str(
            meta.get("effective_embedding_variant")
            or meta.get("embedding_variant")
            or ""
        )
        if variant.endswith("_raw"):
            return False
        scale = meta.get("prior_scale")
        if scale is not None and not math.isclose(float(scale), MANUSCRIPT_PRIOR_SCALE):
            return False
        return True

    return [record for record in records if _keep(record)]


def _paper_model_labels(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """(runner key, short label) of the models drawn, in the paper's order."""
    from src.evaluation.reports.model_style import (
        EXCLUDED_MODELS,
        canonical_model_key,
        label_for_key,
        model_sort_key,
    )

    keys = sorted({str(record["meta"].get("model")) for record in records})
    labelled = []
    for key in keys:
        label = label_for_key(canonical_model_key(key))
        if label in EXCLUDED_MODELS:
            continue
        labelled.append((key, label))
    return sorted(labelled, key=lambda item: model_sort_key(item[1]))


def _pooled_values(
    records: Sequence[Mapping[str, Any]], panel: Mapping[str, Any]
) -> np.ndarray:
    if panel["source"] == "per_document":
        values = [
            value
            for record in records
            for value in _read_doc_entropy_values(Path(record["archive_dir"]))
        ]
    else:
        values = [
            value
            for record in records
            for entry in _record_iterations(record)
            for value in (entry.get("per_topic") or {}).get(panel["key"], [])
        ]
    return _finite(values)


def build_paper_entropy_figure(
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    num_topics: int = PAPER_NUM_TOPICS,
    encoder_variant: str = PAPER_ENCODER,
    colormap: str = "tab10",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DPI,
) -> list[Path]:
    """Box plots of the per-topic and per-document diagnostics, one row per panel.

    Rows are the three diagnostics, columns the datasets, boxes the models in
    the paper's order and colours; each box pools every category of the
    dataset and every run. Reference lines mark the diffuse cut-off (0.95) and,
    on the rank-1 panel, the dead cut-off (0.01) and the equal-use value 1/K.
    The figure is solved to the page width of the other manuscript grids and
    saved without a tight bounding box, so it is included at natural size.
    """
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # noqa: PLC0415 - import after backend selection

    from src.evaluation.reports.model_style import model_style

    selected = select_paper_records(
        records, num_topics=num_topics, encoder_variant=encoder_variant
    )
    if not selected:
        logger.warning(
            "paper figure skipped: no conditions at K=%s under %s",
            num_topics,
            encoder_variant,
        )
        return []
    datasets = sorted({str(record["meta"].get("dataset")) for record in selected})
    models = _paper_model_labels(selected)
    if not models:
        return []
    nrows, ncols = len(PAPER_PANELS), len(datasets)
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in selected:
        by_cell[
            (str(record["meta"].get("dataset")), str(record["meta"].get("model")))
        ].append(record)

    with plt.rc_context(PAPER_RC):
        figsize, adjust, _ = grid_geometry(
            nrows,
            ncols,
            legend_rows=0,
            margin_top=GRID_MARGIN_TOP,
            margin_bottom=PAPER_MARGIN_BOTTOM,
            hspace=PAPER_HSPACE,
        )
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(**adjust)
        positions = np.arange(len(models))
        tick_labels = [label.replace(" (proposed)", "") for _, label in models]
        for row, panel in enumerate(PAPER_PANELS):
            for col, dataset in enumerate(datasets):
                ax = axes[row][col]
                data = [
                    _pooled_values(by_cell.get((dataset, key), []), panel)
                    for key, _ in models
                ]
                data = [
                    values if values.size else np.asarray([np.nan]) for values in data
                ]
                box = ax.boxplot(
                    data,
                    positions=positions,
                    widths=0.62,
                    patch_artist=True,
                    showfliers=panel["showfliers"],
                    whis=1.5,
                    flierprops=dict(
                        marker=".", markersize=1.6, alpha=0.45, markeredgewidth=0.0
                    ),
                    medianprops=dict(color="black", linewidth=0.8),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6),
                    boxprops=dict(linewidth=0.7),
                )
                for index, (_, label) in enumerate(models):
                    style = model_style(label, colormap)
                    patch = box["boxes"][index]
                    patch.set_facecolor(style.color)
                    patch.set_alpha(PAPER_BOX_ALPHA)
                    patch.set_edgecolor(style.color)
                    for artist in (
                        box["whiskers"][2 * index],
                        box["whiskers"][2 * index + 1],
                        box["caps"][2 * index],
                        box["caps"][2 * index + 1],
                    ):
                        artist.set_color(style.color)
                    if panel["showfliers"] and index < len(box["fliers"]):
                        box["fliers"][index].set_markerfacecolor(style.color)
                        box["fliers"][index].set_markeredgecolor(style.color)
                if panel["key"] == "topic_doc_entropy_normalized":
                    ax.axhline(
                        0.95, color="0.35", linestyle=(0, (4.0, 1.5)), linewidth=0.6
                    )
                if panel["key"] == "topic_rank1_doc_fraction":
                    ax.set_yscale("symlog", linthresh=0.01, linscale=0.5)
                    ax.axhline(
                        0.01, color="0.35", linestyle=(0, (4.0, 1.5)), linewidth=0.6
                    )
                    ax.axhline(
                        1.0 / num_topics,
                        color="0.35",
                        linestyle=(0, (1.0, 1.5)),
                        linewidth=0.6,
                    )
                    ax.set_ylim(-0.0005, 1.0)
                    ax.set_yticks([0.0, 0.01, 0.1, 1.0])
                    ax.set_yticklabels(["0", "0.01", "0.1", "1"])
                    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                elif panel["ylim"] is not None:
                    ax.set_ylim(*panel["ylim"])
                ax.set_xticks(positions)
                ax.set_xticklabels(
                    tick_labels, rotation=45, ha="right", rotation_mode="anchor"
                )
                ax.set_xlim(-0.6, len(models) - 0.4)
                if col == 0:
                    ax.set_ylabel(panel["ylabel"])
                ax.set_title(
                    PAPER_DATASET_LABELS.get(dataset, dataset),
                    fontstyle="italic",
                    loc="left",
                )
                ax.tick_params(axis="x", length=0)
                panel_frame(ax, grid=False)
        stem = Path(output_dir) / f"entropy_boxplot_{num_topics}topic_{encoder_variant}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        written = []
        for fmt in formats:
            path = stem.parent / f"{stem.name}.{fmt}"
            fig.savefig(path, dpi=dpi)
            written.append(path)
            logger.info("[write] %s", path)
        plt.close(fig)
    return written


def write_entropy_based_summary(
    *,
    out_root: Path = DEFAULT_OUT_ROOT,
    output_dir: Path | None = None,
    datasets: Sequence[str] | None = None,
    data_runs: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    topics: Sequence[int] | None = None,
    split: str | None = None,
    doc_topic_source: str | None = None,
    exclude_categories: Sequence[str] = DEFAULT_EXCLUDE_CATEGORIES,
    metric_keys: Sequence[str] = SUMMARY_METRIC_KEYS,
    write_tex: bool = True,
    write_plots: bool = True,
    paper: bool = False,
    paper_num_topics: int = PAPER_NUM_TOPICS,
    paper_encoder: str = PAPER_ENCODER,
    paper_vmf_doc_topic_source: str | None = None,
) -> Path:
    """Write the summary tables, the sidecars and, with ``paper``, the box-plot figure.

    ``paper_vmf_doc_topic_source`` draws the boxes of the vMF family (vSLDA and
    MvTM / vLDA) from the conditions computed under that doc-topic source (the
    manuscript's fold-in estimator) while every other model keeps ``doc_topic_source``; the tables
    and sidecars written here are unaffected, so a sidecar never pools two
    sources.
    """
    out_root = Path(out_root)
    output_dir = Path(output_dir) if output_dir is not None else out_root / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_entropy_conditions(
        out_root=out_root,
        datasets=datasets,
        data_runs=data_runs,
        models=models,
        topics=topics,
        split=split,
        doc_topic_source=doc_topic_source,
        exclude_categories=exclude_categories,
    )
    wide_rows, long_rows = build_entropy_summary_table(records, metric_keys=metric_keys)

    wide_columns = [*GROUP_COLUMNS, "n_categories", "n_iterations", "categories"]
    for metric in metric_keys:
        wide_columns.extend([f"{metric}_mean", f"{metric}_std"])
    wide_path = output_dir / "entropy_summary_wide.csv"
    write_csv_rows(fieldnames=wide_columns, rows=wide_rows, path=wide_path)
    long_columns = [
        *GROUP_COLUMNS,
        "metric",
        "mean",
        "std",
        "n_categories",
        "n_iterations",
    ]
    write_csv_rows(
        fieldnames=long_columns,
        rows=long_rows,
        path=output_dir / "entropy_summary_long.csv",
    )
    write_tabular_report_json(
        meta={
            "task": SUMMARY_TASK_NAME,
            "out_root": str(out_root),
            "datasets": None if datasets is None else list(datasets),
            "models": None if models is None else list(models),
            "topics": None if topics is None else [int(t) for t in topics],
            "split": split,
            "doc_topic_source": doc_topic_source,
            "exclude_categories": list(exclude_categories),
            "metric_keys": list(metric_keys),
            "num_conditions": len(records),
            "condition_ids": sorted(
                str(record["meta"].get("condition_id")) for record in records
            ),
        },
        columns=wide_columns,
        rows=wide_rows,
        path=output_dir / "entropy_summary.json",
    )
    if write_tex:
        for (dataset, num_topics), tex in build_latex_tables(
            wide_rows, metric_keys=metric_keys
        ).items():
            (output_dir / f"entropy_{dataset}_k{num_topics}.tex").write_text(
                tex, encoding="utf-8"
            )
    if write_plots:
        build_entropy_histograms(records, output_dir=output_dir / "plots")
    sidecars = write_entropy_scores(records, output_dir=output_dir)
    logger.info("wrote %d scores sidecars under %s", len(sidecars), output_dir)
    if paper:
        figure_records = list(records)
        if paper_vmf_doc_topic_source is not None and str(
            paper_vmf_doc_topic_source
        ) != str(doc_topic_source):
            # The vMF family (vSLDA and MvTM / vLDA) shares the estimator, so both are
            # drawn from the conditions of the paper source; the other baselines keep
            # ``doc_topic_source``.
            vmf_records = collect_entropy_conditions(
                out_root=out_root,
                datasets=datasets,
                data_runs=data_runs,
                models=["vmf", "mvtm"],
                topics=topics,
                split=split,
                doc_topic_source=paper_vmf_doc_topic_source,
                exclude_categories=exclude_categories,
            )
            if not vmf_records:
                raise EntropySummaryError(
                    "paper figure: no vMF conditions computed under doc_topic_source "
                    f"{paper_vmf_doc_topic_source!r}"
                )
            figure_records = [
                record
                for record in records
                if normalize_model_name(str(record["meta"].get("model")))
                not in {"vmf", "mvtm"}
            ] + vmf_records
        build_paper_entropy_figure(
            figure_records,
            output_dir=output_dir / "figures",
            num_topics=paper_num_topics,
            encoder_variant=paper_encoder,
        )
    logger.info(
        "entropy summary written to %s (%d conditions, %d rows)",
        output_dir,
        len(records),
        len(wide_rows),
    )
    return wide_path


run_entropy_based_summary = write_entropy_based_summary

__all__ = [
    "EntropySummaryError",
    "collect_entropy_conditions",
    "build_entropy_summary_table",
    "build_latex_tables",
    "build_entropy_histograms",
    "build_entropy_scores_payload",
    "build_paper_entropy_figure",
    "encoder_group",
    "group_records_for_scores",
    "scores_sidecar_path",
    "select_paper_records",
    "write_entropy_scores",
    "write_entropy_based_summary",
    "run_entropy_based_summary",
]

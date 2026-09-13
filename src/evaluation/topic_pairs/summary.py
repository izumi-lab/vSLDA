"""Cross-condition summary of the topic-pair metrics (task ``topic_pair_summary``).

Reads every condition reachable from ``latest/**/CURRENT.json`` under the
topic-pair root and writes

* a review table (``topic_pairs_summary_wide.csv`` / ``.json``) with the mean
  and pooled standard deviation of the per-run scalars, for reading here; and
* ``*.scores.json`` sidecars, one per (dataset, data run, sentence encoder, K),
  carrying the raw per-run values -- per-topic arrays, the K x K per-pair
  matrices, the cross-model overlap matrices, the label names and the
  provenance of every cell -- and nothing aggregated. That is the interface
  the paper repository reads; it ranks, selects and averages on its own side,
  so no rounded or ranked number crosses the boundary. The layout mirrors the
  entropy sidecars of :mod:`src.evaluation.entropy_based.summary`.

``--paper`` restricts the sidecars to the manuscript grid and refuses to write
when any of its cells or runs is missing, so a partial grid cannot be shipped;
it also writes the representative-word sidecars of the reference runs
(:mod:`.words`) that the appendix's all-topic tables are built from.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.core.artifacts import load_json
from src.core.paths import resolve_project_path
from src.evaluation.reporting import (
    read_evaluation_json,
    write_csv_rows,
    write_tabular_report_json,
)
from src.evaluation.topic_pairs.inputs import TASK_MODELS
from src.evaluation.topic_pairs.metrics import (
    CROSS_MODEL_KEY,
    DEFAULT_OUT_ROOT,
    METRICS_FILENAME,
    PROTOCOL_FIELDS,
    TASK_NAME,
)
from src.evaluation.topic_pairs.numerics import (
    MODEL_OWN_KEYS,
    PER_PAIR_KEYS,
    PER_TOPIC_KEYS,
    PER_TOPIC_MATRIX_KEYS,
    SCALAR_SUMMARY_KEYS,
)
from src.evaluation.topic_pairs.words import (
    DEFAULT_COHERENCE_ROOT,
    write_reference_words,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

SUMMARY_TASK_NAME = "topic_pair_summary"
DEFAULT_EXCLUDE_CATEGORIES: tuple[str, ...] = ("all",)
GROUP_COLUMNS: tuple[str, ...] = (
    "dataset",
    "data_run",
    "model",
    "embedding_variant",
    "parameter_variant",
    "num_topics",
    "split",
)
# Scalar per-run values the sidecar carries (all consumers recompute anything
# aggregated from the per-topic / per-pair blocks; these are for checking).
SCORES_METRIC_KEYS: tuple[str, ...] = ("num_documents", *SCALAR_SUMMARY_KEYS)
SCORES_PER_TOPIC_KEYS: tuple[str, ...] = (*PER_TOPIC_KEYS, *PER_TOPIC_MATRIX_KEYS)
SCORES_PER_PAIR_KEYS: tuple[str, ...] = PER_PAIR_KEYS
SCORES_MODEL_OWN_KEYS: tuple[str, ...] = MODEL_OWN_KEYS
SCORES_PROTOCOL_FIELDS: tuple[str, ...] = PROTOCOL_FIELDS
NO_ENCODER_GROUP = "no_encoder"
SCORES_MODEL_ORDER: tuple[str, ...] = ("sentlda", "sentence_gaussianlda", "vmf")
# The manuscript grid: what ``--paper`` requires to be complete.
PAPER_DATASETS: tuple[str, ...] = ("20newsgroup", "nyt")
PAPER_DATA_RUN = "default"
PAPER_ENCODER = "minilm"
PAPER_TOPICS: tuple[int, ...] = (10, 20, 30)
PAPER_ITERATIONS: tuple[int, ...] = (0, 1, 2, 3, 4)
PAPER_MODELS: tuple[str, ...] = TASK_MODELS
PAPER_CATEGORIES: dict[str, tuple[str, ...]] = {
    "20newsgroup": ("computer", "politics", "religion", "ride", "science", "sports"),
    "nyt": ("arts", "business", "politics", "sports"),
}


class TopicPairSummaryError(RuntimeError):
    """Conditions that cannot be pooled into one sidecar, or an incomplete grid."""


def _matches(value: Any, allowed: Sequence[Any] | None) -> bool:
    if allowed is None:
        return True
    return str(value) in {str(item) for item in allowed}


def _record_models(meta: Mapping[str, Any]) -> list[str]:
    if str(meta.get("model")) == CROSS_MODEL_KEY:
        return [str(model) for model in meta.get("models") or []]
    return [str(meta.get("model"))]


def collect_topic_pair_conditions(
    *,
    out_root: Path = DEFAULT_OUT_ROOT,
    datasets: Sequence[str] | None = None,
    data_runs: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    topics: Sequence[int] | None = None,
    split: str | None = None,
    exclude_categories: Sequence[str] = DEFAULT_EXCLUDE_CATEGORIES,
) -> list[dict[str, Any]]:
    """Read every condition reachable from ``latest/**/CURRENT.json``.

    Cross-model conditions are kept when every model they compare is in
    ``models`` (or when no model filter is given).
    """

    latest_root = Path(out_root) / "latest"
    excluded = {str(value) for value in exclude_categories}
    allowed_models = None if models is None else {str(model) for model in models}
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
        if payload.get("task") not in {None, TASK_NAME}:
            continue
        archive_dir = resolve_project_path(str(payload["archive_dir"]))
        metrics_path = archive_dir / METRICS_FILENAME
        if not metrics_path.exists():
            logger.warning("missing metrics file for pointer %s", pointer_path)
            continue
        meta, results = read_evaluation_json(metrics_path)
        if not isinstance(results, dict) or meta.get("task") != TASK_NAME:
            continue
        if not _matches(meta.get("dataset"), datasets):
            continue
        if not _matches(meta.get("data_run"), data_runs):
            continue
        if (
            allowed_models is not None
            and not set(_record_models(meta)) <= allowed_models
        ):
            continue
        if not _matches(meta.get("num_topics"), topics):
            continue
        if split is not None and str(meta.get("split")) != str(split):
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


def is_cross_record(record: Mapping[str, Any]) -> bool:
    return str(record["meta"].get("model")) == CROSS_MODEL_KEY


def encoder_group(meta: Mapping[str, Any]) -> str | None:
    """Sentence-encoder group of a condition, or None when it records none.

    Conditions are grouped by their variant with the ``_raw`` or ``_norm``
    suffix removed. Every metric of this task is computed in the requested
    sentence-encoder space, so a model that reads no embeddings of its own
    (SentLDA) is still grouped by the space it was evaluated in, which its meta
    records as ``embedding_variant``; the cross-model condition carries the
    requested variant. Only a condition with no recorded space at all joins
    every group.
    """

    variant = meta.get("effective_embedding_variant") or meta.get("embedding_variant")
    if variant in {None, "", "None"}:
        return None
    return str(variant).removesuffix("_raw").removesuffix("_norm")


def _record_iterations(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = list(record["results"].get("per_iteration", []))
    return sorted(entries, key=lambda entry: int(entry.get("iteration", 0)))


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _group_key(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(meta.get("dataset")),
        str(meta.get("data_run")),
        str(meta.get("model")),
        meta.get("effective_embedding_variant"),
        meta.get("parameter_variant"),
        int(meta.get("num_topics")),
        str(meta.get("split")),
    )


def build_topic_pair_summary_table(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = SCALAR_SUMMARY_KEYS,
) -> list[dict[str, Any]]:
    """Review rows per (dataset, model, K, ...): category means, pooled std."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if is_cross_record(record):
            continue
        grouped[_group_key(record["meta"])].append(record)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        group = grouped[key]
        row: dict[str, Any] = dict(zip(GROUP_COLUMNS, key))
        categories = sorted({str(record["category"]) for record in group})
        row["n_categories"] = len(categories)
        row["n_iterations"] = sum(len(_record_iterations(r)) for r in group)
        row["categories"] = ";".join(categories)
        for metric in metric_keys:
            condition_means = _finite(
                record["results"]
                .get("aggregate", {})
                .get(metric, {})
                .get("mean", np.nan)
                for record in group
            )
            values = _finite(
                entry.get(metric, np.nan)
                for record in group
                for entry in _record_iterations(record)
            )
            row[f"{metric}_mean"] = (
                float(condition_means.mean()) if condition_means.size else float("nan")
            )
            if values.size > 1:
                row[f"{metric}_std"] = float(values.std(ddof=1))
            elif values.size == 1:
                row[f"{metric}_std"] = 0.0
            else:
                row[f"{metric}_std"] = float("nan")
        rows.append(row)
    return rows


def _json_number(value: Any) -> float | int | None:
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


def _json_array(values: Any) -> Any:
    """Nested lists with non-finite numbers replaced by ``null``."""

    if values is None:
        return None
    if isinstance(values, (list, tuple, np.ndarray)):
        return [_json_array(value) for value in values]
    return _json_number(values)


def _model_order_index(model: str) -> tuple[int, str]:
    try:
        return (SCORES_MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(SCORES_MODEL_ORDER), model)


def _cell_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    meta = record["meta"]
    entries = _record_iterations(record)
    model_provenance = meta.get("model_provenance") or {}
    provenance: dict[str, Any] = {
        "condition_ids": [str(meta.get("condition_id") or "")],
        "condition_fingerprint": meta.get("condition_fingerprint"),
        "iterations": [int(entry.get("iteration", 0)) for entry in entries],
        "encoder_model": meta.get("effective_embedding_variant"),
        "embedding_variant": meta.get("effective_embedding_variant"),
        "encoder_model_name": meta.get("encoder_model_name"),
        "encoder_config_fingerprint": meta.get("encoder_config_fingerprint"),
        "sentence_sha1": meta.get("sentence_sha1"),
        "prior_scale": meta.get("prior_scale"),
        "parameter_variant": meta.get("parameter_variant"),
        "max_kappa": meta.get("max_kappa"),
        "label_source_model": meta.get("label_source_model"),
        "source_condition_dirs": meta.get("source_condition_dirs"),
        "posterior_metadata": meta.get("posterior_metadata"),
        "num_sentences": meta.get("num_sentences"),
        "num_documents": meta.get("num_documents"),
        "archive_dir": str(meta.get("archive_dir") or record.get("archive_dir") or ""),
        "model_provenance": model_provenance,
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
            raise TopicPairSummaryError(
                f"{where}: conditions disagree on {field_name}: {listed}; "
                "filter with --split or re-run the metrics"
            )


def build_topic_pair_scores_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    data_run: str,
    num_topics: int,
    encoder_variant: str | None,
) -> dict[str, Any]:
    """The per-run values and provenance of one sidecar (the downstream interface)."""

    where = f"{dataset}/{data_run}/{encoder_variant or NO_ENCODER_GROUP}/K={num_topics}"
    _check_protocol(records, where)
    cells: dict[tuple[str, str], Mapping[str, Any]] = {}
    cross_cells: dict[str, Mapping[str, Any]] = {}
    for record in records:
        category = str(record["category"])
        if is_cross_record(record):
            if category in cross_cells:
                raise TopicPairSummaryError(
                    f"{where}: two cross-model conditions for {category} "
                    f"({cross_cells[category]['meta'].get('condition_id')} and "
                    f"{record['meta'].get('condition_id')})"
                )
            cross_cells[category] = record
            continue
        key = (category, str(record["meta"].get("model")))
        if key in cells:
            # GSLDA は未正規化(_raw)と L2 正規化(_norm)の両方の run が残り、
            # encoder_group が同じグループに畳むのでここで衝突する。本稿が
            # 報告するのは norm なので変種で明示的に選ぶ。
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
            raise TopicPairSummaryError(
                f"{where}: two conditions for {key[0]}/{key[1]} "
                f"({cells[key]['meta'].get('condition_id')} and "
                f"{record['meta'].get('condition_id')}); narrow the selection with "
                "--model/--data-run or a parameter variant"
            )
        cells[key] = record
    categories = sorted({category for category, _ in cells} | set(cross_cells))
    models = sorted({model for _, model in cells}, key=_model_order_index)
    iterations = sorted(
        {
            int(entry.get("iteration", 0))
            for record in records
            for entry in _record_iterations(record)
        }
    )

    scores: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for metric in SCORES_METRIC_KEYS:
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

    per_topic: dict[str, dict[str, dict[str, list[Any]]]] = {}
    per_pair: dict[str, dict[str, dict[str, list[Any]]]] = {}
    model_own: dict[str, dict[str, dict[str, list[Any]] | None]] = {}
    cross_model: dict[str, dict[str, list[Any]]] = {}
    label_names: dict[str, list[str]] = {}
    run_iterations: dict[str, dict[str, list[int]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for category in categories:
        per_topic[category] = {}
        per_pair[category] = {}
        model_own[category] = {}
        run_iterations[category] = {}
        provenance[category] = {}
        for model in models:
            record = cells.get((category, model))
            if record is None:
                continue
            entries = _record_iterations(record)
            label_names.setdefault(
                category,
                [str(name) for name in record["meta"].get("label_names") or []],
            )
            per_topic[category][model] = {
                key: [
                    _json_array((entry.get("per_topic") or {}).get(key))
                    for entry in entries
                ]
                for key in SCORES_PER_TOPIC_KEYS
            }
            per_pair[category][model] = {
                key: [
                    _json_array((entry.get("per_pair") or {}).get(key))
                    for entry in entries
                ]
                for key in SCORES_PER_PAIR_KEYS
            }
            if all(entry.get("model_own") for entry in entries):
                model_own[category][model] = {
                    key: [_json_array(entry["model_own"].get(key)) for entry in entries]
                    for key in SCORES_MODEL_OWN_KEYS
                }
            else:
                model_own[category][model] = None
            run_iterations[category][model] = [
                int(entry.get("iteration", 0)) for entry in entries
            ]
            provenance[category][model] = _cell_provenance(record)
        cross = cross_cells.get(category)
        if cross is not None:
            entries = _record_iterations(cross)
            pairs = [str(pair) for pair in cross["meta"].get("pairs") or []]
            cross_model[category] = {
                pair: [
                    _json_array((entry.get("overlap") or {}).get(pair))
                    for entry in entries
                ]
                for pair in pairs
            }
            run_iterations[category][CROSS_MODEL_KEY] = [
                int(entry.get("iteration", 0)) for entry in entries
            ]
            provenance[category][CROSS_MODEL_KEY] = _cell_provenance(cross)
            label_names.setdefault(
                category, [str(name) for name in cross["meta"].get("label_names") or []]
            )

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
        "metrics": list(SCORES_METRIC_KEYS),
        "per_topic_metrics": list(SCORES_PER_TOPIC_KEYS),
        "per_pair_metrics": list(SCORES_PER_PAIR_KEYS),
        "model_own_metrics": list(SCORES_MODEL_OWN_KEYS),
        "protocol": protocol,
        "models": models,
        "categories": categories,
        "label_names": label_names,
        "scores": scores,
        "per_topic": per_topic,
        "per_pair": per_pair,
        "model_own": model_own,
        "cross_model": cross_model,
        "run_iterations": run_iterations,
        "provenance": provenance,
    }


def group_records_for_scores(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int], list[Mapping[str, Any]]]:
    """(dataset, data_run, encoder group, K) -> its conditions.

    Encoder-independent conditions (SentLDA) join every encoder group of their
    (dataset, data_run, K); with no sentence-encoder model they form a
    ``no_encoder`` group of their own.
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
            grouped[(dataset, data_run, encoder, num_topics)] = [
                record
                for record in scoped
                if encoder_group(record["meta"]) in {None, encoder}
            ]
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
        / f"topic_pairs_{_slug(dataset)}_{_slug(data_run)}_{_slug(encoder)}_{num_topics}topic.scores.json"
    )


def write_topic_pair_scores(
    records: Sequence[Mapping[str, Any]], *, output_dir: Path, compact: bool = True
) -> list[Path]:
    """Write one ``*.scores.json`` per (dataset, data run, encoder, K).

    The K x K blocks make these files large, so they are written without
    indentation by default (``compact``); the content is the same.
    """

    written: list[Path] = []
    for (dataset, data_run, encoder, num_topics), members in group_records_for_scores(
        records
    ).items():
        payload = build_topic_pair_scores_payload(
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
            if compact:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        written.append(path)
    return written


def select_paper_records(
    records: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = PAPER_DATASETS,
    data_run: str = PAPER_DATA_RUN,
    encoder_variant: str = PAPER_ENCODER,
    topics: Sequence[int] = PAPER_TOPICS,
) -> list[Mapping[str, Any]]:
    """The conditions of the manuscript grid."""

    return [
        record
        for record in records
        if str(record["meta"].get("dataset")) in set(datasets)
        and str(record["meta"].get("data_run")) == data_run
        and int(record["meta"].get("num_topics")) in {int(k) for k in topics}
        and encoder_group(record["meta"]) in {None, encoder_variant}
    ]


def check_paper_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = PAPER_DATASETS,
    data_run: str = PAPER_DATA_RUN,
    encoder_variant: str = PAPER_ENCODER,
    topics: Sequence[int] = PAPER_TOPICS,
    iterations: Sequence[int] = PAPER_ITERATIONS,
    models: Sequence[str] = PAPER_MODELS,
    categories: Mapping[str, Sequence[str]] = PAPER_CATEGORIES,
) -> None:
    """Raise unless every cell and run of the manuscript grid is present."""

    expected_iterations = [int(value) for value in iterations]
    found: dict[tuple[str, int, str, str], list[int]] = {}
    for record in select_paper_records(
        records,
        datasets=datasets,
        data_run=data_run,
        encoder_variant=encoder_variant,
        topics=topics,
    ):
        meta = record["meta"]
        key = (
            str(meta.get("dataset")),
            int(meta.get("num_topics")),
            str(record["category"]),
            str(meta.get("model")),
        )
        found[key] = [
            int(entry.get("iteration", 0)) for entry in _record_iterations(record)
        ]
    missing: list[str] = []
    for dataset in datasets:
        for num_topics in topics:
            for category in categories.get(dataset, ()):
                for model in (*models, CROSS_MODEL_KEY):
                    key = (str(dataset), int(num_topics), str(category), str(model))
                    runs = found.get(key)
                    if runs is None:
                        missing.append(
                            f"{dataset}/K={num_topics}/{category}/{model}: no condition"
                        )
                    elif sorted(runs) != expected_iterations:
                        missing.append(
                            f"{dataset}/K={num_topics}/{category}/{model}: iterations {sorted(runs)}"
                        )
    if missing:
        shown = "\n  ".join(missing[:40])
        more = "" if len(missing) <= 40 else f"\n  ... and {len(missing) - 40} more"
        raise TopicPairSummaryError(
            f"the manuscript grid is incomplete ({len(missing)} cells):\n  {shown}{more}"
        )


def write_topic_pair_summary(
    *,
    out_root: Path = DEFAULT_OUT_ROOT,
    output_dir: Path | None = None,
    datasets: Sequence[str] | None = None,
    data_runs: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    topics: Sequence[int] | None = None,
    split: str | None = None,
    exclude_categories: Sequence[str] = DEFAULT_EXCLUDE_CATEGORIES,
    paper: bool = False,
    compact: bool = True,
    coherence_root: Path = DEFAULT_COHERENCE_ROOT,
) -> Path:
    out_root = Path(out_root)
    output_dir = Path(output_dir) if output_dir is not None else out_root / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_topic_pair_conditions(
        out_root=out_root,
        datasets=datasets,
        data_runs=data_runs,
        models=models,
        topics=topics,
        split=split,
        exclude_categories=exclude_categories,
    )
    if paper:
        records = select_paper_records(records)
        check_paper_grid(records)

    rows = build_topic_pair_summary_table(records)
    columns = [*GROUP_COLUMNS, "n_categories", "n_iterations", "categories"]
    for metric in SCALAR_SUMMARY_KEYS:
        columns.extend([f"{metric}_mean", f"{metric}_std"])
    wide_path = output_dir / "topic_pairs_summary_wide.csv"
    write_csv_rows(fieldnames=columns, rows=rows, path=wide_path)
    write_tabular_report_json(
        meta={
            "task": SUMMARY_TASK_NAME,
            "out_root": str(out_root),
            "datasets": None if datasets is None else list(datasets),
            "models": None if models is None else list(models),
            "topics": None if topics is None else [int(t) for t in topics],
            "split": split,
            "paper": bool(paper),
            "exclude_categories": list(exclude_categories),
            "num_conditions": len(records),
            "condition_ids": sorted(
                str(r["meta"].get("condition_id")) for r in records
            ),
        },
        columns=columns,
        rows=rows,
        path=output_dir / "topic_pairs_summary.json",
    )
    sidecars = write_topic_pair_scores(records, output_dir=output_dir, compact=compact)
    if paper:
        words = write_reference_words(
            output_dir=output_dir, coherence_root=coherence_root
        )
        logger.info("wrote %d reference-word sidecars under %s", len(words), output_dir)
    logger.info(
        "topic-pair summary written to %s (%d conditions, %d rows, %d sidecars)",
        output_dir,
        len(records),
        len(rows),
        len(sidecars),
    )
    return wide_path


run_topic_pair_summary = write_topic_pair_summary

__all__ = [
    "PAPER_CATEGORIES",
    "PAPER_DATASETS",
    "PAPER_ENCODER",
    "PAPER_ITERATIONS",
    "PAPER_MODELS",
    "PAPER_TOPICS",
    "SCORES_MODEL_ORDER",
    "SCORES_PER_PAIR_KEYS",
    "SCORES_PER_TOPIC_KEYS",
    "SCORES_PROTOCOL_FIELDS",
    "TopicPairSummaryError",
    "build_topic_pair_scores_payload",
    "build_topic_pair_summary_table",
    "check_paper_grid",
    "collect_topic_pair_conditions",
    "encoder_group",
    "group_records_for_scores",
    "is_cross_record",
    "run_topic_pair_summary",
    "scores_sidecar_path",
    "select_paper_records",
    "write_topic_pair_scores",
    "write_topic_pair_summary",
]

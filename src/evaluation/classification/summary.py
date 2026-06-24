from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np

from src.core.artifacts import CURRENT_POINTER_FILENAME, load_json, save_json
from src.core.paths import resolve_project_path
from src.evaluation.reporting import read_evaluation_json, write_csv_rows
from src.evaluation.schema import build_evaluation_meta

from .config import (
    DEFAULT_ALIGNMENT_MODE,
    DEFAULT_FEATURE_RESOLVE_MODE,
    MODEL_NAMES,
    RESULT_ROOT,
    get_dataset_targets,
)
from .workflow import (
    ClassificationCondition,
    build_classification_latest_dir,
    build_classification_output_dir_from_condition,
)


def _metric_filename(metric: str, dataset: str, topics: int) -> str:
    if metric == "acc":
        return f"acc_{dataset}_{topics}topic.json"
    return f"f1_{dataset}_{topics}topic.json"


def _sorted_models(models: List[str]) -> List[str]:
    base_order = {name: idx for idx, name in enumerate(MODEL_NAMES)}

    def key_fn(name: str) -> Tuple[int, str]:
        base = name.split(" [", 1)[0]
        return (base_order.get(base, len(base_order)), name)

    return sorted(models, key=key_fn)


def _aggregate(
    metric: str,
    dataset: str,
    data_run: str,
    topics: int,
    iterations: List[int],
    vmf_assignment: str,
    alignment_mode: str,
    *,
    classifiers: List[str] | None,
    result_root: Path,
    resolve_mode: Literal["latest", "strict"],
    embedding_variants: Sequence[str] | None,
    feature_resolve_mode: str,
    selected_models: Sequence[str] | None,
) -> tuple[
    Dict[str, Dict[str, List[float]]],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    results: Dict[str, Dict[str, List[float]]] = {}
    any_found = False
    summary_meta: dict[str, Any] = {}
    feature_catalog_by_category: dict[str, list[dict[str, Any]]] = {}
    for i in iterations:
        file_paths = _resolve_metric_paths(
            metric=metric,
            dataset=dataset,
            data_run=data_run,
            topics=topics,
            iteration=i,
            vmf_assignment=vmf_assignment,
            classifiers=classifiers,
            alignment_mode=alignment_mode,
            result_root=result_root,
            resolve_mode=resolve_mode,
            embedding_variants=embedding_variants,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=selected_models,
        )
        for file_path in file_paths:
            if not file_path.exists():
                print(f"[skip] missing {file_path}")
                continue
            file_meta, file_results = read_evaluation_json(file_path)
            any_found = True
            if file_meta and not summary_meta:
                summary_meta = dict(file_meta)
            _merge_feature_catalogs(feature_catalog_by_category, file_meta)
            for cat, payload in file_results.items():
                items = (
                    payload.items()
                    if metric == "acc"
                    else (
                        payload["macro"].items()
                        if metric == "f1mac"
                        else payload["micro"].items()
                    )
                )
                for model_name, score in items:
                    results.setdefault(cat, {})
                    results[cat].setdefault(model_name, [])
                    results[cat][model_name].append(score)
    if not any_found:
        return {}, {}, {}
    return results, summary_meta, feature_catalog_by_category


def _matches_metric_meta(
    meta: dict[str, Any],
    *,
    iteration: int,
    topics: int,
    vmf_assignment: str,
    data_run: str,
    alignment_mode: str,
    classifiers: List[str] | None,
    embedding_variants: Sequence[str] | None,
    feature_resolve_mode: str,
    selected_models: Sequence[str] | None,
) -> bool:
    if not meta:
        return False
    if meta.get("mode") is not None or meta.get("value") is not None:
        return False
    if int(meta.get("iteration", -1)) != int(iteration):
        return False
    if int(meta.get("topics", -1)) != int(topics):
        return False
    if str(meta.get("vmf_assignment", "hard")) != str(vmf_assignment):
        return False
    if str(meta.get("data_run", "default")) != str(data_run):
        return False
    if str(meta.get("alignment_mode", DEFAULT_ALIGNMENT_MODE)) != str(alignment_mode):
        return False
    if classifiers is not None:
        if sorted(str(item) for item in meta.get("classifiers", [])) != sorted(
            str(item) for item in classifiers
        ):
            return False
    if selected_models is not None:
        meta_selected_models = [
            str(item) for item in (meta.get("selected_models") or []) if str(item)
        ]
        if meta_selected_models:
            requested = {_normalize_model_selector(item) for item in selected_models}
            available = {
                _normalize_model_selector(item) for item in meta_selected_models
            }
            if requested.isdisjoint(available):
                return False
    if embedding_variants is not None:
        meta_variants = [str(item) for item in meta.get("embedding_variants") or []]
        if meta_variants and sorted(meta_variants) != sorted(
            str(item) for item in embedding_variants
        ):
            return False
    if str(meta.get("feature_resolve_mode", DEFAULT_FEATURE_RESOLVE_MODE)) != str(
        feature_resolve_mode
    ):
        return False
    return True


def _started_at_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    text = str(value).strip()
    if not text:
        return (0, "")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return (1, datetime.fromisoformat(normalized).isoformat())
    except ValueError:
        return (0, text)


def _pick_latest_metric_match(matches: list[tuple[Path, dict[str, Any]]]) -> Path:
    ranked = sorted(
        matches,
        key=lambda item: (
            _started_at_key(item[1].get("started_at")),
            str(item[0]),
        ),
    )
    return ranked[-1][0]


def _format_match_paths(matches: list[tuple[Path, dict[str, Any]]]) -> str:
    return ", ".join(str(path) for path, _meta in matches)


def _select_metric_match(
    *,
    matches: list[tuple[Path, dict[str, Any]]],
    resolve_mode: Literal["latest", "strict"],
    source_label: str,
) -> Path | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]
    if resolve_mode == "strict":
        raise ValueError(
            f"Multiple classification metric matches found in {source_label}: "
            f"{_format_match_paths(matches)}"
        )
    return _pick_latest_metric_match(matches)


def _resolve_metric_path_from_latest(
    *,
    filename: str,
    dataset: str,
    data_run: str,
    topics: int,
    iteration: int,
    vmf_assignment: str,
    alignment_mode: str,
    classifiers: List[str] | None,
    result_root: Path,
    resolve_mode: Literal["latest", "strict"],
    embedding_variants: Sequence[str] | None,
    feature_resolve_mode: str,
    selected_models: Sequence[str] | None,
) -> Path | None:
    latest_root = build_classification_latest_dir(
        result_root=result_root,
        dataset=dataset,
        data_run=data_run,
        display_key="placeholder",
    ).parent
    if not latest_root.exists():
        return None
    matches: list[tuple[Path, dict[str, Any]]] = []
    for pointer_dir in sorted(latest_root.iterdir()):
        if not pointer_dir.is_dir():
            continue
        pointer_path = pointer_dir / CURRENT_POINTER_FILENAME
        if not pointer_path.exists():
            continue
        payload = load_json(pointer_path)
        if not isinstance(payload, dict):
            continue
        archive_dir_raw = payload.get("archive_dir")
        if not archive_dir_raw:
            continue
        archive_dir = resolve_project_path(str(archive_dir_raw))
        candidate = archive_dir / filename
        if not candidate.exists():
            continue
        meta, _ = read_evaluation_json(candidate)
        if _matches_metric_meta(
            meta,
            iteration=iteration,
            topics=topics,
            vmf_assignment=vmf_assignment,
            data_run=data_run,
            alignment_mode=alignment_mode,
            classifiers=classifiers,
            embedding_variants=embedding_variants,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=selected_models,
        ):
            matches.append((candidate, meta))
    if not matches:
        return None
    return _select_metric_match(
        matches=matches,
        resolve_mode=resolve_mode,
        source_label=str(latest_root),
    )


def _resolve_metric_paths_from_latest(
    *,
    filename: str,
    dataset: str,
    data_run: str,
    topics: int,
    iteration: int,
    vmf_assignment: str,
    alignment_mode: str,
    classifiers: List[str] | None,
    result_root: Path,
    resolve_mode: Literal["latest", "strict"],
    embedding_variants: Sequence[str] | None,
    feature_resolve_mode: str,
    selected_models: Sequence[str] | None,
) -> list[Path]:
    latest_root = build_classification_latest_dir(
        result_root=result_root,
        dataset=dataset,
        data_run=data_run,
        display_key="placeholder",
    ).parent
    if not latest_root.exists():
        return []
    matches: list[tuple[Path, dict[str, Any]]] = []
    for pointer_dir in sorted(latest_root.iterdir()):
        if not pointer_dir.is_dir():
            continue
        pointer_path = pointer_dir / CURRENT_POINTER_FILENAME
        if not pointer_path.exists():
            continue
        payload = load_json(pointer_path)
        if not isinstance(payload, dict):
            continue
        archive_dir_raw = payload.get("archive_dir")
        if not archive_dir_raw:
            continue
        archive_dir = resolve_project_path(str(archive_dir_raw))
        candidate = archive_dir / filename
        if not candidate.exists():
            continue
        meta, _ = read_evaluation_json(candidate)
        if _matches_metric_meta(
            meta,
            iteration=iteration,
            topics=topics,
            vmf_assignment=vmf_assignment,
            data_run=data_run,
            alignment_mode=alignment_mode,
            classifiers=classifiers,
            embedding_variants=embedding_variants,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=selected_models,
        ):
            matches.append((candidate, meta))
    if not matches:
        return []
    if selected_models is not None:
        return [path for path, _meta in matches]
    selected_match = _select_metric_match(
        matches=matches,
        resolve_mode=resolve_mode,
        source_label=str(latest_root),
    )
    return [] if selected_match is None else [selected_match]


def _resolve_metric_path(
    *,
    metric: str,
    dataset: str,
    data_run: str,
    topics: int,
    iteration: int,
    vmf_assignment: str,
    alignment_mode: str,
    classifiers: List[str] | None,
    result_root: Path,
    resolve_mode: Literal["latest", "strict"],
    embedding_variants: Sequence[str] | None,
    feature_resolve_mode: str,
    selected_models: Sequence[str] | None = None,
) -> Path:
    filename = _metric_filename(metric, dataset, topics)
    latest_candidate = _resolve_metric_path_from_latest(
        filename=filename,
        dataset=dataset,
        data_run=data_run,
        topics=topics,
        iteration=iteration,
        vmf_assignment=vmf_assignment,
        alignment_mode=alignment_mode,
        classifiers=classifiers,
        result_root=result_root,
        resolve_mode=resolve_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
    )
    if latest_candidate is not None:
        return latest_candidate
    try:
        condition = ClassificationCondition(
            dataset=dataset,
            data_run=data_run,
            topics=topics,
            iteration=iteration,
            classifiers=[] if classifiers is None else classifiers,
            vmf_assignment=vmf_assignment,
            target_column="target_str",
            label_schema="identity",
            embedding_variants=embedding_variants,
            feature_resolve_mode=feature_resolve_mode,
        )
        candidate = (
            build_classification_output_dir_from_condition(
                result_root=result_root,
                condition=condition,
            )
            / filename
        )
        if candidate.exists() and classifiers:
            return candidate
    except Exception:
        pass

    dataset_root = result_root / dataset / data_run
    category_root = dataset_root / "all"
    search_roots = []
    if category_root.exists():
        search_roots.append(category_root)
    if dataset_root.exists():
        search_roots.append(dataset_root)
    for search_root in search_roots:
        matches: list[tuple[Path, dict[str, Any]]] = []
        for condition_dir in sorted(search_root.iterdir()):
            if not condition_dir.is_dir():
                continue
            candidate = condition_dir / filename
            if not candidate.exists():
                continue
            meta, _ = read_evaluation_json(candidate)
            if _matches_metric_meta(
                meta,
                iteration=iteration,
                topics=topics,
                vmf_assignment=vmf_assignment,
                data_run=data_run,
                alignment_mode=alignment_mode,
                classifiers=classifiers,
                embedding_variants=embedding_variants,
                feature_resolve_mode=feature_resolve_mode,
                selected_models=selected_models,
            ):
                matches.append((candidate, meta))
        selected_match = _select_metric_match(
            matches=matches,
            resolve_mode=resolve_mode,
            source_label=str(search_root),
        )
        if selected_match is not None:
            return selected_match

    legacy_path = result_root / f"iter{iteration}" / filename
    return legacy_path


def _resolve_metric_paths(
    *,
    metric: str,
    dataset: str,
    data_run: str,
    topics: int,
    iteration: int,
    vmf_assignment: str,
    alignment_mode: str,
    classifiers: List[str] | None,
    result_root: Path,
    resolve_mode: Literal["latest", "strict"],
    embedding_variants: Sequence[str] | None,
    feature_resolve_mode: str,
    selected_models: Sequence[str] | None,
) -> list[Path]:
    filename = _metric_filename(metric, dataset, topics)
    latest_candidates = _resolve_metric_paths_from_latest(
        filename=filename,
        dataset=dataset,
        data_run=data_run,
        topics=topics,
        iteration=iteration,
        vmf_assignment=vmf_assignment,
        alignment_mode=alignment_mode,
        classifiers=classifiers,
        result_root=result_root,
        resolve_mode=resolve_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
    )
    if latest_candidates:
        return latest_candidates
    if selected_models is None:
        return [
            _resolve_metric_path(
                metric=metric,
                dataset=dataset,
                data_run=data_run,
                topics=topics,
                iteration=iteration,
                vmf_assignment=vmf_assignment,
                alignment_mode=alignment_mode,
                classifiers=classifiers,
                result_root=result_root,
                resolve_mode=resolve_mode,
                embedding_variants=embedding_variants,
                feature_resolve_mode=feature_resolve_mode,
            )
        ]

    dataset_root = result_root / dataset / data_run
    category_root = dataset_root / "all"
    search_roots = []
    if category_root.exists():
        search_roots.append(category_root)
    if dataset_root.exists():
        search_roots.append(dataset_root)
    for search_root in search_roots:
        matches: list[tuple[Path, dict[str, Any]]] = []
        for condition_dir in sorted(search_root.iterdir()):
            if not condition_dir.is_dir():
                continue
            candidate = condition_dir / filename
            if not candidate.exists():
                continue
            meta, _ = read_evaluation_json(candidate)
            if _matches_metric_meta(
                meta,
                iteration=iteration,
                topics=topics,
                vmf_assignment=vmf_assignment,
                data_run=data_run,
                alignment_mode=alignment_mode,
                classifiers=classifiers,
                embedding_variants=embedding_variants,
                feature_resolve_mode=feature_resolve_mode,
                selected_models=selected_models,
            ):
                matches.append((candidate, meta))
        if matches:
            return [path for path, _meta in matches]

    legacy_path = result_root / f"iter{iteration}" / filename
    return [legacy_path]


def _merge_feature_catalogs(
    merged: dict[str, list[dict[str, Any]]],
    file_meta: dict[str, Any],
) -> None:
    categories = file_meta.get("categories")
    if not isinstance(categories, dict):
        return
    for category, payload in categories.items():
        if not isinstance(payload, dict):
            continue
        feature_catalog = payload.get("feature_catalog")
        if not isinstance(feature_catalog, list):
            continue
        for entry in feature_catalog:
            if not isinstance(entry, dict):
                continue
            bucket = merged.setdefault(category, [])
            normalized_entry = dict(entry)
            if normalized_entry not in bucket:
                bucket.append(normalized_entry)


def _format_results(values: List[float]) -> str:
    mean = round(np.mean(values), 2)
    std = round(np.std(values), 2)
    return f"{mean:.2f}~\\ensuremath{{\\pm}}~{std:.2f}"


def _rank_and_mark(row: Dict[str, str], means: Dict[str, float]) -> Dict[str, str]:
    if not means:
        return row
    ordered = sorted(means.items(), key=lambda x: x[1])
    best = ordered[-1][0]
    second = ordered[-2][0] if len(ordered) > 1 else best
    for model, text in row.items():
        if model == best:
            row[model] = r"\textbf{" + text + "}"
        elif model == second:
            row[model] = r"\underline{" + text + "}"
    return row


def _latex_escape_text(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


MODEL_TABLE_LABELS = {
    "Blei LDA": "LDA",
    "sentLDA": "SLDA",
    "Gaussian k-means": "GCLU",
    "Spherical k-means": "SCLU",
    "Gaussian mixture": "MGCLU",
    "movMF": "MSCLU",
    "Gaussian LDA": "GLDA",
    "MvTM": "vLDA",
    "ETM": "ETM",
    "Contextual TM": "CTM",
    "SenClu": "SenClu",
    "BERTopic (UMAP + k-means)": "BERTopic",
    "Sentence LDA": "GSLDA",
    "vMF Sentence LDA": "vSLDA",
}

MODEL_SELECTOR_ALIASES = {
    "BERTopic (UMAP + k-means)": ["bertopic_kmeans", "bertopic"],
    "Sentence LDA": ["sentence_gaussianlda", "gslda"],
}


def _model_table_label(model_name: str) -> str:
    base_name = str(model_name).split(" [", 1)[0]
    return MODEL_TABLE_LABELS.get(base_name, str(model_name))


def _normalize_model_selector(value: str) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _model_matches_selector(model_name: str, selector: str) -> bool:
    normalized_selector = _normalize_model_selector(selector)
    if not normalized_selector:
        return False
    base_name = str(model_name).split(" [", 1)[0]
    candidates = {
        str(model_name),
        base_name,
        _model_table_label(model_name),
    }
    candidates.update(MODEL_SELECTOR_ALIASES.get(base_name, []))
    return normalized_selector in {
        _normalize_model_selector(candidate) for candidate in candidates
    }


def _filter_models_by_selector(
    models: list[str],
    selected_models: Sequence[str] | None,
) -> list[str]:
    selectors = [
        str(selector) for selector in (selected_models or []) if str(selector).strip()
    ]
    if not selectors:
        return models
    return [
        model
        for model in models
        if any(_model_matches_selector(model, selector) for selector in selectors)
    ]


def _coverage_status(run_count: int, expected_runs: int) -> str:
    if run_count == 0:
        return "missing"
    if run_count < expected_runs:
        return "partial"
    if run_count == expected_runs:
        return "complete"
    return "extra"


def _build_run_coverage(
    *,
    metric: str,
    dataset: str,
    data_run: str,
    topics: int,
    iterations: Sequence[int],
    classifiers: Sequence[str] | None,
    vmf_assignment: str,
    embedding_variants: Sequence[str] | None,
    categories: Sequence[str],
    results: dict[str, dict[str, list[float]]],
    models: Sequence[str],
    selected_models: Sequence[str] | None,
) -> dict[str, Any]:
    expected_runs = len(iterations)
    rows: list[dict[str, Any]] = []
    selectors = [
        str(selector) for selector in (selected_models or []) if str(selector).strip()
    ]
    if not selectors:
        selectors = list(models)

    for selector in selectors:
        matched_models = [
            model for model in models if _model_matches_selector(model, selector)
        ]
        if not matched_models:
            rows.append(
                {
                    "metric": metric,
                    "dataset": dataset,
                    "data_run": data_run,
                    "topics": topics,
                    "classifiers": (
                        "" if classifiers is None else ",".join(map(str, classifiers))
                    ),
                    "vmf_assignment": vmf_assignment,
                    "embedding_variants": (
                        ""
                        if embedding_variants is None
                        else ",".join(map(str, embedding_variants))
                    ),
                    "selector": selector,
                    "model": "",
                    "category": "",
                    "run_count": 0,
                    "expected_runs": expected_runs,
                    "missing_runs": expected_runs,
                    "status": "missing",
                }
            )
            continue

        for model in matched_models:
            for category in categories:
                run_count = len(results.get(category, {}).get(model, []))
                rows.append(
                    {
                        "metric": metric,
                        "dataset": dataset,
                        "data_run": data_run,
                        "topics": topics,
                        "classifiers": (
                            ""
                            if classifiers is None
                            else ",".join(map(str, classifiers))
                        ),
                        "vmf_assignment": vmf_assignment,
                        "embedding_variants": (
                            ""
                            if embedding_variants is None
                            else ",".join(map(str, embedding_variants))
                        ),
                        "selector": selector,
                        "model": model,
                        "category": category,
                        "run_count": run_count,
                        "expected_runs": expected_runs,
                        "missing_runs": max(expected_runs - run_count, 0),
                        "status": _coverage_status(run_count, expected_runs),
                    }
                )

    return {
        "expected_runs": expected_runs,
        "iterations": list(iterations),
        "rows": rows,
        "incomplete": [
            row
            for row in rows
            if str(row.get("status")) in {"missing", "partial", "extra"}
        ],
    }


RUN_COVERAGE_CSV_FIELDS = [
    "metric",
    "dataset",
    "data_run",
    "topics",
    "classifiers",
    "vmf_assignment",
    "embedding_variants",
    "selector",
    "model",
    "category",
    "run_count",
    "expected_runs",
    "missing_runs",
    "status",
]


def _empty_run_coverage(
    *,
    metric: str,
    dataset: str,
    data_run: str,
    topics: int,
    iterations: Sequence[int],
    classifiers: Sequence[str] | None,
    vmf_assignment: str,
    embedding_variants: Sequence[str] | None,
    selected_models: Sequence[str] | None,
) -> dict[str, Any]:
    expected_runs = len(iterations)
    rows = [
        {
            "metric": metric,
            "dataset": dataset,
            "data_run": data_run,
            "topics": topics,
            "classifiers": (
                "" if classifiers is None else ",".join(map(str, classifiers))
            ),
            "vmf_assignment": vmf_assignment,
            "embedding_variants": (
                ""
                if embedding_variants is None
                else ",".join(map(str, embedding_variants))
            ),
            "selector": str(selector),
            "model": "",
            "category": "",
            "run_count": 0,
            "expected_runs": expected_runs,
            "missing_runs": expected_runs,
            "status": "missing",
        }
        for selector in (selected_models or [])
    ]
    return {
        "expected_runs": expected_runs,
        "iterations": list(iterations),
        "rows": rows,
        "incomplete": list(rows),
    }


def _write_run_coverage_outputs(
    *,
    output_path: Path,
    coverage: dict[str, Any],
) -> None:
    json_path = output_path.with_suffix(".runs.json")
    csv_path = output_path.with_suffix(".runs.csv")
    save_json(coverage, json_path)
    rows = [row for row in coverage.get("rows", []) if isinstance(row, dict)]
    write_csv_rows(
        fieldnames=RUN_COVERAGE_CSV_FIELDS,
        rows=rows,
        path=csv_path,
    )


def build_latex_table(report: dict[str, Any]) -> str:
    meta = report.get("_meta", {})
    results = report.get("results", {})
    models = [str(model) for model in results.get("models", [])]
    rows = [row for row in results.get("rows", []) if isinstance(row, dict)]
    caption_parts = [
        str(meta.get("dataset", "dataset")),
        f"k={meta.get('topics', 'topic')}",
        str(meta.get("metric", "metric")),
    ]
    classifiers = meta.get("classifiers") or []
    if classifiers:
        caption_parts.append("/".join(str(item) for item in classifiers))
    embedding_variants = meta.get("embedding_variants") or []
    if embedding_variants:
        caption_parts.append(
            "emb=" + "/".join(str(item) for item in embedding_variants)
        )
    caption = "Classification summary: " + ", ".join(caption_parts)

    column_spec = "l" + ("c" * len(models))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        f"\\caption{{{_latex_escape_text(caption)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        "Category & "
        + " & ".join(_latex_escape_text(_model_table_label(model)) for model in models)
        + r" \\",
        r"\hline",
    ]
    for row in rows:
        category = row.get("category", "")
        values = row.get("values", {})
        category_label = str(category).capitalize()
        cell_values = [
            str(values.get(model, "-")) if isinstance(values, dict) else "-"
            for model in models
        ]
        lines.append(
            _latex_escape_text(category_label)
            + " & "
            + " & ".join(cell_values)
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_summary_report(
    metric: str,
    dataset: str,
    topics: int,
    iterations: List[int],
    *,
    data_run: str = "default",
    classifiers: List[str] | None = None,
    vmf_assignment: str = "hard",
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    result_root: Path = RESULT_ROOT,
    target_column: str = "target_str",
    label_schema: str = "identity",
    resolve_mode: Literal["latest", "strict"] = "latest",
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    excluded_categories: Sequence[str] | None = None,
    include_all_category: bool = False,
) -> dict[str, Any]:
    metric_key = metric.lower()
    results, source_meta, feature_catalog_by_category = _aggregate(
        metric_key,
        dataset,
        data_run,
        topics,
        iterations,
        vmf_assignment,
        alignment_mode,
        classifiers=classifiers,
        result_root=result_root,
        resolve_mode=resolve_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
    )
    if not results:
        return {}
    dataset_targets = get_dataset_targets(
        dataset,
        target_column=target_column,
        label_schema=label_schema,
    )
    categories = (
        list(dataset_targets.keys()) if dataset_targets is not None else sorted(results)
    )
    excluded_category_set = {
        str(category) for category in (excluded_categories or []) if str(category)
    }
    if not include_all_category:
        excluded_category_set.add("all")
    categories = [
        category
        for category in categories
        if str(category) not in excluded_category_set
    ]

    all_models = set()
    for cat_data in results.values():
        all_models.update(cat_data.keys())
    ordered_models = _filter_models_by_selector(
        _sorted_models(list(all_models)),
        selected_models,
    )

    rows: list[dict[str, Any]] = []
    for category in categories:
        row = {}
        means_for_rank = {}
        for model in ordered_models:
            scores = results.get(category, {}).get(model, [])
            if scores:
                row[model] = _format_results(scores)
                means_for_rank[model] = round(np.mean(scores), 2)
            else:
                row[model] = "-"
        row = _rank_and_mark(row, means_for_rank)
        class_count = (
            len(dataset_targets.get(category, []))
            if dataset_targets is not None and category in dataset_targets
            else 0
        )
        rows.append(
            {
                "category": category,
                "class_count": class_count,
                "values": {model: row[model] for model in ordered_models},
            }
        )

    return {
        "_meta": build_evaluation_meta(
            task="classification_summary",
            output_kind="tabular",
            metric=metric_key,
            dataset=dataset,
            data_run=data_run,
            topics=topics,
            iterations=list(iterations),
            classifiers=None if classifiers is None else list(classifiers),
            vmf_assignment=vmf_assignment,
            alignment_mode=alignment_mode,
            target_column=target_column,
            label_schema=label_schema,
            resolve_mode=resolve_mode,
            embedding_variants=(
                None if embedding_variants is None else list(embedding_variants)
            ),
            feature_resolve_mode=feature_resolve_mode,
            selected_models=(
                None if selected_models is None else list(selected_models)
            ),
            excluded_categories=(
                None if excluded_categories is None else list(excluded_categories)
            ),
            include_all_category=include_all_category,
            source_meta=source_meta,
            feature_catalog_by_category=feature_catalog_by_category,
            run_coverage=_build_run_coverage(
                metric=metric_key,
                dataset=dataset,
                data_run=data_run,
                topics=topics,
                iterations=iterations,
                classifiers=classifiers,
                vmf_assignment=vmf_assignment,
                embedding_variants=embedding_variants,
                categories=categories,
                results=results,
                models=ordered_models,
                selected_models=selected_models,
            ),
        ),
        "results": {
            "models": ordered_models,
            "rows": rows,
        },
    }


def write_summary(
    metric: str,
    dataset: str,
    topics: int,
    iterations: List[int],
    *,
    data_run: str = "default",
    classifiers: List[str] | None = None,
    vmf_assignment: str = "hard",
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    result_root: Path = RESULT_ROOT,
    target_column: str = "target_str",
    label_schema: str = "identity",
    resolve_mode: Literal["latest", "strict"] = "latest",
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    excluded_categories: Sequence[str] | None = None,
    include_all_category: bool = False,
    output_path: Path | None = None,
) -> None:
    report = build_summary_report(
        metric=metric,
        dataset=dataset,
        topics=topics,
        iterations=iterations,
        data_run=data_run,
        classifiers=classifiers,
        vmf_assignment=vmf_assignment,
        alignment_mode=alignment_mode,
        result_root=result_root,
        target_column=target_column,
        label_schema=label_schema,
        resolve_mode=resolve_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
        excluded_categories=excluded_categories,
        include_all_category=include_all_category,
    )
    if not report:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                f"% no results for {dataset} {topics}topic {metric}\n",
                encoding="utf-8",
            )
            _write_run_coverage_outputs(
                output_path=output_path,
                coverage=_empty_run_coverage(
                    metric=metric.lower(),
                    dataset=dataset,
                    data_run=data_run,
                    topics=topics,
                    iterations=iterations,
                    classifiers=classifiers,
                    vmf_assignment=vmf_assignment,
                    embedding_variants=embedding_variants,
                    selected_models=selected_models,
                ),
            )
        print(f"[skip] no results for {dataset} {topics}topic")
        return
    latex_table = build_latex_table(report)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_table + "\n", encoding="utf-8")
        coverage = report.get("_meta", {}).get("run_coverage")
        if isinstance(coverage, dict):
            _write_run_coverage_outputs(output_path=output_path, coverage=coverage)
        print(f"[write] {output_path}")
        return
    print(latex_table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize classification results as LaTeX rows."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
        choices=["acc", "f1mac", "f1mic"],
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["20newsgroup", "nyt"],
    )
    parser.add_argument(
        "--topics",
        type=int,
        nargs="+",
        default=[10, 20, 50],
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=list(range(5)),
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--target-column", type=str, default="target_str")
    parser.add_argument(
        "--resolve-mode",
        type=str,
        default="latest",
        choices=["latest", "strict"],
    )
    parser.add_argument("--embedding-variant", type=str, nargs="*", default=None)
    parser.add_argument(
        "--feature-resolve-mode",
        type=str,
        default=DEFAULT_FEATURE_RESOLVE_MODE,
        choices=["all", "strict"],
    )
    parser.add_argument(
        "--label-schema",
        type=str,
        default="identity",
    )
    args = parser.parse_args()

    for topic in args.topics:
        for dataset in args.datasets:
            print(topic, dataset)
            write_summary(
                args.metric,
                dataset,
                topic,
                iterations=list(args.iterations),
                result_root=args.result_root,
                target_column=args.target_column,
                label_schema=args.label_schema,
                resolve_mode=args.resolve_mode,
                embedding_variants=args.embedding_variant,
                feature_resolve_mode=args.feature_resolve_mode,
            )

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.artifacts import load_artifact_json, save_csv_rows, save_json
from src.evaluation.schema import (
    build_evaluation_payload,
    normalize_evaluation_meta,
    split_evaluation_payload,
)


def write_json(payload: Any, path: Path) -> None:
    save_json(payload, path)


def write_evaluation_json(
    *,
    meta: dict[str, Any],
    results: Any,
    path: Path,
) -> None:
    write_json(build_evaluation_payload(meta=meta, results=results), path)


def read_json(path: Path) -> Any:
    return load_artifact_json(path)


def read_evaluation_json(path: Path) -> tuple[dict[str, Any], Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {}, payload
    return split_evaluation_payload(payload)


def build_tabular_report(
    *,
    meta: dict[str, Any],
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return build_evaluation_payload(
        meta=normalize_evaluation_meta(meta, default_output_kind="tabular"),
        results={
            "columns": list(columns),
            "rows": [dict(row) for row in rows],
        },
    )


def write_tabular_report_json(
    *,
    meta: dict[str, Any],
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    write_json(build_tabular_report(meta=meta, columns=columns, rows=rows), path)


def write_csv_rows(
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    save_csv_rows(fieldnames=fieldnames, rows=rows, path=path)

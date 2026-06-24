from __future__ import annotations

from pathlib import Path

import pytest

from src.core.errors import MissingArtifactError
from src.evaluation.reporting import (
    build_tabular_report,
    read_evaluation_json,
    read_json,
    write_csv_rows,
    write_tabular_report_json,
)


def test_build_tabular_report_wraps_columns_and_rows() -> None:
    payload = build_tabular_report(
        meta={"task": "summary"},
        columns=["dataset", "score"],
        rows=[{"dataset": "dummy", "score": 1.0}],
    )

    assert payload["_meta"]["task"] == "summary"
    assert payload["_meta"]["output_kind"] == "tabular"
    assert payload["results"]["columns"] == ["dataset", "score"]
    assert payload["results"]["rows"] == [{"dataset": "dummy", "score": 1.0}]


def test_write_helpers_persist_csv_and_json_reports(tmp_path: Path) -> None:
    rows = [{"dataset": "dummy", "score": 1.0}]
    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "summary.json"

    write_csv_rows(fieldnames=["dataset", "score"], rows=rows, path=csv_path)
    write_tabular_report_json(
        meta={"task": "summary"},
        columns=["dataset", "score"],
        rows=rows,
        path=json_path,
    )

    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "dataset,score"
    meta, results = read_evaluation_json(json_path)
    assert meta["task"] == "summary"
    assert meta["output_kind"] == "tabular"
    assert results["columns"] == ["dataset", "score"]
    assert results["rows"] == rows


def test_read_json_raises_missing_artifact_error(tmp_path: Path) -> None:
    missing_json = tmp_path / "missing.json"

    with pytest.raises(MissingArtifactError) as exc_info:
        read_json(missing_json)

    assert str(missing_json) in str(exc_info.value)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.core.artifacts import load_json
from src.evaluation.reporting import write_csv_rows

from .summary import RUN_COVERAGE_CSV_FIELDS

INDEX_FIELDS = ["summary_path", *RUN_COVERAGE_CSV_FIELDS]


def _coverage_rows(summary_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(summary_root.rglob("*.runs.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        source_rows = payload.get("rows")
        if not isinstance(source_rows, list):
            continue
        summary_name = path.name
        if summary_name.endswith(".runs.json"):
            summary_name = summary_name[: -len(".runs.json")] + ".tex"
        summary_path = path.with_name(summary_name)
        for row in source_rows:
            if not isinstance(row, dict):
                continue
            indexed_row = {field: "" for field in INDEX_FIELDS}
            indexed_row["summary_path"] = str(summary_path)
            for field in RUN_COVERAGE_CSV_FIELDS:
                indexed_row[field] = row.get(field, "")
            rows.append(indexed_row)
    return rows


def write_summary_coverage_index(
    *,
    summary_root: Path,
    output_path: Path | None = None,
    incomplete_output_path: Path | None = None,
) -> tuple[Path, Path, int, int]:
    output_path = output_path or summary_root / "run_coverage.csv"
    incomplete_output_path = (
        incomplete_output_path or summary_root / "run_coverage_incomplete.csv"
    )
    rows = _coverage_rows(summary_root)
    incomplete_rows = [row for row in rows if str(row.get("status", "")) != "complete"]
    write_csv_rows(fieldnames=INDEX_FIELDS, rows=rows, path=output_path)
    write_csv_rows(
        fieldnames=INDEX_FIELDS,
        rows=incomplete_rows,
        path=incomplete_output_path,
    )
    return output_path, incomplete_output_path, len(rows), len(incomplete_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an index of classification summary run coverage sidecars."
    )
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--incomplete-output-path", type=Path, default=None)
    args = parser.parse_args()

    output_path, incomplete_output_path, row_count, incomplete_count = (
        write_summary_coverage_index(
            summary_root=args.summary_root,
            output_path=args.output_path,
            incomplete_output_path=args.incomplete_output_path,
        )
    )
    print(f"[write] {output_path} rows={row_count}")
    print(f"[write] {incomplete_output_path} rows={incomplete_count}")


if __name__ == "__main__":
    main()

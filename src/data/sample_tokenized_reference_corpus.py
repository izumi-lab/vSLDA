from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uniformly sample lines from a tokenized JSONL reference corpus without "
            "loading the corpus into memory."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-lines",
        type=int,
        default=None,
        help="Known input line count. If omitted, the input is counted first.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Metadata JSON path. Defaults to OUTPUT with .metadata.json suffix.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as handle:
        for _line in handle:
            total += 1
    return total


def _metadata_path(output_path: Path, metadata_output: Path | None) -> Path:
    if metadata_output is not None:
        return metadata_output
    return output_path.with_suffix(".metadata.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def _sample_lines(
    *,
    input_path: Path,
    output_path: Path,
    sample_size: int,
    total_lines: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_lines), sample_size))
    selected_pos = 0
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(output_path.name + ".tmp")

    started_at = datetime.now(UTC).isoformat()
    with input_path.open("rb") as src, temp_output.open("wb") as dst:
        next_selected = selected[selected_pos]
        for line_index, line in enumerate(src):
            if line_index != next_selected:
                continue
            dst.write(line)
            written += 1
            selected_pos += 1
            if selected_pos >= sample_size:
                break
            next_selected = selected[selected_pos]

    if written != sample_size:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"Expected to write {sample_size} lines, wrote {written}.")

    temp_output.replace(output_path)
    finished_at = datetime.now(UTC).isoformat()
    return {
        "task": "sample_tokenized_reference_corpus",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sample_size": int(sample_size),
        "total_lines": int(total_lines),
        "seed": int(seed),
        "selection": "uniform_without_replacement_by_line",
        "output_order": "input_order",
        "first_selected_line_index": int(selected[0]),
        "last_selected_line_index": int(selected[-1]),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def main() -> None:
    args = _parse_args()
    input_path = args.input
    output_path = args.output
    metadata_path = _metadata_path(output_path, args.metadata_output)

    if args.sample_size < 1:
        raise ValueError(f"sample-size must be >= 1, got {args.sample_size}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}")
    if metadata_path.exists() and not args.force:
        raise FileExistsError(f"Metadata output already exists: {metadata_path}")

    total_lines = args.total_lines
    if total_lines is None:
        print(f"[sample-reference] counting lines: {input_path}", file=sys.stderr)
        total_lines = _count_lines(input_path)
    if total_lines < args.sample_size:
        raise ValueError(
            f"sample-size {args.sample_size} exceeds total lines {total_lines}."
        )

    print(
        "[sample-reference] sampling "
        f"{args.sample_size} / {total_lines} lines seed={args.seed}",
        file=sys.stderr,
    )
    metadata = _sample_lines(
        input_path=input_path,
        output_path=output_path,
        sample_size=args.sample_size,
        total_lines=total_lines,
        seed=args.seed,
    )
    _write_json(metadata_path, metadata)
    print(f"[sample-reference] wrote {output_path}", file=sys.stderr)
    print(f"[sample-reference] wrote {metadata_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

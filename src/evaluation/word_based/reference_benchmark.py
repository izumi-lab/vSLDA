from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter, process_time

from .reference_counts import (
    build_shared_reference_counts,
    effective_reference_count_backend,
)
from .reference_query import ReferenceCountQuery


def _load_query(path: Path) -> ReferenceCountQuery:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReferenceCountQuery(
        target_words=tuple(payload["target_words"]),
        requested_pairs=tuple(tuple(pair) for pair in payload["requested_pairs"]),
        window_sizes=tuple(payload["window_sizes"]),
        need_document_counts=bool(payload.get("need_document_counts", False)),
    )


def _serialize_counts(counts) -> dict[str, object]:
    return {
        "num_docs": counts.num_docs,
        "window_counts": {
            str(window_size): {
                "num_windows": values.num_windows,
                "word_window_counts": dict(values.word_window_counts),
                "pair_window_counts": [
                    [word_i, word_j, count]
                    for (word_i, word_j), count in sorted(
                        values.pair_window_counts.items()
                    )
                ],
            }
            for window_size, values in sorted(counts.counts_by_window_size.items())
        },
        "doc_word_counts": dict(counts.doc_word_counts),
        "doc_pair_counts": [
            [word_i, word_j, count]
            for (word_i, word_j), count in sorted(counts.doc_pair_counts.items())
        ],
    }


def run_reference_benchmark(
    *,
    reference_path: Path,
    query: ReferenceCountQuery,
    backend: str,
    workers: int,
    chunk_size: int,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
) -> tuple[dict[str, object], dict[str, object]]:
    wall_started = perf_counter()
    cpu_started = process_time()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    counts = build_shared_reference_counts(
        reference_path=reference_path,
        target_words=set(query.target_words),
        window_sizes=set(query.window_sizes),
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
        backend=backend,  # type: ignore[arg-type]
        workers=workers,
        chunk_size=chunk_size,
        query=query,
        progress_label="reference benchmark",
    )
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    benchmark = {
        "reference_path": str(reference_path.resolve()),
        "reference_size": reference_path.stat().st_size,
        "query_fingerprint": query.fingerprint,
        "requested_backend": backend,
        # max_docs silently downgrades numba_interval to numba, so labelling the
        # row with the requested backend would misattribute the timings.
        "backend": effective_reference_count_backend(
            backend,  # type: ignore[arg-type]
            max_docs=max_docs,
        ),
        "workers": workers,
        "chunk_size": chunk_size,
        "max_docs": max_docs,
        "min_doc_tokens": min_doc_tokens,
        "target_words": len(query.target_words),
        "requested_pairs": len(query.requested_pairs),
        "window_sizes": list(query.window_sizes),
        "num_docs": counts.num_docs,
        "num_windows": {
            str(size): values.num_windows
            for size, values in counts.counts_by_window_size.items()
        },
        "wall_seconds": perf_counter() - wall_started,
        "parent_cpu_seconds": process_time() - cpu_started,
        "parent_user_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "parent_system_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "parent_peak_rss_kib": usage_after.ru_maxrss,
    }
    return benchmark, _serialize_counts(counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact word-based reference counting."
    )
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["python", "numba", "numba_interval"],
        default="numba_interval",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--min-doc-tokens", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden-counts", type=Path, default=None)
    args = parser.parse_args()
    query = _load_query(args.query)
    benchmark, counts = run_reference_benchmark(
        reference_path=args.reference_path,
        query=query,
        backend=args.backend,
        workers=args.workers,
        chunk_size=args.chunk_size,
        max_docs=args.max_docs,
        min_doc_tokens=args.min_doc_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.golden_counts is not None:
        args.golden_counts.parent.mkdir(parents=True, exist_ok=True)
        args.golden_counts.write_text(
            json.dumps(counts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

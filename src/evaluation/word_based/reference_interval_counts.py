from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from .reference_query import ReferenceCountQuery
from .topic_word_metrics import SlidingWindowCounts


def occurrence_window_intervals(
    positions: Iterable[int],
    *,
    document_length: int,
    window_size: int,
) -> tuple[tuple[int, int], ...]:
    """Return merged, inclusive window-start intervals for word occurrences."""
    if document_length <= 0:
        return ()
    normalized = sorted(set(int(position) for position in positions))
    if not normalized:
        return ()
    if document_length <= window_size:
        return ((0, 0),)
    last_start = document_length - window_size
    merged: list[tuple[int, int]] = []
    for position in normalized:
        if position < 0 or position >= document_length:
            raise ValueError(
                f"occurrence position {position} outside document length "
                f"{document_length}"
            )
        start = max(0, position - window_size + 1)
        end = min(position, last_start)
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def interval_union_size(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


def interval_intersection_size(
    left: Iterable[tuple[int, int]],
    right: Iterable[tuple[int, int]],
) -> int:
    left_values = tuple(left)
    right_values = tuple(right)
    left_idx = right_idx = total = 0
    while left_idx < len(left_values) and right_idx < len(right_values):
        left_start, left_end = left_values[left_idx]
        right_start, right_end = right_values[right_idx]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if start <= end:
            total += end - start + 1
        if left_end <= right_end:
            left_idx += 1
        else:
            right_idx += 1
    return total


def byte_shard_ranges(path: Path, shard_count: int) -> tuple[tuple[int, int], ...]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    size = path.stat().st_size
    return tuple(
        (size * shard_id // shard_count, size * (shard_id + 1) // shard_count)
        for shard_id in range(shard_count)
    )


def _tokens_from_line(line: bytes, *, path: Path, offset: int) -> list[str]:
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Invalid JSON in reference corpus at {path} byte {offset}: {exc}"
        ) from exc
    raw_tokens = payload.get("tokens") if isinstance(payload, dict) else payload
    if not isinstance(raw_tokens, list):
        raise ValueError(
            "Reference corpus JSONL rows must be token lists or objects with a "
            f"list-valued 'tokens' field ({path} byte {offset})."
        )
    tokens: list[str] = []
    for token in raw_tokens:
        if not isinstance(token, str):
            raise ValueError(
                f"Reference corpus tokens must be strings ({path} byte {offset})."
            )
        normalized = token.strip()
        if normalized:
            tokens.append(normalized)
    return tokens


def iter_byte_shard_tokens(
    path: Path,
    *,
    start: int,
    end: int,
    min_doc_tokens: int,
):
    with path.open("rb") as handle:
        if start > 0:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        else:
            handle.seek(0)
        while True:
            line_start = handle.tell()
            if line_start >= end:
                break
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            tokens = _tokens_from_line(line, path=path, offset=line_start)
            if len(tokens) >= min_doc_tokens:
                yield tokens


def _build_pair_adjacency(
    query: ReferenceCountQuery,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    word_to_id = {word: idx for idx, word in enumerate(query.target_words)}
    edges: list[list[tuple[int, int]]] = [[] for _ in query.target_words]
    for pair_id, (word_i, word_j) in enumerate(query.requested_pairs):
        id_i, id_j = word_to_id[word_i], word_to_id[word_j]
        if id_j < id_i:
            id_i, id_j = id_j, id_i
        edges[id_i].append((id_j, pair_id))
    indptr = np.zeros(len(edges) + 1, dtype=np.int64)
    other: list[int] = []
    pair_ids: list[int] = []
    for word_id, values in enumerate(edges):
        values.sort()
        other.extend(value[0] for value in values)
        pair_ids.extend(value[1] for value in values)
        indptr[word_id + 1] = len(other)
    return (
        indptr,
        np.asarray(other, dtype=np.int64),
        np.asarray(pair_ids, dtype=np.int64),
    )


def _encode_documents(
    tokens_by_doc: list[list[str]],
    word_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray([len(tokens) for tokens in tokens_by_doc], dtype=np.int64)
    offsets = np.zeros(len(tokens_by_doc) + 1, dtype=np.int64)
    positions: list[int] = []
    word_ids: list[int] = []
    for doc_idx, tokens in enumerate(tokens_by_doc):
        for position, token in enumerate(tokens):
            word_id = word_to_id.get(token)
            if word_id is not None:
                positions.append(position)
                word_ids.append(word_id)
        offsets[doc_idx + 1] = len(positions)
    return (
        lengths,
        offsets,
        np.asarray(positions, dtype=np.int64),
        np.asarray(word_ids, dtype=np.int64),
    )


@lru_cache(maxsize=1)
def _get_interval_kernel():
    try:
        from numba import njit
    except ImportError as exc:
        raise RuntimeError(
            "numba_interval backend requested but numba is not installed"
        ) from exc

    @njit
    def _intersection_size(head_i, head_j, starts, ends, next_interval):
        total = 0
        left = head_i
        right = head_j
        while left >= 0 and right >= 0:
            start = max(starts[left], starts[right])
            end = min(ends[left], ends[right])
            if start <= end:
                total += end - start + 1
            if ends[left] <= ends[right]:
                left = next_interval[left]
            else:
                right = next_interval[right]
        return total

    @njit
    def _kernel(
        document_lengths,
        doc_offsets,
        positions,
        word_ids,
        window_sizes,
        num_targets,
        pair_indptr,
        pair_other_ids,
        pair_ids,
        need_document_counts,
    ):
        num_sizes = len(window_sizes)
        num_pairs = len(pair_ids)
        word_counts = np.zeros((num_sizes, num_targets), np.int64)
        pair_counts = np.zeros((num_sizes, num_pairs), np.int64)
        num_windows = np.zeros(num_sizes, np.int64)
        doc_word_counts = np.zeros(num_targets, np.int64)
        doc_pair_counts = np.zeros(num_pairs, np.int64)
        occurrence_next = np.full(len(positions), -1, np.int64)
        word_heads = np.full(num_targets, -1, np.int64)
        word_tails = np.full(num_targets, -1, np.int64)
        interval_heads = np.full(num_targets, -1, np.int64)
        interval_tails = np.full(num_targets, -1, np.int64)
        interval_starts = np.empty(len(positions), np.int64)
        interval_ends = np.empty(len(positions), np.int64)
        interval_next = np.full(len(positions), -1, np.int64)
        touched = np.empty(num_targets, np.int64)

        for doc_idx in range(len(document_lengths)):
            doc_length = document_lengths[doc_idx]
            occ_start = doc_offsets[doc_idx]
            occ_end = doc_offsets[doc_idx + 1]
            touched_count = 0
            for occ_idx in range(occ_start, occ_end):
                word_id = word_ids[occ_idx]
                if word_heads[word_id] < 0:
                    word_heads[word_id] = occ_idx
                    touched[touched_count] = word_id
                    touched_count += 1
                else:
                    occurrence_next[word_tails[word_id]] = occ_idx
                word_tails[word_id] = occ_idx

            if need_document_counts:
                for touched_idx in range(touched_count):
                    word_id = touched[touched_idx]
                    doc_word_counts[word_id] += 1
                    for edge_idx in range(
                        pair_indptr[word_id], pair_indptr[word_id + 1]
                    ):
                        if word_heads[pair_other_ids[edge_idx]] >= 0:
                            doc_pair_counts[pair_ids[edge_idx]] += 1

            for size_idx in range(num_sizes):
                window_size = window_sizes[size_idx]
                if doc_length > 0:
                    num_windows[size_idx] += max(1, doc_length - window_size + 1)
                interval_cursor = occ_start
                for touched_idx in range(touched_count):
                    word_id = touched[touched_idx]
                    occ_idx = word_heads[word_id]
                    head = -1
                    tail = -1
                    while occ_idx >= 0:
                        if doc_length <= window_size:
                            start = 0
                            end = 0
                        else:
                            start = max(0, positions[occ_idx] - window_size + 1)
                            end = min(positions[occ_idx], doc_length - window_size)
                        if tail >= 0 and start <= interval_ends[tail] + 1:
                            if end > interval_ends[tail]:
                                interval_ends[tail] = end
                        else:
                            interval_starts[interval_cursor] = start
                            interval_ends[interval_cursor] = end
                            interval_next[interval_cursor] = -1
                            if head < 0:
                                head = interval_cursor
                            else:
                                interval_next[tail] = interval_cursor
                            tail = interval_cursor
                            interval_cursor += 1
                        occ_idx = occurrence_next[occ_idx]
                    interval_heads[word_id] = head
                    interval_tails[word_id] = tail
                    current = head
                    while current >= 0:
                        word_counts[size_idx, word_id] += (
                            interval_ends[current] - interval_starts[current] + 1
                        )
                        current = interval_next[current]

                for touched_idx in range(touched_count):
                    word_id = touched[touched_idx]
                    for edge_idx in range(
                        pair_indptr[word_id], pair_indptr[word_id + 1]
                    ):
                        other_id = pair_other_ids[edge_idx]
                        if interval_heads[other_id] >= 0:
                            pair_counts[
                                size_idx, pair_ids[edge_idx]
                            ] += _intersection_size(
                                interval_heads[word_id],
                                interval_heads[other_id],
                                interval_starts,
                                interval_ends,
                                interval_next,
                            )
                for touched_idx in range(touched_count):
                    word_id = touched[touched_idx]
                    interval_heads[word_id] = -1
                    interval_tails[word_id] = -1

            for touched_idx in range(touched_count):
                word_id = touched[touched_idx]
                word_heads[word_id] = -1
                word_tails[word_id] = -1
            for occ_idx in range(occ_start, occ_end):
                occurrence_next[occ_idx] = -1
        return (
            word_counts,
            pair_counts,
            num_windows,
            doc_word_counts,
            doc_pair_counts,
        )

    return _kernel


def _count_encoded(
    *,
    tokens_by_doc: list[list[str]],
    query: ReferenceCountQuery,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    word_to_id = {word: idx for idx, word in enumerate(query.target_words)}
    lengths, offsets, positions, word_ids = _encode_documents(tokens_by_doc, word_to_id)
    indptr, other_ids, pair_ids = _build_pair_adjacency(query)
    kernel = _get_interval_kernel()
    result = kernel(
        lengths,
        offsets,
        positions,
        word_ids,
        np.asarray(query.window_sizes, dtype=np.int64),
        len(query.target_words),
        indptr,
        other_ids,
        pair_ids,
        query.need_document_counts,
    )
    return (*result, len(tokens_by_doc))


def _count_shard(
    *,
    reference_path: Path,
    start: int,
    end: int,
    query: ReferenceCountQuery,
    min_doc_tokens: int,
    flush_docs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    shape_words = (len(query.window_sizes), len(query.target_words))
    shape_pairs = (len(query.window_sizes), len(query.requested_pairs))
    word_counts = np.zeros(shape_words, dtype=np.int64)
    pair_counts = np.zeros(shape_pairs, dtype=np.int64)
    num_windows = np.zeros(len(query.window_sizes), dtype=np.int64)
    doc_word_counts = np.zeros(len(query.target_words), dtype=np.int64)
    doc_pair_counts = np.zeros(len(query.requested_pairs), dtype=np.int64)
    num_docs = 0
    chunk: list[list[str]] = []
    for tokens in iter_byte_shard_tokens(
        reference_path,
        start=start,
        end=end,
        min_doc_tokens=min_doc_tokens,
    ):
        chunk.append(tokens)
        if len(chunk) < flush_docs:
            continue
        result = _count_encoded(tokens_by_doc=chunk, query=query)
        word_counts += result[0]
        pair_counts += result[1]
        num_windows += result[2]
        doc_word_counts += result[3]
        doc_pair_counts += result[4]
        num_docs += result[5]
        chunk = []
    if chunk:
        result = _count_encoded(tokens_by_doc=chunk, query=query)
        word_counts += result[0]
        pair_counts += result[1]
        num_windows += result[2]
        doc_word_counts += result[3]
        doc_pair_counts += result[4]
        num_docs += result[5]
    return (
        word_counts,
        pair_counts,
        num_windows,
        doc_word_counts,
        doc_pair_counts,
        num_docs,
    )


def _to_public_counts(
    *,
    query: ReferenceCountQuery,
    word_counts: np.ndarray,
    pair_counts: np.ndarray,
    num_windows: np.ndarray,
    doc_word_counts: np.ndarray,
    doc_pair_counts: np.ndarray,
) -> tuple[
    dict[int, SlidingWindowCounts],
    Counter[str],
    Counter[tuple[str, str]],
]:
    counts_by_window_size: dict[int, SlidingWindowCounts] = {}
    for size_idx, window_size in enumerate(query.window_sizes):
        counts_by_window_size[window_size] = SlidingWindowCounts(
            word_window_counts=Counter(
                {
                    word: int(word_counts[size_idx, word_id])
                    for word_id, word in enumerate(query.target_words)
                    if word_counts[size_idx, word_id]
                }
            ),
            pair_window_counts=Counter(
                {
                    pair: int(pair_counts[size_idx, pair_id])
                    for pair_id, pair in enumerate(query.requested_pairs)
                    if pair_counts[size_idx, pair_id]
                }
            ),
            num_windows=int(num_windows[size_idx]),
        )
    return (
        counts_by_window_size,
        Counter(
            {
                word: int(doc_word_counts[word_id])
                for word_id, word in enumerate(query.target_words)
                if doc_word_counts[word_id]
            }
        ),
        Counter(
            {
                pair: int(doc_pair_counts[pair_id])
                for pair_id, pair in enumerate(query.requested_pairs)
                if doc_pair_counts[pair_id]
            }
        ),
    )


def build_interval_reference_counts(
    *,
    reference_path: Path,
    query: ReferenceCountQuery,
    min_doc_tokens: int,
    workers: int,
    flush_docs: int,
) -> tuple[
    dict[int, SlidingWindowCounts],
    Counter[str],
    Counter[tuple[str, str]],
    int,
]:
    # Compile once in the parent before forking so workers inherit the ready
    # machine code instead of compiling the same kernel concurrently.
    _count_encoded(tokens_by_doc=[], query=query)
    ranges = byte_shard_ranges(reference_path, workers)
    if workers == 1:
        results = [
            _count_shard(
                reference_path=reference_path,
                start=ranges[0][0],
                end=ranges[0][1],
                query=query,
                min_doc_tokens=min_doc_tokens,
                flush_docs=flush_docs,
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _count_shard,
                    reference_path=reference_path,
                    start=start,
                    end=end,
                    query=query,
                    min_doc_tokens=min_doc_tokens,
                    flush_docs=flush_docs,
                )
                for start, end in ranges
            ]
            # Merge in deterministic shard order.
            results = [future.result() for future in futures]
    word_counts = sum(
        (result[0] for result in results), start=np.zeros_like(results[0][0])
    )
    pair_counts = sum(
        (result[1] for result in results), start=np.zeros_like(results[0][1])
    )
    num_windows = sum(
        (result[2] for result in results), start=np.zeros_like(results[0][2])
    )
    doc_word_counts = sum(
        (result[3] for result in results), start=np.zeros_like(results[0][3])
    )
    doc_pair_counts = sum(
        (result[4] for result in results), start=np.zeros_like(results[0][4])
    )
    public = _to_public_counts(
        query=query,
        word_counts=word_counts,
        pair_counts=pair_counts,
        num_windows=num_windows,
        doc_word_counts=doc_word_counts,
        doc_pair_counts=doc_pair_counts,
    )
    return (*public, sum(int(result[5]) for result in results))

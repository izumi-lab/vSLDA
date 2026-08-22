from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .reference_counts import (
    ReferenceCountKey,
    SharedReferenceCounts,
    fingerprint_target_words,
)
from .reference_query import ReferenceCountQuery
from .topic_word_metrics import SlidingWindowCounts

REFERENCE_CACHE_SCHEMA_VERSION = 2


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reference_cache_v2_dir(
    *,
    cache_root: Path,
    query: ReferenceCountQuery,
) -> Path:
    return (
        cache_root
        / "reference_counts"
        / f"v{REFERENCE_CACHE_SCHEMA_VERSION}"
        / query.fingerprint
    )


def _query_payload(query: ReferenceCountQuery) -> dict[str, Any]:
    return {
        "target_words": list(query.target_words),
        "requested_pairs": [list(pair) for pair in query.requested_pairs],
        "window_sizes": list(query.window_sizes),
        "need_document_counts": query.need_document_counts,
        "fingerprint": query.fingerprint,
    }


def _query_from_payload(payload: Mapping[str, Any]) -> ReferenceCountQuery:
    return ReferenceCountQuery(
        target_words=tuple(str(word) for word in payload["target_words"]),
        requested_pairs=tuple(
            (str(pair[0]), str(pair[1])) for pair in payload["requested_pairs"]
        ),
        window_sizes=tuple(int(value) for value in payload["window_sizes"]),
        need_document_counts=bool(payload["need_document_counts"]),
    )


def save_reference_count_cache_v2(
    *,
    cache_root: Path,
    reference_identity: Mapping[str, Any],
    max_docs: int | None,
    min_doc_tokens: int,
    query: ReferenceCountQuery,
    counts: SharedReferenceCounts,
) -> Path:
    cache_dir = reference_cache_v2_dir(cache_root=cache_root, query=query)
    cache_dir.mkdir(parents=True, exist_ok=True)
    completion = cache_dir / "COMPLETE.json"
    if completion.exists():
        completion.unlink()
    word_to_id = {word: idx for idx, word in enumerate(query.target_words)}
    pair_to_id = {pair: idx for idx, pair in enumerate(query.requested_pairs)}
    word_counts = np.zeros(
        (len(query.window_sizes), len(query.target_words)), dtype=np.int64
    )
    pair_counts = np.zeros(
        (len(query.window_sizes), len(query.requested_pairs)), dtype=np.int64
    )
    num_windows = np.zeros(len(query.window_sizes), dtype=np.int64)
    for size_idx, window_size in enumerate(query.window_sizes):
        values = counts.counts_by_window_size[window_size]
        num_windows[size_idx] = values.num_windows
        for word, count in values.word_window_counts.items():
            if word in word_to_id:
                word_counts[size_idx, word_to_id[word]] = count
        for pair, count in values.pair_window_counts.items():
            if pair in pair_to_id:
                pair_counts[size_idx, pair_to_id[pair]] = count
    doc_word_counts = np.asarray(
        [counts.doc_word_counts[word] for word in query.target_words],
        dtype=np.int64,
    )
    doc_pair_counts = np.asarray(
        [counts.doc_pair_counts[pair] for pair in query.requested_pairs],
        dtype=np.int64,
    )
    for name, values in (
        ("word_counts.npy", word_counts),
        ("pair_counts.npy", pair_counts),
        ("num_windows.npy", num_windows),
        ("doc_word_counts.npy", doc_word_counts),
        ("doc_pair_counts.npy", doc_pair_counts),
    ):
        _atomic_npy(cache_dir / name, values)
    _atomic_json(cache_dir / "query.json", _query_payload(query))
    _atomic_json(
        cache_dir / "manifest.json",
        {
            "schema": "word_based_reference_count_cache",
            "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
            "reference": _jsonable(dict(reference_identity)),
            "max_docs": max_docs,
            "min_doc_tokens": int(min_doc_tokens),
            "num_docs": int(counts.num_docs),
            "counter_backend": counts.key.backend,
            "interval_algorithm_version": 1,
            "count_dtype": "int64",
            "query_fingerprint": query.fingerprint,
        },
    )
    _atomic_json(
        completion,
        {
            "schema": "word_based_reference_count_cache_completion",
            "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
            "query_fingerprint": query.fingerprint,
        },
    )
    return cache_dir


def _load_candidate(
    *,
    cache_dir: Path,
    reference_identity: Mapping[str, Any],
    max_docs: int | None,
    min_doc_tokens: int,
    query: ReferenceCountQuery,
) -> SharedReferenceCounts | None:
    if not (cache_dir / "COMPLETE.json").exists():
        return None
    try:
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != REFERENCE_CACHE_SCHEMA_VERSION:
            return None
        if manifest.get("reference") != _jsonable(dict(reference_identity)):
            return None
        if manifest.get("max_docs") != max_docs:
            return None
        if int(manifest.get("min_doc_tokens", -1)) != int(min_doc_tokens):
            return None
        cached_query = _query_from_payload(
            json.loads((cache_dir / "query.json").read_text(encoding="utf-8"))
        )
        if not query.is_subset_of(cached_query):
            return None
        word_counts = np.load(cache_dir / "word_counts.npy", allow_pickle=False)
        pair_counts = np.load(cache_dir / "pair_counts.npy", allow_pickle=False)
        num_windows = np.load(cache_dir / "num_windows.npy", allow_pickle=False)
        cached_word_ids = {
            word: idx for idx, word in enumerate(cached_query.target_words)
        }
        cached_pair_ids = {
            pair: idx for idx, pair in enumerate(cached_query.requested_pairs)
        }
        cached_size_ids = {
            size: idx for idx, size in enumerate(cached_query.window_sizes)
        }
        counts_by_window_size: dict[int, SlidingWindowCounts] = {}
        for window_size in query.window_sizes:
            size_idx = cached_size_ids[window_size]
            counts_by_window_size[window_size] = SlidingWindowCounts(
                word_window_counts=Counter(
                    {
                        word: int(word_counts[size_idx, cached_word_ids[word]])
                        for word in query.target_words
                        if word_counts[size_idx, cached_word_ids[word]]
                    }
                ),
                pair_window_counts=Counter(
                    {
                        pair: int(pair_counts[size_idx, cached_pair_ids[pair]])
                        for pair in query.requested_pairs
                        if pair_counts[size_idx, cached_pair_ids[pair]]
                    }
                ),
                num_windows=int(num_windows[size_idx]),
            )
        doc_word_counts: Counter[str] = Counter()
        doc_pair_counts: Counter[tuple[str, str]] = Counter()
        if query.need_document_counts:
            doc_words = np.load(cache_dir / "doc_word_counts.npy", allow_pickle=False)
            doc_pairs = np.load(cache_dir / "doc_pair_counts.npy", allow_pickle=False)
            doc_word_counts.update(
                {
                    word: int(doc_words[cached_word_ids[word]])
                    for word in query.target_words
                    if doc_words[cached_word_ids[word]]
                }
            )
            doc_pair_counts.update(
                {
                    pair: int(doc_pairs[cached_pair_ids[pair]])
                    for pair in query.requested_pairs
                    if doc_pairs[cached_pair_ids[pair]]
                }
            )
        reference_path = Path(str(reference_identity["path"]))
        backend = str(manifest.get("counter_backend", "numba_interval"))
        if backend not in {"python", "numba", "numba_interval"}:
            backend = "numba_interval"
        return SharedReferenceCounts(
            key=ReferenceCountKey(
                reference_path=reference_path,
                max_docs=max_docs,
                min_doc_tokens=int(min_doc_tokens),
                window_sizes=query.window_sizes,
                target_words_fingerprint=fingerprint_target_words(query.target_words),
                query_fingerprint=query.fingerprint,
                backend=backend,  # type: ignore[arg-type]
            ),
            counts_by_window_size=counts_by_window_size,
            doc_word_counts=doc_word_counts,
            doc_pair_counts=doc_pair_counts,
            num_docs=int(manifest["num_docs"]),
            target_words=set(query.target_words),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def load_reference_count_cache_v2(
    *,
    cache_root: Path,
    reference_identity: Mapping[str, Any],
    max_docs: int | None,
    min_doc_tokens: int,
    query: ReferenceCountQuery,
) -> tuple[SharedReferenceCounts, Path] | None:
    root = cache_root / "reference_counts" / f"v{REFERENCE_CACHE_SCHEMA_VERSION}"
    exact = reference_cache_v2_dir(cache_root=cache_root, query=query)
    candidates = [exact]
    if root.exists():
        candidates.extend(
            path for path in sorted(root.iterdir()) if path.is_dir() and path != exact
        )
    for candidate in candidates:
        counts = _load_candidate(
            cache_dir=candidate,
            reference_identity=reference_identity,
            max_docs=max_docs,
            min_doc_tokens=min_doc_tokens,
            query=query,
        )
        if counts is not None:
            return counts, candidate
    return None

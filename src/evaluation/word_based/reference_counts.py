from __future__ import annotations

import hashlib
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import log
from pathlib import Path
from sys import stderr
from time import perf_counter
from typing import Iterable, Literal

import numpy as np

from src.utils.logging import get_logger, get_progress_bar

from .corpus_bundle import iter_tokenized_reference_corpus
from .topic_word_metrics import (
    DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT,
    EPSILON_SMOOTHED_COHERENCES,
    PMI_SMOOTHING_EPSILON,
    SlidingWindowCounts,
    _compute_epsilon_smoothed_pmi_score_from_counts,
    _compute_palmetto_cv_score_from_counts,
    _effective_sliding_window_size,
    coherence_metric_key,
    compute_topic_diversity,
    normalize_coherences,
    topic_words_to_word_lists,
    truncate_topic_words,
)
from .topic_words import TopicWords

logger = get_logger(__name__)

ReferenceCountBackend = Literal["python", "numba"]
REFERENCE_PROGRESS_DOC_INTERVAL = 100_000
DEFAULT_REFERENCE_COUNT_WORKERS = 8
DEFAULT_REFERENCE_COUNT_CHUNK_SIZE = 25_000


@dataclass(frozen=True)
class ReferenceCountKey:
    reference_path: Path
    max_docs: int | None
    min_doc_tokens: int
    window_sizes: tuple[int, ...]
    target_words_fingerprint: str
    backend: ReferenceCountBackend = "numba"
    workers: int = DEFAULT_REFERENCE_COUNT_WORKERS
    chunk_size: int = DEFAULT_REFERENCE_COUNT_CHUNK_SIZE


@dataclass
class SharedReferenceCounts:
    key: ReferenceCountKey
    counts_by_window_size: dict[int, SlidingWindowCounts]
    doc_word_counts: Counter[str]
    doc_pair_counts: Counter[tuple[str, str]]
    num_docs: int
    target_words: set[str]

    @property
    def vocab_size(self) -> int:
        return len(self.target_words)


def fingerprint_target_words(target_words: Iterable[str]) -> str:
    encoded = "\n".join(sorted(set(target_words))).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:10]


def collect_target_words(topic_words_by_condition: Iterable[TopicWords]) -> set[str]:
    return {
        word
        for topic_words in topic_words_by_condition
        for words in topic_words_to_word_lists(topic_words)
        for word in words
    }


def effective_window_sizes_for_coherences(
    coherences: Iterable[str],
    *,
    window_size: int | None = None,
) -> set[int]:
    return {
        _effective_sliding_window_size(coherence, window_size)
        for coherence in normalize_coherences(list(coherences))
        if coherence == "c_v" or coherence in EPSILON_SMOOTHED_COHERENCES
    }


def _iter_boolean_sliding_window_sets(
    tokens: list[str],
    *,
    window_size: int,
):
    if not tokens:
        return
    if len(tokens) <= window_size:
        yield set(tokens)
        return
    for start in range(0, len(tokens) - window_size + 1):
        yield set(tokens[start : start + window_size])


def _build_python_reference_counts(
    *,
    reference_path: Path,
    target_words: set[str],
    window_sizes: set[int],
    max_docs: int | None,
    min_doc_tokens: int,
    progress_label: str,
) -> tuple[dict[int, SlidingWindowCounts], Counter[str], Counter[tuple[str, str]], int]:
    counts_by_window_size = {
        int(window_size): SlidingWindowCounts(Counter(), Counter(), 0)
        for window_size in window_sizes
    }
    doc_word_counts: Counter[str] = Counter()
    doc_pair_counts: Counter[tuple[str, str]] = Counter()
    num_docs = 0
    started = perf_counter()
    reference_iter = iter_tokenized_reference_corpus(
        path=reference_path,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
    )
    docs_iter = get_progress_bar(
        reference_iter,
        total=max_docs,
        desc=f"{progress_label} scan",
        unit="docs",
        mininterval=1.0,
        disable=not stderr.isatty(),
    )
    for num_docs, tokens in enumerate(docs_iter, start=1):
        present_doc_words = sorted(set(tokens) & target_words)
        for word in present_doc_words:
            doc_word_counts[word] += 1
        for word_i, word_j in combinations(present_doc_words, 2):
            doc_pair_counts[(word_i, word_j)] += 1

        for window_size, counts in counts_by_window_size.items():
            for window_words in _iter_boolean_sliding_window_sets(
                tokens,
                window_size=window_size,
            ):
                counts.num_windows += 1
                present_words = sorted(window_words & target_words)
                for word in present_words:
                    counts.word_window_counts[word] += 1
                for word_i, word_j in combinations(present_words, 2):
                    counts.pair_window_counts[(word_i, word_j)] += 1
        if num_docs % REFERENCE_PROGRESS_DOC_INTERVAL == 0:
            total_windows = sum(
                counts.num_windows for counts in counts_by_window_size.values()
            )
            logger.info(
                "%s progress docs=%s/%s windows=%s sec=%.1f",
                progress_label,
                num_docs,
                "unknown" if max_docs is None else max_docs,
                total_windows,
                perf_counter() - started,
            )
    return counts_by_window_size, doc_word_counts, doc_pair_counts, num_docs


def _count_tokens_chunk_python(
    tokens_by_doc: list[list[str]],
    target_words: set[str],
    window_sizes: set[int],
) -> tuple[dict[int, SlidingWindowCounts], Counter[str], Counter[tuple[str, str]], int]:
    counts_by_window_size = {
        int(window_size): SlidingWindowCounts(Counter(), Counter(), 0)
        for window_size in window_sizes
    }
    doc_word_counts: Counter[str] = Counter()
    doc_pair_counts: Counter[tuple[str, str]] = Counter()
    for tokens in tokens_by_doc:
        present_doc_words = sorted(set(tokens) & target_words)
        for word in present_doc_words:
            doc_word_counts[word] += 1
        for word_i, word_j in combinations(present_doc_words, 2):
            doc_pair_counts[(word_i, word_j)] += 1

        for window_size, counts in counts_by_window_size.items():
            for window_words in _iter_boolean_sliding_window_sets(
                tokens,
                window_size=window_size,
            ):
                counts.num_windows += 1
                present_words = sorted(window_words & target_words)
                for word in present_words:
                    counts.word_window_counts[word] += 1
                for word_i, word_j in combinations(present_words, 2):
                    counts.pair_window_counts[(word_i, word_j)] += 1
    return counts_by_window_size, doc_word_counts, doc_pair_counts, len(tokens_by_doc)


def _encode_reference_docs(
    *,
    reference_path: Path,
    target_word_to_id: dict[str, int],
    max_docs: int | None,
    min_doc_tokens: int,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    encoded_docs: list[np.ndarray] = []
    num_docs = 0
    started = perf_counter()
    reference_iter = iter_tokenized_reference_corpus(
        path=reference_path,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
    )
    docs_iter = get_progress_bar(
        reference_iter,
        total=max_docs,
        desc=f"{progress_label} encode",
        unit="docs",
        mininterval=1.0,
        disable=not stderr.isatty(),
    )
    for num_docs, tokens in enumerate(docs_iter, start=1):
        encoded_docs.append(
            np.array(
                [target_word_to_id.get(token, -1) for token in tokens],
                dtype=np.int64,
            )
        )
        if num_docs % REFERENCE_PROGRESS_DOC_INTERVAL == 0:
            logger.info(
                "%s encode progress docs=%s/%s sec=%.1f",
                progress_label,
                num_docs,
                "unknown" if max_docs is None else max_docs,
                perf_counter() - started,
            )
    offsets = np.zeros(len(encoded_docs) + 1, dtype=np.int64)
    if encoded_docs:
        lengths = np.array([len(doc) for doc in encoded_docs], dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        tokens_flat = np.concatenate(encoded_docs).astype(np.int64, copy=False)
    else:
        tokens_flat = np.array([], dtype=np.int64)
    return tokens_flat, offsets, num_docs


def _encode_tokens_by_doc(
    *,
    tokens_by_doc: list[list[str]],
    target_word_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    encoded_docs = [
        np.array(
            [target_word_to_id.get(token, -1) for token in tokens],
            dtype=np.int64,
        )
        for tokens in tokens_by_doc
    ]
    offsets = np.zeros(len(encoded_docs) + 1, dtype=np.int64)
    if encoded_docs:
        lengths = np.array([len(doc) for doc in encoded_docs], dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        tokens_flat = np.concatenate(encoded_docs).astype(np.int64, copy=False)
    else:
        tokens_flat = np.array([], dtype=np.int64)
    return tokens_flat, offsets


@lru_cache(maxsize=1)
def _get_numba_reference_count_functions():
    try:
        from numba import njit
        from numba.typed import Dict
        from numba.types import int64
    except ImportError as exc:
        raise RuntimeError(
            "numba backend requested but numba is not installed"
        ) from exc

    @njit
    def _increment(counter, key):
        counter[key] = counter.get(key, 0) + 1

    @njit
    def _unique_sorted_target_ids(tokens, start, end, num_targets):
        present = np.zeros(num_targets, np.uint8)
        count = 0
        for pos in range(start, end):
            token_id = tokens[pos]
            if token_id >= 0 and present[token_id] == 0:
                present[token_id] = 1
                count += 1
        result = np.empty(count, np.int64)
        out_idx = 0
        for token_id in range(num_targets):
            if present[token_id] == 1:
                result[out_idx] = token_id
                out_idx += 1
        return result

    @njit
    def _count_doc_pairs(tokens, offsets, num_targets):
        word_counts = Dict.empty(key_type=int64, value_type=int64)
        pair_counts = Dict.empty(key_type=int64, value_type=int64)
        for doc_idx in range(len(offsets) - 1):
            doc_start = offsets[doc_idx]
            doc_end = offsets[doc_idx + 1]
            present = _unique_sorted_target_ids(
                tokens,
                doc_start,
                doc_end,
                num_targets,
            )
            for i in range(len(present)):
                word_id = present[i]
                _increment(word_counts, word_id)
                for j in range(i + 1, len(present)):
                    pair_key = word_id * num_targets + present[j]
                    _increment(pair_counts, pair_key)
        return word_counts, pair_counts

    @njit
    def _count_windows(tokens, offsets, window_size, num_targets):
        word_counts = Dict.empty(key_type=int64, value_type=int64)
        pair_counts = Dict.empty(key_type=int64, value_type=int64)
        num_windows = 0
        for doc_idx in range(len(offsets) - 1):
            doc_start = offsets[doc_idx]
            doc_end = offsets[doc_idx + 1]
            doc_len = doc_end - doc_start
            if doc_len <= 0:
                continue
            if doc_len <= window_size:
                last_start = doc_start
                stop = doc_start + 1
            else:
                last_start = doc_end - window_size
                stop = last_start + 1
            for window_start in range(doc_start, stop):
                if doc_len <= window_size:
                    window_end = doc_end
                else:
                    window_end = window_start + window_size
                present = _unique_sorted_target_ids(
                    tokens,
                    window_start,
                    window_end,
                    num_targets,
                )
                num_windows += 1
                for i in range(len(present)):
                    word_id = present[i]
                    _increment(word_counts, word_id)
                    for j in range(i + 1, len(present)):
                        pair_key = word_id * num_targets + present[j]
                        _increment(pair_counts, pair_key)
        return word_counts, pair_counts, num_windows

    return _count_doc_pairs, _count_windows


def _counts_from_encoded_docs_numba(
    *,
    tokens_flat: np.ndarray,
    offsets: np.ndarray,
    target_word_list: list[str],
    window_sizes: set[int],
) -> tuple[dict[int, SlidingWindowCounts], Counter[str], Counter[tuple[str, str]], int]:
    _count_doc_pairs, _count_windows = _get_numba_reference_count_functions()
    id_to_word = {idx: word for idx, word in enumerate(target_word_list)}
    num_targets = len(target_word_list)
    doc_word_counts_raw, doc_pair_counts_raw = _count_doc_pairs(
        tokens_flat,
        offsets,
        num_targets,
    )
    doc_word_counts: Counter[str] = Counter()
    for word_id, count in doc_word_counts_raw.items():
        doc_word_counts[id_to_word[int(word_id)]] = int(count)
    doc_pair_counts: Counter[tuple[str, str]] = Counter()
    for pair_key, count in doc_pair_counts_raw.items():
        word_i = id_to_word[int(pair_key) // num_targets]
        word_j = id_to_word[int(pair_key) % num_targets]
        doc_pair_counts[tuple(sorted((word_i, word_j)))] = int(count)

    counts_by_window_size: dict[int, SlidingWindowCounts] = {}
    for window_size in sorted(window_sizes):
        word_counts_raw, pair_counts_raw, num_windows = _count_windows(
            tokens_flat,
            offsets,
            int(window_size),
            num_targets,
        )
        word_counts: Counter[str] = Counter()
        for word_id, count in word_counts_raw.items():
            word_counts[id_to_word[int(word_id)]] = int(count)
        pair_counts: Counter[tuple[str, str]] = Counter()
        for pair_key, count in pair_counts_raw.items():
            word_i = id_to_word[int(pair_key) // num_targets]
            word_j = id_to_word[int(pair_key) % num_targets]
            pair_counts[tuple(sorted((word_i, word_j)))] = int(count)
        counts_by_window_size[int(window_size)] = SlidingWindowCounts(
            word_window_counts=word_counts,
            pair_window_counts=pair_counts,
            num_windows=int(num_windows),
        )
    return (
        counts_by_window_size,
        doc_word_counts,
        doc_pair_counts,
        max(0, len(offsets) - 1),
    )


def _count_tokens_chunk_numba(
    tokens_by_doc: list[list[str]],
    target_word_list: list[str],
    window_sizes: set[int],
) -> tuple[dict[int, SlidingWindowCounts], Counter[str], Counter[tuple[str, str]], int]:
    target_word_to_id = {word: idx for idx, word in enumerate(target_word_list)}
    tokens_flat, offsets = _encode_tokens_by_doc(
        tokens_by_doc=tokens_by_doc,
        target_word_to_id=target_word_to_id,
    )
    return _counts_from_encoded_docs_numba(
        tokens_flat=tokens_flat,
        offsets=offsets,
        target_word_list=target_word_list,
        window_sizes=window_sizes,
    )


def _build_numba_reference_counts(
    *,
    reference_path: Path,
    target_words: set[str],
    window_sizes: set[int],
    max_docs: int | None,
    min_doc_tokens: int,
    progress_label: str,
) -> tuple[dict[int, SlidingWindowCounts], Counter[str], Counter[tuple[str, str]], int]:
    target_word_list = sorted(target_words)
    target_word_to_id = {word: idx for idx, word in enumerate(target_word_list)}
    tokens_flat, offsets, num_docs = _encode_reference_docs(
        reference_path=reference_path,
        target_word_to_id=target_word_to_id,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
        progress_label=progress_label,
    )
    logger.info(
        "%s numba count start docs=%s targets=%s window_sizes=%s",
        progress_label,
        num_docs,
        len(target_words),
        sorted(window_sizes),
    )
    counts_by_window_size, doc_word_counts, doc_pair_counts, _num_docs = (
        _counts_from_encoded_docs_numba(
            tokens_flat=tokens_flat,
            offsets=offsets,
            target_word_list=target_word_list,
            window_sizes=window_sizes,
        )
    )
    return counts_by_window_size, doc_word_counts, doc_pair_counts, num_docs


def _merge_reference_count_result(
    *,
    target: tuple[
        dict[int, SlidingWindowCounts],
        Counter[str],
        Counter[tuple[str, str]],
    ],
    result: tuple[
        dict[int, SlidingWindowCounts],
        Counter[str],
        Counter[tuple[str, str]],
        int,
    ],
) -> int:
    target_counts_by_window_size, target_doc_word_counts, target_doc_pair_counts = (
        target
    )
    counts_by_window_size, doc_word_counts, doc_pair_counts, num_docs = result
    target_doc_word_counts.update(doc_word_counts)
    target_doc_pair_counts.update(doc_pair_counts)
    for window_size, counts in counts_by_window_size.items():
        target_counts = target_counts_by_window_size[window_size]
        target_counts.word_window_counts.update(counts.word_window_counts)
        target_counts.pair_window_counts.update(counts.pair_window_counts)
        target_counts.num_windows += counts.num_windows
    return int(num_docs)


def _count_reference_chunk(
    *,
    tokens_by_doc: list[list[str]],
    target_words: set[str],
    target_word_list: list[str],
    window_sizes: set[int],
    backend: ReferenceCountBackend,
) -> tuple[dict[int, SlidingWindowCounts], Counter[str], Counter[tuple[str, str]], int]:
    if backend == "python":
        return _count_tokens_chunk_python(
            tokens_by_doc=tokens_by_doc,
            target_words=target_words,
            window_sizes=window_sizes,
        )
    if backend == "numba":
        return _count_tokens_chunk_numba(
            tokens_by_doc=tokens_by_doc,
            target_word_list=target_word_list,
            window_sizes=window_sizes,
        )
    raise ValueError(f"Unsupported reference count backend: {backend!r}")


def _submit_reference_chunk(
    *,
    executor: ProcessPoolExecutor,
    tokens_by_doc: list[list[str]],
    target_words: set[str],
    target_word_list: list[str],
    window_sizes: set[int],
    backend: ReferenceCountBackend,
) -> Future[
    tuple[dict[int, SlidingWindowCounts], Counter[str], Counter[tuple[str, str]], int]
]:
    return executor.submit(
        _count_reference_chunk,
        tokens_by_doc=tokens_by_doc,
        target_words=target_words,
        target_word_list=target_word_list,
        window_sizes=window_sizes,
        backend=backend,
    )


def _build_parallel_reference_counts(
    *,
    reference_path: Path,
    target_words: set[str],
    window_sizes: set[int],
    max_docs: int | None,
    min_doc_tokens: int,
    backend: ReferenceCountBackend,
    workers: int,
    chunk_size: int,
    progress_label: str,
) -> tuple[dict[int, SlidingWindowCounts], Counter[str], Counter[tuple[str, str]], int]:
    counts_by_window_size = {
        int(window_size): SlidingWindowCounts(Counter(), Counter(), 0)
        for window_size in window_sizes
    }
    doc_word_counts: Counter[str] = Counter()
    doc_pair_counts: Counter[tuple[str, str]] = Counter()
    target: tuple[
        dict[int, SlidingWindowCounts],
        Counter[str],
        Counter[tuple[str, str]],
    ] = (counts_by_window_size, doc_word_counts, doc_pair_counts)
    submitted_docs = 0
    completed_docs = 0
    next_progress_docs = REFERENCE_PROGRESS_DOC_INTERVAL
    started = perf_counter()
    pending: set[
        Future[
            tuple[
                dict[int, SlidingWindowCounts],
                Counter[str],
                Counter[tuple[str, str]],
                int,
            ]
        ]
    ] = set()
    target_word_list = sorted(target_words)
    reference_iter = iter_tokenized_reference_corpus(
        path=reference_path,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
    )
    docs_iter = get_progress_bar(
        reference_iter,
        total=max_docs,
        desc=f"{progress_label} submit",
        unit="docs",
        mininterval=1.0,
        disable=not stderr.isatty(),
    )
    max_pending = max(1, workers * 2)
    chunk: list[list[str]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for tokens in docs_iter:
            chunk.append(tokens)
            if len(chunk) < chunk_size:
                continue
            pending.add(
                _submit_reference_chunk(
                    executor=executor,
                    tokens_by_doc=chunk,
                    target_words=target_words,
                    target_word_list=target_word_list,
                    window_sizes=window_sizes,
                    backend=backend,
                )
            )
            submitted_docs += len(chunk)
            chunk = []
            while len(pending) >= max_pending:
                done = next(as_completed(pending))
                pending.remove(done)
                completed_docs += _merge_reference_count_result(
                    target=target,
                    result=done.result(),
                )
                if completed_docs >= next_progress_docs:
                    logger.info(
                        "%s parallel progress docs=%s/%s submitted=%s sec=%.1f",
                        progress_label,
                        completed_docs,
                        "unknown" if max_docs is None else max_docs,
                        submitted_docs,
                        perf_counter() - started,
                    )
                    next_progress_docs += REFERENCE_PROGRESS_DOC_INTERVAL
        if chunk:
            pending.add(
                _submit_reference_chunk(
                    executor=executor,
                    tokens_by_doc=chunk,
                    target_words=target_words,
                    target_word_list=target_word_list,
                    window_sizes=window_sizes,
                    backend=backend,
                )
            )
            submitted_docs += len(chunk)
        for done in as_completed(pending):
            completed_docs += _merge_reference_count_result(
                target=target,
                result=done.result(),
            )
    return counts_by_window_size, doc_word_counts, doc_pair_counts, completed_docs


def build_shared_reference_counts(
    *,
    reference_path: Path,
    target_words: set[str],
    window_sizes: set[int],
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
    backend: ReferenceCountBackend = "numba",
    workers: int = DEFAULT_REFERENCE_COUNT_WORKERS,
    chunk_size: int = DEFAULT_REFERENCE_COUNT_CHUNK_SIZE,
    progress_label: str = "reference_counts",
) -> SharedReferenceCounts:
    target_words = set(target_words)
    window_sizes = {int(window_size) for window_size in window_sizes}
    workers = int(workers)
    chunk_size = int(chunk_size)
    if not target_words:
        raise ValueError("target_words must not be empty.")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    key = ReferenceCountKey(
        reference_path=Path(reference_path),
        max_docs=max_docs,
        min_doc_tokens=int(min_doc_tokens),
        window_sizes=tuple(sorted(window_sizes)),
        target_words_fingerprint=fingerprint_target_words(target_words),
        backend=backend,
        workers=workers,
        chunk_size=chunk_size,
    )
    logger.info(
        "%s scan start backend=%s workers=%s chunk_size=%s path=%s target_words=%s "
        "window_sizes=%s",
        progress_label,
        backend,
        workers,
        chunk_size,
        reference_path,
        len(target_words),
        sorted(window_sizes),
    )
    started = perf_counter()
    if workers > 1:
        counts_by_window_size, doc_word_counts, doc_pair_counts, num_docs = (
            _build_parallel_reference_counts(
                reference_path=Path(reference_path),
                target_words=target_words,
                window_sizes=window_sizes,
                max_docs=max_docs,
                min_doc_tokens=int(min_doc_tokens),
                backend=backend,
                workers=workers,
                chunk_size=chunk_size,
                progress_label=progress_label,
            )
        )
    elif backend == "python":
        counts_by_window_size, doc_word_counts, doc_pair_counts, num_docs = (
            _build_python_reference_counts(
                reference_path=Path(reference_path),
                target_words=target_words,
                window_sizes=window_sizes,
                max_docs=max_docs,
                min_doc_tokens=int(min_doc_tokens),
                progress_label=progress_label,
            )
        )
    elif backend == "numba":
        counts_by_window_size, doc_word_counts, doc_pair_counts, num_docs = (
            _build_numba_reference_counts(
                reference_path=Path(reference_path),
                target_words=target_words,
                window_sizes=window_sizes,
                max_docs=max_docs,
                min_doc_tokens=int(min_doc_tokens),
                progress_label=progress_label,
            )
        )
    else:
        raise ValueError(f"Unsupported reference count backend: {backend!r}")
    logger.info(
        "%s scan done backend=%s workers=%s docs=%s windows=%s sec=%.1f",
        progress_label,
        backend,
        workers,
        num_docs,
        {
            window_size: counts.num_windows
            for window_size, counts in counts_by_window_size.items()
        },
        perf_counter() - started,
    )
    return SharedReferenceCounts(
        key=key,
        counts_by_window_size=counts_by_window_size,
        doc_word_counts=doc_word_counts,
        doc_pair_counts=doc_pair_counts,
        num_docs=num_docs,
        target_words=target_words,
    )


def _compute_doc_npmi_score_from_counts(
    *,
    word_lists: list[list[str]],
    counts: SharedReferenceCounts,
) -> float:
    if counts.num_docs == 0:
        return float("nan")
    topic_scores: list[float] = []
    for words in word_lists:
        pair_scores: list[float] = []
        for word_i, word_j in combinations(words, 2):
            pair_key = tuple(sorted((word_i, word_j)))
            co_doc_count = counts.doc_pair_counts.get(pair_key, 0)
            if co_doc_count == 0:
                pair_scores.append(-1.0)
                continue
            p_i = counts.doc_word_counts[word_i] / counts.num_docs
            p_j = counts.doc_word_counts[word_j] / counts.num_docs
            p_ij = co_doc_count / counts.num_docs
            if p_ij >= 1.0:
                pair_scores.append(1.0)
                continue
            pair_scores.append(log(p_ij / (p_i * p_j)) / (-log(p_ij)))
        if not pair_scores:
            return float("nan")
        topic_scores.append(float(np.mean(pair_scores)))
    return float(np.mean(topic_scores))


def compute_shared_reference_coherence_scores(
    *,
    topic_words: TopicWords,
    metric_names: list[str] | tuple[str, ...],
    coherences: list[str] | tuple[str, ...],
    counts: SharedReferenceCounts,
    coherence_topn: int | None = None,
    diversity_topn: int | None = None,
    window_size: int | None = None,
    min_window_count: int | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    requested = normalize_coherences(coherences)
    multiple_coherences = len(requested) > 1
    coherence_metric_names = {
        coherence_metric_key(coherence_name, multiple=multiple_coherences)
        for coherence_name in requested
    }
    coherence_evaluated = False
    for metric_name in metric_names:
        if metric_name == "coherence" or metric_name in coherence_metric_names:
            if coherence_evaluated:
                continue
            coherence_topic_words = (
                truncate_topic_words(topic_words, coherence_topn)
                if coherence_topn is not None
                else topic_words
            )
            if not coherence_topic_words or any(
                len(topic) == 0 for topic in coherence_topic_words
            ):
                for coherence_name in requested:
                    metrics[
                        coherence_metric_key(
                            coherence_name,
                            multiple=multiple_coherences,
                        )
                    ] = float("nan")
                coherence_evaluated = True
                continue
            word_lists = topic_words_to_word_lists(coherence_topic_words)
            for coherence_name in requested:
                score_started = perf_counter()
                logger.info("shared reference score start metric=%s", coherence_name)
                if coherence_name == "c_v":
                    resolved_window_size = _effective_sliding_window_size(
                        coherence_name,
                        window_size,
                    )
                    resolved_min_frequency = (
                        DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT
                        if min_window_count is None
                        else int(min_window_count)
                    )
                    score = _compute_palmetto_cv_score_from_counts(
                        word_lists=word_lists,
                        counts=counts.counts_by_window_size[resolved_window_size],
                        min_frequency=resolved_min_frequency,
                        epsilon=PMI_SMOOTHING_EPSILON,
                    )
                elif coherence_name in EPSILON_SMOOTHED_COHERENCES:
                    resolved_window_size = _effective_sliding_window_size(
                        coherence_name,
                        window_size,
                    )
                    score = _compute_epsilon_smoothed_pmi_score_from_counts(
                        word_lists=word_lists,
                        counts=counts.counts_by_window_size[resolved_window_size],
                        coherence=coherence_name,
                        epsilon=PMI_SMOOTHING_EPSILON,
                    )
                elif coherence_name == "doc_npmi":
                    score = _compute_doc_npmi_score_from_counts(
                        word_lists=word_lists,
                        counts=counts,
                    )
                else:
                    raise ValueError(
                        f"Unsupported shared reference coherence: {coherence_name!r}"
                    )
                metrics[
                    coherence_metric_key(
                        coherence_name,
                        multiple=multiple_coherences,
                    )
                ] = score
                logger.info(
                    "shared reference score done metric=%s value=%s sec=%.1f",
                    coherence_name,
                    score,
                    perf_counter() - score_started,
                )
            coherence_evaluated = True
        elif metric_name == "diversity":
            diversity_topic_words = (
                truncate_topic_words(topic_words, diversity_topn)
                if diversity_topn is not None
                else topic_words
            )
            metrics["diversity"] = compute_topic_diversity(diversity_topic_words)
        else:
            raise ValueError(f"Unsupported topic-word metric '{metric_name}'.")
    return metrics

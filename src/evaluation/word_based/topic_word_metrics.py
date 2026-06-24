from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import log
from pathlib import Path
from sys import stderr
from time import perf_counter

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from src.utils.logging import get_logger, get_progress_bar

from .corpus_bundle import CountingTokenizedReferenceCorpus
from .topic_words import TopicWords

MetricName = str
DEFAULT_METRIC_NAMES: tuple[MetricName, ...] = ("coherence", "diversity")
STREAMING_REFERENCE_COHERENCES = {"c_v", "c_npmi", "c_uci", "doc_npmi"}
MULTI_COHERENCE_CHOICES = ("c_v", "c_npmi", "c_uci")
EPSILON_SMOOTHED_COHERENCES = {"c_npmi", "c_uci"}
PMI_SMOOTHING_EPSILON = 1e-12
PALMETTO_CV_IMPLEMENTATION = "palmetto_compatible"
DEFAULT_PALMETTO_CV_WINDOW_SIZE = 110
DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT = 1
COHERENCE_PROGRESS_DOC_INTERVAL = 100_000

logger = get_logger(__name__)


@dataclass(frozen=True)
class StreamingReferenceCoherenceResult:
    score: float
    num_docs: int
    vocab_size: int


@dataclass(frozen=True)
class StreamingReferenceCoherenceScoresResult:
    scores: dict[str, float]
    num_docs: int
    vocab_size: int


@dataclass
class SlidingWindowCounts:
    word_window_counts: Counter[str]
    pair_window_counts: Counter[tuple[str, str]]
    num_windows: int = 0


def describe_coherence_metric(coherence: str) -> dict[str, str]:
    if coherence == "c_v":
        return {
            "definition": (
                "Palmetto-compatible C_V using boolean sliding windows, NPMI "
                "topic-word vectors, one-set segmentation, cosine similarity, "
                "and arithmetic mean aggregation."
            ),
            "cooccurrence_unit": "sliding_window",
            "zero_cooccurrence_policy": "undefined_npmi_and_zero_vector_as_zero",
            "pmi_smoothing_epsilon": PMI_SMOOTHING_EPSILON,
        }
    if coherence == "doc_npmi":
        return {
            "definition": (
                "Average pairwise NPMI over top-N topic words using document-level "
                "boolean co-occurrence."
            ),
            "cooccurrence_unit": "document",
            "zero_cooccurrence_policy": "minus_one",
        }
    if coherence in EPSILON_SMOOTHED_COHERENCES:
        return {
            "definition": "epsilon-smoothed direct PMI/NPMI over sliding windows",
            "cooccurrence_unit": "sliding_window",
            "zero_cooccurrence_policy": "epsilon_smoothing",
            "pmi_smoothing_epsilon": PMI_SMOOTHING_EPSILON,
        }
    return {
        "definition": "gensim CoherenceModel",
        "cooccurrence_unit": "",
        "zero_cooccurrence_policy": "",
        "pmi_smoothing_epsilon": None,
    }


def topic_words_to_word_lists(topic_words: TopicWords) -> list[list[str]]:
    return [[word for word, _score in topic] for topic in topic_words]


def truncate_topic_words(topic_words: TopicWords, topn: int) -> TopicWords:
    return [topic[:topn] for topic in topic_words]


def normalize_coherences(coherence: str | list[str] | tuple[str, ...]) -> list[str]:
    values = [coherence] if isinstance(coherence, str) else list(coherence)
    normalized: list[str] = []
    for value in values:
        metric = str(value).strip()
        if metric and metric not in normalized:
            normalized.append(metric)
    if not normalized:
        raise ValueError("At least one coherence metric is required.")
    return normalized


def coherence_metric_key(coherence: str, *, multiple: bool) -> str:
    return f"coherence_{coherence}" if multiple else "coherence"


def compute_doc_npmi_score(
    topic_words: TopicWords,
    texts: list[list[str]],
) -> float:
    if not topic_words or not texts:
        return float("nan")
    word_lists = topic_words_to_word_lists(topic_words)
    if any(len(words) == 0 for words in word_lists):
        return float("nan")

    doc_sets = [set(doc) for doc in texts]
    num_docs = len(doc_sets)
    if num_docs == 0:
        return float("nan")

    target_words = {word for words in word_lists for word in words}
    word_doc_counts: Counter[str] = Counter()
    pair_doc_counts: Counter[tuple[str, str]] = Counter()
    for doc_words in doc_sets:
        present_words = sorted(doc_words & target_words)
        for word in present_words:
            word_doc_counts[word] += 1
        for word_i, word_j in combinations(present_words, 2):
            pair_doc_counts[(word_i, word_j)] += 1

    topic_scores: list[float] = []
    for words in word_lists:
        pair_scores: list[float] = []
        for word_i, word_j in combinations(words, 2):
            pair_key = tuple(sorted((word_i, word_j)))
            co_doc_count = pair_doc_counts.get(pair_key, 0)
            if co_doc_count == 0:
                pair_scores.append(-1.0)
                continue
            p_i = word_doc_counts[word_i] / num_docs
            p_j = word_doc_counts[word_j] / num_docs
            p_ij = co_doc_count / num_docs
            if p_ij >= 1.0:
                pair_scores.append(1.0)
                continue
            pair_scores.append(log(p_ij / (p_i * p_j)) / (-log(p_ij)))
        if not pair_scores:
            return float("nan")
        topic_scores.append(float(np.mean(pair_scores)))
    return float(np.mean(topic_scores))


def compute_streaming_doc_npmi_score(
    topic_words: TopicWords,
    texts,
) -> tuple[float, int]:
    if not topic_words:
        return float("nan"), 0
    word_lists = topic_words_to_word_lists(topic_words)
    if any(len(words) == 0 for words in word_lists):
        return float("nan"), 0

    num_docs = 0
    target_words = {word for words in word_lists for word in words}
    word_doc_counts: Counter[str] = Counter()
    pair_doc_counts: Counter[tuple[str, str]] = Counter()
    for doc in texts:
        num_docs += 1
        present_words = sorted(set(doc) & target_words)
        for word in present_words:
            word_doc_counts[word] += 1
        for word_i, word_j in combinations(present_words, 2):
            pair_doc_counts[(word_i, word_j)] += 1

    if num_docs == 0:
        return float("nan"), 0

    topic_scores: list[float] = []
    for words in word_lists:
        pair_scores: list[float] = []
        for word_i, word_j in combinations(words, 2):
            pair_key = tuple(sorted((word_i, word_j)))
            co_doc_count = pair_doc_counts.get(pair_key, 0)
            if co_doc_count == 0:
                pair_scores.append(-1.0)
                continue
            p_i = word_doc_counts[word_i] / num_docs
            p_j = word_doc_counts[word_j] / num_docs
            p_ij = co_doc_count / num_docs
            if p_ij >= 1.0:
                pair_scores.append(1.0)
                continue
            pair_scores.append(log(p_ij / (p_i * p_j)) / (-log(p_ij)))
        if not pair_scores:
            return float("nan"), num_docs
        topic_scores.append(float(np.mean(pair_scores)))
    return float(np.mean(topic_scores)), num_docs


def _effective_sliding_window_size(coherence: str, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    defaults = {
        "c_v": DEFAULT_PALMETTO_CV_WINDOW_SIZE,
        "c_uci": 10,
        "c_npmi": 10,
    }
    return defaults.get(coherence, DEFAULT_PALMETTO_CV_WINDOW_SIZE)


def _iter_boolean_sliding_windows(
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


def _build_sliding_window_counts(
    *,
    word_lists: list[list[str]],
    texts,
    window_sizes: set[int],
    progress_label: str = "coherence",
) -> dict[int, SlidingWindowCounts]:
    if not window_sizes:
        return {}
    target_words = {word for words in word_lists for word in words}
    counts_by_window_size = {
        int(window_size): SlidingWindowCounts(Counter(), Counter(), 0)
        for window_size in window_sizes
    }
    total_docs = len(texts) if hasattr(texts, "__len__") else None
    started = perf_counter()
    logger.info(
        "%s window_counts start docs=%s window_sizes=%s target_words=%s",
        progress_label,
        "unknown" if total_docs is None else total_docs,
        sorted(counts_by_window_size),
        len(target_words),
    )
    docs_seen = 0
    docs_iter = get_progress_bar(
        texts,
        total=total_docs,
        desc=f"{progress_label} windows",
        unit="docs",
        mininterval=1.0,
        disable=not stderr.isatty(),
    )
    for docs_seen, doc in enumerate(docs_iter, start=1):
        doc_tokens = list(doc)
        for window_size, counts in counts_by_window_size.items():
            for window_words in _iter_boolean_sliding_windows(
                doc_tokens,
                window_size=window_size,
            ):
                counts.num_windows += 1
                present_words = sorted(window_words & target_words)
                for word in present_words:
                    counts.word_window_counts[word] += 1
                for word_i, word_j in combinations(present_words, 2):
                    counts.pair_window_counts[(word_i, word_j)] += 1
        if docs_seen % COHERENCE_PROGRESS_DOC_INTERVAL == 0:
            total_windows = sum(
                counts.num_windows for counts in counts_by_window_size.values()
            )
            logger.info(
                "%s window_counts progress docs=%s/%s windows=%s sec=%.1f",
                progress_label,
                docs_seen,
                "unknown" if total_docs is None else total_docs,
                total_windows,
                perf_counter() - started,
            )
    total_windows = sum(counts.num_windows for counts in counts_by_window_size.values())
    logger.info(
        "%s window_counts done docs=%s windows=%s sec=%.1f",
        progress_label,
        docs_seen,
        total_windows,
        perf_counter() - started,
    )
    return counts_by_window_size


def _compute_epsilon_smoothed_pmi_score_from_counts(
    *,
    word_lists: list[list[str]],
    counts: SlidingWindowCounts,
    coherence: str,
    epsilon: float,
) -> float:
    if counts.num_windows == 0:
        return float("nan")

    topic_scores: list[float] = []
    for words in word_lists:
        pair_scores: list[float] = []
        for word_i, word_j in combinations(words, 2):
            pair_key = tuple(sorted((word_i, word_j)))
            p_i = counts.word_window_counts[word_i] / counts.num_windows
            p_j = counts.word_window_counts[word_j] / counts.num_windows
            if p_i == 0.0 or p_j == 0.0:
                pair_scores.append(-1.0)
                continue
            p_ij = counts.pair_window_counts.get(pair_key, 0) / counts.num_windows
            product = p_i * p_j
            pmi = log((p_ij + epsilon) / (product + epsilon))
            if coherence == "c_uci":
                pair_scores.append(pmi)
                continue
            if p_ij >= 1.0:
                pair_scores.append(1.0)
                continue
            pair_scores.append(pmi / (-log(p_ij + epsilon)))
        if not pair_scores:
            return float("nan")
        topic_scores.append(float(np.mean(pair_scores)))
    return float(np.mean(topic_scores))


def compute_epsilon_smoothed_pmi_score(
    *,
    topic_words: TopicWords,
    texts,
    coherence: str,
    window_size: int | None = None,
    epsilon: float = PMI_SMOOTHING_EPSILON,
    progress_label: str = "coherence",
) -> float:
    if coherence not in EPSILON_SMOOTHED_COHERENCES:
        raise ValueError(
            "epsilon-smoothed PMI coherence supports only "
            f"{sorted(EPSILON_SMOOTHED_COHERENCES)}, got {coherence!r}."
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if not topic_words:
        return float("nan")
    word_lists = topic_words_to_word_lists(topic_words)
    if any(len(words) < 2 for words in word_lists):
        return float("nan")

    resolved_window_size = _effective_sliding_window_size(coherence, window_size)
    if resolved_window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {resolved_window_size}")

    counts = _build_sliding_window_counts(
        word_lists=word_lists,
        texts=texts,
        window_sizes={resolved_window_size},
        progress_label=progress_label,
    )[resolved_window_size]
    return _compute_epsilon_smoothed_pmi_score_from_counts(
        word_lists=word_lists,
        counts=counts,
        coherence=coherence,
        epsilon=epsilon,
    )


def _safe_cosine_similarity(vector: np.ndarray, reference: np.ndarray) -> float:
    vector_norm = float(np.linalg.norm(vector))
    reference_norm = float(np.linalg.norm(reference))
    if vector_norm == 0.0 or reference_norm == 0.0:
        return 0.0
    return float(np.dot(vector, reference) / (vector_norm * reference_norm))


def _npmi_from_window_counts(
    *,
    word_i: str,
    word_j: str,
    word_window_counts: Counter[str],
    pair_window_counts: Counter[tuple[str, str]],
    num_windows: int,
    min_frequency: int,
    epsilon: float,
) -> float:
    count_i = word_window_counts[word_i]
    count_j = word_window_counts[word_j]
    if count_i < min_frequency or count_j < min_frequency:
        return 0.0

    if word_i == word_j:
        co_count = count_i
    else:
        co_count = pair_window_counts.get(tuple(sorted((word_i, word_j))), 0)
    if co_count < min_frequency:
        co_count = 0

    p_i = count_i / num_windows
    p_j = count_j / num_windows
    if p_i == 0.0 or p_j == 0.0:
        return 0.0

    p_ij = co_count / num_windows
    if p_ij >= 1.0:
        return 1.0
    denominator = -log(p_ij + epsilon)
    if denominator == 0.0:
        return 0.0
    return float(log((p_ij + epsilon) / (p_i * p_j)) / denominator)


def _compute_palmetto_cv_score_from_counts(
    *,
    word_lists: list[list[str]],
    counts: SlidingWindowCounts,
    min_frequency: int,
    epsilon: float,
) -> float:
    if counts.num_windows == 0:
        return float("nan")

    topic_scores: list[float] = []
    for words in word_lists:
        word_vectors = np.array(
            [
                [
                    _npmi_from_window_counts(
                        word_i=word_i,
                        word_j=word_j,
                        word_window_counts=counts.word_window_counts,
                        pair_window_counts=counts.pair_window_counts,
                        num_windows=counts.num_windows,
                        min_frequency=min_frequency,
                        epsilon=epsilon,
                    )
                    for word_j in words
                ]
                for word_i in words
            ],
            dtype=float,
        )
        topic_vector = word_vectors.sum(axis=0)
        topic_scores.append(
            float(
                np.mean(
                    [
                        _safe_cosine_similarity(word_vector, topic_vector)
                        for word_vector in word_vectors
                    ]
                )
            )
        )
    return float(np.mean(topic_scores))


def compute_palmetto_cv_score(
    *,
    topic_words: TopicWords,
    texts,
    window_size: int | None = None,
    epsilon: float = PMI_SMOOTHING_EPSILON,
    min_frequency: int | None = None,
    progress_label: str = "coherence",
) -> float:
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if not topic_words:
        return float("nan")
    word_lists = topic_words_to_word_lists(topic_words)
    if any(len(words) == 0 for words in word_lists):
        return float("nan")

    resolved_window_size = _effective_sliding_window_size("c_v", window_size)
    if resolved_window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {resolved_window_size}")
    resolved_min_frequency = (
        DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT
        if min_frequency is None
        else int(min_frequency)
    )
    if resolved_min_frequency < 1:
        raise ValueError(f"min_frequency must be >= 1, got {resolved_min_frequency}")

    counts = _build_sliding_window_counts(
        word_lists=word_lists,
        texts=texts,
        window_sizes={resolved_window_size},
        progress_label=progress_label,
    )[resolved_window_size]
    return _compute_palmetto_cv_score_from_counts(
        word_lists=word_lists,
        counts=counts,
        min_frequency=resolved_min_frequency,
        epsilon=epsilon,
    )


def compute_coherence_scores(
    *,
    topic_words: TopicWords,
    texts: list[list[str]],
    dictionary: Dictionary,
    corpus_bow: list[list[tuple[int, int]]],
    coherences: list[str] | tuple[str, ...],
    window_size: int | None = None,
    min_window_count: int | None = None,
    progress_label: str = "coherence",
) -> dict[str, float]:
    scores: dict[str, float] = {}
    requested = list(dict.fromkeys(str(coherence) for coherence in coherences))
    if not requested:
        return scores
    if not topic_words or any(len(words) == 0 for words in topic_words):
        return {coherence: float("nan") for coherence in requested}

    word_lists = topic_words_to_word_lists(topic_words)
    sliding_coherences = [
        coherence
        for coherence in requested
        if coherence == "c_v" or coherence in EPSILON_SMOOTHED_COHERENCES
    ]
    window_size_by_coherence = {
        coherence: _effective_sliding_window_size(coherence, window_size)
        for coherence in sliding_coherences
    }
    for resolved_window_size in window_size_by_coherence.values():
        if resolved_window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {resolved_window_size}")
    counts_by_window_size = _build_sliding_window_counts(
        word_lists=word_lists,
        texts=texts,
        window_sizes=set(window_size_by_coherence.values()),
        progress_label=progress_label,
    )

    for coherence in requested:
        score_started = perf_counter()
        logger.info("%s score start metric=%s", progress_label, coherence)
        if coherence != "c_v" and len(dictionary) == 0:
            scores[coherence] = float("nan")
        elif coherence == "c_v":
            resolved_min_frequency = (
                DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT
                if min_window_count is None
                else int(min_window_count)
            )
            if resolved_min_frequency < 1:
                raise ValueError(
                    f"min_frequency must be >= 1, got {resolved_min_frequency}"
                )
            scores[coherence] = _compute_palmetto_cv_score_from_counts(
                word_lists=word_lists,
                counts=counts_by_window_size[window_size_by_coherence[coherence]],
                min_frequency=resolved_min_frequency,
                epsilon=PMI_SMOOTHING_EPSILON,
            )
        elif coherence in EPSILON_SMOOTHED_COHERENCES:
            scores[coherence] = _compute_epsilon_smoothed_pmi_score_from_counts(
                word_lists=word_lists,
                counts=counts_by_window_size[window_size_by_coherence[coherence]],
                coherence=coherence,
                epsilon=PMI_SMOOTHING_EPSILON,
            )
        elif coherence == "doc_npmi":
            scores[coherence] = compute_doc_npmi_score(
                topic_words=topic_words,
                texts=texts,
            )
        else:
            kwargs = {} if window_size is None else {"window_size": int(window_size)}
            model = CoherenceModel(
                topics=word_lists,
                texts=texts,
                dictionary=dictionary,
                corpus=corpus_bow,
                coherence=coherence,
                **kwargs,
            )
            scores[coherence] = float(model.get_coherence())
        logger.info(
            "%s score done metric=%s value=%s sec=%.1f",
            progress_label,
            coherence,
            scores[coherence],
            perf_counter() - score_started,
        )
    return scores


def compute_coherence_score(
    topic_words: TopicWords,
    texts: list[list[str]],
    dictionary: Dictionary,
    corpus_bow: list[list[tuple[int, int]]],
    coherence: str,
    window_size: int | None = None,
    min_window_count: int | None = None,
    progress_label: str = "coherence",
) -> float:
    if not topic_words:
        return float("nan")
    if coherence != "c_v" and len(dictionary) == 0:
        return float("nan")
    if any(len(words) == 0 for words in topic_words):
        return float("nan")
    if coherence == "c_v":
        return compute_palmetto_cv_score(
            topic_words=topic_words,
            texts=texts,
            window_size=window_size,
            min_frequency=min_window_count,
            progress_label=progress_label,
        )
    if coherence == "doc_npmi":
        return compute_doc_npmi_score(topic_words=topic_words, texts=texts)
    if coherence in EPSILON_SMOOTHED_COHERENCES:
        return compute_epsilon_smoothed_pmi_score(
            topic_words=topic_words,
            texts=texts,
            coherence=coherence,
            window_size=window_size,
            progress_label=progress_label,
        )
    word_lists = topic_words_to_word_lists(topic_words)
    kwargs = {} if window_size is None else {"window_size": int(window_size)}
    model = CoherenceModel(
        topics=word_lists,
        texts=texts,
        dictionary=dictionary,
        corpus=corpus_bow,
        coherence=coherence,
        **kwargs,
    )
    return float(model.get_coherence())


def compute_streaming_reference_coherence_score(
    *,
    topic_words: TopicWords,
    reference_path: Path,
    coherence: str,
    window_size: int | None = None,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
    min_window_count: int | None = None,
    progress_label: str = "coherence",
) -> StreamingReferenceCoherenceResult:
    if coherence not in STREAMING_REFERENCE_COHERENCES:
        raise ValueError(
            "Streaming reference coherence supports only "
            f"{sorted(STREAMING_REFERENCE_COHERENCES)}, got {coherence!r}."
        )
    if not topic_words:
        return StreamingReferenceCoherenceResult(
            score=float("nan"),
            num_docs=0,
            vocab_size=0,
        )
    if any(len(words) == 0 for words in topic_words):
        return StreamingReferenceCoherenceResult(
            score=float("nan"),
            num_docs=0,
            vocab_size=0,
        )

    word_lists = topic_words_to_word_lists(topic_words)
    dictionary = Dictionary(word_lists)
    texts = CountingTokenizedReferenceCorpus(
        path=reference_path,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
    )
    if coherence == "c_v":
        score = compute_palmetto_cv_score(
            topic_words=topic_words,
            texts=texts,
            window_size=window_size,
            min_frequency=min_window_count,
            progress_label=progress_label,
        )
        return StreamingReferenceCoherenceResult(
            score=score,
            num_docs=texts.num_docs,
            vocab_size=len(dictionary),
        )
    if coherence == "doc_npmi":
        score, num_docs = compute_streaming_doc_npmi_score(
            topic_words=topic_words,
            texts=texts,
        )
        return StreamingReferenceCoherenceResult(
            score=score,
            num_docs=num_docs,
            vocab_size=len(dictionary),
        )
    if coherence in EPSILON_SMOOTHED_COHERENCES:
        score = compute_epsilon_smoothed_pmi_score(
            topic_words=topic_words,
            texts=texts,
            coherence=coherence,
            window_size=window_size,
            progress_label=progress_label,
        )
        return StreamingReferenceCoherenceResult(
            score=score,
            num_docs=texts.num_docs,
            vocab_size=len(dictionary),
        )

    kwargs = {} if window_size is None else {"window_size": int(window_size)}
    model = CoherenceModel(
        topics=word_lists,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence,
        **kwargs,
    )
    score = float(model.get_coherence())
    return StreamingReferenceCoherenceResult(
        score=score,
        num_docs=texts.num_docs,
        vocab_size=len(dictionary),
    )


def compute_streaming_reference_coherence_scores(
    *,
    topic_words: TopicWords,
    reference_path: Path,
    coherences: list[str] | tuple[str, ...],
    window_size: int | None = None,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
    min_window_count: int | None = None,
    progress_label: str = "coherence",
) -> StreamingReferenceCoherenceScoresResult:
    requested = normalize_coherences(coherences)
    unsupported = [
        coherence
        for coherence in requested
        if coherence not in STREAMING_REFERENCE_COHERENCES
    ]
    if unsupported:
        raise ValueError(
            "Streaming reference coherence supports only "
            f"{sorted(STREAMING_REFERENCE_COHERENCES)}, got {unsupported!r}."
        )
    if not topic_words or any(len(words) == 0 for words in topic_words):
        return StreamingReferenceCoherenceScoresResult(
            scores={coherence: float("nan") for coherence in requested},
            num_docs=0,
            vocab_size=0,
        )

    word_lists = topic_words_to_word_lists(topic_words)
    dictionary = Dictionary(word_lists)
    texts = CountingTokenizedReferenceCorpus(
        path=reference_path,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
    )
    sliding_coherences = [
        coherence
        for coherence in requested
        if coherence == "c_v" or coherence in EPSILON_SMOOTHED_COHERENCES
    ]
    window_size_by_coherence = {
        coherence: _effective_sliding_window_size(coherence, window_size)
        for coherence in sliding_coherences
    }
    for resolved_window_size in window_size_by_coherence.values():
        if resolved_window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {resolved_window_size}")
    counts_by_window_size = _build_sliding_window_counts(
        word_lists=word_lists,
        texts=texts,
        window_sizes=set(window_size_by_coherence.values()),
        progress_label=progress_label,
    )
    num_docs = texts.num_docs

    scores: dict[str, float] = {}
    for coherence in requested:
        score_started = perf_counter()
        logger.info("%s score start metric=%s", progress_label, coherence)
        if coherence == "c_v":
            resolved_min_frequency = (
                DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT
                if min_window_count is None
                else int(min_window_count)
            )
            if resolved_min_frequency < 1:
                raise ValueError(
                    f"min_frequency must be >= 1, got {resolved_min_frequency}"
                )
            scores[coherence] = _compute_palmetto_cv_score_from_counts(
                word_lists=word_lists,
                counts=counts_by_window_size[window_size_by_coherence[coherence]],
                min_frequency=resolved_min_frequency,
                epsilon=PMI_SMOOTHING_EPSILON,
            )
        elif coherence in EPSILON_SMOOTHED_COHERENCES:
            scores[coherence] = _compute_epsilon_smoothed_pmi_score_from_counts(
                word_lists=word_lists,
                counts=counts_by_window_size[window_size_by_coherence[coherence]],
                coherence=coherence,
                epsilon=PMI_SMOOTHING_EPSILON,
            )
        elif coherence == "doc_npmi":
            doc_texts = CountingTokenizedReferenceCorpus(
                path=reference_path,
                max_docs=max_docs,
                min_doc_tokens=min_doc_tokens,
            )
            score, _num_docs = compute_streaming_doc_npmi_score(
                topic_words=topic_words,
                texts=doc_texts,
            )
            scores[coherence] = score
            if num_docs == 0:
                num_docs = _num_docs
        else:
            raise ValueError(f"Unsupported streaming coherence metric: {coherence!r}")
        logger.info(
            "%s score done metric=%s value=%s sec=%.1f",
            progress_label,
            coherence,
            scores[coherence],
            perf_counter() - score_started,
        )

    return StreamingReferenceCoherenceScoresResult(
        scores=scores,
        num_docs=num_docs,
        vocab_size=len(dictionary),
    )


def compute_topic_diversity(topic_words: TopicWords) -> float:
    word_lists = topic_words_to_word_lists(topic_words)
    total_words = sum(len(words) for words in word_lists)
    if total_words == 0:
        return float("nan")
    if any(len(words) == 0 for words in word_lists):
        return float("nan")
    unique_words = {word for words in word_lists for word in words}
    return float(len(unique_words) / total_words)


def evaluate_topic_words(
    *,
    topic_words: TopicWords,
    metric_names: list[MetricName] | tuple[MetricName, ...] = DEFAULT_METRIC_NAMES,
    texts: list[list[str]] | None = None,
    dictionary: Dictionary | None = None,
    corpus_bow: list[list[tuple[int, int]]] | None = None,
    coherence: str | list[str] | tuple[str, ...] = "c_v",
    coherence_topn: int | None = None,
    diversity_topn: int | None = None,
    coherence_window_size: int | None = None,
    coherence_min_window_count: int | None = None,
    progress_label: str = "coherence",
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    coherences = normalize_coherences(coherence)
    multiple_coherences = len(coherences) > 1
    coherence_metric_names = {
        coherence_metric_key(coherence_name, multiple=multiple_coherences)
        for coherence_name in coherences
    }
    coherence_evaluated = False
    for metric_name in metric_names:
        if metric_name == "coherence" or metric_name in coherence_metric_names:
            if coherence_evaluated:
                continue
            if texts is None or dictionary is None or corpus_bow is None:
                raise ValueError(
                    "texts, dictionary, and corpus_bow are required for coherence."
                )
            coherence_topic_words = (
                truncate_topic_words(topic_words, coherence_topn)
                if coherence_topn is not None
                else topic_words
            )
            if multiple_coherences:
                coherence_scores = compute_coherence_scores(
                    topic_words=coherence_topic_words,
                    texts=texts,
                    dictionary=dictionary,
                    corpus_bow=corpus_bow,
                    coherences=coherences,
                    window_size=coherence_window_size,
                    min_window_count=coherence_min_window_count,
                    progress_label=progress_label,
                )
            else:
                coherence_scores = {
                    coherences[0]: compute_coherence_score(
                        topic_words=coherence_topic_words,
                        texts=texts,
                        dictionary=dictionary,
                        corpus_bow=corpus_bow,
                        coherence=coherences[0],
                        window_size=coherence_window_size,
                        min_window_count=coherence_min_window_count,
                        progress_label=progress_label,
                    )
                }
            for coherence_name, score in coherence_scores.items():
                metrics[
                    coherence_metric_key(
                        coherence_name,
                        multiple=multiple_coherences,
                    )
                ] = score
            coherence_evaluated = True
        elif metric_name == "diversity":
            diversity_topic_words = (
                truncate_topic_words(topic_words, diversity_topn)
                if diversity_topn is not None
                else topic_words
            )
            metrics["diversity"] = compute_topic_diversity(
                topic_words=diversity_topic_words
            )
        else:
            raise ValueError(f"Unsupported topic-word metric '{metric_name}'.")
    return metrics


def aggregate_metrics(
    per_iter_metrics: list[dict[str, float]],
    metric_names: list[MetricName] | tuple[MetricName, ...] = DEFAULT_METRIC_NAMES,
) -> dict[str, dict[str, float]]:
    def _mean_std(values: np.ndarray) -> dict[str, float]:
        values = values.astype(float)
        if values.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        if values.size == 1:
            return {"mean": float(values[0]), "std": 0.0}
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }

    agg: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        agg[metric_name] = _mean_std(
            np.array(
                [
                    metrics.get(metric_name, float("nan"))
                    for metrics in per_iter_metrics
                ],
                dtype=float,
            )
        )
    return agg

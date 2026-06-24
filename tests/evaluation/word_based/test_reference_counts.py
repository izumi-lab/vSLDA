from __future__ import annotations

import json
from pathlib import Path

import pytest
from gensim.corpora import Dictionary

from src.evaluation.word_based.reference_counts import (
    build_shared_reference_counts,
    collect_target_words,
    compute_shared_reference_coherence_scores,
    effective_window_sizes_for_coherences,
)
from src.evaluation.word_based.topic_word_metrics import compute_coherence_scores


def _write_reference(path: Path, docs: list[list[str]]) -> None:
    path.write_text(
        "\n".join(json.dumps({"tokens": doc}) for doc in docs) + "\n",
        encoding="utf-8",
    )


def _assert_reference_counts_equal(actual, expected) -> None:
    assert actual.num_docs == expected.num_docs
    assert actual.doc_word_counts == expected.doc_word_counts
    assert actual.doc_pair_counts == expected.doc_pair_counts
    assert actual.counts_by_window_size.keys() == expected.counts_by_window_size.keys()
    for window_size in expected.counts_by_window_size:
        assert (
            actual.counts_by_window_size[window_size].num_windows
            == expected.counts_by_window_size[window_size].num_windows
        )
        assert (
            actual.counts_by_window_size[window_size].word_window_counts
            == expected.counts_by_window_size[window_size].word_window_counts
        )
        assert (
            actual.counts_by_window_size[window_size].pair_window_counts
            == expected.counts_by_window_size[window_size].pair_window_counts
        )


def test_shared_reference_counts_match_existing_coherence_scores(
    tmp_path: Path,
) -> None:
    docs = [
        ["alpha", "beta", "gamma", "delta"],
        ["alpha", "beta", "epsilon"],
        ["beta", "gamma", "delta"],
        ["alpha", "gamma", "epsilon"],
    ]
    reference_path = tmp_path / "reference.jsonl"
    _write_reference(reference_path, docs)
    topic_words = [
        [("alpha", 1.0), ("beta", 0.9), ("gamma", 0.8)],
        [("gamma", 1.0), ("delta", 0.9), ("epsilon", 0.8)],
    ]
    coherences = ["c_v", "c_npmi", "c_uci"]
    dictionary = Dictionary(docs)
    corpus_bow = [dictionary.doc2bow(doc) for doc in docs]

    expected = compute_coherence_scores(
        topic_words=topic_words,
        texts=docs,
        dictionary=dictionary,
        corpus_bow=corpus_bow,
        coherences=coherences,
    )
    counts = build_shared_reference_counts(
        reference_path=reference_path,
        target_words=collect_target_words([topic_words]),
        window_sizes=effective_window_sizes_for_coherences(coherences),
        backend="python",
    )
    actual = compute_shared_reference_coherence_scores(
        topic_words=topic_words,
        metric_names=["coherence_c_v", "coherence_c_npmi", "coherence_c_uci"],
        coherences=coherences,
        counts=counts,
    )

    assert actual["coherence_c_v"] == pytest.approx(expected["c_v"])
    assert actual["coherence_c_npmi"] == pytest.approx(expected["c_npmi"])
    assert actual["coherence_c_uci"] == pytest.approx(expected["c_uci"])


def test_numba_reference_counts_match_python_backend(tmp_path: Path) -> None:
    pytest.importorskip("numba")
    docs = [
        ["alpha", "beta", "gamma", "delta"],
        ["alpha", "beta", "epsilon"],
        ["beta", "gamma", "delta"],
        ["alpha", "gamma", "epsilon"],
    ]
    reference_path = tmp_path / "reference.jsonl"
    _write_reference(reference_path, docs)
    topic_words = [
        [("alpha", 1.0), ("beta", 0.9), ("gamma", 0.8)],
        [("gamma", 1.0), ("delta", 0.9), ("epsilon", 0.8)],
    ]
    coherences = ["c_v", "c_npmi", "c_uci"]
    target_words = collect_target_words([topic_words])
    window_sizes = effective_window_sizes_for_coherences(coherences)

    python_counts = build_shared_reference_counts(
        reference_path=reference_path,
        target_words=target_words,
        window_sizes=window_sizes,
        backend="python",
    )
    numba_counts = build_shared_reference_counts(
        reference_path=reference_path,
        target_words=target_words,
        window_sizes=window_sizes,
        backend="numba",
    )

    _assert_reference_counts_equal(numba_counts, python_counts)

    python_scores = compute_shared_reference_coherence_scores(
        topic_words=topic_words,
        metric_names=["coherence_c_v", "coherence_c_npmi", "coherence_c_uci"],
        coherences=coherences,
        counts=python_counts,
    )
    numba_scores = compute_shared_reference_coherence_scores(
        topic_words=topic_words,
        metric_names=["coherence_c_v", "coherence_c_npmi", "coherence_c_uci"],
        coherences=coherences,
        counts=numba_counts,
    )

    assert numba_scores == pytest.approx(python_scores)


def test_parallel_reference_counts_match_serial_backends(tmp_path: Path) -> None:
    pytest.importorskip("numba")
    docs = [
        ["alpha", "beta", "gamma", "delta"],
        ["alpha", "beta", "epsilon"],
        ["beta", "gamma", "delta"],
        ["alpha", "gamma", "epsilon"],
        ["zeta", "alpha", "delta"],
    ]
    reference_path = tmp_path / "reference.jsonl"
    _write_reference(reference_path, docs)
    topic_words = [
        [("alpha", 1.0), ("beta", 0.9), ("gamma", 0.8)],
        [("gamma", 1.0), ("delta", 0.9), ("epsilon", 0.8)],
    ]
    coherences = ["c_v", "c_npmi", "c_uci"]
    target_words = collect_target_words([topic_words])
    window_sizes = effective_window_sizes_for_coherences(coherences)

    python_serial = build_shared_reference_counts(
        reference_path=reference_path,
        target_words=target_words,
        window_sizes=window_sizes,
        backend="python",
    )
    python_parallel = build_shared_reference_counts(
        reference_path=reference_path,
        target_words=target_words,
        window_sizes=window_sizes,
        backend="python",
        workers=2,
        chunk_size=2,
    )
    numba_parallel = build_shared_reference_counts(
        reference_path=reference_path,
        target_words=target_words,
        window_sizes=window_sizes,
        backend="numba",
        workers=2,
        chunk_size=2,
    )

    _assert_reference_counts_equal(python_parallel, python_serial)
    _assert_reference_counts_equal(numba_parallel, python_serial)

    python_scores = compute_shared_reference_coherence_scores(
        topic_words=topic_words,
        metric_names=["coherence_c_v", "coherence_c_npmi", "coherence_c_uci"],
        coherences=coherences,
        counts=python_parallel,
    )
    numba_scores = compute_shared_reference_coherence_scores(
        topic_words=topic_words,
        metric_names=["coherence_c_v", "coherence_c_npmi", "coherence_c_uci"],
        coherences=coherences,
        counts=numba_parallel,
    )
    assert numba_scores == pytest.approx(python_scores)

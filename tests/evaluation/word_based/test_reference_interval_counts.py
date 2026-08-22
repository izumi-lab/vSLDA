from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src.evaluation.word_based.reference_counts import build_shared_reference_counts
from src.evaluation.word_based.reference_interval_counts import (
    byte_shard_ranges,
    interval_intersection_size,
    interval_union_size,
    iter_byte_shard_tokens,
    occurrence_window_intervals,
)
from src.evaluation.word_based.reference_query import (
    ReferenceCountQuery,
    build_reference_count_query,
)


def _write_reference(path: Path, docs: list[list[str]]) -> None:
    path.write_text(
        "\n".join(json.dumps({"tokens": doc}) for doc in docs) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("positions", "length", "window_size", "expected"),
    [
        ([], 5, 3, ()),
        ([0], 3, 3, ((0, 0),)),
        ([0], 5, 3, ((0, 0),)),
        ([4], 5, 3, ((2, 2),)),
        ([1, 3], 5, 3, ((0, 2),)),
        ([0, 4], 8, 3, ((0, 0), (2, 4))),
    ],
)
def test_occurrence_window_intervals(positions, length, window_size, expected) -> None:
    assert (
        occurrence_window_intervals(
            positions,
            document_length=length,
            window_size=window_size,
        )
        == expected
    )


def test_interval_sizes() -> None:
    assert interval_union_size(((0, 2), (5, 5))) == 4
    assert interval_intersection_size(((0, 2), (5, 7)), ((2, 5),)) == 2


def test_byte_shards_cover_each_jsonl_row_once(tmp_path: Path) -> None:
    docs = [[f"document-{idx}", "x" * (idx + 1)] for idx in range(17)]
    path = tmp_path / "reference.jsonl"
    _write_reference(path, docs)
    observed: list[list[str]] = []
    for start, end in byte_shard_ranges(path, 7):
        observed.extend(
            iter_byte_shard_tokens(
                path,
                start=start,
                end=end,
                min_doc_tokens=1,
            )
        )
    assert observed == docs


def test_query_plan_only_requests_topic_local_pairs() -> None:
    query = build_reference_count_query(
        topic_words_by_condition=[
            [[("alpha", 1.0), ("beta", 0.5)], [("gamma", 1.0), ("delta", 0.5)]]
        ],
        window_sizes={10, 110},
    )
    assert query.requested_pairs == (("alpha", "beta"), ("delta", "gamma"))
    assert ("alpha", "gamma") not in query.requested_pairs


def _assert_counts_equal(actual, expected, query: ReferenceCountQuery) -> None:
    assert actual.num_docs == expected.num_docs
    if query.need_document_counts:
        assert actual.doc_word_counts == expected.doc_word_counts
        assert actual.doc_pair_counts == {
            pair: expected.doc_pair_counts[pair]
            for pair in query.requested_pairs
            if expected.doc_pair_counts[pair]
        }
    for window_size in query.window_sizes:
        actual_counts = actual.counts_by_window_size[window_size]
        expected_counts = expected.counts_by_window_size[window_size]
        assert actual_counts.num_windows == expected_counts.num_windows
        assert actual_counts.word_window_counts == expected_counts.word_window_counts
        assert actual_counts.pair_window_counts == {
            pair: expected_counts.pair_window_counts[pair]
            for pair in query.requested_pairs
            if expected_counts.pair_window_counts[pair]
        }


def test_numba_interval_matches_enumeration_on_random_corpora(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numba")
    rng = random.Random(1729)
    vocabulary = ["a", "b", "c", "d", "other"]
    for trial in range(12):
        docs = [
            [rng.choice(vocabulary) for _ in range(rng.randint(1, 14))]
            for _ in range(rng.randint(1, 12))
        ]
        path = tmp_path / f"reference-{trial}.jsonl"
        _write_reference(path, docs)
        query = ReferenceCountQuery(
            target_words=("a", "b", "c", "d"),
            requested_pairs=(("a", "b"), ("a", "d"), ("c", "d")),
            window_sizes=(1, 3, 7, 20),
            need_document_counts=True,
        )
        expected = build_shared_reference_counts(
            reference_path=path,
            target_words=set(query.target_words),
            window_sizes=set(query.window_sizes),
            backend="python",
            query=query,
        )
        actual = build_shared_reference_counts(
            reference_path=path,
            target_words=set(query.target_words),
            window_sizes=set(query.window_sizes),
            backend="numba_interval",
            workers=3,
            chunk_size=3,
            query=query,
        )
        _assert_counts_equal(actual, expected, query)


def test_numba_interval_max_docs_preserves_prefix_semantics(tmp_path: Path) -> None:
    pytest.importorskip("numba")
    docs = [["a", str(idx), "b"] for idx in range(8)]
    path = tmp_path / "reference.jsonl"
    _write_reference(path, docs)
    query = ReferenceCountQuery(
        target_words=("a", "b"),
        requested_pairs=(("a", "b"),),
        window_sizes=(2,),
        need_document_counts=False,
    )
    expected = build_shared_reference_counts(
        reference_path=path,
        target_words={"a", "b"},
        window_sizes={2},
        max_docs=3,
        backend="python",
        query=query,
    )
    actual = build_shared_reference_counts(
        reference_path=path,
        target_words={"a", "b"},
        window_sizes={2},
        max_docs=3,
        backend="numba_interval",
        workers=2,
        chunk_size=2,
        query=query,
    )
    _assert_counts_equal(actual, expected, query)

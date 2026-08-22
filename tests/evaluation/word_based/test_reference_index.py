from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.word_based.reference_counts import build_shared_reference_counts
from src.evaluation.word_based.reference_index import (
    build_reference_index,
    is_reference_index_valid,
    query_reference_index,
)
from src.evaluation.word_based.reference_query import ReferenceCountQuery


def test_positional_index_matches_direct_scan_and_detects_stale_source(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numba")
    path = tmp_path / "reference.jsonl"
    docs = [
        ["a", "x", "b", "a"],
        ["c", "a"],
        ["b", "c", "c"],
        ["x"],
    ]
    path.write_text(
        "\n".join(json.dumps({"tokens": doc}) for doc in docs) + "\n",
        encoding="utf-8",
    )
    query = ReferenceCountQuery(
        target_words=("a", "b", "c", "missing"),
        requested_pairs=(("a", "b"), ("a", "c"), ("b", "c")),
        window_sizes=(2, 10),
        need_document_counts=True,
    )
    expected = build_shared_reference_counts(
        reference_path=path,
        target_words=set(query.target_words),
        window_sizes=set(query.window_sizes),
        backend="python",
        query=query,
    )
    index_root = tmp_path / "index"
    build_reference_index(
        reference_path=path,
        index_root=index_root,
        commit_docs=2,
    )
    assert is_reference_index_valid(
        reference_path=path,
        index_root=index_root,
    )
    window_counts, doc_words, doc_pairs, num_docs = query_reference_index(
        reference_path=path,
        index_root=index_root,
        query=query,
    )
    assert num_docs == expected.num_docs
    assert doc_words == expected.doc_word_counts
    assert doc_pairs == expected.doc_pair_counts
    for window_size in query.window_sizes:
        assert (
            window_counts[window_size].num_windows
            == expected.counts_by_window_size[window_size].num_windows
        )
        assert (
            window_counts[window_size].word_window_counts
            == expected.counts_by_window_size[window_size].word_window_counts
        )
        assert (
            window_counts[window_size].pair_window_counts
            == expected.counts_by_window_size[window_size].pair_window_counts
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tokens": ["new"]}) + "\n")
    assert not is_reference_index_valid(
        reference_path=path,
        index_root=index_root,
    )


def test_build_shared_counts_uses_valid_index_in_auto_mode(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numba")
    path = tmp_path / "reference.jsonl"
    path.write_text(
        json.dumps({"tokens": ["a", "b", "a"]}) + "\n",
        encoding="utf-8",
    )
    index_root = tmp_path / "index"
    build_reference_index(reference_path=path, index_root=index_root)
    query = ReferenceCountQuery(
        target_words=("a", "b"),
        requested_pairs=(("a", "b"),),
        window_sizes=(2,),
    )
    counts = build_shared_reference_counts(
        reference_path=path,
        target_words={"a", "b"},
        window_sizes={2},
        backend="numba_interval",
        query=query,
        reference_index_mode="auto",
        reference_index_root=index_root,
    )
    assert counts.counts_by_window_size[2].word_window_counts == {
        "a": 2,
        "b": 2,
    }
    assert counts.counts_by_window_size[2].pair_window_counts == {("a", "b"): 2}


def test_resume_refuses_progress_recorded_for_a_different_corpus(
    tmp_path: Path,
) -> None:
    """A missing manifest must not let a stale byte offset be replayed.

    The resume offset lives in index.sqlite3 while the corpus identity used to
    live only in manifest.json, so deleting the manifest -- which the stale-index
    error message invites -- made the next build rewrite the manifest for the new
    corpus, resume at the old offset and mark the result COMPLETE.
    """

    path = tmp_path / "reference.jsonl"
    index_root = tmp_path / "index"

    def _write(documents: list[list[str]]) -> None:
        path.write_text(
            "\n".join(json.dumps(tokens) for tokens in documents) + "\n",
            encoding="utf-8",
        )

    _write([["a", "b"], ["a", "b"], ["b", "a"]])
    build_reference_index(reference_path=path, index_root=index_root)

    (index_root / "manifest.json").unlink()
    _write([["c", "c"], ["c", "c"], ["c", "c"], ["a", "c"], ["c", "a"]])

    with pytest.raises(ValueError, match="different reference corpus"):
        build_reference_index(reference_path=path, index_root=index_root)

    assert not is_reference_index_valid(
        reference_path=path, index_root=index_root, min_doc_tokens=1
    )

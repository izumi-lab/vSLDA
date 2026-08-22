from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.word_based.reference_cache import (
    load_reference_count_cache_v2,
    save_reference_count_cache_v2,
)
from src.evaluation.word_based.reference_counts import build_shared_reference_counts
from src.evaluation.word_based.reference_query import ReferenceCountQuery
from src.evaluation.word_based.resumability import reference_corpus_identity


def test_v2_cache_reuses_superset_without_scan(tmp_path: Path) -> None:
    path = tmp_path / "reference.jsonl"
    docs = [["a", "b", "c"], ["a", "c"], ["b", "d"]]
    path.write_text(
        "\n".join(json.dumps({"tokens": doc}) for doc in docs) + "\n",
        encoding="utf-8",
    )
    superset = ReferenceCountQuery(
        target_words=("a", "b", "c", "d"),
        requested_pairs=(("a", "b"), ("a", "c"), ("b", "d")),
        window_sizes=(2, 5),
        need_document_counts=True,
    )
    counts = build_shared_reference_counts(
        reference_path=path,
        target_words=set(superset.target_words),
        window_sizes=set(superset.window_sizes),
        backend="python",
        query=superset,
    )
    cache_root = tmp_path / ".cache"
    saved = save_reference_count_cache_v2(
        cache_root=cache_root,
        reference_identity=reference_corpus_identity(path),
        max_docs=None,
        min_doc_tokens=1,
        query=superset,
        counts=counts,
    )
    subset = ReferenceCountQuery(
        target_words=("a", "c"),
        requested_pairs=(("a", "c"),),
        window_sizes=(2,),
        need_document_counts=False,
    )
    loaded = load_reference_count_cache_v2(
        cache_root=cache_root,
        reference_identity=reference_corpus_identity(path),
        max_docs=None,
        min_doc_tokens=1,
        query=subset,
    )
    assert loaded is not None
    subset_counts, source = loaded
    assert source == saved
    assert subset_counts.counts_by_window_size[2].word_window_counts == {
        "a": counts.counts_by_window_size[2].word_window_counts["a"],
        "c": counts.counts_by_window_size[2].word_window_counts["c"],
    }
    assert subset_counts.counts_by_window_size[2].pair_window_counts == {
        ("a", "c"): counts.counts_by_window_size[2].pair_window_counts[("a", "c")]
    }
    assert subset_counts.doc_word_counts == {}

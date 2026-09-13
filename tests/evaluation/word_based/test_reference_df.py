from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.word_based import reference_df as reference_df_module
from src.evaluation.word_based.reference_df import (
    build_reference_document_frequencies,
    load_reference_document_frequencies,
    reference_df_fingerprint,
)


def _write_reference(path: Path, documents: list[list[str]]) -> Path:
    path.write_text(
        "\n".join(json.dumps({"tokens": tokens}) for tokens in documents) + "\n",
        encoding="utf-8",
    )
    return path


def test_document_frequency_counts_documents_not_occurrences(tmp_path: Path) -> None:
    reference = _write_reference(
        tmp_path / "ref.jsonl",
        [
            ["alpha", "alpha", "beta"],
            ["alpha", "gamma"],
            ["beta"],
        ],
    )

    table = build_reference_document_frequencies(reference)

    assert table.num_docs == 3
    assert table.document_frequencies == {"alpha": 2, "beta": 2, "gamma": 1}
    assert table.ratio("alpha") == 2 / 3
    assert table.ratio("missing") == 0.0


def test_document_frequency_honours_max_docs(tmp_path: Path) -> None:
    reference = _write_reference(
        tmp_path / "ref.jsonl", [["alpha"], ["beta"], ["gamma"]]
    )

    table = build_reference_document_frequencies(reference, max_docs=2)

    assert table.num_docs == 2
    assert "gamma" not in table.document_frequencies


def test_cache_round_trip_avoids_a_second_scan(tmp_path: Path, monkeypatch) -> None:
    reference = _write_reference(tmp_path / "ref.jsonl", [["alpha"], ["alpha", "beta"]])
    cache_root = tmp_path / "cache"

    first = load_reference_document_frequencies(reference, cache_root=cache_root)
    fingerprint = reference_df_fingerprint(
        path=reference, max_docs=None, min_doc_tokens=1
    )
    assert (cache_root / "reference_df" / f"{fingerprint}.json").exists()

    def _fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the cached table should have been reused")

    monkeypatch.setattr(
        reference_df_module, "build_reference_document_frequencies", _fail
    )
    second = load_reference_document_frequencies(reference, cache_root=cache_root)

    assert second.document_frequencies == first.document_frequencies
    assert second.num_docs == first.num_docs


def test_cache_is_invalidated_when_the_reference_changes(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path / "ref.jsonl", [["alpha"]])
    cache_root = tmp_path / "cache"
    load_reference_document_frequencies(reference, cache_root=cache_root)

    _write_reference(reference, [["alpha"], ["beta"], ["gamma"]])
    refreshed = load_reference_document_frequencies(reference, cache_root=cache_root)

    assert refreshed.num_docs == 3
    assert "gamma" in refreshed.document_frequencies


def test_cache_key_separates_different_scan_settings(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path / "ref.jsonl", [["alpha"], ["beta"]])

    assert reference_df_fingerprint(
        path=reference, max_docs=None, min_doc_tokens=1
    ) != reference_df_fingerprint(path=reference, max_docs=1, min_doc_tokens=1)


def test_empty_reference_is_rejected(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path / "ref.jsonl", [])

    try:
        build_reference_document_frequencies(reference)
    except ValueError as error:
        assert "empty" in str(error)
    else:  # pragma: no cover - guarded by the assertion above
        raise AssertionError("expected ValueError for an empty reference corpus")

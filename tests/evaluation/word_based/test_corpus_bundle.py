from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.word_based.corpus_bundle import (
    build_dictionary_and_corpus,
    load_filtered_split_texts_from_csvs,
    load_tokenized_reference_corpus,
)


def test_dictionary_exact_token_exclusion_does_not_remove_plain_num() -> None:
    dictionary, corpus = build_dictionary_and_corpus(
        [["<NUM>", "num", "alpha"], ["<NUM>", "num", "beta"]],
        dict_no_below=1,
        dict_no_above=1.0,
        dict_exclude_tokens=frozenset({"<NUM>"}),
    )

    assert "<NUM>" not in dictionary.token2id
    assert "num" in dictionary.token2id
    assert all(
        dictionary[token_id] != "<NUM>"
        for document in corpus
        for token_id, _count in document
    )


def test_load_filtered_split_texts_from_csvs_filters_all_to_mapped_targets(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "data": ["dance text", "cosmos text", "environment text", "baseball text"],
            "target_str": ["dance", "cosmos", "environment", "baseball"],
        }
    ).to_csv(csv_path, index=False)

    texts = load_filtered_split_texts_from_csvs(
        dataset="nyt",
        csv_paths=(str(csv_path),),
        category="all",
    )

    assert texts == ["dance text", "baseball text"]


def test_load_tokenized_reference_corpus_preserves_japanese_tokens(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "jawiki-tokenized.jsonl"
    reference_path.write_text(
        '{"id":"1","tokens":["企業","成長","リスク"]}\n' '{"id":"2","tokens":["短"]}\n',
        encoding="utf-8",
    )

    texts = load_tokenized_reference_corpus(
        reference_path,
        min_doc_tokens=2,
    )

    assert texts == [["企業", "成長", "リスク"]]


def _band_texts() -> list[list[str]]:
    return [
        ["common", "midrange", "rare", "alpha"],
        ["common", "midrange", "rare", "beta"],
        ["common", "midrange", "gamma"],
    ]


_BAND_DF = {
    "common": 900,  # 90% of the reference corpus -> above the upper bound
    "midrange": 200,  # 20% -> inside the band
    "rare": 3,  # below the lower bound
    "alpha": 250,
    "beta": 250,
    "gamma": 250,
}


def test_reference_band_defaults_do_not_change_the_vocabulary() -> None:
    baseline, baseline_bow = build_dictionary_and_corpus(
        _band_texts(), dict_no_below=1, dict_no_above=1.0
    )
    restricted, restricted_bow = build_dictionary_and_corpus(
        _band_texts(),
        dict_no_below=1,
        dict_no_above=1.0,
        reference_document_frequencies=_BAND_DF,
        reference_num_docs=1000,
        reference_min_df=0,
        reference_max_df_ratio=1.0,
    )

    ordered_baseline = [baseline[i] for i in range(len(baseline))]
    ordered_restricted = [restricted[i] for i in range(len(restricted))]
    assert ordered_baseline == ordered_restricted
    assert baseline_bow == restricted_bow


def test_reference_band_drops_words_outside_the_band() -> None:
    dictionary, _ = build_dictionary_and_corpus(
        _band_texts(),
        dict_no_below=1,
        dict_no_above=1.0,
        reference_document_frequencies=_BAND_DF,
        reference_num_docs=1000,
        reference_min_df=50,
        reference_max_df_ratio=0.30,
    )

    assert "rare" not in dictionary.token2id, "below reference_min_df"
    assert "common" not in dictionary.token2id, "at or above reference_max_df_ratio"
    assert "midrange" in dictionary.token2id
    assert "alpha" in dictionary.token2id


def test_reference_band_treats_unseen_words_as_zero_document_frequency() -> None:
    dictionary, _ = build_dictionary_and_corpus(
        _band_texts(),
        dict_no_below=1,
        dict_no_above=1.0,
        reference_document_frequencies={"midrange": 200},
        reference_num_docs=1000,
        reference_min_df=50,
        reference_max_df_ratio=1.0,
    )

    assert list(dictionary.token2id) == ["midrange"]


def test_reference_max_df_ratio_of_one_keeps_ubiquitous_words() -> None:
    # ratio 1.0 documents "no upper bound"; a token present in every reference
    # document must survive once only a lower bound is requested.
    dictionary, _ = build_dictionary_and_corpus(
        _band_texts(),
        dict_no_below=1,
        dict_no_above=1.0,
        reference_document_frequencies={**_BAND_DF, "common": 1000},
        reference_num_docs=1000,
        reference_min_df=50,
        reference_max_df_ratio=1.0,
    )

    assert "common" in dictionary.token2id
    assert "rare" not in dictionary.token2id, "still below reference_min_df"


def test_reference_band_requires_the_document_frequency_table() -> None:
    try:
        build_dictionary_and_corpus(
            _band_texts(), dict_no_below=1, dict_no_above=1.0, reference_min_df=50
        )
    except ValueError as error:
        assert "reference_document_frequencies" in str(error)
    else:  # pragma: no cover - guarded by the assertion above
        raise AssertionError("expected ValueError for a missing frequency table")


def test_reference_band_rejects_an_empty_result() -> None:
    try:
        build_dictionary_and_corpus(
            _band_texts(),
            dict_no_below=1,
            dict_no_above=1.0,
            reference_document_frequencies=_BAND_DF,
            reference_num_docs=1000,
            reference_min_df=10_000,
        )
    except ValueError as error:
        assert "empty after filtering" in str(error)
    else:  # pragma: no cover - guarded by the assertion above
        raise AssertionError("expected ValueError for an empty dictionary")

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.word_based.corpus_bundle import (
    build_dictionary_and_corpus,
    load_filtered_split_texts_from_csvs,
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

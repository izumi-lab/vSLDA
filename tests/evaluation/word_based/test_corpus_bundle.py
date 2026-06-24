from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.word_based.corpus_bundle import load_filtered_split_texts_from_csvs


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

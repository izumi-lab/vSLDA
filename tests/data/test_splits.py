from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import splits


def test_load_filtered_split_texts_filters_all_to_mapped_targets(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "data": ["dance text", "cosmos text", "environment text", "baseball text"],
            "target_str": ["dance", "cosmos", "environment", "baseball"],
        }
    )
    monkeypatch.setattr(
        splits,
        "load_dataset_split",
        lambda dataset, split: (Path("test.csv"), frame),
    )

    texts = splits.load_filtered_split_texts("nyt", "all", "test")

    assert texts == ["dance text", "baseball text"]


def test_load_filtered_split_texts_without_targets_keeps_all_rows(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "data": ["first text", "second text"],
            "target_str": ["unknown_a", "unknown_b"],
        }
    )
    monkeypatch.setattr(
        splits,
        "load_dataset_split",
        lambda dataset, split: (Path("test.csv"), frame),
    )
    monkeypatch.setattr(splits, "get_dataset_targets", lambda dataset: None)

    texts = splits.load_filtered_split_texts("custom", "all", "test")

    assert texts == ["first text", "second text"]

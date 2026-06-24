from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.nyt import prepare_nyt


def test_prepare_nyt_builds_stratified_sixty_forty_split(tmp_path: Path) -> None:
    raw_path = tmp_path / "df_fine.pkl"
    output_dir = tmp_path / "nyt"
    raw = pd.DataFrame(
        {
            "text": [
                f"Alpha {label} story {idx}. Beta follows after review."
                for label in ("arts", "sports")
                for idx in range(10)
            ],
            "label": [label for label in ("arts", "sports") for _ in range(10)],
        }
    )
    raw.to_pickle(raw_path)

    prepare_nyt(raw_path=raw_path, output_dir=output_dir)

    train = pd.read_csv(output_dir / "train.csv")
    test = pd.read_csv(output_dir / "test.csv")
    manifest = json.loads((output_dir / "manifest.json").read_text("utf-8"))

    assert train.columns.tolist() == ["data", "target_str"]
    assert test.columns.tolist() == ["data", "target_str"]
    assert len(train) == 12
    assert len(test) == 8
    assert train["target_str"].value_counts().to_dict() == {"arts": 6, "sports": 6}
    assert test["target_str"].value_counts().to_dict() == {"arts": 4, "sports": 4}
    assert train["data"].str.contains(" / ").any()
    assert manifest["test_size"] == 0.4
    assert manifest["random_state"] == 42
    assert manifest["stratify"] == "target_str"
    assert manifest["train_rows"] == 12
    assert manifest["test_rows"] == 8


def test_prepare_nyt_rejects_missing_required_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "df_fine.pkl"
    pd.DataFrame({"data": ["text"], "target_str": ["label"]}).to_pickle(raw_path)

    try:
        prepare_nyt(raw_path=raw_path, output_dir=tmp_path / "out")
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "label" in str(exc)
        assert "text" in str(exc)
    else:
        raise AssertionError("prepare_nyt should reject invalid raw pickle columns")

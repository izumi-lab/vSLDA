from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.baselines.dataset_adapters import (
    load_document_texts,
    load_filtered_texts,
    load_sentence_corpus,
    resolve_baseline_targets,
    resolve_split_csv_paths,
    use_legacy_category_behavior,
)


def test_use_legacy_category_behavior_for_builtin_english_dataset() -> None:
    assert use_legacy_category_behavior("20newsgroup", "english") is True
    assert use_legacy_category_behavior("20newsgroup", "ja") is False


def test_resolve_baseline_targets_allows_unfiltered_all_for_custom_dataset() -> None:
    assert (
        resolve_baseline_targets(
            "custom_dataset",
            "all",
            None,
            language="ja",
        )
        is None
    )


def test_resolve_split_csv_paths_defaults_to_data_root() -> None:
    paths = resolve_split_csv_paths("dummy", "train", None)

    assert paths == [str(Path("data/dummy/train.csv").resolve())]


def test_load_filtered_texts_filters_by_target(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    frame = pd.DataFrame(
        {
            "data": ["first", "second", "   ", "third"],
            "target_str": ["a", "b", "a", "a"],
        }
    )
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    texts = load_filtered_texts(
        csv_paths=[str(csv_path)],
        text_column="data",
        target_column="target_str",
        targets=["a"],
    )

    assert texts == ["first", "third"]


def test_load_filtered_texts_requires_target_column_when_filtering(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"data": ["first"]}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    with pytest.raises(ValueError):
        load_filtered_texts(
            csv_paths=[str(csv_path)],
            text_column="data",
            target_column=None,
            targets=["a"],
        )


def test_load_sentence_corpus_splits_english_by_delimiter(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "data": ["alpha / beta", "gamma"],
            "target_str": ["a", "a"],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    corpus = load_sentence_corpus(
        csv_paths=[str(csv_path)],
        text_column="data",
        target_column="target_str",
        targets=["a"],
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
    )

    assert corpus == [["alpha", "beta"], ["gamma"]]


def test_load_document_texts_joins_english_segments(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "data": ["alpha / beta", "gamma"],
            "target_str": ["a", "a"],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    docs = load_document_texts(
        csv_paths=[str(csv_path)],
        text_column="data",
        target_column="target_str",
        targets=["a"],
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
    )

    assert docs == ["alpha beta", "gamma"]

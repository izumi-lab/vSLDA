from __future__ import annotations

import numpy as np
import pandas as pd

import src.evaluation.classification.alignment as alignment_module
from src.evaluation.classification.alignment import (
    build_baseline_available_indices,
    build_common_feature_alignment,
    build_label_space_indices,
)


def test_build_common_feature_alignment_uses_shared_intersection() -> None:
    alignments, common_train, common_test = build_common_feature_alignment(
        train_label_source_indices=np.asarray([0, 1, 2, 3, 4]),
        test_label_source_indices=np.asarray([10, 11, 12, 13]),
        feature_available_indices={
            "ctm": (np.asarray([0, 1, 2, 4]), np.asarray([10, 11, 13])),
            "bleilda": (np.asarray([1, 2, 4]), np.asarray([11, 13])),
        },
    )

    assert common_train.tolist() == [1, 2, 4]
    assert common_test.tolist() == [11, 13]
    assert alignments["ctm"].train_indices_in_label_space.tolist() == [1, 2, 4]
    assert alignments["ctm"].train_row_selector.tolist() == [1, 2, 3]
    assert alignments["bleilda"].train_row_selector.tolist() == [0, 1, 2]


def test_build_common_feature_alignment_ignores_missing_models_not_in_input() -> None:
    alignments, common_train, common_test = build_common_feature_alignment(
        train_label_source_indices=np.asarray([0, 1, 2]),
        test_label_source_indices=np.asarray([0, 1]),
        feature_available_indices={
            "vmf": (np.asarray([0, 1, 2]), np.asarray([0, 1])),
        },
    )

    assert list(alignments.keys()) == ["vmf"]
    assert common_train.tolist() == [0, 1, 2]
    assert common_test.tolist() == [0, 1]


def test_build_label_space_indices_caches_split_loading(monkeypatch) -> None:
    alignment_module._build_label_space_split_alignment.cache_clear()
    calls: list[tuple[str, str]] = []

    def _fake_load_dataset_split(dataset: str, split: str):
        calls.append((dataset, split))
        frame = pd.DataFrame(
            {
                "data": ["a / b", "c / d"],
                "target_str": ["science", "science"],
            }
        )
        return f"{dataset}-{split}.csv", frame

    monkeypatch.setattr(
        "src.evaluation.classification.alignment.load_dataset_split",
        _fake_load_dataset_split,
    )
    monkeypatch.setattr(
        "src.evaluation.classification.alignment.get_dataset_targets",
        lambda *args, **kwargs: {"science": ["science"]},
    )

    first = build_label_space_indices("dummy", "science")
    second = build_label_space_indices("dummy", "science")

    assert first[0].available_indices.tolist() == [0, 1]
    assert second[1].available_indices.tolist() == [0, 1]
    assert calls == [("dummy", "train"), ("dummy", "test")]


def test_build_baseline_available_indices_caches_preprocessing(monkeypatch) -> None:
    alignment_module._build_preprocessed_split_alignment_cached.cache_clear()
    calls: list[tuple[str, str]] = []

    def _fake_load_dataset_split(dataset: str, split: str):
        frame = pd.DataFrame(
            {
                "data": ["a / b", "c / d"],
                "target_str": ["science", "science"],
            }
        )
        return f"{dataset}-{split}.csv", frame

    def _fake_preprocess_document(text: str, **kwargs):
        calls.append((text, str(kwargs.get("segmenter"))))
        return type(
            "Doc",
            (),
            {
                "sentences_raw": ["x"],
                "document_tokens": ["x"],
                "contextual_text": "x",
            },
        )()

    monkeypatch.setattr(
        "src.evaluation.classification.alignment.load_dataset_split",
        _fake_load_dataset_split,
    )
    monkeypatch.setattr(
        "src.evaluation.classification.alignment.preprocess_document",
        _fake_preprocess_document,
    )

    metadata = {
        "language": "english",
        "legacy_preprocessing": False,
        "text_column": "data",
        "target_column": "target_str",
        "targets": ["science"],
        "delimiter": " / ",
        "segmenter": "delimiter",
        "tokenizer": "default",
        "ja_replace_num": True,
        "ja_dicdir": None,
        "ja_require_unidic": False,
    }

    first = build_baseline_available_indices(
        "dummy",
        "science",
        metadata,
        require_document_tokens=False,
        require_contextual_text=False,
        require_sentences=True,
    )
    second = build_baseline_available_indices(
        "dummy",
        "science",
        metadata,
        require_document_tokens=False,
        require_contextual_text=False,
        require_sentences=True,
    )

    assert first[0].available_indices.tolist() == [0, 1]
    assert second[1].available_indices.tolist() == [0, 1]
    assert len(calls) == 4

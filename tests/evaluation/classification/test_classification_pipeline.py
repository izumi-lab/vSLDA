from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import src.evaluation.classification.pipeline as pipeline_module
from src.evaluation.classification.alignment import SplitAlignment
from src.evaluation.classification.pipeline import (
    ClassificationLabelBundle,
    FeatureSet,
    build_label_bundle,
    collect_feature_sets,
    run_classification_task,
)


def test_build_label_bundle_applies_train_indices(monkeypatch) -> None:
    def _fake_load_labels(
        dataset: str,
        category: str,
        split: str,
        *,
        target_column: str = "target_str",
        label_schema: str = "identity",
    ) -> list[str]:
        assert dataset == "dummy"
        assert category == "science"
        _ = target_column, label_schema
        return ["a", "b", "a"] if split == "train" else ["b", "a"]

    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.load_classification_labels",
        _fake_load_labels,
    )
    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.build_label_space_indices",
        lambda *args, **kwargs: (
            SplitAlignment(
                raw_indices=np.asarray([0, 1, 2, 3]),
                available_indices=np.asarray([0, 2, 3]),
            ),
            SplitAlignment(
                raw_indices=np.asarray([0, 1, 2]),
                available_indices=np.asarray([0, 2]),
            ),
        ),
    )

    bundle = build_label_bundle(
        dataset="dummy",
        category="science",
        category_labels=["a", "b"],
        train_indices=[0, 2],
    )

    assert bundle.label_map == {"a": 0, "b": 1}
    assert bundle.train_y == [0, 0]
    assert bundle.test_y == [1, 0]
    assert bundle.train_indices is not None
    assert bundle.train_indices.tolist() == [0, 2]
    assert bundle.train_source_indices.tolist() == [0, 3]
    assert bundle.test_source_indices.tolist() == [0, 2]
    assert bundle.raw_train_count == 4
    assert bundle.label_filtered_train_count == 3


def test_run_classification_task_uses_aligned_labels_and_returns_coverage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.build_label_bundle",
        lambda **kwargs: ClassificationLabelBundle(
            category_labels=["a", "b"],
            label_map={"a": 0, "b": 1},
            train_y=[0, 1, 0],
            test_y=[1, 0],
            train_indices=None,
            train_source_indices=np.asarray([0, 1, 2]),
            test_source_indices=np.asarray([0, 1]),
            raw_train_count=4,
            raw_test_count=3,
            label_filtered_train_count=3,
            label_filtered_test_count=2,
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.collect_feature_sets",
        lambda **kwargs: (
            [
                FeatureSet(
                    name="FeatureA",
                    train_x=np.asarray([[0.0], [1.0]]),
                    test_x=np.asarray([[0.5]]),
                    catalog_entry={"feature_name": "FeatureA", "model_key": "dummy"},
                    available_train_docs=2,
                    available_test_docs=1,
                )
            ],
            [0, 1],
            [1],
            {
                "raw_train_docs": 4,
                "raw_test_docs": 3,
                "label_filtered_train_docs": 3,
                "label_filtered_test_docs": 2,
                "selected_train_docs": 3,
                "selected_test_docs": 2,
                "common_train_docs": 2,
                "common_test_docs": 1,
                "available_train_docs": {"FeatureA": 2},
                "available_test_docs": {"FeatureA": 1},
            },
        ),
    )

    class _DummyClassifier:
        coef_ = np.asarray([[0.1, 0.9], [0.8, 0.2]])

    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.get_classifier_specs",
        lambda classifiers_to_use: [object()] if classifiers_to_use == ["svm"] else [],
    )

    def _fake_fit(feature_sets, train_y, classifier_specs):
        assert train_y == [0, 1]
        _ = feature_sets, classifier_specs
        return (
            {"FeatureA [Dummy]": _DummyClassifier()},
            {"FeatureA [Dummy]": np.asarray([1])},
        )

    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.fit_classifiers",
        _fake_fit,
    )

    result = run_classification_task(
        dataset="dummy",
        category="science",
        num_topics=20,
        iteration=0,
        category_labels=["a", "b"],
        classifiers_to_use=["svm"],
        vmf_assignment="hard",
    )

    assert result is not None
    acc_result, f1_result, feature_importance, feature_catalog, coverage = result
    assert acc_result["FeatureA [Dummy]"] == 100.0
    assert "macro" in f1_result
    assert feature_importance["FeatureA [Dummy]"]["a"] == [0, 1]
    assert feature_catalog == [{"feature_name": "FeatureA", "model_key": "dummy"}]
    assert coverage["common_train_docs"] == 2
    assert coverage["available_test_docs"] == {"FeatureA": 1}


def test_run_classification_task_logs_when_no_feature_sets(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.build_label_bundle",
        lambda **kwargs: ClassificationLabelBundle(
            category_labels=["a", "b"],
            label_map={"a": 0, "b": 1},
            train_y=[0, 1],
            test_y=[1],
            train_indices=None,
            train_source_indices=np.asarray([0, 1]),
            test_source_indices=np.asarray([0]),
            raw_train_count=2,
            raw_test_count=1,
            label_filtered_train_count=2,
            label_filtered_test_count=1,
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.collect_feature_sets",
        lambda **kwargs: ([], [], [], {}),
    )

    with caplog.at_level(logging.WARNING):
        result = run_classification_task(
            dataset="dummy",
            category="science",
            num_topics=20,
            iteration=0,
            category_labels=["a", "b"],
            classifiers_to_use=["svm"],
            vmf_assignment="hard",
        )

    assert result is None
    assert "no feature sets found" in caplog.text


def test_run_classification_task_logs_when_no_classifiers(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.build_label_bundle",
        lambda **kwargs: ClassificationLabelBundle(
            category_labels=["a", "b"],
            label_map={"a": 0, "b": 1},
            train_y=[0, 1],
            test_y=[1],
            train_indices=None,
            train_source_indices=np.asarray([0, 1]),
            test_source_indices=np.asarray([0]),
            raw_train_count=2,
            raw_test_count=1,
            label_filtered_train_count=2,
            label_filtered_test_count=1,
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.collect_feature_sets",
        lambda **kwargs: (
            [
                FeatureSet(
                    name="FeatureA",
                    train_x=np.asarray([[0.0], [1.0]]),
                    test_x=np.asarray([[0.5]]),
                    catalog_entry=None,
                    available_train_docs=2,
                    available_test_docs=1,
                )
            ],
            [0, 1],
            [1],
            {},
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.get_classifier_specs",
        lambda classifiers_to_use: [],
    )

    with caplog.at_level(logging.WARNING):
        result = run_classification_task(
            dataset="dummy",
            category="science",
            num_topics=20,
            iteration=0,
            category_labels=["a", "b"],
            classifiers_to_use=["svm"],
            vmf_assignment="hard",
        )

    assert result is None
    assert "no classifiers selected" in caplog.text


def test_load_classification_labels_caches_filtered_split_loading(monkeypatch) -> None:
    pipeline_module._load_classification_labels_cached.cache_clear()
    calls: list[tuple[str, str, str]] = []

    def _fake_load_filtered_split_labels(
        dataset: str,
        category: str,
        split: str,
        *,
        data_column: str = "data",
        target_column: str = "target_str",
        label_schema: str = "identity",
        delimiter: str = " / ",
    ) -> list[str]:
        _ = data_column, target_column, label_schema, delimiter
        calls.append((dataset, category, split))
        return ["a", "b"]

    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.load_filtered_split_labels",
        _fake_load_filtered_split_labels,
    )

    first = pipeline_module.load_classification_labels("dummy", "science", "train")
    second = pipeline_module.load_classification_labels("dummy", "science", "train")

    assert first == ["a", "b"]
    assert second == ["a", "b"]
    assert first is not second
    assert calls == [("dummy", "science", "train")]


def test_collect_feature_sets_applies_intersection_alignment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    feature_a_train = tmp_path / "params_a" / "train.pkl"
    feature_a_test = tmp_path / "params_a" / "test.pkl"
    feature_b_train = tmp_path / "params_b" / "train.pkl"
    feature_b_test = tmp_path / "params_b" / "test.pkl"
    for path in [feature_a_train, feature_a_test, feature_b_train, feature_b_test]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    spec_a = type(
        "Spec",
        (),
        {
            "model_key": "ctm",
            "display_name": "Contextual TM",
            "train_loader": lambda self, _path: np.asarray([[1.0], [2.0], [3.0]]),
            "test_loader": lambda self, _path: np.asarray([[10.0], [11.0], [12.0]]),
            "available_index_resolver": lambda self, *_args: (
                SplitAlignment(
                    raw_indices=np.asarray([0, 1, 2, 3]),
                    available_indices=np.asarray([0, 1, 3]),
                ),
                SplitAlignment(
                    raw_indices=np.asarray([0, 1, 2]),
                    available_indices=np.asarray([0, 1, 2]),
                ),
            ),
        },
    )()
    spec_b = type(
        "Spec",
        (),
        {
            "model_key": "bleilda",
            "display_name": "Blei LDA",
            "train_loader": lambda self, _path: np.asarray([[5.0], [6.0]]),
            "test_loader": lambda self, _path: np.asarray([[20.0], [21.0]]),
            "available_index_resolver": lambda self, *_args: (
                SplitAlignment(
                    raw_indices=np.asarray([0, 1, 2, 3]),
                    available_indices=np.asarray([1, 3]),
                ),
                SplitAlignment(
                    raw_indices=np.asarray([0, 1, 2]),
                    available_indices=np.asarray([1, 2]),
                ),
            ),
        },
    )()

    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.iter_available_features",
        lambda **kwargs: [
            (spec_a, feature_a_train, feature_a_test),
            (spec_b, feature_b_train, feature_b_test),
        ],
    )

    feature_sets, aligned_train_y, aligned_test_y, coverage = collect_feature_sets(
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="science",
        vmf_assignment="hard",
        label_bundle=ClassificationLabelBundle(
            category_labels=["a", "b"],
            label_map={"a": 0, "b": 1},
            train_y=[0, 1, 0, 1],
            test_y=[1, 0, 1],
            train_indices=None,
            train_source_indices=np.asarray([0, 1, 2, 3]),
            test_source_indices=np.asarray([0, 1, 2]),
            raw_train_count=4,
            raw_test_count=3,
            label_filtered_train_count=4,
            label_filtered_test_count=3,
        ),
    )

    assert aligned_train_y == [1, 1]
    assert aligned_test_y == [0, 1]
    assert [feature.train_x[:, 0].tolist() for feature in feature_sets] == [
        [2.0, 3.0],
        [5.0, 6.0],
    ]
    assert [feature.test_x[:, 0].tolist() for feature in feature_sets] == [
        [11.0, 12.0],
        [20.0, 21.0],
    ]
    assert coverage["common_train_docs"] == 2
    assert coverage["available_train_docs"] == {
        "Contextual TM": 3,
        "Blei LDA": 2,
    }


def test_collect_feature_sets_filters_selected_models_by_short_label(
    monkeypatch,
    tmp_path: Path,
) -> None:
    feature_a_train = tmp_path / "params_a" / "train.pkl"
    feature_a_test = tmp_path / "params_a" / "test.pkl"
    feature_b_train = tmp_path / "params_b" / "train.pkl"
    feature_b_test = tmp_path / "params_b" / "test.pkl"
    for path in [feature_a_train, feature_a_test, feature_b_train, feature_b_test]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    spec_gslda = type(
        "Spec",
        (),
        {
            "model_key": "sentence_gaussianlda",
            "display_name": "Sentence LDA",
            "train_loader": lambda self, _path: np.asarray([[1.0], [2.0]]),
            "test_loader": lambda self, _path: np.asarray([[10.0], [11.0]]),
            "available_index_resolver": lambda self, *_args: (
                SplitAlignment(
                    raw_indices=np.asarray([0, 1]),
                    available_indices=np.asarray([0, 1]),
                ),
                SplitAlignment(
                    raw_indices=np.asarray([0, 1]),
                    available_indices=np.asarray([0, 1]),
                ),
            ),
        },
    )()
    spec_lda = type(
        "Spec",
        (),
        {
            "model_key": "bleilda",
            "display_name": "Blei LDA",
            "train_loader": lambda self, _path: np.asarray([[5.0], [6.0]]),
            "test_loader": lambda self, _path: np.asarray([[20.0], [21.0]]),
            "available_index_resolver": lambda self, *_args: (
                SplitAlignment(
                    raw_indices=np.asarray([0, 1]),
                    available_indices=np.asarray([0, 1]),
                ),
                SplitAlignment(
                    raw_indices=np.asarray([0, 1]),
                    available_indices=np.asarray([0, 1]),
                ),
            ),
        },
    )()

    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.iter_available_features",
        lambda **kwargs: [
            (spec_gslda, feature_a_train, feature_a_test),
            (spec_lda, feature_b_train, feature_b_test),
        ],
    )

    feature_sets, aligned_train_y, aligned_test_y, coverage = collect_feature_sets(
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="science",
        vmf_assignment="hard",
        label_bundle=ClassificationLabelBundle(
            category_labels=["a", "b"],
            label_map={"a": 0, "b": 1},
            train_y=[0, 1],
            test_y=[1, 0],
            train_indices=None,
            train_source_indices=np.asarray([0, 1]),
            test_source_indices=np.asarray([0, 1]),
            raw_train_count=2,
            raw_test_count=2,
            label_filtered_train_count=2,
            label_filtered_test_count=2,
        ),
        selected_models=["GSLDA"],
    )

    assert [feature.name for feature in feature_sets] == ["Sentence LDA"]
    assert aligned_train_y == [0, 1]
    assert aligned_test_y == [1, 0]
    assert coverage["available_train_docs"] == {"Sentence LDA": 2}


def test_collect_feature_sets_strict_skip_preserves_legacy_behavior(
    monkeypatch,
    tmp_path: Path,
) -> None:
    feature_a_train = tmp_path / "params_a" / "train.pkl"
    feature_a_test = tmp_path / "params_a" / "test.pkl"
    feature_b_train = tmp_path / "params_b" / "train.pkl"
    feature_b_test = tmp_path / "params_b" / "test.pkl"
    for path in [feature_a_train, feature_a_test, feature_b_train, feature_b_test]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    spec_ok = type(
        "Spec",
        (),
        {
            "model_key": "vmf_sentence_lda",
            "display_name": "vMF Sentence LDA",
            "train_loader": lambda self, _path: np.asarray([[1.0], [2.0], [3.0]]),
            "test_loader": lambda self, _path: np.asarray([[4.0], [5.0]]),
            "available_index_resolver": lambda self, *_args: (
                SplitAlignment(
                    raw_indices=np.asarray([0, 1, 2]),
                    available_indices=np.asarray([0, 1, 2]),
                ),
                SplitAlignment(
                    raw_indices=np.asarray([0, 1]),
                    available_indices=np.asarray([0, 1]),
                ),
            ),
        },
    )()
    spec_skip = type(
        "Spec",
        (),
        {
            "model_key": "ctm",
            "display_name": "Contextual TM",
            "train_loader": lambda self, _path: np.asarray([[9.0], [10.0]]),
            "test_loader": lambda self, _path: np.asarray([[11.0], [12.0]]),
            "available_index_resolver": lambda self, *_args: (
                SplitAlignment(
                    raw_indices=np.asarray([0, 1, 2]),
                    available_indices=np.asarray([0, 1]),
                ),
                SplitAlignment(
                    raw_indices=np.asarray([0, 1]),
                    available_indices=np.asarray([0, 1]),
                ),
            ),
        },
    )()

    monkeypatch.setattr(
        "src.evaluation.classification.pipeline.iter_available_features",
        lambda **kwargs: [
            (spec_ok, feature_a_train, feature_a_test),
            (spec_skip, feature_b_train, feature_b_test),
        ],
    )

    feature_sets, aligned_train_y, aligned_test_y, coverage = collect_feature_sets(
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="science",
        vmf_assignment="hard",
        label_bundle=ClassificationLabelBundle(
            category_labels=["a", "b"],
            label_map={"a": 0, "b": 1},
            train_y=[0, 1, 0],
            test_y=[1, 0],
            train_indices=None,
            train_source_indices=np.asarray([0, 1, 2]),
            test_source_indices=np.asarray([0, 1]),
            raw_train_count=3,
            raw_test_count=2,
            label_filtered_train_count=3,
            label_filtered_test_count=2,
        ),
        alignment_mode="strict_skip",
    )

    assert [feature.name for feature in feature_sets] == ["vMF Sentence LDA"]
    assert aligned_train_y == [0, 1, 0]
    assert aligned_test_y == [1, 0]
    assert coverage["common_train_docs"] == 3

from __future__ import annotations

import logging
from pathlib import Path

from src.core.artifacts import load_json
from src.core.paths_roots import resolve_project_path
from src.evaluation.classification.workflow import (
    ClassificationCondition,
    EvaluationWriteSpec,
    build_classification_archive_dir_from_condition,
    build_classification_condition_id,
    build_classification_condition_payload,
    build_classification_latest_dir,
    build_classification_meta,
    build_classification_output_dir,
    build_classification_write_spec,
    run_classification_grid,
)
from src.evaluation.reporting import read_evaluation_json


def test_run_classification_grid_logs_unknown_dataset(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        run_classification_grid(
            iterations=[0],
            datasets=["missing_dataset"],
            topics=[10],
            classifiers=["svm"],
            vmf_assignment="hard",
            target_column="target_str",
            label_schema="identity",
            write_spec_builder=lambda iteration, dataset, topics: None,  # pragma: no cover
            train_runner=lambda *args, **kwargs: None,
        )

    assert "unknown dataset missing_dataset" in caplog.text


def test_run_classification_grid_persists_feature_catalog_in_meta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.classification.workflow.resolve_dataset_name",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(
        "src.evaluation.classification.workflow.get_dataset_targets",
        lambda dataset, **kwargs: {"science": ["a", "b"]},
    )

    run_classification_grid(
        iterations=[0],
        datasets=["dummy"],
        topics=[10],
        classifiers=["svm"],
        vmf_assignment="hard",
        target_column="target_str",
        label_schema="identity",
        write_spec_builder=lambda iteration, dataset, topics: EvaluationWriteSpec(
            output_dir=tmp_path / f"iter{iteration}",
            acc_filename=f"acc_{dataset}_{topics}topic.json",
            f1_filename=f"f1_{dataset}_{topics}topic.json",
            feature_filename=f"feat_{dataset}_{topics}topic.json",
            meta={"task": "classification", "categories": {}},
        ),
        train_runner=lambda *args, **kwargs: (
            {"Contextual TM [num_epochs=12] [SVM]": 80.0},
            {
                "macro": {"Contextual TM [num_epochs=12] [SVM]": 80.0},
                "micro": {"Contextual TM [num_epochs=12] [SVM]": 80.0},
            },
            {"Contextual TM [num_epochs=12] [SVM]": {}},
            [
                {
                    "feature_name": "Contextual TM [num_epochs=12]",
                    "runner_family": "ctm",
                    "parameter_variant": "num_epochs=12",
                }
            ],
            {
                "common_train_docs": 12,
                "common_test_docs": 6,
                "available_train_docs": {"Contextual TM [num_epochs=12]": 12},
                "available_test_docs": {"Contextual TM [num_epochs=12]": 6},
            },
        ),
    )

    meta, _results = read_evaluation_json(tmp_path / "iter0" / "acc_dummy_10topic.json")

    assert meta["categories"]["science"]["feature_catalog"] == [
        {
            "feature_name": "Contextual TM [num_epochs=12]",
            "runner_family": "ctm",
            "parameter_variant": "num_epochs=12",
        }
    ]
    assert meta["categories"]["science"]["coverage"]["common_train_docs"] == 12


def test_run_classification_grid_retries_single_class_limited_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.classification.workflow.resolve_dataset_name",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(
        "src.evaluation.classification.workflow.get_dataset_targets",
        lambda dataset, **kwargs: {"science": ["a", "b"]},
    )

    attempts: list[int] = []

    def _train_index_resolver(
        _dataset: str,
        _category: str,
        _iteration: int,
        _topics: int,
        retry_attempt: int,
    ):
        attempts.append(retry_attempt)
        if retry_attempt >= 2:
            return [], {"skip_reason": "sampling retry attempts exhausted"}
        return [retry_attempt], {"sampling_retry_attempt": retry_attempt}

    def _train_runner(*_args, train_indices, **_kwargs):
        if train_indices == [0]:
            raise ValueError(
                "This solver needs samples of at least 2 classes in the data, "
                "but the data contains only one class: np.int64(1)"
            )
        return (
            {"ModelA": 80.0},
            {"macro": {"ModelA": 80.0}, "micro": {"ModelA": 80.0}},
            {"ModelA": {}},
            [{"feature_name": "ModelA"}],
            {"common_train_docs": 3, "common_test_docs": 2},
        )

    run_classification_grid(
        iterations=[0],
        datasets=["dummy"],
        topics=[10],
        classifiers=["svm"],
        vmf_assignment="hard",
        target_column="target_str",
        label_schema="identity",
        write_spec_builder=lambda iteration, dataset, topics: EvaluationWriteSpec(
            output_dir=tmp_path / f"iter{iteration}",
            acc_filename=f"acc_{dataset}_{topics}topic.json",
            f1_filename=f"f1_{dataset}_{topics}topic.json",
            feature_filename=f"feat_{dataset}_{topics}topic.json",
            meta={"task": "classification", "categories": {}},
        ),
        train_runner=_train_runner,
        train_index_resolver=_train_index_resolver,
    )

    meta, results = read_evaluation_json(tmp_path / "iter0" / "acc_dummy_10topic.json")

    assert attempts == [0, 1]
    assert results == {"science": {"ModelA": 80.0}}
    assert meta["categories"]["science"]["sampling_retry_attempt"] == 1


def test_run_classification_grid_respects_category_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.classification.workflow.resolve_dataset_name",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(
        "src.evaluation.classification.workflow.get_dataset_targets",
        lambda dataset, **kwargs: {
            "science": ["a"],
            "sports": ["b"],
            "all": ["a", "b"],
        },
    )

    seen_categories: list[str] = []

    def _train_runner(category, *_args, **_kwargs):
        seen_categories.append(category)
        return (
            {"ModelA": 80.0},
            {"macro": {"ModelA": 80.0}, "micro": {"ModelA": 80.0}},
            {"ModelA": {}},
            [{"feature_name": "ModelA"}],
            {"common_train_docs": 3, "common_test_docs": 2},
        )

    run_classification_grid(
        iterations=[0],
        datasets=["dummy"],
        categories=["science", "sports"],
        topics=[10],
        classifiers=["svm"],
        vmf_assignment="hard",
        target_column="target_str",
        label_schema="identity",
        write_spec_builder=lambda iteration, dataset, topics: EvaluationWriteSpec(
            output_dir=tmp_path / f"iter{iteration}",
            acc_filename=f"acc_{dataset}_{topics}topic.json",
            f1_filename=f"f1_{dataset}_{topics}topic.json",
            feature_filename=f"feat_{dataset}_{topics}topic.json",
            meta={"task": "classification", "categories": {}},
        ),
        train_runner=_train_runner,
    )

    assert seen_categories == ["science", "sports"]


def test_build_classification_output_dir_uses_dataset_data_run_and_condition_id(
    tmp_path: Path,
) -> None:
    condition_id, _ = build_classification_condition_id(
        dataset="dummy",
        data_run="fy2024",
        topics=10,
        iteration=0,
        classifiers=["svm"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
    )

    out_dir = build_classification_output_dir(
        result_root=tmp_path,
        dataset="dummy",
        data_run="fy2024",
        condition_id=condition_id,
    )

    assert out_dir == tmp_path / "dummy" / "fy2024" / "all" / condition_id


def test_build_classification_write_spec_uses_archive_and_latest_layout(
    tmp_path: Path,
) -> None:
    condition = ClassificationCondition(
        dataset="dummy",
        data_run="fy2024",
        topics=10,
        iteration=0,
        classifiers=["svm"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
    )

    spec = build_classification_write_spec(
        result_root=tmp_path,
        condition=condition,
        acc_filename="acc_dummy_10topic.json",
        f1_filename="f1_dummy_10topic.json",
        feature_filename="feat_dummy_10topic.json",
    )

    assert spec.output_dir == build_classification_archive_dir_from_condition(
        result_root=tmp_path,
        condition=condition,
        started_at=str(spec.started_at),
        execution_id=str(spec.execution_id),
    )
    assert spec.display_key == "svm_soft_k10_it0"
    assert spec.meta["display_key"] == "svm_soft_k10_it0"
    assert spec.meta["latest_dir"] == str(
        build_classification_latest_dir(
            result_root=tmp_path,
            dataset="dummy",
            data_run="fy2024",
            display_key="svm_soft_k10_it0",
        )
    )


def test_limited_classification_condition_records_sampling_repeat() -> None:
    condition = ClassificationCondition(
        dataset="dummy",
        data_run="fy2024",
        topics=10,
        iteration=2,
        classifiers=["svm"],
        vmf_assignment="hard",
        target_column="target_str",
        label_schema="identity",
        mode="ratio",
        value=0.05,
        stratified=True,
        sampling_repeat=3,
    )

    assert condition.display_key() == "svm_hard_ratio_0-05_stratified_sample-r3_k10_it2"
    condition_id, _ = condition.condition_id()
    assert condition_id.startswith("it2__k10__hard__ratio__0-05__sample-r3__")
    meta = condition.meta(seed=3044)
    assert meta["sampling_repeat"] == 3
    assert meta["seed"] == 3044
    assert meta["sampling_seed"] == 3044


def test_run_classification_grid_writes_latest_pointer_in_new_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.classification.workflow.resolve_dataset_name",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(
        "src.evaluation.classification.workflow.get_dataset_targets",
        lambda dataset, **kwargs: {"science": ["a", "b"]},
    )

    condition = ClassificationCondition(
        dataset="dummy",
        data_run="fy2024",
        topics=10,
        iteration=0,
        classifiers=["svm"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
    )

    run_classification_grid(
        iterations=[0],
        datasets=["dummy"],
        data_run="fy2024",
        topics=[10],
        classifiers=["svm"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
        write_spec_builder=lambda iteration, dataset, topics: build_classification_write_spec(
            result_root=tmp_path,
            condition=condition,
            acc_filename=f"acc_{dataset}_{topics}topic.json",
            f1_filename=f"f1_{dataset}_{topics}topic.json",
            feature_filename=f"feat_{dataset}_{topics}topic.json",
        ),
        train_runner=lambda *args, **kwargs: (
            {"ModelA": 80.0},
            {"macro": {"ModelA": 80.0}, "micro": {"ModelA": 80.0}},
            {"ModelA": {}},
            [{"feature_name": "ModelA"}],
            {"common_train_docs": 3, "common_test_docs": 2},
        ),
    )

    latest_dir = build_classification_latest_dir(
        result_root=tmp_path,
        dataset="dummy",
        data_run="fy2024",
        display_key="svm_soft_k10_it0",
    )
    pointer_path = latest_dir / "CURRENT.json"
    assert pointer_path.exists()
    pointer = load_json(pointer_path)

    assert pointer["task"] == "classification"
    assert pointer["display_key"] == "svm_soft_k10_it0"
    assert pointer["dataset"] == "dummy"
    assert pointer["data_run"] == "fy2024"
    assert pointer["category"] == "all"
    assert pointer["artifacts"] == {
        "acc": "acc_dummy_10topic.json",
        "f1": "f1_dummy_10topic.json",
        "feature": "feature/feat_dummy_10topic.json",
    }

    archive_dir = resolve_project_path(pointer["archive_dir"])
    assert archive_dir.exists()

    acc_meta, acc_results = read_evaluation_json(archive_dir / "acc_dummy_10topic.json")
    assert acc_meta["display_key"] == "svm_soft_k10_it0"
    assert acc_results == {"science": {"ModelA": 80.0}}
    assert (archive_dir / "f1_dummy_10topic.json").exists()
    assert (archive_dir / "feature" / "feat_dummy_10topic.json").exists()


def test_build_classification_meta_records_data_run_and_condition_identity() -> None:
    meta = build_classification_meta(
        dataset="dummy",
        data_run="fy2024",
        topics=10,
        iteration=0,
        classifiers=["svm"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
    )

    assert meta["data_run"] == "fy2024"
    assert meta["vmf_assignment"] == "soft"
    assert meta["alignment_mode"] == "intersection"
    assert meta["condition_id"].startswith("it0__k10__soft__")


def test_classification_condition_id_differs_for_hard_and_soft_assignment() -> None:
    hard_id, _ = build_classification_condition_id(
        dataset="dummy",
        data_run="default",
        topics=10,
        iteration=0,
        classifiers=["svm"],
        vmf_assignment="hard",
        target_column="target_str",
        label_schema="identity",
    )
    soft_id, _ = build_classification_condition_id(
        dataset="dummy",
        data_run="default",
        topics=10,
        iteration=0,
        classifiers=["svm"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
    )

    assert hard_id != soft_id


def test_build_classification_meta_uses_same_condition_id_as_helper() -> None:
    condition_id, condition_fingerprint = build_classification_condition_id(
        dataset="dummy",
        data_run="fy2024",
        topics=10,
        iteration=2,
        classifiers=["svm", "logreg"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
        mode="ratio",
        value=0.3,
        stratified=True,
    )

    meta = build_classification_meta(
        dataset="dummy",
        data_run="fy2024",
        topics=10,
        iteration=2,
        classifiers=["svm", "logreg"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
        mode="ratio",
        value=0.3,
        stratified=True,
    )

    assert meta["condition_id"] == condition_id
    assert meta["condition_fingerprint"] == condition_fingerprint
    assert meta["mode"] == "ratio"
    assert meta["value"] == 0.3
    assert meta["stratified"] is True


def test_build_classification_condition_payload_sorts_classifiers() -> None:
    payload = build_classification_condition_payload(
        dataset="dummy",
        data_run="default",
        topics=10,
        iteration=0,
        classifiers=["svm", "logreg"],
        vmf_assignment="hard",
        target_column="target_str",
        label_schema="identity",
    )

    assert payload["classifiers"] == ["logreg", "svm"]
    assert payload["alignment_mode"] == "intersection"

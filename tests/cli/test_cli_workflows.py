from __future__ import annotations

from pathlib import Path

import yaml

from src.cli.workflows import (
    run_all_experiments_workflow,
    run_evaluation_from_config_workflow,
)


def test_run_all_experiments_workflow_does_not_run_evaluation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10, 20], "num_iterations": 3},
        "experiments": {"iterations": [0, 1]},
        "evaluation": {"classifiers": ["svm", "logreg"]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: dict[str, object] = {"run_comparison_calls": 0}

    def _fake_run_comparison(**_kwargs) -> Path:
        captured["run_comparison_calls"] = int(captured["run_comparison_calls"]) + 1
        return Path("results/experiments/dummy/summary.json")

    monkeypatch.setattr("src.cli.workflows.run_comparison", _fake_run_comparison)
    monkeypatch.setattr(
        "src.evaluation.registry.run_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("run_task must not be called")
        ),
    )

    run_all_experiments_workflow(
        configs=[config_path],
        models="vmf_sentence_lda",
        seed_base=42,
        num_workers=1,
        vmf_soft_temp=1.0,
        include_all_category_runs=False,
        all_category_topics=[50],
        all_category_iterations=[0],
    )

    assert captured["run_comparison_calls"] == 1


def test_run_evaluation_from_config_workflow_uses_config_tasks_and_classifiers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10, 20], "num_iterations": 3},
        "experiments": {"iterations": [0, 1]},
        "evaluation": {
            "tasks": ["classification"],
            "classifiers": ["svm", "logreg"],
            "embedding_variants": ["mpnet"],
            "feature_resolve_mode": "strict",
        },
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured.append((task_name, kwargs))

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    run_evaluation_from_config_workflow(
        config=config_path,
        task=None,
        classifiers=[],
        vmf_assignment="hard",
        result_root=tmp_path / "classification",
        target_column=None,
        label_schema="identity",
    )

    assert len(captured) == 1
    assert captured[0][0] == "classification"
    assert captured[0][1]["classifiers"] == ["svm", "logreg"]
    assert captured[0][1]["categories"] == ["all"]
    assert captured[0][1]["topics"] == [10, 20]
    assert captured[0][1]["iterations"] == [0, 1]
    assert captured[0][1]["embedding_variants"] == ["mpnet"]
    assert captured[0][1]["feature_resolve_mode"] == "strict"


def test_run_evaluation_from_config_workflow_passes_explicit_categories_to_classification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"science": ["sci.space"], "sports": ["rec.sport.baseball"]},
        },
        "train": {"num_topics": [10], "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "evaluation": {
            "tasks": ["classification"],
            "classifiers": ["svm"],
        },
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured.append((task_name, kwargs))

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    run_evaluation_from_config_workflow(
        config=config_path,
        task=None,
        classifiers=[],
        vmf_assignment="hard",
        result_root=tmp_path / "classification",
        target_column=None,
        label_schema="identity",
    )

    assert len(captured) == 1
    assert captured[0][0] == "classification"
    assert captured[0][1]["categories"] == ["science", "sports"]


def test_run_evaluation_from_config_workflow_cli_embedding_options_override_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10], "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "evaluation": {
            "tasks": ["classification"],
            "classifiers": ["svm"],
            "embedding_variants": ["e5"],
            "feature_resolve_mode": "all",
        },
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured.append((task_name, kwargs))

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    run_evaluation_from_config_workflow(
        config=config_path,
        task=None,
        classifiers=[],
        vmf_assignment="hard",
        result_root=tmp_path / "classification",
        target_column=None,
        label_schema="identity",
        embedding_variants=["mpnet"],
        feature_resolve_mode="strict",
    )

    assert len(captured) == 1
    assert captured[0][1]["embedding_variants"] == ["mpnet"]
    assert captured[0][1]["feature_resolve_mode"] == "strict"


def test_run_evaluation_from_config_workflow_cli_task_overrides_config_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10], "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "evaluation": {
            "tasks": ["word-based-metrics"],
            "classifiers": ["svm"],
        },
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured.append((task_name, kwargs))

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    run_evaluation_from_config_workflow(
        config=config_path,
        task="classification",
        classifiers=["logreg"],
        vmf_assignment="soft",
        result_root=tmp_path / "classification",
        target_column="label",
        label_schema="custom",
    )

    assert len(captured) == 1
    assert captured[0][0] == "classification"
    assert captured[0][1]["classifiers"] == ["logreg"]
    assert captured[0][1]["vmf_assignment"] == "soft"
    assert captured[0][1]["target_column"] == "label"


def test_run_evaluation_from_config_workflow_rejects_unsupported_tasks(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10], "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "evaluation": {"tasks": ["word_based_label_profile"]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    try:
        run_evaluation_from_config_workflow(
            config=config_path,
            task=None,
            classifiers=[],
            vmf_assignment="hard",
            result_root=tmp_path / "classification",
            target_column=None,
            label_schema="identity",
        )
    except ValueError as exc:
        assert "not supported by run-from-config" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_run_evaluation_from_config_workflow_uses_configured_baselines_for_word_based_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10], "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "evaluation": {"tasks": ["word_based_metrics"]},
        "baselines": [
            {"runner": "ctm", "params": {}},
            {"runner": "sentence_gaussianlda", "params": {}},
        ],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured.append((task_name, kwargs))

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    run_evaluation_from_config_workflow(
        config=config_path,
        task=None,
        classifiers=[],
        vmf_assignment="hard",
        result_root=tmp_path / "classification",
        target_column=None,
        label_schema="identity",
    )

    assert len(captured) == 1
    assert captured[0][0] == "word_based_metrics"
    assert captured[0][1]["models"] == ["vmf", "ctm", "sentence_gaussianlda"]
    assert captured[0][1]["data_runs"] == ["default"]


def test_run_evaluation_from_config_workflow_uses_supported_models_for_geometry_based_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10], "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "evaluation": {"tasks": ["geometry_based_metrics"]},
        "baselines": [
            {"runner": "ctm", "params": {}},
            {"runner": "sentence_gaussianlda", "params": {}},
        ],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured.append((task_name, kwargs))

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    run_evaluation_from_config_workflow(
        config=config_path,
        task=None,
        classifiers=[],
        vmf_assignment="hard",
        result_root=tmp_path / "classification",
        target_column=None,
        label_schema="identity",
    )

    assert len(captured) == 1
    assert captured[0][0] == "geometry_based_metrics"
    assert captured[0][1]["models"] == ["vmf", "gaussian"]
    assert captured[0][1]["data_runs"] == ["default"]
    assert captured[0][1]["embedding_variant"] == "mpnet"
    assert captured[0][1]["encoder_model"] == "sentence-transformers/all-mpnet-base-v2"


def test_run_evaluation_from_config_workflow_rejects_legacy_task_aliases(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10], "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "evaluation": {"tasks": ["topic-coherence"]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    try:
        run_evaluation_from_config_workflow(
            config=config_path,
            task=None,
            classifiers=[],
            vmf_assignment="hard",
            result_root=tmp_path / "classification",
            target_column=None,
            label_schema="identity",
        )
    except ValueError as exc:
        assert "Use 'word_based_metrics' instead" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")


def test_run_evaluation_from_config_workflow_dispatches_topic_count_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"science": ["sci.space"], "all": None},
        },
        "train": {"num_topics": [10, 20], "num_iterations": 3},
        "experiments": {"iterations": [0, 1]},
        "evaluation": {"tasks": ["topic_count_diagnostics"]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured.append((task_name, kwargs))

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    run_evaluation_from_config_workflow(
        config=config_path,
        task=None,
        classifiers=[],
        vmf_assignment="hard",
        result_root=tmp_path / "classification",
        target_column=None,
        label_schema="identity",
    )

    assert len(captured) == 1
    assert captured[0][0] == "topic_count_diagnostics"
    assert captured[0][1]["dataset"] == "dummy"
    assert captured[0][1]["topics"] == [10, 20]
    assert captured[0][1]["iterations"] == [0, 1]
    assert set(captured[0][1]["categories"]) == {"science", "all"}
    assert captured[0][1]["data_runs"] == ["default"]


def test_run_evaluation_from_config_workflow_dispatches_classification_summary_per_topic_and_data_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "by_fy_root": str(tmp_path / "by_fy"),
            "fiscal_years": [2023, 2024],
            "fiscal_year_mode": "per_year",
            "categories": {"all": None},
        },
        "train": {"num_topics": [10, 20], "num_iterations": 3},
        "experiments": {"iterations": [0, 1]},
        "evaluation": {
            "tasks": ["classification_summary"],
            "classifiers": ["svm"],
        },
        "baselines": [],
        "output_root": "results/experiments",
    }
    for year in (2023, 2024):
        fy_dir = tmp_path / "by_fy" / f"fy{year}"
        fy_dir.mkdir(parents=True)
        (fy_dir / "train.csv").write_text("data,target_str\nx,a\n", encoding="utf-8")
        (fy_dir / "test.csv").write_text("data,target_str\ny,a\n", encoding="utf-8")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    captured: list[tuple[str, dict[str, object]]] = []

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured.append((task_name, kwargs))

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    run_evaluation_from_config_workflow(
        config=config_path,
        task=None,
        classifiers=[],
        vmf_assignment="hard",
        result_root=tmp_path / "classification",
        target_column=None,
        label_schema="identity",
    )

    assert len(captured) == 4
    assert all(task_name == "classification_summary" for task_name, _kwargs in captured)
    assert {(item[1]["topics"], item[1]["data_run"]) for item in captured} == {
        (10, "fy2023"),
        (10, "fy2024"),
        (20, "fy2023"),
        (20, "fy2024"),
    }

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.app import app
from src.cli.data_commands import _default_audit_review_output_path

runner = CliRunner()


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["experiments", "--help"],
        ["experiments", "run", "--help"],
        ["experiments", "smoke", "--help"],
        ["experiments", "run-all", "--help"],
        ["data", "--help"],
        ["data", "prepare-nyt", "--help"],
        ["data", "audit-preprocessing", "--help"],
        ["evaluation", "--help"],
        ["evaluation", "classify", "--help"],
        ["evaluation", "classify-limited", "--help"],
        ["evaluation", "summarize-classification", "--help"],
        ["evaluation", "run-from-config", "--help"],
        ["evaluation", "list-tasks", "--help"],
        ["evaluation", "word-based-metrics", "--help"],
        ["evaluation", "word-based-topic-word-table", "--help"],
        ["evaluation", "word-based-label-profile", "--help"],
        ["evaluation", "geometry-based-metrics", "--help"],
        ["evaluation", "topic-count-diagnostics", "--help"],
        ["evaluation", "cross-model-pair-diagnostics", "--help"],
        ["evaluation", "sentence-topic-inspection", "--help"],
    ],
)
def test_cli_help_commands_are_registered(argv: list[str]) -> None:
    result = runner.invoke(app, argv)
    assert result.exit_code == 0


def test_root_help_lists_top_level_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "experiments" in result.stdout
    assert "evaluation" in result.stdout
    assert "data" in result.stdout


def test_default_audit_review_output_path_uses_dataset_and_split() -> None:
    path = _default_audit_review_output_path(
        Path("data/20newsgroup/test.csv"),
        Path("scripts/audit_review"),
    )

    assert path == Path("scripts/audit_review/20newsgroup_test_audit_review.csv")


def test_experiments_run_uses_default_seed_base(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset: {}\ntrain: {}\nexperiments: {}\nbaselines: []\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _fake_run_experiments_workflow(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.cli.workflows.run_experiments_workflow",
        _fake_run_experiments_workflow,
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "run",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["config"] == config_path
    assert captured["seed"] is None
    assert captured["seed_base"] is None


def test_experiments_run_dispatches_to_comparison_runner(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_experiments_workflow(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.cli.workflows.run_experiments_workflow",
        _fake_run_experiments_workflow,
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "run",
            "--config",
            str(config_path),
            "--category",
            "all",
            "--topic",
            "20",
            "--iteration",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"] == config_path
    assert captured["categories"] == ["all"]
    assert captured["topics"] == [20]
    assert captured["iterations"] == [0]


def test_experiments_run_passes_none_for_empty_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_experiments_workflow(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.cli.workflows.run_experiments_workflow",
        _fake_run_experiments_workflow,
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "run",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["categories"] is None
    assert captured["topics"] is None
    assert captured["iterations"] is None


def test_experiments_run_accepts_underscore_option_aliases(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_experiments_workflow(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.cli.workflows.run_experiments_workflow",
        _fake_run_experiments_workflow,
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "run",
            "--config",
            str(config_path),
            "--num_workers",
            "3",
            "--vmf_soft_temp",
            "0.7",
        ],
    )

    assert result.exit_code == 0
    assert captured["num_workers"] == 3
    assert captured["vmf_soft_temp"] == 0.7


def test_experiments_run_accepts_encoder_overrides(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_experiments_workflow(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.cli.workflows.run_experiments_workflow",
        _fake_run_experiments_workflow,
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "run",
            "--config",
            str(config_path),
            "--encoder-model",
            "baai/bge-base-en-v1.5",
            "--keep-terminal-normalize",
        ],
    )

    assert result.exit_code == 0
    assert captured["encoder_model"] == "baai/bge-base-en-v1.5"
    assert captured["strip_terminal_normalize"] is False


def test_evaluation_classify_limited_requires_ratio_or_count() -> None:
    result = runner.invoke(app, ["evaluation", "classify-limited"])
    assert result.exit_code != 0
    assert "either --ratio or --count is required" in result.output


def test_evaluation_classify_limited_rejects_both_ratio_and_count() -> None:
    result = runner.invoke(
        app,
        ["evaluation", "classify-limited", "--ratio", "0.5", "--count", "10"],
    )
    assert result.exit_code != 0
    assert "use only one of --ratio or --count" in result.output


def test_evaluation_classify_limited_dispatches_categories_to_registry(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.evaluation.registry.register_builtin_tasks", lambda: None)

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured["task_name"] = task_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    result = runner.invoke(
        app,
        [
            "evaluation",
            "classify-limited",
            "--dataset",
            "dummy",
            "--category",
            "science",
            "--topic",
            "20",
            "--iteration",
            "0",
            "--classifier",
            "svm",
            "--ratio",
            "0.2",
            "--sampling-repeat",
            "0",
            "--sampling-repeat",
            "4",
            "--sampling-seed-stride",
            "2000",
            "--embedding-variant",
            "mpnet",
            "--model",
            "GSLDA",
            "--model",
            "vSLDA",
            "--result-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["task_name"] == "classification_limited"
    assert captured["kwargs"]["datasets"] == ["dummy"]
    assert captured["kwargs"]["categories"] == ["science"]
    assert captured["kwargs"]["topics"] == [20]
    assert captured["kwargs"]["iterations"] == [0]
    assert captured["kwargs"]["mode"] == "ratio"
    assert captured["kwargs"]["value"] == 0.2
    assert captured["kwargs"]["sampling_repeats"] == [0, 4]
    assert captured["kwargs"]["sampling_seed_stride"] == 2000
    assert captured["kwargs"]["embedding_variants"] == ["mpnet"]
    assert captured["kwargs"]["selected_models"] == ["GSLDA", "vSLDA"]


def test_experiments_smoke_dispatches_to_smoke_workflow(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_smoke_workflow(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.cli.workflows.run_smoke_workflow", _fake_run_smoke_workflow
    )

    result = runner.invoke(
        app,
        [
            "experiments",
            "smoke",
            "--config",
            str(config_path),
            "--topic",
            "5",
            "--iteration",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"] == config_path
    assert captured["topic"] == [5]
    assert captured["iteration"] == [0]


def test_evaluation_list_tasks_shows_builtin_tasks() -> None:
    result = runner.invoke(app, ["evaluation", "list-tasks"])
    assert result.exit_code == 0
    assert "run_from_config" in result.stdout
    assert "classification" in result.stdout
    assert "classification_limited" in result.stdout
    assert "classification_summary" in result.stdout
    assert "geometry_based_metrics" in result.stdout
    assert "sentence_topic_inspection" in result.stdout
    assert "word_based_metrics" in result.stdout
    assert "topic_count_diagnostics" in result.stdout
    assert "word_based_label_profile" in result.stdout
    assert "word_based_topic_word_table" in result.stdout
    assert "cross_model_pair_diagnostics" in result.stdout


def test_evaluation_classify_dispatches_to_registry(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.evaluation.registry.register_builtin_tasks", lambda: None)

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured["task_name"] = task_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    result = runner.invoke(
        app,
        [
            "evaluation",
            "classify",
            "--dataset",
            "dummy",
            "--category",
            "science",
            "--topic",
            "20",
            "--iteration",
            "0",
            "--classifier",
            "svm",
            "--embedding-variant",
            "mpnet",
            "--model",
            "GSLDA",
            "--model",
            "vSLDA",
            "--feature-resolve-mode",
            "strict",
            "--result-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["task_name"] == "classification"
    assert captured["kwargs"]["datasets"] == ["dummy"]
    assert captured["kwargs"]["categories"] == ["science"]
    assert captured["kwargs"]["topics"] == [20]
    assert captured["kwargs"]["iterations"] == [0]
    assert captured["kwargs"]["embedding_variants"] == ["mpnet"]
    assert captured["kwargs"]["feature_resolve_mode"] == "strict"
    assert captured["kwargs"]["selected_models"] == ["GSLDA", "vSLDA"]


def test_evaluation_summarize_classification_dispatches_to_registry(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.evaluation.registry.register_builtin_tasks", lambda: None)

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured["task_name"] = task_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)
    output_path = tmp_path / "summary.tex"

    result = runner.invoke(
        app,
        [
            "evaluation",
            "summarize-classification",
            "--dataset",
            "dummy",
            "--topic",
            "20",
            "--iteration",
            "0",
            "--embedding-variant",
            "e5",
            "--model",
            "LDA",
            "--model",
            "vSLDA",
            "--exclude-category",
            "science",
            "--include-all-category",
            "--feature-resolve-mode",
            "strict",
            "--output-path",
            str(output_path),
            "--result-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["task_name"] == "classification_summary"
    assert captured["kwargs"]["dataset"] == "dummy"
    assert captured["kwargs"]["topics"] == 20
    assert captured["kwargs"]["iterations"] == [0]
    assert captured["kwargs"]["resolve_mode"] == "latest"
    assert captured["kwargs"]["embedding_variants"] == ["e5"]
    assert captured["kwargs"]["feature_resolve_mode"] == "strict"
    assert captured["kwargs"]["selected_models"] == ["LDA", "vSLDA"]
    assert captured["kwargs"]["excluded_categories"] == ["science"]
    assert captured["kwargs"]["include_all_category"] is True
    assert captured["kwargs"]["output_path"] == output_path


def test_evaluation_run_from_config_dispatches_to_workflow(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset: {}\ntrain: {}\nexperiments: {}\nbaselines: []\n", "utf-8"
    )

    captured: dict[str, object] = {}

    def _fake_run_evaluation_from_config_workflow(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.cli.workflows.run_evaluation_from_config_workflow",
        _fake_run_evaluation_from_config_workflow,
    )

    result = runner.invoke(
        app,
        [
            "evaluation",
            "run-from-config",
            "--config",
            str(config_path),
            "--task",
            "classification",
            "--classifier",
            "svm",
            "--vmf-assignment",
            "soft",
            "--result-root",
            str(tmp_path),
            "--target-column",
            "label",
            "--embedding-variant",
            "mpnet",
            "--feature-resolve-mode",
            "strict",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"] == config_path
    assert captured["task"] == "classification"
    assert captured["classifiers"] == ["svm"]
    assert captured["vmf_assignment"] == "soft"
    assert captured["target_column"] == "label"
    assert captured["embedding_variants"] == ["mpnet"]
    assert captured["feature_resolve_mode"] == "strict"


def test_evaluation_sentence_topic_inspection_dispatches_to_registry(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.evaluation.registry.register_builtin_tasks", lambda: None)

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured["task_name"] = task_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    result = runner.invoke(
        app,
        [
            "evaluation",
            "sentence-topic-inspection",
            "--model",
            "sentlda",
            "--dataset",
            "dummy",
            "--data-run",
            "fy2024",
            "--category",
            "all",
            "--iteration",
            "0",
            "--topic",
            "2",
            "--encoder",
            "fake-model",
            "--embedding-variant",
            "mpnet",
            "--condition-id",
            "it0__k2__abcd1234",
            "--num-components",
            "1",
            "--gaussian-embedding-variant",
            "mpnet_raw",
            "--results-root",
            str(tmp_path / "results"),
            "--out-root",
            str(tmp_path / "viz"),
        ],
    )

    assert result.exit_code == 0
    assert captured["task_name"] == "sentence_topic_inspection"
    assert captured["kwargs"]["model"] == "sentlda"
    assert captured["kwargs"]["dataset"] == "dummy"
    assert captured["kwargs"]["data_run"] == "fy2024"
    assert captured["kwargs"]["categories"] == ["all"]
    assert captured["kwargs"]["iterations"] == [0]
    assert captured["kwargs"]["num_topics_list"] == [2]
    assert captured["kwargs"]["encoder_model"] == "fake-model"
    assert captured["kwargs"]["embedding_variant"] == "mpnet"
    assert captured["kwargs"]["source_condition_id"] == "it0__k2__abcd1234"
    assert captured["kwargs"]["num_components"] == 1
    assert captured["kwargs"]["gaussian_embedding_variant"] == "mpnet_raw"


def test_evaluation_geometry_based_metrics_dispatches_data_runs_to_registry(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.evaluation.registry.register_builtin_tasks", lambda: None)

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured["task_name"] = task_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    result = runner.invoke(
        app,
        [
            "evaluation",
            "geometry-based-metrics",
            "--dataset",
            "dummy",
            "--data-run",
            "fy2024",
            "--topic",
            "20",
            "--iteration",
            "0",
            "--embedding-variant",
            "mpnet",
            "--out-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["task_name"] == "geometry_based_metrics"
    assert captured["kwargs"]["data_runs"] == ["fy2024"]
    assert captured["kwargs"]["embedding_variant"] == "mpnet"


def test_evaluation_word_based_metrics_dispatches_data_runs_to_registry(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.evaluation.registry.register_builtin_tasks", lambda: None)

    def _fake_run_task(task_name: str, **kwargs) -> None:
        captured["task_name"] = task_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr("src.evaluation.registry.run_task", _fake_run_task)

    result = runner.invoke(
        app,
        [
            "evaluation",
            "word-based-metrics",
            "--dataset",
            "dummy",
            "--data-run",
            "fy2024",
            "--topic",
            "20",
            "--iteration",
            "0",
            "--coherence",
            "doc_npmi",
            "--diversity-topn",
            "30",
            "--proxy-word-score-mode",
            "word_npmi",
            "--coherence-reference",
            "wikipedia",
            "--coherence-reference-path",
            str(tmp_path / "wiki.jsonl"),
            "--coherence-window-size",
            "110",
            "--coherence-min-window-count",
            "9",
            "--coherence-count-backend",
            "numba",
            "--coherence-count-workers",
            "4",
            "--coherence-count-chunk-size",
            "17",
            "--coherence-topic-word-workers",
            "3",
            "--coherence-score-workers",
            "2",
            "--out-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["task_name"] == "word_based_metrics"
    assert captured["kwargs"]["data_runs"] == ["fy2024"]
    assert captured["kwargs"]["coherence"] == "doc_npmi"
    assert captured["kwargs"]["diversity_topn"] == 30
    assert captured["kwargs"]["dict_no_below"] == 3
    assert captured["kwargs"]["dict_no_above"] == 0.7
    assert captured["kwargs"]["proxy_word_score_mode"] == "word_npmi"
    assert captured["kwargs"]["coherence_reference"] == "wikipedia"
    assert captured["kwargs"]["coherence_reference_path"] == tmp_path / "wiki.jsonl"
    assert captured["kwargs"]["coherence_window_size"] == 110
    assert captured["kwargs"]["coherence_min_window_count"] == 9
    assert captured["kwargs"]["coherence_count_backend"] == "numba"
    assert captured["kwargs"]["coherence_count_workers"] == 4
    assert captured["kwargs"]["coherence_count_chunk_size"] == 17
    assert captured["kwargs"]["coherence_topic_word_workers"] == 3
    assert captured["kwargs"]["coherence_score_workers"] == 2

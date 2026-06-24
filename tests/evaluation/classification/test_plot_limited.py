from __future__ import annotations

from pathlib import Path

from src.core.artifacts import CURRENT_POINTER_FILENAME, save_json
from src.evaluation.classification.plot_limited import (
    MODEL_ORDER,
    _build_average_category_data,
    _collect_files,
    _collect_legend_models,
    _display_model_name,
    _format_x_tick,
    _load_scores,
    _metric_axis_label,
    _mode_axis_label,
    _model_color,
    _model_sort_key,
    _write_legend_figure,
)
from src.evaluation.reporting import write_evaluation_json


def _write_limited_result(path: Path, *, score: float) -> None:
    write_evaluation_json(
        meta={"task": "classification", "started_at": "2026-01-01T00:00:00Z"},
        results={"computer": {"Contextual TM [mpnet]": score}},
        path=path,
    )


def _write_full_result(path: Path, *, score: float) -> None:
    write_evaluation_json(
        meta={"task": "classification", "started_at": "2026-01-01T00:00:00Z"},
        results={"computer": {"Contextual TM [mpnet]": score}},
        path=path,
    )


def test_collect_files_prefers_latest_pointers_over_archive_history(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "classification"
    current_archive_dir = (
        base_dir
        / "archive"
        / "2026-01-02"
        / "dummy"
        / "default"
        / "all"
        / "k2_it0"
        / "exec_current"
    )
    old_archive_dir = (
        base_dir
        / "archive"
        / "2026-01-01"
        / "dummy"
        / "default"
        / "all"
        / "k2_it0"
        / "exec_old"
    )
    current_result = current_archive_dir / "acc_dummy_2topic_ratio0.5.json"
    old_result = old_archive_dir / "acc_dummy_2topic_ratio0.5.json"
    _write_limited_result(current_result, score=90.0)
    _write_limited_result(old_result, score=10.0)
    save_json(
        {
            "schema": "latest_result_pointer",
            "schema_version": 1,
            "task": "classification",
            "display_key": "k2_it0",
            "dataset": "dummy",
            "data_run": "default",
            "category": "all",
            "archive_dir": str(current_archive_dir),
            "started_at": "2026-01-02T00:00:00Z",
            "execution_id": "exec_current",
            "condition_fingerprint": "fingerprint",
            "artifacts": {"acc": current_result.name},
        },
        base_dir
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / "k2_it0"
        / CURRENT_POINTER_FILENAME,
    )

    assert _collect_files(base_dir) == [current_result]
    assert _collect_files(base_dir, archive_history=True) == [
        old_result,
        current_result,
    ]

    scores = _load_scores(base_dir)
    values = scores["acc"]["dummy"][2]["ratio"]["computer"]["Contextual TM [mpnet]"]
    assert values == {0.5: [90.0]}


def test_load_scores_uses_full_classification_as_ratio_one(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "classification"
    archive_dir = (
        base_dir
        / "archive"
        / "2026-01-01"
        / "dummy"
        / "default"
        / "all"
        / "k2_it0"
        / "exec_full"
    )
    full_result = archive_dir / "acc_dummy_2topic.json"
    _write_full_result(full_result, score=95.0)
    save_json(
        {
            "schema": "latest_result_pointer",
            "schema_version": 1,
            "task": "classification",
            "display_key": "k2_it0",
            "dataset": "dummy",
            "data_run": "default",
            "category": "all",
            "archive_dir": str(archive_dir),
            "started_at": "2026-01-01T00:00:00Z",
            "execution_id": "exec_full",
            "condition_fingerprint": "fingerprint",
            "artifacts": {"acc": full_result.name},
        },
        base_dir
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / "k2_it0"
        / CURRENT_POINTER_FILENAME,
    )

    scores = _load_scores(base_dir)
    values = scores["acc"]["dummy"][2]["ratio"]["computer"]["Contextual TM [mpnet]"]
    assert values == {1.0: [95.0]}


def test_load_scores_collects_sampling_repeat_latest_pointers(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "classification"
    for repeat, score in [(0, 80.0), (1, 90.0)]:
        display_key = f"ratio_0-5_stratified_sample-r{repeat}_k2_it0"
        archive_dir = (
            base_dir
            / "archive"
            / "2026-01-01"
            / "dummy"
            / "default"
            / "all"
            / display_key
            / f"exec_r{repeat}"
        )
        result_path = archive_dir / "acc_dummy_2topic_ratio0.5.json"
        _write_limited_result(result_path, score=score)
        save_json(
            {
                "schema": "latest_result_pointer",
                "schema_version": 1,
                "task": "classification",
                "display_key": display_key,
                "dataset": "dummy",
                "data_run": "default",
                "category": "all",
                "archive_dir": str(archive_dir),
                "started_at": "2026-01-01T00:00:00Z",
                "execution_id": f"exec_r{repeat}",
                "condition_fingerprint": "fingerprint",
                "artifacts": {"acc": result_path.name},
            },
            base_dir
            / "latest"
            / "dummy"
            / "default"
            / "all"
            / display_key
            / CURRENT_POINTER_FILENAME,
        )

    scores = _load_scores(base_dir)
    values = scores["acc"]["dummy"][2]["ratio"]["computer"]["Contextual TM [mpnet]"]
    assert values == {0.5: [80.0, 90.0]}


def test_display_labels_and_sort_order_match_paper_model_order() -> None:
    models = [
        "vMF Sentence LDA [c1_mpnet] [SVM]",
        "Contextual TM [mpnet] [SVM]",
        "Blei LDA [SVM]",
        "MvTM [c1_glove100] [SVM]",
        "ETM [glove100] [SVM]",
        "Sentence LDA [mpnet_raw] [SVM]",
        "sentLDA [SVM]",
        "SenClu [mpnet] [SVM]",
        "Gaussian LDA [glove100] [SVM]",
    ]

    sorted_labels = [
        _display_model_name(model) for model in sorted(models, key=_model_sort_key)
    ]

    assert sorted_labels == [
        "LDA",
        "SLDA",
        "GLDA",
        "vLDA",
        "ETM",
        "CTM",
        "SenClu",
        "GSLDA",
        "vSLDA(proposed)",
    ]


def test_model_colors_are_mechanically_distinct_for_default_order() -> None:
    colors = [_model_color(label, "tab10") for label in MODEL_ORDER]

    assert len(set(colors)) == len(MODEL_ORDER)
    assert _model_color("vSLDA(proposed)", "tab10") == "#d62728"
    assert _model_color("vLDA", "tab10") != _model_color("vSLDA(proposed)", "tab10")


def test_axis_labels_are_paper_friendly() -> None:
    assert _metric_axis_label("acc") == "Accuracy"
    assert _mode_axis_label("ratio") == "Training data used"
    assert _format_x_tick(0.05, "ratio") == "5%"
    assert _format_x_tick(1.0, "ratio") == "100%"


def test_average_category_data_excludes_all_category() -> None:
    mode_data = {
        "computer": {"Model": {0.5: [70.0, 90.0]}},
        "science": {"Model": {0.5: [60.0]}},
        "all": {"Model": {0.5: [10.0]}},
    }

    averaged = _build_average_category_data(mode_data, categories=None)

    assert averaged == {"Model": {0.5: [80.0, 60.0]}}


def test_average_category_data_returns_empty_when_only_all_selected() -> None:
    mode_data = {
        "computer": {"Model": {0.5: [70.0]}},
        "all": {"Model": {0.5: [10.0]}},
    }

    averaged = _build_average_category_data(mode_data, categories=["all"])

    assert averaged == {}


def test_collect_legend_models_and_write_legend_figure(
    tmp_path: Path,
) -> None:
    data = {
        "acc": {
            "dummy": {
                2: {
                    "ratio": {
                        "computer": {
                            "vMF Sentence LDA [c1_mpnet] [SVM]": {
                                0.5: [70.0],
                                1.0: [90.0],
                            },
                            "vMF Sentence LDA [c1_bge] [SVM]": {
                                0.5: [60.0],
                                1.0: [80.0],
                            },
                            "ETM [glove100] [SVM]": {
                                0.5: [50.0],
                                1.0: [75.0],
                            },
                        },
                        "science": {
                            "vMF Sentence LDA [c1_mpnet] [SVM]": {
                                0.5: [65.0],
                                1.0: [85.0],
                            }
                        },
                        "all": {
                            "vMF Sentence LDA [c1_mpnet] [SVM]": {
                                0.5: [1.0],
                                1.0: [2.0],
                            }
                        },
                    }
                }
            }
        }
    }
    outdir = tmp_path / "figures"

    legend_models = _collect_legend_models(
        data,
        metrics=["acc"],
        datasets=["dummy"],
        topics_list=[2],
        modes=["ratio"],
        categories=None,
        models=None,
        include_average=False,
    )
    _write_legend_figure(models=legend_models, outdir=outdir, colormap="tab10")

    assert legend_models == [
        "ETM [glove100] [SVM]",
        "vMF Sentence LDA [c1_bge] [SVM]",
        "vMF Sentence LDA [c1_mpnet] [SVM]",
    ]
    assert (outdir / "legend.png").is_file()

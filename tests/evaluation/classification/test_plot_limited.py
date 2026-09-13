from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pytest
from matplotlib.colors import to_rgb

from src.core.artifacts import CURRENT_POINTER_FILENAME, save_json
from src.evaluation.classification.plot_limited import (
    PAPER_DATASET_CATEGORIES,
    PAPER_GRID_COLS,
    PAPER_GRID_LEGEND_FONTSIZE,
    PAPER_MODELS,
    PAPER_OUTDIR,
    PAPER_TOPICS,
    _build_average_category_data,
    _collect_files,
    _collect_legend_models,
    _display_model_name,
    _format_x_tick,
    _grid_columns,
    _grid_geometry,
    _legend_handles,
    _load_scores,
    _metric_axis_label,
    _mode_axis_label,
    _model_color,
    _model_sort_key,
    _model_style,
    _plot_category,
    _plot_grid,
    _row_major,
    _write_legend_figure,
    main,
    paper_models,
)
from src.evaluation.reporting import write_evaluation_json
from src.evaluation.reports.grid_layout import (
    GRID_NCOLS,
    GRID_TEXT_WIDTH,
    GRID_WSPACE,
)
from src.evaluation.reports.model_style import (
    EXCLUDED_MODELS,
    FAMILY_COLORS,
    MARKER,
    MODEL_MARKERS,
    MODEL_ORDER,
    MODEL_TAXONOMY,
    PROPOSED_LABEL,
)


def _gray_luminance(color: object) -> float:
    r, g, b = to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _write_limited_result(path: Path, *, score: float, **meta: object) -> None:
    write_evaluation_json(
        meta={"task": "classification", "started_at": "2026-01-01T00:00:00Z", **meta},
        results={"computer": {"Contextual TM [mpnet]": score}},
        path=path,
    )


def _write_full_result(path: Path, *, score: float, **meta: object) -> None:
    write_evaluation_json(
        meta={"task": "classification", "started_at": "2026-01-01T00:00:00Z", **meta},
        results={"computer": {"Contextual TM [mpnet]": score}},
        path=path,
    )


def _write_full_pointer(
    base_dir: Path, *, display_key: str, score: float, started_at: str, **meta: object
) -> None:
    archive_dir = (
        base_dir
        / "archive"
        / started_at[:10]
        / "dummy"
        / "default"
        / "all"
        / display_key
    )
    result_path = archive_dir / "acc_dummy_2topic.json"
    _write_full_result(result_path, score=score, started_at=started_at, **meta)
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
            "started_at": started_at,
            "execution_id": f"exec_{display_key}",
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


def test_load_scores_keeps_newest_score_per_iteration_and_repeat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_dir = tmp_path / "classification"
    # Stale pilot batch and the proper per-model batch both describe it0.
    _write_full_pointer(
        base_dir,
        display_key="logreg-svm_models-bundle_k2_it0",
        score=10.0,
        started_at="2026-08-18T06:00:00+00:00",
        iteration=0,
        sampling_repeat=None,
    )
    _write_full_pointer(
        base_dir,
        display_key="logreg_models-ctm_k2_it0",
        score=90.0,
        started_at="2026-08-22T19:00:00+00:00",
        iteration=0,
        sampling_repeat=None,
    )
    _write_full_pointer(
        base_dir,
        display_key="logreg_models-ctm_k2_it1",
        score=80.0,
        started_at="2026-08-22T19:01:00+00:00",
        iteration=1,
        sampling_repeat=None,
    )

    scores = _load_scores(base_dir)
    values = scores["acc"]["dummy"][2]["ratio"]["computer"]["Contextual TM [mpnet]"]
    assert sorted(values[1.0]) == [80.0, 90.0]
    assert "dropped 1 superseded or re-recorded score" in capsys.readouterr().err

    scores = _load_scores(base_dir, keep_duplicates=True)
    values = scores["acc"]["dummy"][2]["ratio"]["computer"]["Contextual TM [mpnet]"]
    assert sorted(values[1.0]) == [10.0, 80.0, 90.0]
    assert capsys.readouterr().err == ""


def test_load_scores_keeps_distinct_conditions_but_collapses_identical_scores(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "classification"
    # Same iteration, different feature_resolve_mode and different score: both stay.
    _write_full_pointer(
        base_dir,
        display_key="logreg_models-ctm_k2_it0",
        score=70.0,
        started_at="2026-07-01T10:00:00+00:00",
        iteration=0,
        feature_resolve_mode="all",
    )
    _write_full_pointer(
        base_dir,
        display_key="logreg_strict-skip_models-ctm_k2_it0",
        score=75.0,
        started_at="2026-07-01T11:00:00+00:00",
        iteration=0,
        feature_resolve_mode="strict-skip",
    )
    # Same iteration, different resolve mode, identical score: a re-recording.
    _write_full_pointer(
        base_dir,
        display_key="logreg-svm_strict_models-ctm_k2_it0",
        score=70.0,
        started_at="2026-06-01T10:00:00+00:00",
        iteration=0,
        feature_resolve_mode="strict",
    )

    scores = _load_scores(base_dir)
    values = scores["acc"]["dummy"][2]["ratio"]["computer"]["Contextual TM [mpnet]"]
    assert sorted(values[1.0]) == [70.0, 75.0]


def test_load_scores_never_merges_files_without_iteration_meta(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "classification"
    _write_full_pointer(
        base_dir,
        display_key="legacy_a_k2_it0",
        score=10.0,
        started_at="2026-08-18T06:00:00+00:00",
    )
    _write_full_pointer(
        base_dir,
        display_key="legacy_b_k2_it0",
        score=90.0,
        started_at="2026-08-22T19:00:00+00:00",
    )

    scores = _load_scores(base_dir)
    values = scores["acc"]["dummy"][2]["ratio"]["computer"]["Contextual TM [mpnet]"]
    assert sorted(values[1.0]) == [10.0, 90.0]


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

    # SenClu is not part of the paper order and sorts after the known models.
    assert sorted_labels == [
        "LDA",
        "SentLDA",
        "GLDA",
        "vLDA",
        "ETM",
        "ConTM",
        "GSLDA",
        "vSLDA (proposed)",
        "SenClu",
    ]
    assert PROPOSED_LABEL == "vSLDA (proposed)"
    assert "SenClu" in EXCLUDED_MODELS


def test_model_styles_follow_taxonomy_and_are_unique() -> None:
    assert set(MODEL_TAXONOMY) == set(MODEL_ORDER)

    styles = {label: _model_style(label, "tab10") for label in MODEL_ORDER}
    cells = {(style.linestyle, style.color) for style in styles.values()}
    assert len(cells) == len(MODEL_ORDER)
    # Marker shapes are unique as well, so series survive grayscale printing.
    assert len({style.marker for style in styles.values()}) == len(MODEL_ORDER)

    # Line style encodes sentence-level assignment; colour encodes the family.
    for label, (sentence_level, family) in MODEL_TAXONOMY.items():
        style = styles[label]
        assert (style.linestyle == "-") is sentence_level
        for other, (_, other_family) in MODEL_TAXONOMY.items():
            assert (styles[other].color == style.color) is (family == other_family)

    # Proposed method: red (vMF family), solid, emphasised.
    proposed = styles[PROPOSED_LABEL]
    assert proposed.color == FAMILY_COLORS["vmf"] == "#D62728"
    assert proposed.linestyle == "-"
    assert proposed.linewidth > styles["GSLDA"].linewidth
    assert proposed.zorder > styles["GSLDA"].zorder
    # Red identifies the proposed method alone.  vLDA and SAM are the other two
    # vMF-family models; they used to share this red *and*, being document level,
    # its dashed counterpart, leaving only the marker to separate them.  They now
    # sit on neighbouring warm hues, so the family still reads as a group.
    assert styles["vLDA"].color != proposed.color
    assert styles["SAM"].color != proposed.color
    assert styles["vLDA"].color != styles["SAM"].color
    assert styles["vLDA"].linestyle != proposed.linestyle
    assert styles["SAM"].linestyle != proposed.linestyle

    # Solid (sentence-level) series stay separable in grayscale.
    solid = sorted(
        _gray_luminance(style.color)
        for style in styles.values()
        if style.linestyle == "-"
    )
    assert all(b - a >= 0.1 for a, b in zip(solid, solid[1:]))


def test_model_color_matches_style_and_falls_back_for_unknown_labels() -> None:
    assert _model_color(PROPOSED_LABEL, "tab10") == "#D62728"
    assert _model_color("vLDA", "tab10") == _model_style("vLDA", "tab10").color

    unknown = _model_style("Some New Model", "tab10")
    assert unknown.linestyle == "-"
    assert _model_color("Some New Model", "tab10") == unknown.color


def test_axis_labels_are_paper_friendly() -> None:
    assert _metric_axis_label("acc") == "Accuracy (%)"
    assert _mode_axis_label("ratio") == "Fraction of labeled documents"
    assert _format_x_tick(0.05, "ratio") == "5%"
    assert _format_x_tick(1.0, "ratio") == "100%"
    assert _format_x_tick(0.05, "ratio", percent_sign=False) == "5"


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
    _write_legend_figure(
        models=legend_models,
        outdir=outdir,
        colormap="tab10",
        formats=("png", "pdf"),
    )
    _write_legend_figure(
        models=legend_models,
        outdir=outdir,
        colormap="tab10",
        filename="legend_h.png",
        ncol=8,
        formats=("png",),
    )

    assert legend_models == [
        "ETM [glove100] [SVM]",
        "vMF Sentence LDA [c1_bge] [SVM]",
        "vMF Sentence LDA [c1_mpnet] [SVM]",
    ]
    assert (outdir / "legend.png").is_file()
    assert (outdir / "legend.pdf").is_file()
    assert (outdir / "legend_h.png").is_file()
    assert not (outdir / "legend_h.pdf").exists()


def _category_data_with_senclu() -> dict:
    return {
        "vMF Sentence LDA [c1_mpnet] [LogReg]": {0.05: [60.0, 62.0], 1.0: [90.0]},
        "SenClu [mpnet] [LogReg]": {0.05: [40.0], 1.0: [70.0]},
        "Blei LDA [LogReg]": {0.05: [50.0], 1.0: [80.0]},
    }


def test_excluded_models_are_not_plotted_or_listed_in_legend(
    tmp_path: Path, monkeypatch
) -> None:
    category_data = _category_data_with_senclu()
    data = {"acc": {"dummy": {2: {"ratio": {"computer": category_data}}}}}

    legend_models = _collect_legend_models(
        data,
        metrics=["acc"],
        datasets=["dummy"],
        topics_list=[2],
        modes=["ratio"],
        categories=None,
        models=None,
        include_average=True,
    )
    assert legend_models == [
        "Blei LDA [LogReg]",
        "vMF Sentence LDA [c1_mpnet] [LogReg]",
    ]

    plotted: list[str] = []
    import matplotlib.axes

    original_plot = matplotlib.axes.Axes.plot

    def _record_plot(self, *args, **kwargs):
        plotted.append(kwargs.get("label"))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _record_plot)
    _plot_category(
        metric="acc",
        dataset="dummy",
        topics=2,
        mode="ratio",
        category="computer",
        category_data=category_data,
        outdir=tmp_path,
        models=None,
        no_errorbar=False,
        ylim=None,
        colormap="tab10",
        formats=("png",),
    )
    assert plotted == ["LDA", "vSLDA (proposed)"]


def test_plot_category_writes_requested_formats_and_label_variants(
    tmp_path: Path, monkeypatch
) -> None:
    category_data = _category_data_with_senclu()
    xlabels: list[str] = []
    ylabels: list[str] = []
    import matplotlib.figure

    original_savefig = matplotlib.figure.Figure.savefig

    def _record_savefig(self, fname, *args, **kwargs):
        ax = self.axes[0]
        xlabels.append(ax.get_xlabel())
        ylabels.append(ax.get_ylabel())
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _record_savefig)

    common = dict(
        metric="acc",
        dataset="dummy",
        topics=2,
        mode="ratio",
        category="computer",
        category_data=category_data,
        outdir=tmp_path,
        models=None,
        no_errorbar=False,
        ylim=None,
        colormap="tab10",
    )
    stem_both = _plot_category(**common, formats=("png", "pdf"))
    stem_nox = _plot_category(**common, show_xlabel=False, formats=("png",))
    stem_noxy = _plot_category(
        **common, show_xlabel=False, show_ylabel=False, formats=("pdf",)
    )

    assert stem_both == tmp_path / "dummy_2topic_acc_ratio_computer"
    assert stem_nox == tmp_path / "dummy_2topic_acc_ratio_computer_nox"
    assert stem_noxy == tmp_path / "dummy_2topic_acc_ratio_computer_noxy"
    assert (tmp_path / "dummy_2topic_acc_ratio_computer.png").is_file()
    assert (tmp_path / "dummy_2topic_acc_ratio_computer.pdf").is_file()
    assert (tmp_path / "dummy_2topic_acc_ratio_computer_nox.png").is_file()
    assert not (tmp_path / "dummy_2topic_acc_ratio_computer_nox.pdf").exists()
    assert (tmp_path / "dummy_2topic_acc_ratio_computer_noxy.pdf").is_file()

    # savefig is called once per format: both(2) + nox(1) + noxy(1).
    assert xlabels == ["Fraction of labeled documents"] * 2 + ["", ""]
    assert ylabels == ["Accuracy (%)"] * 2 + ["Accuracy (%)", ""]


def test_model_markers_are_deterministic_per_model_order() -> None:
    assert list(MODEL_MARKERS) == MODEL_ORDER
    for label in MODEL_ORDER:
        kwargs = _model_style(label, "tab10").line_kwargs()
        assert kwargs["marker"] == MODEL_MARKERS[label]
    assert _model_style(PROPOSED_LABEL, "tab10").marker == "*"
    assert _model_style("Some New Model", "tab10").marker == MARKER


def test_legend_handles_match_model_styles_and_read_row_major() -> None:
    handles = _legend_handles(MODEL_ORDER, "tab10")
    assert [handle.get_label() for handle in handles] == MODEL_ORDER
    for handle, label in zip(handles, MODEL_ORDER):
        style = _model_style(label, "tab10")
        assert handle.get_marker() == style.marker
        assert handle.get_color() == style.color
        assert handle.get_linewidth() == style.linewidth

    assert _row_major(list("ABCDEFGH"), 4) == list("AEBFCGDH")
    assert _row_major(list("ABCDE"), 4) == list("AEBCD")
    assert _row_major(list("AB"), 4) == list("AB")


def test_grid_columns_fill_the_last_row_when_possible() -> None:
    assert _grid_columns(6, 3) == 3
    assert _grid_columns(4, 3) == 2
    assert _grid_columns(5, 3) == 3
    assert _grid_columns(2, 3) == 2
    assert _grid_columns(1, 3) == 1


def test_grid_geometry_matches_text_width_and_keeps_panel_size() -> None:
    (width3, height3), adjust3, legend_h = _grid_geometry(2, GRID_NCOLS, 2)
    (width2, height2), adjust2, _ = _grid_geometry(2, 2, 2)
    assert width3 == pytest.approx(GRID_TEXT_WIDTH)
    assert height2 == pytest.approx(height3)
    assert legend_h > 0
    # Dropping one column removes exactly one panel plus one gap.
    panel_w = adjust3["wspace"] and GRID_WSPACE / adjust3["wspace"]
    assert width3 - width2 == pytest.approx(panel_w + GRID_WSPACE)
    for adjust in (adjust3, adjust2):
        assert 0 < adjust["left"] < adjust["right"] < 1
        assert 0 < adjust["bottom"] < adjust["top"] < 1


def _grid_mode_data() -> dict:
    return {
        "computer": _category_data_with_senclu(),
        "ride": _category_data_with_senclu(),
        "science": _category_data_with_senclu(),
        "all": _category_data_with_senclu(),
        "average": _category_data_with_senclu(),
    }


def test_plot_grid_writes_files_with_edge_labels_and_shared_legend(
    tmp_path: Path, monkeypatch
) -> None:
    import matplotlib.figure

    captured: list = []
    original_savefig = matplotlib.figure.Figure.savefig

    def _record_savefig(self, fname, *args, **kwargs):
        captured.append(self)
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _record_savefig)

    stem = _plot_grid(
        metric="acc",
        dataset="dummy",
        topics=2,
        mode="ratio",
        mode_data=_grid_mode_data(),
        categories=["computer", "ride", "science"],
        outdir=tmp_path,
        models=None,
        no_errorbar=False,
        ylim=None,
        colormap="tab10",
        max_cols=2,
        legend_fontsize=10.0,
        formats=("png", "pdf"),
    )

    assert stem == tmp_path / "dummy_2topic_acc_ratio_grid"
    assert (tmp_path / "dummy_2topic_acc_ratio_grid.png").is_file()
    assert (tmp_path / "dummy_2topic_acc_ratio_grid.pdf").is_file()
    assert len(captured) == 2
    fig = captured[0]
    axes = fig.axes
    assert len(axes) == 4
    assert [ax.get_visible() for ax in axes] == [True, True, True, False]
    assert [ax.get_title() for ax in axes[:3]] == ["computer", "ride", "science"]
    assert all(ax.title.get_fontstyle() == "italic" for ax in axes[:3])
    # y label on the left column only; one shared x label for the figure;
    # x tick labels wherever no panel sits below.
    assert [ax.get_ylabel() for ax in axes[:3]] == ["Accuracy (%)", "", "Accuracy (%)"]
    assert [ax.get_xlabel() for ax in axes[:3]] == ["", "", ""]
    assert fig.get_supxlabel() == "Fraction of labeled documents (%)"
    assert [t.get_text() for t in axes[2].get_xticklabels()] == ["5", "100"]
    assert not axes[0].xaxis.get_tick_params()["labelbottom"]
    assert axes[1].xaxis.get_tick_params()["labelbottom"]
    assert fig.get_size_inches()[0] == pytest.approx(_grid_geometry(2, 2, 1)[0][0])
    assert fig.get_size_inches()[1] == pytest.approx(
        _grid_geometry(2, 2, 1, 10.0)[0][1]
    )
    assert fig.get_size_inches()[1] > _grid_geometry(2, 2, 1)[0][1]
    legend = fig.legends[0]
    assert [t.get_text() for t in legend.get_texts()] == ["LDA", "vSLDA (proposed)"]
    assert all(t.get_fontsize() == 10.0 for t in legend.get_texts())


def test_plot_grid_follows_category_order_and_skips_missing(
    tmp_path: Path, monkeypatch
) -> None:
    import matplotlib.figure

    captured: list = []
    original_savefig = matplotlib.figure.Figure.savefig

    def _record_savefig(self, fname, *args, **kwargs):
        captured.append(self)
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _record_savefig)

    common = dict(
        metric="acc",
        dataset="dummy",
        topics=2,
        mode="ratio",
        mode_data=_grid_mode_data(),
        outdir=tmp_path,
        models=None,
        no_errorbar=False,
        ylim=None,
        colormap="tab10",
        formats=("png",),
    )
    stem = _plot_grid(**common, categories=["science", "computer", "missing", "all"])
    assert stem == tmp_path / "dummy_2topic_acc_ratio_grid"
    titles = [ax.get_title() for ax in captured[0].axes if ax.get_visible()]
    assert titles == ["science", "computer"]

    assert _plot_grid(**common, categories=["missing", "all", "average"]) is None
    assert len(captured) == 1


def _write_paper_results(
    base_dir: Path, *, dataset: str, categories: Sequence[str], models: Sequence[str]
) -> None:
    """Limited-label and full runs for ``dataset`` at K=2, for the paper models."""
    scores = {
        category: {model: 60.0 + index for index, model in enumerate(models)}
        for category in categories
    }
    for name, display_key in [
        (
            f"acc_{dataset}_2topic_ratio0.5.json",
            "ratio_0-5_stratified_sample-r0_k2_it0",
        ),
        (f"acc_{dataset}_2topic.json", "k2_it0"),
    ]:
        archive_dir = (
            base_dir
            / "archive"
            / "2026-01-01"
            / dataset
            / "default"
            / "all"
            / display_key
            / "exec_0"
        )
        result = archive_dir / name
        write_evaluation_json(
            meta={"task": "classification", "started_at": "2026-01-01T00:00:00Z"},
            results=scores,
            path=result,
        )
        save_json(
            {
                "schema": "latest_result_pointer",
                "schema_version": 1,
                "task": "classification",
                "display_key": display_key,
                "dataset": dataset,
                "data_run": "default",
                "category": "all",
                "archive_dir": str(archive_dir),
                "started_at": "2026-01-01T00:00:00Z",
                "execution_id": "exec_0",
                "condition_fingerprint": "fingerprint",
                "artifacts": {"acc": result.name},
            },
            base_dir
            / "latest"
            / dataset
            / "default"
            / "all"
            / display_key
            / CURRENT_POINTER_FILENAME,
        )


def test_paper_configuration_matches_the_manuscript() -> None:
    assert set(PAPER_DATASET_CATEGORIES) == {"20newsgroup", "nyt"}
    assert PAPER_DATASET_CATEGORIES["20newsgroup"] == (
        "computer",
        "ride",
        "sports",
        "science",
        "religion",
        "politics",
    )
    assert PAPER_DATASET_CATEGORIES["nyt"] == ("arts", "business", "politics", "sports")
    assert PAPER_TOPICS == (10, 20, 30)
    assert PAPER_GRID_COLS == 2 and PAPER_GRID_LEGEND_FONTSIZE == 9.0
    assert PAPER_OUTDIR.endswith("limited_minilm_acc_logreg")

    # Every raw key of the paper's comparison must be recognised, and the
    # proposed method must be in it.  ``PAPER_MODELS`` is an editorial selection
    # for one manuscript figure, not the registry of every model the reports know
    # about, so it is deliberately a *subset* of ``MODEL_ORDER``: SAM is
    # registered for figures and tables but stays out of this typeset grid until
    # its numbers have been reviewed.
    paper_labels = [_display_model_name(model) for model in PAPER_MODELS]
    assert len(set(paper_labels)) == len(PAPER_MODELS)
    assert set(paper_labels).issubset(set(MODEL_ORDER))
    assert PROPOSED_LABEL in paper_labels


def _run_main(argv: Sequence[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["plot_limited", *argv])
    main()


def test_paper_flag_writes_grids_and_fails_on_a_missing_one(
    tmp_path: Path, monkeypatch
) -> None:
    base_dir = tmp_path / "classification"
    outdir = tmp_path / "figures"
    models = list(PAPER_MODELS[:2])
    _write_paper_results(
        base_dir, dataset="nyt", categories=("arts", "business"), models=models
    )
    common = [
        "--paper",
        "--base_dir",
        str(base_dir),
        "--outdir",
        str(outdir),
        "--datasets",
        "nyt",
        "--formats",
        "png",
    ]

    _run_main([*common, "--topics", "2"], monkeypatch)

    assert (outdir / "nyt_2topic_acc_ratio_grid.png").is_file()
    # The per-category panels and the legend files come with the preset.
    assert (outdir / "nyt_2topic_acc_ratio_arts.png").is_file()
    assert (outdir / "nyt_2topic_acc_ratio_arts_nox.png").is_file()
    assert (outdir / "legend_h.png").is_file()

    # A topic count without results produces no grid; the run must fail loudly.
    with pytest.raises(SystemExit) as excinfo:
        _run_main([*common, "--topics", "2", "3"], monkeypatch)
    assert "nyt_3topic_acc_ratio_grid.png" in str(excinfo.value)


def test_paper_flag_rejects_datasets_outside_the_preset(
    tmp_path: Path, monkeypatch
) -> None:
    base_dir = tmp_path / "classification"
    _write_paper_results(
        base_dir, dataset="nyt", categories=("arts",), models=[PAPER_MODELS[0]]
    )
    with pytest.raises(SystemExit) as excinfo:
        _run_main(
            [
                "--paper",
                "--base_dir",
                str(base_dir),
                "--outdir",
                str(tmp_path / "figures"),
                "--datasets",
                "dummy",
            ],
            monkeypatch,
        )
    assert "selects none of them" in str(excinfo.value)


def _write_full_pointer_with_models(
    base_dir: Path,
    *,
    display_key: str,
    scores: dict[str, float],
    started_at: str,
    **meta: object,
) -> None:
    archive_dir = (
        base_dir
        / "archive"
        / started_at[:10]
        / "dummy"
        / "default"
        / "all"
        / display_key
    )
    result_path = archive_dir / "acc_dummy_2topic.json"
    write_evaluation_json(
        meta={"task": "classification", "started_at": started_at, **meta},
        results={"computer": dict(scores)},
        path=result_path,
    )
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
            "started_at": started_at,
            "execution_id": f"exec_{display_key}",
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


def test_load_scores_selects_the_vslda_estimator_and_keeps_the_baselines(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "classification"
    hard_key = "vMF Sentence LDA [c1_minilm] [LogReg]"
    foldin_key = "vMF Sentence LDA (fold-in) [c1_minilm] [LogReg]"
    baseline = "Blei LDA [LogReg]"
    _write_full_pointer_with_models(
        base_dir,
        display_key="logreg_hard_k2_it0",
        scores={hard_key: 70.0, baseline: 60.0},
        started_at="2026-07-01T10:00:00+00:00",
        iteration=0,
        vmf_assignment="hard",
    )
    _write_full_pointer_with_models(
        base_dir,
        display_key="logreg_foldin_models-vmf_k2_it0",
        scores={foldin_key: 72.0},
        started_at="2026-07-01T11:00:00+00:00",
        iteration=0,
        vmf_assignment="foldin",
    )

    def _series(data):
        return data["acc"]["dummy"][2]["ratio"]["computer"]

    unfiltered = _series(_load_scores(base_dir))
    assert set(unfiltered) == {hard_key, foldin_key, baseline}

    hard = _series(_load_scores(base_dir, vmf_assignment="hard"))
    assert set(hard) == {hard_key, baseline}
    assert hard[hard_key][1.0] == [70.0]

    foldin = _series(_load_scores(base_dir, vmf_assignment="foldin"))
    assert set(foldin) == {foldin_key, baseline}
    assert foldin[foldin_key][1.0] == [72.0]
    assert foldin[baseline][1.0] == [60.0]


def test_load_scores_reads_mvtm_from_the_selected_estimator_only(
    tmp_path: Path,
) -> None:
    """MvTM (vLDA) shares the estimator: its scores come from the files of the
    selected assignment alone, unlike the other baselines."""

    base_dir = tmp_path / "classification"
    mvtm = "MvTM [c1_googlenews300] [LogReg]"
    baseline = "Blei LDA [LogReg]"
    _write_full_pointer_with_models(
        base_dir,
        display_key="logreg_hard_models-mvtm_k2_it0",
        scores={mvtm: 40.0, baseline: 60.0},
        started_at="2026-07-01T10:00:00+00:00",
        iteration=0,
        vmf_assignment="hard",
    )
    _write_full_pointer_with_models(
        base_dir,
        display_key="logreg_foldincounts_strict_models-mvtm_k2_it0",
        scores={mvtm: 41.0},
        started_at="2026-09-06T10:00:00+00:00",
        iteration=0,
        vmf_assignment="foldincounts",
    )

    def _series(data):
        return data["acc"]["dummy"][2]["ratio"]["computer"]

    foldin = _series(_load_scores(base_dir, vmf_assignment="foldincounts"))
    assert foldin[mvtm][1.0] == [41.0]
    assert foldin[baseline][1.0] == [60.0]
    hard = _series(_load_scores(base_dir, vmf_assignment="hard"))
    assert hard[mvtm][1.0] == [40.0]


def test_paper_models_swaps_only_the_vslda_key() -> None:
    assert paper_models("hard") == PAPER_MODELS
    swapped = paper_models("foldin")
    assert swapped[-1] == "vMF Sentence LDA (fold-in) [c1_minilm] [LogReg]"
    assert swapped[:-1] == PAPER_MODELS[:-1]
    with pytest.raises(ValueError):
        paper_models("argmax")

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from src.evaluation.reports.grid_layout import legend_row_major, reference_width
from src.evaluation.reports.topic_sweep_plot import (
    FIGURE_A_METRICS,
    FIGURE_B_MAX_TOPICS,
    FIGURE_B_METRICS,
    METRIC_AXIS_LABELS,
    PAPER_GRID_COLS,
    _paper_legend_rows,
    run_topic_sweep_plots,
)

MODELS = ("vmf_sentence_lda", "bleilda", "sentlda", "etm", "ctm")
CATEGORIES = ("computer", "ride", "sports", "science")
TOPICS = (10, 20, 30, 50, 100, 200, 300)
COLUMNS = (
    "dataset",
    "category",
    "model",
    "num_topics",
    "iteration",
    "metric",
    "value",
    "topics_per_label",
)


def _write_sweep_csv(path: Path) -> None:
    """A sweep shaped like the real one: ``all`` spans every K, categories stop
    at ``FIGURE_B_MAX_TOPICS``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for category in ("all", *CATEGORIES):
            for num_topics in TOPICS:
                if category != "all" and num_topics > FIGURE_B_MAX_TOPICS:
                    continue
                for model_index, model in enumerate(MODELS):
                    for metric in (*FIGURE_A_METRICS, "acc"):
                        for iteration in range(2):
                            writer.writerow(
                                {
                                    "dataset": "ds",
                                    "category": category,
                                    "model": model,
                                    "num_topics": num_topics,
                                    "iteration": iteration,
                                    "metric": metric,
                                    "value": 10.0 + model_index + iteration,
                                    "topics_per_label": num_topics / 4.0,
                                }
                            )


def _pdf_width_points(path: Path) -> float:
    """Width of a one-page PDF, read from its MediaBox."""
    match = re.search(
        rb"/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)",
        path.read_bytes(),
    )
    assert match, f"no MediaBox in {path}"
    return float(match.group(3)) - float(match.group(1))


def test_paper_figures_are_one_page_width(tmp_path: Path) -> None:
    """Both paper figures must be exactly as wide as the manuscript's text block.

    They are included at natural size, so a width that drifts silently changes
    how large the type prints. A tight bounding box would trim each figure to
    its ink and break that, which is the failure this guards.
    """
    pytest.importorskip("matplotlib")
    _write_sweep_csv(tmp_path / "ds" / "topic_sweep.csv")

    written = run_topic_sweep_plots(
        dataset="ds", sweep_root=tmp_path, formats=("pdf",), paper=True
    )
    # Figure A, plus one Figure B per per-category metric.
    assert len(written) == 1 + 1 + len(FIGURE_B_METRICS)

    expected_points = reference_width(PAPER_GRID_COLS) * 72.0
    for path in written:
        assert _pdf_width_points(path) == pytest.approx(expected_points, abs=0.05)


def test_screen_figures_keep_the_tight_bounding_box(tmp_path: Path) -> None:
    """Without ``paper`` the default layout is untouched, so the code repo's own
    review figures do not shift when the manuscript's layout changes."""
    pytest.importorskip("matplotlib")
    _write_sweep_csv(tmp_path / "ds" / "topic_sweep.csv")

    written = run_topic_sweep_plots(
        dataset="ds", sweep_root=tmp_path, formats=("pdf",), paper=False
    )
    expected_points = reference_width(PAPER_GRID_COLS) * 72.0
    assert all(
        _pdf_width_points(path) != pytest.approx(expected_points) for path in written
    )


def test_paper_legend_is_balanced_and_reads_in_model_order() -> None:
    """Five models must split 3 + 2, and read left to right in model order.

    matplotlib fills a legend column by column, so the entries are reordered
    before being handed over; getting that backwards scrambles the legend
    without failing anything.
    """
    labels = ["LDA", "SentLDA", "ETM", "ConTM", "vSLDA (proposed)"]
    ncol, rows = _paper_legend_rows(labels, 2)
    assert (ncol, rows) == (3, 2)

    entries = legend_row_major(labels, ncol)
    # Rebuild what matplotlib renders: column-major, remainder in leading columns.
    per_column = [rows] * (len(labels) % ncol or ncol) + [rows - 1] * (
        ncol - (len(labels) % ncol or ncol)
    )
    columns, cursor = [], 0
    for height in per_column:
        columns.append(entries[cursor : cursor + height])
        cursor += height
    assert [column[0] for column in columns] == ["LDA", "SentLDA", "ETM"]
    assert [column[1] for column in columns if len(column) > 1] == [
        "ConTM",
        "vSLDA (proposed)",
    ]


def test_per_category_figures_cover_accuracy_and_topic_quality(tmp_path: Path) -> None:
    """The appendix reports both per coarse category, so both must be emitted, each in
    its own file named by its metric."""
    pytest.importorskip("matplotlib")
    _write_sweep_csv(tmp_path / "ds" / "topic_sweep.csv")
    written = run_topic_sweep_plots(
        dataset="ds", sweep_root=tmp_path, formats=("pdf",), paper=True
    )
    names = {path.stem for path in written}
    assert "topic_sweep_b_acc_by_category_ds" in names
    assert "topic_sweep_b_topic_quality_by_category_ds" in names


def test_figure_a_reports_cv_coherence() -> None:
    """The manuscript's primary coherence measure is C_V; C_NPMI has its own
    appendix and must not stand in for it here."""
    assert "coherence_c_v" in FIGURE_A_METRICS
    assert "coherence_c_npmi" not in FIGURE_A_METRICS


def test_the_primary_coherence_carries_the_plain_axis_label() -> None:
    """C_V is the measure the manuscript reports, so its panel is labelled simply
    "Topic coherence"; C_NPMI names itself, so the two labels stay distinct."""
    assert METRIC_AXIS_LABELS["coherence_c_v"] == "Topic coherence"
    assert METRIC_AXIS_LABELS["coherence_c_npmi"] != METRIC_AXIS_LABELS["coherence_c_v"]
    assert "NPMI" in METRIC_AXIS_LABELS["coherence_c_npmi"]


def test_paper_legend_columns_follow_the_grid_width() -> None:
    """The legend sits inside the fixed figure width, so its column count is capped by
    the panel columns (:func:`legend_ncol_for`): the seven sweep models under a
    two-column grid become 3 + 3 + 1, not a 4 + 3 that clips at the TMLR width, while a
    three-column grid may still use four columns."""
    seven = ["LDA", "SentLDA", "SAM", "vLDA", "ETM", "ConTM", "vSLDA (proposed)"]
    assert _paper_legend_rows(seven, 2) == (3, 3)
    assert _paper_legend_rows(seven, 3) == (4, 2)
    assert _paper_legend_rows([], 2) == (1, 0)

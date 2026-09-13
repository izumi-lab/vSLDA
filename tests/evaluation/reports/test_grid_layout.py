"""The shared legend of the manuscript grids must stay inside the fixed figure width."""

from __future__ import annotations

import pytest

from src.evaluation.reports.grid_layout import (
    GRID_LEGEND_NCOL,
    check_legend_fits,
    legend_ncol_for,
)


def test_legend_columns_are_capped_by_the_panel_columns() -> None:
    assert legend_ncol_for(2) == 3
    assert legend_ncol_for(3) == GRID_LEGEND_NCOL
    assert legend_ncol_for(1) == 2
    assert legend_ncol_for(2, max_ncol=2) == 2


def test_a_legend_wider_than_the_figure_aborts_the_build() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plt.figure(figsize=(1.0, 1.0))
    handles = [
        plt.Line2D([0], [0], label="a label that is far too wide") for _ in range(4)
    ]
    legend = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    with pytest.raises(SystemExit):
        check_legend_fits(fig, legend)
    plt.close(fig)


def test_a_legend_inside_the_figure_passes() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plt.figure(figsize=(4.0, 1.0))
    legend = fig.legend(
        handles=[plt.Line2D([0], [0], label="LDA")],
        loc="lower center",
        ncol=1,
        frameon=False,
    )
    check_legend_fits(fig, legend)
    plt.close(fig)

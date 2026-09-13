"""Page geometry for the multi-panel figures the manuscript includes at 1:1.

Both figure families in the paper -- the sample-efficiency grids
(:mod:`src.evaluation.classification.plot_limited`) and the topic-count sweep
(:mod:`src.evaluation.reports.topic_sweep_plot`) -- are pasted into the
manuscript at their natural size, with no ``width=`` on the LaTeX side. Their
panels and fonts must therefore match on the page, which they can only do if
one module owns the lengths. That module is this one; it is the geometry
counterpart of :mod:`src.evaluation.reports.model_style`, which owns colour and
line style for the same two families.

Every length here is in inches, i.e. a physical length on the printed page.
``GRID_TEXT_WIDTH`` is the text width of the ``elsarticle``
``preprint,12pt,a4paper`` layout (override with the ``PAPER_TEXT_WIDTH``
environment variable, e.g. ``6.5`` for the TMLR layout), and the panel size is
derived from the
``GRID_NCOLS``-column case so that grids with fewer columns keep identical
panels rather than stretching to fill the line.

Figures laid out through :func:`grid_geometry` must be saved *without*
``bbox_inches="tight"``: the returned width is the promise made to the
manuscript, and a tight bounding box would trim it to whatever the ink happens
to span.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "DEFAULT_FORMATS",
    "DPI",
    "GRID_HSPACE",
    "GRID_LEGEND_FONTSIZE",
    "GRID_LEGEND_NCOL",
    "GRID_LEGEND_PAD",
    "GRID_LEGEND_ROW_HEIGHT",
    "GRID_MARGIN_BOTTOM",
    "GRID_MARGIN_LEFT",
    "GRID_MARGIN_RIGHT",
    "GRID_MARGIN_TOP",
    "GRID_NCOLS",
    "GRID_PANEL_ASPECT",
    "GRID_TEXT_WIDTH",
    "GRID_WSPACE",
    "GRID_XLABEL_OFFSET",
    "PANEL_GRID_KWARGS",
    "PAPER_RC",
    "grid_geometry",
    "panel_frame",
    "legend_row_major",
    "legend_ncol_for",
    "check_legend_fits",
    "reference_panel_width",
    "reference_width",
]

DPI = 300
DEFAULT_FORMATS = ("png", "pdf")

GRID_TEXT_WIDTH = float(os.environ.get("PAPER_TEXT_WIDTH", "7.22"))
# PAPER_GRID_NCOLS: reference column count the panel size is derived from. 2 makes
# the paper's two-column grids span the full text line instead of 2/3 of it.
GRID_NCOLS = int(os.environ.get("PAPER_GRID_NCOLS", "3"))
# PAPER_PANEL_ASPECT: with full-line panels (PAPER_GRID_NCOLS=2) the default 0.85
# aspect makes the three-row grids taller than the TMLR text height; ~0.60 fits.
GRID_PANEL_ASPECT = float(
    os.environ.get("PAPER_PANEL_ASPECT", "0.85")
)  # panel height / panel width
GRID_MARGIN_LEFT = 0.55  # y label (9 pt) + two-digit y tick labels (8 pt)
GRID_MARGIN_RIGHT = 0.06
GRID_MARGIN_TOP = 0.26  # italic panel title (9 pt)
GRID_MARGIN_BOTTOM = 0.42  # x tick labels (8 pt) + x label (9 pt)
GRID_WSPACE = 0.42  # room for the next column's y tick labels
GRID_HSPACE = 0.32  # room for the next row's title
GRID_LEGEND_NCOL = 4
GRID_LEGEND_FONTSIZE = (
    8.0  # pt; wide grids may use a larger size (--grid-legend-fontsize)
)
GRID_LEGEND_ROW_HEIGHT = 0.18  # per legend row at GRID_LEGEND_FONTSIZE
GRID_LEGEND_PAD = 0.12
GRID_XLABEL_OFFSET = 0.06  # gap between the legend block and the shared x label

# rcParams applied locally while drawing; sized for a ~2 in wide panel.
PAPER_RC = {
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.titlesize": 9,
    "axes.titlepad": 3.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# Panel frame shared by every manuscript figure, so that the families cannot
# drift apart: the top and right spines are dropped, and the value axis carries
# a faint dotted grid. The grid sits at the default ``axes.axisbelow`` depth,
# under the series and their bands, and is meant to be read only where the
# panel is otherwise empty.
PANEL_GRID_KWARGS = {"linestyle": ":", "linewidth": 0.6, "alpha": 0.35}


def panel_frame(ax: Any, *, grid: bool = True) -> None:
    """Apply the shared panel frame to ``ax``.

    Pass ``grid=False`` for a panel that already draws horizontal reference
    lines of its own (the entropy box plots), where a dotted grid would read as
    one more reference line.
    """
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis="y", **PANEL_GRID_KWARGS)


def reference_panel_width() -> float:
    """Panel width (in) of the ``GRID_NCOLS``-column full-width layout."""
    return (
        GRID_TEXT_WIDTH
        - GRID_MARGIN_LEFT
        - GRID_MARGIN_RIGHT
        - (GRID_NCOLS - 1) * GRID_WSPACE
    ) / GRID_NCOLS


def reference_width(ncols: int, wspace: float = GRID_WSPACE) -> float:
    """Total width (in) of an ``ncols``-column grid at the reference panel size.

    This is the width a figure should be solved to when its gutters differ from
    the default: the page then sees one width for every figure, and only the
    split between panel and gutter moves.
    """
    panel_w = reference_panel_width()
    return GRID_MARGIN_LEFT + ncols * panel_w + (ncols - 1) * wspace + GRID_MARGIN_RIGHT


def grid_geometry(
    nrows: int,
    ncols: int,
    legend_rows: int,
    legend_fontsize: float = GRID_LEGEND_FONTSIZE,
    *,
    panel_aspect: float = GRID_PANEL_ASPECT,
    wspace: float = GRID_WSPACE,
    margin_top: float = GRID_MARGIN_TOP,
    margin_bottom: float = GRID_MARGIN_BOTTOM,
    hspace: float = GRID_HSPACE,
    total_width: float | None = None,
) -> Tuple[Tuple[float, float], Dict[str, float], float]:
    """Figure size (in), ``subplots_adjust`` fractions and legend height (in).

    By default the panel size is fixed by the ``GRID_NCOLS``-column full-width
    layout, so that grids with fewer columns keep identical panels and fonts and
    the total width follows from them. Pass ``total_width`` to invert that: the
    width is then the fixed quantity and the panels are solved to fill it, which
    is what a grid with wider gutters needs in order to stay on the same page
    width as the others.

    ``panel_aspect``, ``wspace``, ``hspace``, ``margin_top`` and
    ``margin_bottom`` let a caller buy room for what its own panels carry --
    taller panels, a y label in every column, rotated tick labels, no titles --
    without changing the lengths every other figure is built from.
    """
    if total_width is None:
        panel_w = reference_panel_width()
    else:
        panel_w = (
            total_width - GRID_MARGIN_LEFT - GRID_MARGIN_RIGHT - (ncols - 1) * wspace
        ) / ncols
    panel_h = panel_aspect * panel_w
    row_h = GRID_LEGEND_ROW_HEIGHT * legend_fontsize / GRID_LEGEND_FONTSIZE
    legend_h = legend_rows * row_h + GRID_LEGEND_PAD if legend_rows else 0.0
    width = (
        GRID_MARGIN_LEFT + ncols * panel_w + (ncols - 1) * wspace + GRID_MARGIN_RIGHT
    )
    height = (
        margin_top + nrows * panel_h + (nrows - 1) * hspace + margin_bottom + legend_h
    )
    adjust = {
        "left": GRID_MARGIN_LEFT / width,
        "right": 1.0 - GRID_MARGIN_RIGHT / width,
        "top": 1.0 - margin_top / height,
        "bottom": (margin_bottom + legend_h) / height,
        "wspace": wspace / panel_w,
        "hspace": hspace / panel_h,
    }
    return (width, height), adjust, legend_h


def legend_ncol_for(ncols: int, max_ncol: int = GRID_LEGEND_NCOL) -> int:
    """Legend columns that fit under a grid of ``ncols`` panel columns.

    The shared legend sits inside the figure width, which is fixed by the panel
    count (see :func:`reference_width`), so its column count has to follow the
    grid: four 9 pt entries with the nine model labels of the manuscript span
    about 335 pt, more than the 317 pt of a two-column grid at
    ``PAPER_TEXT_WIDTH=6.5``, whereas three (about 265 pt) fit. Wider grids may
    use up to ``max_ncol``.
    """
    return max(1, min(max_ncol, ncols + 1))


def check_legend_fits(fig: Any, legend: Any) -> None:
    """Abort if ``legend`` leaves the figure box.

    The manuscript figures are saved without ``bbox_inches="tight"``, so a
    legend wider than the figure is silently clipped in the PDF instead of
    widening it; the check turns that into a build error.
    """
    fig.canvas.draw()
    box = legend.get_window_extent()
    if box.x0 < 0 or box.x1 > fig.bbox.width:
        raise SystemExit(
            f"legend {box.width / fig.dpi * 72:.0f} pt wider than the "
            f"{fig.bbox.width / fig.dpi * 72:.0f} pt figure; lower the legend ncol "
            "(see legend_ncol_for)"
        )


def legend_row_major(items: Sequence, ncol: int) -> List:
    """Reorder ``items`` so that a matplotlib legend with ``ncol`` columns
    (which fills column by column) reads row by row, left to right.

    A last row that does not fill is handled the way matplotlib packs it, so no
    padding entry is needed to keep the order right.
    """
    rows = [list(items[i : i + ncol]) for i in range(0, len(items), ncol)]
    return [row[col] for col in range(ncol) for row in rows if col < len(row)]

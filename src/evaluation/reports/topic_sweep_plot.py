"""Figures for the topic-count (K) sweep.

Reads the long table written by :mod:`src.evaluation.reports.topic_sweep` and
draws four views of it. They answer different questions and are meant to be read
in this order:

``A`` (main)  ``all`` only, 2x2: accuracy, NPMI coherence, diversity and topic
              utilization against K. This is the figure the analysis is about.
``B``         accuracy against K, one panel per category plus ``all``: how the
              best K moves with corpus size.
``C``         active and empty topics against K for ``all``, with the diagonal
              ``y = K`` drawn: how much of the model's capacity is actually used.
``D``         the same metrics against ``K / num_labels``, every category
              overlaid, so categories of different size land on one axis.

The K axis of figures A-C is ordinal: the swept values (10, 20, 30, 50, 100, 200,
300) are placed at equal intervals and labelled with the value. A linear axis
would compress everything below K=100 into the left margin, and a logarithmic
one still crowds 20/30 and 200/300 until the tick labels have to be rotated.
Figure D plots the continuous ratio K / L and keeps a logarithmic axis.

Series appearance (colour by model family, solid for sentence-level assignment)
comes from :mod:`src.evaluation.reports.model_style`, the same source the
sample-efficiency figures use, so a model looks identical in both.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.reports.grid_layout import (
    DEFAULT_FORMATS,
    DPI,
    GRID_LEGEND_PAD,
    GRID_MARGIN_LEFT,
    GRID_XLABEL_OFFSET,
    PAPER_RC,
    check_legend_fits,
    grid_geometry,
    legend_ncol_for,
    legend_row_major,
    panel_frame,
    reference_width,
)
from src.evaluation.reports.model_style import (
    EXCLUDED_MODELS,
    MODEL_ORDER,
    label_for_key,
    model_style,
)

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import error handling
    raise SystemExit(
        "matplotlib is required to plot figures. "
        "Install it in your environment and rerun."
    ) from exc

__all__ = [
    "FIGURE_A_METRICS",
    "FIGURE_B_METRICS",
    "METRIC_AXIS_LABELS",
    "read_topic_sweep_csv",
    "run_topic_sweep_plots",
]

DEFAULT_COLORMAP = "tab10"
PANEL_SIZE = (2.7, 2.3)
LINE_ALPHA = 0.95
ERROR_ALPHA = 0.12

# ``topic_utilization`` counts topics whose expected word mass is exactly zero,
# which only happens for degenerate runs (MvTM). For the models plotted here it
# is 1.0 everywhere and carries no signal, so the composite takes its slot.
# The coherence panel reports C_V, the measure the manuscript takes as primary
# (C_NPMI is confined to its own appendix there), and the composite below is
# already C_V x diversity, so the two panels agree on the coherence factor.
FIGURE_A_METRICS = (
    "acc",
    "coherence_c_v",
    "diversity",
    "topic_quality",
)
# Figure B is drawn once per metric here. The classification metric shows whether the
# per-category ordering survives the extra step to K=50; topic quality is added because
# the coherence advantage is the one that moves with K, and the composite is what the
# manuscript reads when coherence and diversity pull in opposite directions.
FIGURE_B_METRICS = ("topic_quality",)
FIGURE_C_METRICS = ("num_active_topics", "num_empty_topics")
FIGURE_D_METRICS = ("acc", "topic_quality")
METRIC_AXIS_LABELS = {
    "acc": "Accuracy (%)",
    "f1mac": "Macro F1",
    "f1mic": "Micro F1",
    # C_V is the manuscript's primary coherence measure and carries the plain label; C_NPMI
    # is confined to an appendix there and names itself, so the two never read alike.
    "coherence_c_npmi": r"$C_{\mathrm{NPMI}}$ coherence",
    "coherence_c_v": "Topic coherence",
    "diversity": "Topic diversity",
    "num_active_topics": "Active topics",
    "num_empty_topics": "Empty topics",
    "topic_utilization": "Topic utilization",
    "topic_quality": "Topic quality",
}
ALL_CATEGORY = "all"
# Per-category runs stop here; larger K was only swept on the ``all`` corpus.
FIGURE_B_MAX_TOPICS = 100

# ---------------------------------------------------------------------------
# Manuscript layout (``--paper``)
#
# The paper includes these figures at their natural size, with no ``width=`` on
# the LaTeX side, so their width is fixed here rather than left to whatever the
# ink happens to span. The lengths come from
# :mod:`src.evaluation.reports.grid_layout`, which the sample-efficiency figures
# also use, so both families land on the page with the same panel size and the
# same 8/9 pt type. The default (screen) layout is untouched: it still sizes
# panels by ``PANEL_SIZE`` and saves with a tight bounding box.
# ---------------------------------------------------------------------------
PAPER_LEGEND_NCOL = 4
PAPER_LEGEND_FONTSIZE = 9.0  # matches the sample-efficiency grids
# Maximum panel columns of the paper figures; ``_paper_cols`` picks the largest
# count that fills the last row, mirroring the sample-efficiency grids. The
# default 2 keeps Figure A 2x2 and Figure B 3x2 / 2x2; PAPER_MAX_GRID_COLS=3
# (the TMLR layout) turns the six-panel Figure B grids into full-line 2x3.
PAPER_GRID_COLS = int(os.environ.get("PAPER_MAX_GRID_COLS", "2"))
# Figure A gives every panel its own y label, so the gutter between columns has
# to hold a label as well as tick labels; Figure B repeats one metric and labels
# the left column only, so the narrower default gutter is enough. Both are
# solved to the same total width, which costs Figure A ~3% of its panel width.
PAPER_FIGURE_A_WSPACE = GRID_MARGIN_LEFT
# Figure A identifies its panels by the y label, so it needs no room for titles.
PAPER_FIGURE_A_MARGIN_TOP = 0.06
PAPER_XLABEL = "Number of topics $K$"
# PAPER_SWEEP_COMBINED_METRICS="acc,coherence_c_v,topic_quality" (the TMLR
# layout) additionally draws, with ``--paper``, one full-line figure whose rows
# are the datasets of PAPER_SWEEP_COMBINED_DATASETS and whose columns are the
# listed metrics of the ``all`` unit (``<sweep_root>/figures/topic_sweep_a_combined``);
# the manuscript's main text shows it and keeps the per-dataset Figure A in an
# appendix. Unset by default so the default outputs are unchanged.
PAPER_SWEEP_COMBINED_METRICS = tuple(
    metric
    for metric in os.environ.get("PAPER_SWEEP_COMBINED_METRICS", "").split(",")
    if metric
)
PAPER_SWEEP_COMBINED_DATASETS = tuple(
    dataset
    for dataset in os.environ.get(
        "PAPER_SWEEP_COMBINED_DATASETS", "20newsgroup,nyt"
    ).split(",")
    if dataset
)
DATASET_LABELS = {"20newsgroup": "20 Newsgroups", "nyt": "NYT"}


def _axis_label(metric: str) -> str:
    return METRIC_AXIS_LABELS.get(metric, metric)


def read_topic_sweep_csv(path: Path) -> list[dict[str, Any]]:
    """Load ``topic_sweep.csv``, coercing the numeric columns."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="") as handle:
        for record in csv.DictReader(handle):
            try:
                value = float(record["value"])
                num_topics = int(float(record["num_topics"]))
            except (KeyError, TypeError, ValueError):
                continue
            topics_per_label = record.get("topics_per_label")
            try:
                per_label = float(topics_per_label) if topics_per_label else None
            except ValueError:
                per_label = None
            rows.append(
                {
                    **record,
                    "value": value,
                    "num_topics": num_topics,
                    "topics_per_label": per_label,
                }
            )
    return rows


def _series(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    category: str | None = None,
    x_field: str = "num_topics",
) -> dict[str, dict[float, list[float]]]:
    """``model -> x -> [value per seed]`` for one metric."""
    collected: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("metric") != metric:
            continue
        if category is not None and row.get("category") != category:
            continue
        x = row.get(x_field)
        if x is None:
            continue
        label = label_for_key(row["model"])
        if label in EXCLUDED_MODELS:
            continue
        collected[label][float(x)].append(float(row["value"]))
    return {label: dict(points) for label, points in collected.items()}


def _ordered_labels(series: Mapping[str, Any]) -> list[str]:
    def sort_key(label: str) -> tuple[int, str]:
        try:
            return (MODEL_ORDER.index(label), label)
        except ValueError:
            return (len(MODEL_ORDER), label)

    return sorted(series, key=sort_key)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance**0.5


def _draw_panel(
    ax: Any,
    series: Mapping[str, Mapping[float, Sequence[float]]],
    *,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    colormap: str = DEFAULT_COLORMAP,
    draw_band: bool = True,
    positions: Mapping[int, float] | None = None,
) -> None:
    """Draw one series per model.

    With ``positions`` (``K -> slot``, see :func:`_topic_positions`) the x axis
    is ordinal: every K is drawn at its slot and values at a K outside the
    mapping are left out. Without it the x values are used as they are, on a
    logarithmic axis.
    """
    for label in _ordered_labels(series):
        points = series[label]
        keys = sorted(points)
        if positions is not None:
            keys = [key for key in keys if int(key) in positions]
        if not keys:
            continue
        xs = [positions[int(key)] if positions is not None else key for key in keys]
        means = []
        stds = []
        for key in keys:
            mean, std = _mean_std(list(points[key]))
            means.append(mean)
            stds.append(std)
        style = model_style(label, colormap)
        ax.plot(xs, means, alpha=LINE_ALPHA, zorder=style.zorder, **style.line_kwargs())
        if draw_band and any(std > 0 for std in stds):
            ax.fill_between(
                xs,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=style.color,
                alpha=ERROR_ALPHA,
                linewidth=0,
                zorder=style.zorder - 1,
            )
    if positions is None:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, style="italic")
    panel_frame(ax)


def _topic_positions(topics: Sequence[int]) -> dict[int, int]:
    """``K -> slot`` for the ordinal K axis: the sorted K values at 0, 1, 2, ...."""
    return {topic: slot for slot, topic in enumerate(sorted({int(t) for t in topics}))}


def _set_topic_ticks(ax: Any, positions: Mapping[int, float]) -> None:
    """One tick per swept K at its slot, labelled with the K value."""
    if not positions:
        return
    ticks = sorted(positions.items())
    ax.set_xticks([slot for _, slot in ticks])
    ax.set_xticklabels([str(topic) for topic, _ in ticks])
    ax.minorticks_off()


def _legend_below(fig: Any, labels: Sequence[str], *, colormap: str, ncol: int) -> None:
    handles = [
        plt.Line2D([0], [0], label=label, **model_style(label, colormap).line_kwargs())
        for label in labels
    ]
    if not handles:
        return
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(ncol, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )


def _save(
    fig: Any, stem: Path, formats: Sequence[str], dpi: int, *, tight: bool = True
) -> list[Path]:
    """Write ``stem`` in every format; ``tight=False`` keeps the figure width.

    The paper layout solves the figure to an exact page width, so trimming it to
    the ink would break the 1:1 inclusion the manuscript relies on.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = stem.parent / f"{stem.name}.{fmt}"
        fig.savefig(path, dpi=dpi, **({"bbox_inches": "tight"} if tight else {}))
        written.append(path)
    plt.close(fig)
    return written


def _paper_legend_rows(labels: Sequence[str], ncols: int) -> tuple[int, int]:
    """Legend columns and rows for ``labels``, balanced across the rows.

    The widest the legend may get is ``PAPER_LEGEND_NCOL`` capped by what fits
    under a grid of ``ncols`` panel columns (:func:`legend_ncol_for`); the
    columns are then spread evenly over the rows that many entries need, so the
    five models of the sweep become 3 + 2 rather than a lopsided 4 + 1.
    """
    if not labels:
        return 1, 0
    max_ncol = legend_ncol_for(ncols, PAPER_LEGEND_NCOL)
    rows = -(-len(labels) // max(1, max_ncol))
    ncol = -(-len(labels) // rows)
    return ncol, rows


def _paper_finish(
    fig: Any,
    *,
    figsize: tuple[float, float],
    adjust: Mapping[str, float],
    legend_h: float,
    labels: Sequence[str],
    legend_ncol: int,
    colormap: str,
) -> Any:
    """Shared x label and legend below the panels, as in the other paper grids.

    Returns the legend (or ``None``) after checking that it stays inside the
    figure, which is saved without a tight bounding box.
    """
    fig.supxlabel(
        PAPER_XLABEL,
        x=0.5 * (adjust["left"] + adjust["right"]),
        y=(legend_h + GRID_XLABEL_OFFSET) / figsize[1],
        va="bottom",
        fontsize=PAPER_RC["axes.labelsize"],
    )
    if not labels:
        return None
    handles = [
        plt.Line2D([0], [0], label=label, **model_style(label, colormap).line_kwargs())
        for label in labels
    ]
    legend = fig.legend(
        handles=legend_row_major(handles, legend_ncol),
        labels=legend_row_major(list(labels), legend_ncol),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.5 * GRID_LEGEND_PAD / figsize[1]),
        ncol=legend_ncol,
        fontsize=PAPER_LEGEND_FONTSIZE,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    check_legend_fits(fig, legend)
    return legend


def _grid(panel_count: int, ncols: int) -> tuple[int, int]:
    ncols = max(1, min(ncols, panel_count))
    nrows = (panel_count + ncols - 1) // ncols
    return nrows, ncols


def _paper_cols(panel_count: int) -> int:
    """Largest column count <= ``PAPER_GRID_COLS`` that fills the last row."""
    for cols in range(min(PAPER_GRID_COLS, panel_count), 1, -1):
        if panel_count % cols == 0:
            return cols
    return min(PAPER_GRID_COLS, panel_count)


def _figure_a(
    rows: Sequence[Mapping[str, Any]],
    *,
    topics: Sequence[int],
    stem: Path,
    colormap: str,
    formats: Sequence[str],
    dpi: int,
    paper: bool = False,
) -> list[Path]:
    metrics = [
        metric
        for metric in FIGURE_A_METRICS
        if any(
            row["metric"] == metric and row["category"] == ALL_CATEGORY for row in rows
        )
    ]
    if not metrics:
        return []
    nrows, ncols = _grid(len(metrics), _paper_cols(len(metrics)) if paper else 2)

    labels: list[str] = []
    for metric in metrics:
        labels = (
            _ordered_labels(_series(rows, metric=metric, category=ALL_CATEGORY))
            or labels
        )
    legend_ncol, legend_rows = _paper_legend_rows(labels, ncols)

    if paper:
        figsize, adjust, legend_h = grid_geometry(
            nrows,
            ncols,
            legend_rows,
            PAPER_LEGEND_FONTSIZE,
            wspace=PAPER_FIGURE_A_WSPACE,
            margin_top=PAPER_FIGURE_A_MARGIN_TOP,
            total_width=reference_width(ncols),
        )
    else:
        figsize = (PANEL_SIZE[0] * ncols, PANEL_SIZE[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    if paper:
        fig.subplots_adjust(**adjust)
    positions = _topic_positions(topics)

    for index, metric in enumerate(metrics):
        ax = axes[index // ncols][index % ncols]
        series = _series(rows, metric=metric, category=ALL_CATEGORY)
        _draw_panel(
            ax,
            series,
            # The paper layout carries one shared x label below the panels; the
            # y label differs per panel and stays on the panel either way.
            xlabel="" if paper else PAPER_XLABEL,
            ylabel=_axis_label(metric),
            colormap=colormap,
            positions=positions,
        )
        _set_topic_ticks(ax, positions)
    for index in range(len(metrics), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    if paper:
        _paper_finish(
            fig,
            figsize=figsize,
            adjust=adjust,
            legend_h=legend_h,
            labels=labels,
            legend_ncol=legend_ncol,
            colormap=colormap,
        )
        return _save(fig, stem, formats, dpi, tight=False)

    _legend_below(fig, labels, colormap=colormap, ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, stem, formats, dpi)


def _figure_a_combined(
    rows_by_dataset: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    metrics: Sequence[str],
    topics: Sequence[int],
    stem: Path,
    colormap: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Paper layout only: datasets as rows, ``metrics`` of ``all`` as columns."""
    present = [
        metric
        for metric in metrics
        if all(
            any(
                row["metric"] == metric and row["category"] == ALL_CATEGORY
                for row in rows
            )
            for _, rows in rows_by_dataset
        )
    ]
    if not present or not rows_by_dataset:
        return []
    nrows, ncols = len(rows_by_dataset), len(present)
    labels: list[str] = []
    for _, rows in rows_by_dataset:
        for metric in present:
            labels = (
                _ordered_labels(_series(rows, metric=metric, category=ALL_CATEGORY))
                or labels
            )
    legend_ncol, legend_rows = _paper_legend_rows(labels, ncols)
    figsize, adjust, legend_h = grid_geometry(
        nrows,
        ncols,
        legend_rows,
        PAPER_LEGEND_FONTSIZE,
        wspace=PAPER_FIGURE_A_WSPACE,
        total_width=reference_width(ncols),
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(**adjust)
    positions = _topic_positions(topics)
    for row_index, (dataset, rows) in enumerate(rows_by_dataset):
        for col_index, metric in enumerate(present):
            ax = axes[row_index][col_index]
            _draw_panel(
                ax,
                _series(rows, metric=metric, category=ALL_CATEGORY),
                xlabel="",
                ylabel=_axis_label(metric),
                title=DATASET_LABELS.get(dataset, dataset),
                colormap=colormap,
                positions=positions,
            )
            _set_topic_ticks(ax, positions)
            if row_index < nrows - 1:
                ax.set_xticklabels([])
    _paper_finish(
        fig,
        figsize=figsize,
        adjust=adjust,
        legend_h=legend_h,
        labels=labels,
        legend_ncol=legend_ncol,
        colormap=colormap,
    )
    return _save(fig, stem, formats, dpi, tight=False)


def _figure_b(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    topics: Sequence[int],
    stem: Path,
    colormap: str,
    formats: Sequence[str],
    dpi: int,
    paper: bool = False,
) -> list[Path]:
    # The per-category panels only ever ran up to FIGURE_B_MAX_TOPICS, and the
    # aggregated ``all`` corpus lives on a different scale (19-24 gold labels vs
    # 2-9), so mixing it in here stretches the shared axis for no gain. Figure A
    # already covers ``all`` across the full K range.
    rows = [
        row
        for row in rows
        if row["category"] != ALL_CATEGORY
        and int(row["num_topics"]) <= FIGURE_B_MAX_TOPICS
    ]
    topics = [int(topic) for topic in topics if int(topic) <= FIGURE_B_MAX_TOPICS]
    categories = list(
        dict.fromkeys(row["category"] for row in rows if row["metric"] == metric)
    )
    if not categories:
        return []
    nrows, ncols = _grid(len(categories), _paper_cols(len(categories)) if paper else 3)

    labels: list[str] = []
    for category in categories:
        labels = (
            _ordered_labels(_series(rows, metric=metric, category=category)) or labels
        )
    legend_ncol, legend_rows = _paper_legend_rows(labels, ncols)

    if paper:
        figsize, adjust, legend_h = grid_geometry(
            nrows, ncols, legend_rows, PAPER_LEGEND_FONTSIZE
        )
    else:
        figsize = (PANEL_SIZE[0] * ncols, PANEL_SIZE[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    if paper:
        fig.subplots_adjust(**adjust)
    positions = _topic_positions(topics)

    for index, category in enumerate(categories):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        series = _series(rows, metric=metric, category=category)
        _draw_panel(
            ax,
            series,
            # Every panel shows the same metric against the same axis, so the
            # paper layout labels the left column and the bottom of the figure
            # once instead of repeating both on all six panels.
            xlabel="" if paper else PAPER_XLABEL,
            ylabel=_axis_label(metric) if not paper or col == 0 else "",
            title=category,
            colormap=colormap,
            positions=positions,
        )
        _set_topic_ticks(ax, positions)
        if paper and index + ncols < len(categories):
            ax.tick_params(labelbottom=False)
    for index in range(len(categories), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    if paper:
        _paper_finish(
            fig,
            figsize=figsize,
            adjust=adjust,
            legend_h=legend_h,
            labels=labels,
            legend_ncol=legend_ncol,
            colormap=colormap,
        )
        return _save(fig, stem, formats, dpi, tight=False)

    _legend_below(fig, labels, colormap=colormap, ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, stem, formats, dpi)


def _figure_c(
    rows: Sequence[Mapping[str, Any]],
    *,
    topics: Sequence[int],
    stem: Path,
    colormap: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    metrics = [
        metric
        for metric in FIGURE_C_METRICS
        if any(
            row["metric"] == metric and row["category"] == ALL_CATEGORY for row in rows
        )
    ]
    if not metrics:
        return []
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(PANEL_SIZE[0] * len(metrics), PANEL_SIZE[1]),
        squeeze=False,
    )
    positions = _topic_positions(topics)
    labels: list[str] = []
    for index, metric in enumerate(metrics):
        ax = axes[0][index]
        series = _series(rows, metric=metric, category=ALL_CATEGORY)
        labels = _ordered_labels(series) or labels
        _draw_panel(
            ax,
            series,
            xlabel="Number of topics $K$",
            ylabel=_axis_label(metric),
            colormap=colormap,
            positions=positions,
        )
        if metric == "num_active_topics":
            # y = K is the ceiling: the gap below it is capacity the model left
            # unused at that K.
            ceiling = sorted(positions)
            ax.plot(
                [positions[topic] for topic in ceiling],
                ceiling,
                color="#666666",
                linestyle=(0, (1.0, 1.5)),
                linewidth=0.7,
                zorder=1,
                label="$y = K$",
            )
        _set_topic_ticks(ax, positions)
    _legend_below(fig, labels, colormap=colormap, ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _save(fig, stem, formats, dpi)


def _figure_d(
    rows: Sequence[Mapping[str, Any]],
    *,
    stem: Path,
    colormap: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    metrics = [
        metric
        for metric in FIGURE_D_METRICS
        if any(row["metric"] == metric for row in rows)
    ]
    if not metrics:
        return []
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(PANEL_SIZE[0] * len(metrics), PANEL_SIZE[1]),
        squeeze=False,
    )
    labels: list[str] = []
    for index, metric in enumerate(metrics):
        ax = axes[0][index]
        # Every category at once: on this axis a coarse category at K=50 and
        # ``all`` at K=200 sit at comparable positions.
        series = _series(rows, metric=metric, x_field="topics_per_label")
        labels = _ordered_labels(series) or labels
        _draw_panel(
            ax,
            series,
            xlabel="Topics per label $K / L$",
            ylabel=_axis_label(metric),
            colormap=colormap,
            draw_band=False,
        )
    _legend_below(fig, labels, colormap=colormap, ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _save(fig, stem, formats, dpi)


def run_topic_sweep_plots(
    *,
    dataset: str,
    sweep_root: Path,
    topics: Sequence[int] | None = None,
    primary_metric: str = "acc",
    colormap: str = DEFAULT_COLORMAP,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DPI,
    paper: bool = False,
) -> list[Path]:
    """Draw figures A-D for ``dataset`` from ``<sweep_root>/<dataset>``.

    Panels whose metric has no rows yet are skipped rather than drawn empty, so
    this is safe to run while the sweep is still filling in.

    With ``paper``, the figures are laid out to the manuscript's page width and
    saved without a tight bounding box, so that they can be included at their
    natural size next to the sample-efficiency grids.
    """
    dataset_dir = Path(sweep_root) / dataset
    csv_path = dataset_dir / "topic_sweep.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No topic sweep table at {csv_path}")
    rows = read_topic_sweep_csv(csv_path)
    if not rows:
        return []
    topic_values = sorted({int(row["num_topics"]) for row in rows})
    if topics:
        topic_values = sorted({int(topic) for topic in topics})

    figures_dir = dataset_dir / "figures"
    written: list[Path] = []
    with plt.rc_context(PAPER_RC):
        written += _figure_a(
            rows,
            topics=topic_values,
            stem=figures_dir / f"topic_sweep_a_all_{dataset}",
            colormap=colormap,
            formats=formats,
            dpi=dpi,
            paper=paper,
        )
        for metric in (primary_metric, *FIGURE_B_METRICS):
            written += _figure_b(
                rows,
                metric=metric,
                topics=topic_values,
                stem=figures_dir / f"topic_sweep_b_{metric}_by_category_{dataset}",
                colormap=colormap,
                formats=formats,
                dpi=dpi,
                paper=paper,
            )
    if paper and PAPER_SWEEP_COMBINED_METRICS:
        rows_by_dataset = []
        for other in PAPER_SWEEP_COMBINED_DATASETS:
            other_csv = Path(sweep_root) / other / "topic_sweep.csv"
            if other_csv.is_file():
                other_rows = read_topic_sweep_csv(other_csv)
                if other_rows:
                    rows_by_dataset.append((other, other_rows))
        if len(rows_by_dataset) == len(PAPER_SWEEP_COMBINED_DATASETS):
            combined_dir = Path(sweep_root) / "figures"
            combined_dir.mkdir(parents=True, exist_ok=True)
            with plt.rc_context(PAPER_RC):
                written += _figure_a_combined(
                    rows_by_dataset,
                    metrics=PAPER_SWEEP_COMBINED_METRICS,
                    topics=topic_values,
                    stem=combined_dir / "topic_sweep_a_combined",
                    colormap=colormap,
                    formats=formats,
                    dpi=dpi,
                )
    for path in written:
        print(f"[write] {path}")
    return written

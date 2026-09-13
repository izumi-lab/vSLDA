from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.core.artifacts import CURRENT_POINTER_FILENAME, load_json
from src.core.paths import resolve_project_path
from src.core.vmf_assignment import DEFAULT_VMF_ASSIGNMENT, LEGACY_VMF_ASSIGNMENT
from src.evaluation.reporting import read_evaluation_json
from src.evaluation.reports.grid_layout import (
    DEFAULT_FORMATS,
    DPI,
    GRID_LEGEND_FONTSIZE,
    GRID_LEGEND_NCOL,
    GRID_LEGEND_PAD,
    GRID_MARGIN_TOP,
    GRID_NCOLS,
    GRID_TEXT_WIDTH,
    GRID_XLABEL_OFFSET,
    PAPER_RC,
    check_legend_fits,
    grid_geometry,
    legend_ncol_for,
    legend_row_major,
    panel_frame,
)
from src.evaluation.reports.model_style import (
    EXCLUDED_MODELS,
    base_model_name,
    display_model_name,
    fallback_color,
    model_sort_key,
    model_style,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator
except Exception as exc:  # pragma: no cover - import error handling
    raise SystemExit(
        "matplotlib is required to plot figures. "
        "Install it in your environment and rerun."
    ) from exc


LIMITED_FILE_RE = re.compile(r"^(acc|f1)_(.+)_(\d+)topic_(ratio|count)(.+)\.json$")
FULL_FILE_RE = re.compile(r"^(acc|f1)_(.+)_(\d+)topic\.json$")
# Meta fields (besides iteration and sampling repeat) that distinguish run
# conditions; mirrors what ``summary._matches_metric_meta`` compares. Two
# scores for one model that agree on all of these describe the same evaluation.
CONDITION_META_FIELDS = (
    "data_run",
    "vmf_assignment",
    "alignment_mode",
    "feature_resolve_mode",
    "prior_scale",
    "covariance_type",
)
# Series identity and appearance are shared with the topic-count sweep figures;
# see :mod:`src.evaluation.reports.model_style` for the taxonomy behind them.
DEFAULT_COLORMAP = "tab10"
METRIC_AXIS_LABELS = {
    "acc": "Accuracy (%)",
    "f1mac": "Macro F1",
    "f1mic": "Micro F1",
}
MODE_AXIS_LABELS = {
    "ratio": "Fraction of labeled documents",
    "count": "Training examples",
}
LINE_ALPHA = 0.95
ERROR_ALPHA = 0.12
CATEGORY_FIGSIZE = (2.6, 2.3)
YTICK_STEP = 10.0
MIN_YTICKS = 3
SUPPORTED_FORMATS = ("png", "pdf")
GRID_FILENAME_SUFFIX = "_grid"

# Configuration of the sample-efficiency figures included in the manuscript
# (drawn by ``--paper``). It is kept here, next to the drawing code, so that the
# figures the paper uses can be regenerated with one command and cannot drift
# from the drawing defaults. The paper repository's ``make sync`` runs it before
# copying figures; see also ``--paper`` in the CLI below.
PAPER_OUTDIR = "results/classification/figures/limited_minilm_acc_logreg"
# Coarse categories per dataset, in the order the panels appear in the paper.
PAPER_DATASET_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "20newsgroup": ("computer", "ride", "sports", "science", "religion", "politics"),
    "nyt": ("arts", "business", "politics", "sports"),
}
PAPER_TOPICS: Tuple[int, ...] = (10, 20, 30)
PAPER_METRIC = "acc"
PAPER_MODE = "ratio"
# Raw result keys of the nine compared models (MiniLM sentence embeddings,
# logistic regression) as they appear in the classification result files.
# "SAM [LogReg]" is the tf condition (runner ``sam_tf``); the tf-idf condition is a
# separate key, "SAM (tf-idf) [LogReg]", and is not the one the manuscript reports.
PAPER_MODELS: Tuple[str, ...] = (
    "Blei LDA [LogReg]",
    "sentLDA [LogReg]",
    "SAM [LogReg]",
    "Gaussian LDA [googlenews300] [LogReg]",
    "MvTM [c1_googlenews300] [LogReg]",
    "ETM [googlenews300] [LogReg]",
    "Contextual TM [minilm] [LogReg]",
    "Sentence LDA [minilm_norm_psi0-0p1] [LogReg]",
    "vMF Sentence LDA [c1_minilm] [LogReg]",
)
# The vSLDA key under each document-topic estimator (``--vmf-assignment``); the
# eight baselines are the same whichever estimator vSLDA uses.
PAPER_VMF_MODEL_BY_ASSIGNMENT: Dict[str, str] = {
    "hard": "vMF Sentence LDA [c1_minilm] [LogReg]",
    "soft": "vMF Sentence LDA (soft) [c1_minilm] [LogReg]",
    "foldin": "vMF Sentence LDA (fold-in) [c1_minilm] [LogReg]",
    "foldincounts": "vMF Sentence LDA (fold-in counts) [c1_minilm] [LogReg]",
}
VMF_MODEL_PREFIX = "vMF Sentence LDA"
# Feature names whose scores depend on the vMF-family estimator (``vmf_assignment``):
# vSLDA and, since 2026-09-06, MvTM (vLDA), whose fold-in theta is the same estimator.
VMF_FAMILY_MODEL_PREFIXES: Tuple[str, ...] = (VMF_MODEL_PREFIX, "MvTM")


def paper_models(vmf_assignment: str = DEFAULT_VMF_ASSIGNMENT) -> Tuple[str, ...]:
    """``PAPER_MODELS`` with the vSLDA key of the requested estimator."""

    try:
        vmf_key = PAPER_VMF_MODEL_BY_ASSIGNMENT[str(vmf_assignment)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported vmf_assignment {vmf_assignment!r}; "
            f"use one of {', '.join(PAPER_VMF_MODEL_BY_ASSIGNMENT)}"
        ) from exc
    return tuple(
        vmf_key if model.startswith(VMF_MODEL_PREFIX) else model
        for model in PAPER_MODELS
    )


# Maximum panel columns per dataset; the actual count is the largest one that
# fills the last row (``_grid_columns``), so all figures share one panel size
# and are included at natural size; legend at 9 pt. The default 2 gives
# 20 Newsgroups 3x2 and NYT 2x2. PAPER_MAX_GRID_COLS=3 (the TMLR layout) turns
# the six-panel grids into full-line 2x3 while the four-panel ones stay 2x2.
PAPER_GRID_COLS = int(os.environ.get("PAPER_MAX_GRID_COLS", "2"))
PAPER_GRID_LEGEND_FONTSIZE = 9.0
# PAPER_AVERAGE_GRID=1 (the TMLR layout) additionally writes, per topic count,
# one two-panel grid whose panels are the category-averaged series of each
# dataset (``combined_<K>topic_acc_ratio_average_grid``); the manuscript's main
# text shows that figure and moves the per-category grids to an appendix. Off
# by default so the default (ISwA) outputs are unchanged.
PAPER_AVERAGE_GRID = os.environ.get("PAPER_AVERAGE_GRID", "0") == "1"
# Dataset headings of the combined-figure prototype (--paper-combined).
DATASET_LABELS: Dict[str, str] = {"20newsgroup": "20 Newsgroups", "nyt": "NYT"}


def _read_json_with_meta(path: Path) -> Tuple[Dict, Dict]:
    meta, results = read_evaluation_json(path)
    if isinstance(results, dict) and isinstance(results.get("results"), dict):
        results = results["results"]
    return (meta if isinstance(meta, dict) else {}), results


def _read_json(path: Path) -> Dict:
    return _read_json_with_meta(path)[1]


def _started_at_key(value: object) -> Tuple[int, str]:
    """Sort key for ISO ``started_at`` strings; unparseable values sort oldest."""
    if value is None:
        return (0, "")
    text = str(value).strip()
    if not text:
        return (0, "")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return (1, datetime.fromisoformat(normalized).isoformat())
    except ValueError:
        return (0, text)


def _parse_filename(path: Path) -> Optional[Tuple[str, str, int, str, float]]:
    limited_match = LIMITED_FILE_RE.match(path.name)
    if limited_match:
        metric, dataset, topics, mode, value = limited_match.groups()
        try:
            value_num = float(value)
        except ValueError:
            return None
        return metric, dataset, int(topics), mode, value_num

    full_match = FULL_FILE_RE.match(path.name)
    if full_match:
        metric, dataset, topics = full_match.groups()
        return metric, dataset, int(topics), "ratio", 1.0

    return None


def _collect_archive_history_files(base_dir: Path) -> List[Path]:
    return sorted(path for path in base_dir.rglob("*.json") if _parse_filename(path))


def _collect_latest_files(base_dir: Path) -> List[Path]:
    latest_root = base_dir / "latest"
    if not latest_root.exists():
        return []
    files: list[Path] = []
    for pointer_path in sorted(latest_root.rglob(CURRENT_POINTER_FILENAME)):
        payload = load_json(pointer_path)
        if not isinstance(payload, dict):
            continue
        archive_dir_raw = payload.get("archive_dir")
        if not archive_dir_raw:
            continue
        archive_dir = resolve_project_path(str(archive_dir_raw))
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for artifact_name in ("acc", "f1"):
            artifact_path_raw = artifacts.get(artifact_name)
            if not artifact_path_raw:
                continue
            candidate = archive_dir / str(artifact_path_raw)
            if candidate.exists() and _parse_filename(candidate):
                files.append(candidate)
    return sorted(dict.fromkeys(files))


def _collect_files(base_dir: Path, *, archive_history: bool = False) -> List[Path]:
    if archive_history:
        return _collect_archive_history_files(base_dir)
    latest_files = _collect_latest_files(base_dir)
    if latest_files:
        return latest_files
    return _collect_archive_history_files(base_dir)


def _append_score(
    store: Dict,
    *,
    metric: str,
    dataset: str,
    topics: int,
    mode: str,
    value: float,
    category: str,
    model: str,
    score: float,
) -> None:
    store.setdefault(metric, {})
    store[metric].setdefault(dataset, {})
    store[metric][dataset].setdefault(topics, {})
    store[metric][dataset][topics].setdefault(mode, {})
    store[metric][dataset][topics][mode].setdefault(category, {})
    store[metric][dataset][topics][mode][category].setdefault(model, {})
    store[metric][dataset][topics][mode][category][model].setdefault(value, [])
    store[metric][dataset][topics][mode][category][model][value].append(score)


def _condition_key(meta: Dict) -> Tuple:
    """Run-condition tuple; normalised like ``summary._matches_metric_meta``."""
    values = []
    for field in CONDITION_META_FIELDS:
        recorded = meta.get(field)
        if field == "prior_scale":
            # Runs made before --prior-scale existed store null but used 0.1.
            try:
                recorded = 0.1 if recorded is None else float(recorded)
            except (TypeError, ValueError):
                recorded = str(recorded)
        elif field == "feature_resolve_mode" and recorded is None:
            recorded = "all"
        elif field == "covariance_type" and recorded is None:
            # Runs made before the covariance variants existed are full-covariance.
            recorded = "full"
        values.append(recorded)
    return tuple(values)


def _iter_metric_scores(
    metric_key: str, results: Dict
) -> Iterable[Tuple[str, str, str, float]]:
    """Yield ``(metric, category, model, score)`` from one acc/f1 payload."""
    if metric_key == "acc":
        for category, model_scores in results.items():
            for model, score in model_scores.items():
                yield "acc", category, model, score
        return
    for category, f1_payload in results.items():
        for model, score in f1_payload.get("macro", {}).items():
            yield "f1mac", category, model, score
        for model, score in f1_payload.get("micro", {}).items():
            yield "f1mic", category, model, score


def _load_scores(
    base_dir: Path,
    *,
    archive_history: bool = False,
    keep_duplicates: bool = False,
    vmf_assignment: str | None = None,
) -> Dict:
    """Aggregate scores per (metric, dataset, topics, mode, value, category, model).

    ``vmf_assignment`` selects the document-topic estimator of the vMF family
    (vSLDA and MvTM / vLDA): only result files recorded under that assignment
    contribute their scores, while the other baselines' scores are read from the
    files of every assignment (they do not depend on it). ``None`` keeps every
    file, as before.

    Several ``latest`` pointers with different display keys can describe the
    same run condition; e.g. an early pilot batch bundling many models next to
    the later per-model batch. Unless ``keep_duplicates`` is set, for each
    model and condition (iteration, sampling repeat and the meta fields that
    ``summary.py`` also matches on) only the newest ``started_at`` score is
    kept, and scores that are identical for the same iteration and repeat are
    collapsed as re-recordings of one evaluation. Files without ``iteration``
    in their meta are never merged.
    """
    Entry = Tuple[Tuple[int, str], str, float]
    # (series key, iteration, repeat) -> condition key -> entries
    candidates: Dict[Tuple, Dict[Tuple, List[Entry]]] = {}
    for path in _collect_files(base_dir, archive_history=archive_history):
        parsed = _parse_filename(path)
        if not parsed:
            continue
        metric_key, dataset, topics, mode, value = parsed
        meta, results = _read_json_with_meta(path)
        iteration = meta.get("iteration")
        mergeable = iteration is not None and not keep_duplicates
        started_key = _started_at_key(meta.get("started_at"))
        condition = _condition_key(meta)
        file_assignment = str(meta.get("vmf_assignment") or LEGACY_VMF_ASSIGNMENT)
        for metric, category, model, score in _iter_metric_scores(metric_key, results):
            if (
                vmf_assignment is not None
                and model.startswith(VMF_FAMILY_MODEL_PREFIXES)
                and file_assignment != str(vmf_assignment)
            ):
                continue
            series = (metric, dataset, topics, mode, value, category, model)
            run = (iteration, meta.get("sampling_repeat")) if mergeable else str(path)
            group = candidates.setdefault((series, run), {})
            group.setdefault(condition if mergeable else (str(path),), []).append(
                (started_key, str(path), score)
            )

    data: Dict = {}
    dropped = 0
    for (series, _run), by_condition in candidates.items():
        metric, dataset, topics, mode, value, category, model = series
        newest_per_condition = []
        for entries in by_condition.values():
            entries.sort(key=lambda item: (item[0], item[1]))
            dropped += len(entries) - 1
            newest_per_condition.append(entries[-1])
        newest_per_condition.sort(key=lambda item: (item[0], item[1]))
        seen_scores: set = set()
        for _started, _path, score in newest_per_condition:
            if score in seen_scores:
                dropped += 1
                continue
            seen_scores.add(score)
            _append_score(
                data,
                metric=metric,
                dataset=dataset,
                topics=topics,
                mode=mode,
                value=value,
                category=category,
                model=model,
                score=score,
            )
    if dropped:
        print(
            f"plot_limited: dropped {dropped} superseded or re-recorded score(s) "
            "that share iteration and sampling repeat with another run "
            "(use --keep-duplicates to keep them).",
            file=sys.stderr,
        )
    return data


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


_base_model_name = base_model_name
_display_model_name = display_model_name
_model_sort_key = model_sort_key
_fallback_color = fallback_color
_model_style = model_style


def _model_color(label: str, colormap: str) -> object:
    return _model_style(label, colormap).color


def _is_excluded(model: str) -> bool:
    return _display_model_name(model) in EXCLUDED_MODELS


def _metric_axis_label(metric: str) -> str:
    return METRIC_AXIS_LABELS.get(metric, metric)


def _mode_axis_label(mode: str) -> str:
    return MODE_AXIS_LABELS.get(mode, mode)


def _format_x_tick(value: float, mode: str, *, percent_sign: bool = True) -> str:
    if mode == "ratio":
        return f"{value * 100:g}%" if percent_sign else f"{value * 100:g}"
    return f"{value:g}"


def _label_variant_suffix(show_xlabel: bool, show_ylabel: bool) -> str:
    if show_xlabel and show_ylabel:
        return ""
    if show_xlabel:
        return "_noy"
    if show_ylabel:
        return "_nox"
    return "_noxy"


def _y_major_locator(ax: object, ytick_step: float) -> object:
    """Integer ticks every ``ytick_step`` points; fall back when the range is narrow."""
    low, high = ax.get_ylim()
    if ytick_step > 0:
        ticks_inside = int(np.floor(high / ytick_step) - np.ceil(low / ytick_step)) + 1
        if ticks_inside >= MIN_YTICKS:
            return MultipleLocator(ytick_step)
    return MaxNLocator(nbins=5, integer=True)


def _save_figure(fig: object, stem: Path, formats: Sequence[str], dpi: int) -> None:
    for fmt in formats:
        fig.savefig(stem.parent / f"{stem.name}.{fmt}", dpi=dpi)


def _draw_series(
    ax: object,
    category_data: Dict[str, Dict[float, List[float]]],
    *,
    mode: str,
    models: Optional[List[str]],
    no_errorbar: bool,
    colormap: str,
) -> Tuple[List[float], List[float]]:
    """Plot every selected model on ``ax``; return ``(x_vals, x_plot)``.

    ``x_vals`` are the data x positions (fractions or counts) and ``x_plot``
    the plotted positions (equally spaced indices for ratio mode). Both are
    empty when ``category_data`` holds no scores.
    """
    values = set()
    for model_data in category_data.values():
        values.update(model_data.keys())
    if not values:
        return [], []

    x_vals = sorted(values)
    x_plot = list(range(len(x_vals))) if mode == "ratio" else list(x_vals)
    for model, model_data in sorted(
        category_data.items(), key=lambda item: _model_sort_key(item[0])
    ):
        if models is not None and model not in models:
            continue
        if _is_excluded(model):
            continue
        means = []
        stds = []
        for x in x_vals:
            if x not in model_data:
                means.append(np.nan)
                stds.append(0.0)
            else:
                mean, std = _mean_std(model_data[x])
                means.append(mean)
                stds.append(std)
        if all(np.isnan(means)):
            continue
        label = _display_model_name(model)
        style = _model_style(label, colormap)
        ax.plot(
            x_plot,
            means,
            label=label,
            alpha=LINE_ALPHA,
            zorder=style.zorder,
            **style.line_kwargs(),
        )
        if not no_errorbar:
            lower = np.asarray(means) - np.asarray(stds)
            upper = np.asarray(means) + np.asarray(stds)
            ax.fill_between(
                x_plot,
                lower,
                upper,
                alpha=ERROR_ALPHA,
                color=style.color,
                linewidth=0,
                zorder=1,
            )
    return x_vals, x_plot


def _x_axis_label(mode: str, *, compact_ticks: bool = False) -> str:
    xlabel = _mode_axis_label(mode)
    if compact_ticks and mode == "ratio":
        return f"{xlabel} (%)"
    return xlabel


def _finish_axes(
    ax: object,
    *,
    metric: str,
    mode: str,
    x_vals: Sequence[float],
    x_plot: Sequence[float],
    ylim: Optional[Tuple[float, float]],
    ytick_step: float,
    show_xlabel: bool,
    show_ylabel: bool,
    show_xticklabels: bool = True,
    compact_ticks: bool = False,
) -> None:
    """Axis labels, ticks, spines and grid shared by single panels and grids.

    ``compact_ticks`` moves the percent sign of ratio-mode tick labels into the
    axis label ("... (%)") so that seven tick labels fit a narrow panel.
    """
    if show_xlabel:
        ax.set_xlabel(_x_axis_label(mode, compact_ticks=compact_ticks))
    if show_ylabel:
        ax.set_ylabel(_metric_axis_label(metric))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(_y_major_locator(ax, ytick_step))
    panel_frame(ax)
    ax.set_xticks(list(x_plot))
    ax.set_xticklabels(
        [_format_x_tick(x, mode, percent_sign=not compact_ticks) for x in x_vals]
    )
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)


def _plot_category(
    *,
    metric: str,
    dataset: str,
    topics: int,
    mode: str,
    category: str,
    category_data: Dict[str, Dict[float, List[float]]],
    outdir: Path,
    models: Optional[List[str]],
    no_errorbar: bool,
    ylim: Optional[Tuple[float, float]],
    colormap: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    figsize: Tuple[float, float] = CATEGORY_FIGSIZE,
    dpi: int = DPI,
    formats: Sequence[str] = DEFAULT_FORMATS,
    ytick_step: float = YTICK_STEP,
) -> Optional[Path]:
    """Draw one panel and return the output stem (without extension), or None."""
    if not any(model_data for model_data in category_data.values()):
        return None

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=figsize)
        x_vals, x_plot = _draw_series(
            ax,
            category_data,
            mode=mode,
            models=models,
            no_errorbar=no_errorbar,
            colormap=colormap,
        )
        _finish_axes(
            ax,
            metric=metric,
            mode=mode,
            x_vals=x_vals,
            x_plot=x_plot,
            ylim=ylim,
            ytick_step=ytick_step,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
        )
        fig.tight_layout(pad=0.3)

        suffix = _label_variant_suffix(show_xlabel, show_ylabel)
        stem = outdir / f"{dataset}_{topics}topic_{metric}_{mode}_{category}{suffix}"
        _save_figure(fig, stem, formats, dpi)
        plt.close(fig)
    return stem


def _legend_handles(labels: Sequence[str], colormap: str) -> List[object]:
    """Proxy artists carrying exactly the series styles used in the panels."""
    return [
        plt.Line2D([0], [0], label=label, **_model_style(label, colormap).line_kwargs())
        for label in labels
    ]


def _row_major(items: Sequence, ncol: int) -> List:
    """Reorder for a column-filled legend; see ``grid_layout.legend_row_major``."""
    return legend_row_major(items, ncol)


def _grid_columns(panel_count: int, max_cols: int) -> int:
    """Largest column count <= ``max_cols`` that fills the last row, if any."""
    max_cols = max(1, min(max_cols, panel_count))
    for cols in range(max_cols, 1, -1):
        if panel_count % cols == 0:
            return cols
    return max_cols


def _grid_geometry(
    nrows: int,
    ncols: int,
    legend_rows: int,
    legend_fontsize: float = GRID_LEGEND_FONTSIZE,
) -> Tuple[Tuple[float, float], Dict[str, float], float]:
    """Figure size (in), ``subplots_adjust`` fractions and legend height (in).

    Thin wrapper over :func:`src.evaluation.reports.grid_layout.grid_geometry`,
    which the topic-count sweep figures share so that both families keep the
    same panel size on the page.
    """
    return grid_geometry(nrows, ncols, legend_rows, legend_fontsize)


def _plot_grid(
    *,
    metric: str,
    dataset: str,
    topics: int,
    mode: str,
    mode_data: Dict[str, Dict[str, Dict[float, List[float]]]],
    categories: Sequence[str],
    outdir: Path,
    models: Optional[List[str]],
    no_errorbar: bool,
    ylim: Optional[Tuple[float, float]],
    colormap: str,
    max_cols: int = GRID_NCOLS,
    legend_ncol: int = GRID_LEGEND_NCOL,
    legend_fontsize: float = GRID_LEGEND_FONTSIZE,
    dpi: int = DPI,
    formats: Sequence[str] = DEFAULT_FORMATS,
    ytick_step: float = YTICK_STEP,
) -> Optional[Path]:
    """Draw all ``categories`` (in the given order) as one multi-panel figure.

    Panel titles are the category names; the y label appears on the left
    column only, x tick labels on the bottom row only, and a single shared
    x label and legend are placed below the panels. The synthetic ``all`` and
    ``average`` entries are never part of the grid. Returns the output stem.
    """
    present = [
        category
        for category in categories
        if category not in ("all", "average")
        and any(model_data for model_data in mode_data.get(category, {}).values())
    ]
    if not present:
        return None

    ncols = _grid_columns(len(present), max_cols)
    nrows = int(np.ceil(len(present) / ncols))
    labels: List[str] = []
    for category in present:
        for model in sorted(mode_data[category], key=_model_sort_key):
            if models is not None and model not in models:
                continue
            if _is_excluded(model):
                continue
            label = _display_model_name(model)
            if label not in labels:
                labels.append(label)
    labels.sort(key=_model_sort_key)
    legend_ncol = max(1, min(legend_ncol_for(ncols, legend_ncol), len(labels)))
    legend_rows = int(np.ceil(len(labels) / legend_ncol)) if labels else 0

    with plt.rc_context(PAPER_RC):
        figsize, adjust, legend_h = _grid_geometry(
            nrows, ncols, legend_rows, legend_fontsize
        )
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(**adjust)
        for index, category in enumerate(present):
            row, col = divmod(index, ncols)
            ax = axes[row][col]
            x_vals, x_plot = _draw_series(
                ax,
                mode_data[category],
                mode=mode,
                models=models,
                no_errorbar=no_errorbar,
                colormap=colormap,
            )
            bottom = index + ncols >= len(present)
            _finish_axes(
                ax,
                metric=metric,
                mode=mode,
                x_vals=x_vals,
                x_plot=x_plot,
                ylim=ylim,
                ytick_step=ytick_step,
                show_xlabel=False,
                show_ylabel=col == 0,
                show_xticklabels=bottom,
                compact_ticks=True,
            )
            ax.set_title(category, fontstyle="italic")
        for ax in axes.flat[len(present) :]:
            ax.set_visible(False)
        fig.supxlabel(
            _x_axis_label(mode, compact_ticks=True),
            x=0.5 * (adjust["left"] + adjust["right"]),
            y=(legend_h + GRID_XLABEL_OFFSET) / figsize[1],
            va="bottom",
            fontsize=PAPER_RC["axes.labelsize"],
        )

        if labels:
            legend = fig.legend(
                handles=_row_major(_legend_handles(labels, colormap), legend_ncol),
                labels=_row_major(labels, legend_ncol),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.5 * GRID_LEGEND_PAD / figsize[1]),
                ncol=legend_ncol,
                fontsize=legend_fontsize,
                frameon=False,
                handlelength=2.6,
                columnspacing=1.2,
                handletextpad=0.6,
                borderaxespad=0.0,
            )
            check_legend_fits(fig, legend)

        stem = outdir / f"{dataset}_{topics}topic_{metric}_{mode}{GRID_FILENAME_SUFFIX}"
        _save_figure(fig, stem, formats, dpi)
        plt.close(fig)
    return stem


def _plot_grid_average(
    *,
    metric: str,
    topics: int,
    mode: str,
    dataset_blocks: Sequence[
        Tuple[str, Sequence[str], Dict[str, Dict[str, Dict[float, List[float]]]]]
    ],
    outdir: Path,
    models: Optional[List[str]],
    colormap: str,
    legend_ncol: int = GRID_LEGEND_NCOL,
    legend_fontsize: float = GRID_LEGEND_FONTSIZE,
    dpi: int = DPI,
    formats: Sequence[str] = DEFAULT_FORMATS,
    ytick_step: float = YTICK_STEP,
) -> Optional[Path]:
    """One panel per dataset showing the category-averaged series.

    Each entry of ``dataset_blocks`` is ``(dataset, categories, mode_data)``;
    the panel of a dataset draws, for every model, the mean over its
    ``categories`` of the per-category mean score, with the band spanning one
    standard deviation of those per-category means (the same ``average``
    series as the single-panel ``*_average`` files). Panels share the size of
    every other paper grid, so the figure is as wide as a 2-column grid.
    """
    blocks = []
    for dataset, categories, mode_data in dataset_blocks:
        averaged = _build_average_category_data(mode_data, categories=list(categories))
        if averaged:
            blocks.append((dataset, averaged))
    if not blocks:
        return None
    ncols = len(blocks)
    labels: List[str] = []
    for _, averaged in blocks:
        for model in sorted(averaged, key=_model_sort_key):
            if models is not None and model not in models:
                continue
            if _is_excluded(model):
                continue
            label = _display_model_name(model)
            if label not in labels:
                labels.append(label)
    labels.sort(key=_model_sort_key)
    legend_ncol = max(1, min(legend_ncol_for(ncols, legend_ncol), len(labels)))
    legend_rows = int(np.ceil(len(labels) / legend_ncol)) if labels else 0

    with plt.rc_context(PAPER_RC):
        figsize, adjust, legend_h = _grid_geometry(
            1, ncols, legend_rows, legend_fontsize
        )
        fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(**adjust)
        for col, (dataset, averaged) in enumerate(blocks):
            ax = axes[0][col]
            x_vals, x_plot = _draw_series(
                ax,
                averaged,
                mode=mode,
                models=models,
                no_errorbar=False,
                colormap=colormap,
            )
            _finish_axes(
                ax,
                metric=metric,
                mode=mode,
                x_vals=x_vals,
                x_plot=x_plot,
                ylim=None,
                ytick_step=ytick_step,
                show_xlabel=False,
                show_ylabel=col == 0,
                show_xticklabels=True,
                compact_ticks=True,
            )
            ax.set_title(DATASET_LABELS.get(dataset, dataset), fontstyle="italic")
        fig.supxlabel(
            _x_axis_label(mode, compact_ticks=True),
            x=0.5 * (adjust["left"] + adjust["right"]),
            y=(legend_h + GRID_XLABEL_OFFSET) / figsize[1],
            va="bottom",
            fontsize=PAPER_RC["axes.labelsize"],
        )
        if labels:
            legend = fig.legend(
                handles=_row_major(_legend_handles(labels, colormap), legend_ncol),
                labels=_row_major(labels, legend_ncol),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.5 * GRID_LEGEND_PAD / figsize[1]),
                ncol=legend_ncol,
                fontsize=legend_fontsize,
                frameon=False,
                handlelength=2.6,
                columnspacing=1.2,
                handletextpad=0.6,
                borderaxespad=0.0,
            )
            check_legend_fits(fig, legend)
        stem = (
            outdir
            / f"combined_{topics}topic_{metric}_{mode}_average{GRID_FILENAME_SUFFIX}"
        )
        _save_figure(fig, stem, formats, dpi)
        plt.close(fig)
    return stem


def _plot_grid_combined(
    *,
    metric: str,
    topics: int,
    mode: str,
    dataset_blocks: Sequence[
        Tuple[str, Sequence[str], Dict[str, Dict[str, Dict[float, List[float]]]]]
    ],
    outdir: Path,
    models: Optional[List[str]],
    colormap: str,
    block_cols: int = 2,
    legend_ncol: int = GRID_LEGEND_NCOL,
    legend_fontsize: float = GRID_LEGEND_FONTSIZE,
    dpi: int = DPI,
    formats: Sequence[str] = DEFAULT_FORMATS,
    ytick_step: float = YTICK_STEP,
    xtick_every: int = 2,
) -> Optional[Path]:
    """PROTOTYPE: both datasets side by side in one full-line figure.

    Each entry of ``dataset_blocks`` is ``(dataset, categories, mode_data)`` and
    becomes a ``block_cols``-wide block of panels; the blocks sit next to each
    other under a dataset heading, with one shared legend and x label. The
    figure is solved to the full manuscript line (``GRID_TEXT_WIDTH``), so with
    two 2-column blocks each panel is about 1.16 in wide — x tick labels are
    thinned to every ``xtick_every``-th to stay legible. Kept separate from
    ``_plot_grid`` so the shipped per-dataset figures are untouched while the
    combined layout is evaluated.
    """
    blocks = []
    for dataset, categories, mode_data in dataset_blocks:
        present = [
            category
            for category in categories
            if category not in ("all", "average")
            and any(model_data for model_data in mode_data.get(category, {}).values())
        ]
        if present:
            blocks.append((dataset, present, mode_data))
    if not blocks:
        return None

    ncols = block_cols * len(blocks)
    nrows = max(int(np.ceil(len(present) / block_cols)) for _, present, _ in blocks)
    labels: List[str] = []
    for _, present, mode_data in blocks:
        for category in present:
            for model in sorted(mode_data[category], key=_model_sort_key):
                if models is not None and model not in models:
                    continue
                if _is_excluded(model):
                    continue
                label = _display_model_name(model)
                if label not in labels:
                    labels.append(label)
    labels.sort(key=_model_sort_key)
    legend_ncol = max(1, min(legend_ncol_for(ncols, legend_ncol), len(labels)))
    legend_rows = int(np.ceil(len(labels) / legend_ncol)) if labels else 0

    with plt.rc_context(PAPER_RC):
        # Extra head room for the dataset heading row above the panel titles.
        heading_h = 0.20
        figsize, adjust, legend_h = grid_geometry(
            nrows,
            ncols,
            legend_rows,
            legend_fontsize,
            total_width=GRID_TEXT_WIDTH,
            margin_top=GRID_MARGIN_TOP + heading_h,
        )
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(**adjust)
        for block_index, (dataset, present, mode_data) in enumerate(blocks):
            col0 = block_index * block_cols
            for index, category in enumerate(present):
                row, col = divmod(index, block_cols)
                ax = axes[row][col0 + col]
                x_vals, x_plot = _draw_series(
                    ax,
                    mode_data[category],
                    mode=mode,
                    models=models,
                    no_errorbar=False,
                    colormap=colormap,
                )
                bottom = index + block_cols >= len(present)
                _finish_axes(
                    ax,
                    metric=metric,
                    mode=mode,
                    x_vals=x_vals,
                    x_plot=x_plot,
                    ylim=None,
                    ytick_step=ytick_step,
                    show_xlabel=False,
                    show_ylabel=col0 + col == 0,
                    show_xticklabels=bottom,
                    compact_ticks=True,
                )
                if bottom and xtick_every > 1:
                    tick_labels = [label.get_text() for label in ax.get_xticklabels()]
                    ax.set_xticklabels(
                        [
                            text if i % xtick_every == 0 else ""
                            for i, text in enumerate(tick_labels)
                        ]
                    )
                ax.set_title(category, fontstyle="italic")
            for index in range(len(present), nrows * block_cols):
                row, col = divmod(index, block_cols)
                axes[row][col0 + col].set_visible(False)
            # Dataset heading centered over the block's columns.
            block_axes = [axes[0][col0 + c] for c in range(block_cols)]
            x_left = block_axes[0].get_position().x0
            x_right = block_axes[-1].get_position().x1
            fig.text(
                (x_left + x_right) / 2,
                1.0 - 0.55 * heading_h / figsize[1],
                DATASET_LABELS.get(dataset, dataset),
                ha="center",
                va="top",
                fontsize=PAPER_RC["axes.labelsize"],
                fontweight="bold",
            )
        fig.supxlabel(
            _x_axis_label(mode, compact_ticks=True),
            x=0.5 * (adjust["left"] + adjust["right"]),
            y=(legend_h + GRID_XLABEL_OFFSET) / figsize[1],
            va="bottom",
            fontsize=PAPER_RC["axes.labelsize"],
        )
        if labels:
            legend = fig.legend(
                handles=_row_major(_legend_handles(labels, colormap), legend_ncol),
                labels=_row_major(labels, legend_ncol),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.5 * GRID_LEGEND_PAD / figsize[1]),
                ncol=legend_ncol,
                fontsize=legend_fontsize,
                frameon=False,
                handlelength=2.6,
                columnspacing=1.2,
                handletextpad=0.6,
                borderaxespad=0.0,
            )
            check_legend_fits(fig, legend)
        stem = outdir / f"combined_{topics}topic_{metric}_{mode}{GRID_FILENAME_SUFFIX}"
        _save_figure(fig, stem, formats, dpi)
        plt.close(fig)
    return stem


def _build_average_category_data(
    mode_data: Dict[str, Dict[str, Dict[float, List[float]]]],
    *,
    categories: Optional[List[str]],
) -> Dict[str, Dict[float, List[float]]]:
    averaged: Dict[str, Dict[float, List[float]]] = {}
    for category, category_data in mode_data.items():
        if category == "all":
            continue
        if categories is not None and category not in categories:
            continue
        for model, model_data in category_data.items():
            for x_value, scores in model_data.items():
                mean_score, _ = _mean_std(scores)
                averaged.setdefault(model, {}).setdefault(x_value, []).append(
                    mean_score
                )
    return averaged


def _collect_legend_models(
    data: Dict,
    *,
    metrics: Iterable[str],
    datasets: Iterable[str],
    topics_list: Iterable[int],
    modes: Iterable[str],
    categories: Optional[List[str]],
    models: Optional[List[str]],
    include_average: bool,
) -> List[str]:
    legend_models: set[str] = set()
    for metric in metrics:
        metric_data = data.get(metric, {})
        for dataset in datasets:
            ds_data = metric_data.get(dataset, {})
            for topics in topics_list:
                topics_data = ds_data.get(topics, {})
                for mode in modes:
                    mode_data = topics_data.get(mode, {})
                    for category, category_data in mode_data.items():
                        if categories is not None and category not in categories:
                            continue
                        for model in category_data:
                            if models is None or model in models:
                                if not _is_excluded(model):
                                    legend_models.add(model)
                    if include_average:
                        average_data = _build_average_category_data(
                            mode_data,
                            categories=categories,
                        )
                        for model in average_data:
                            if models is None or model in models:
                                if not _is_excluded(model):
                                    legend_models.add(model)
    return sorted(legend_models, key=lambda model: (*_model_sort_key(model), model))


def _write_legend_figure(
    *,
    models: List[str],
    outdir: Path,
    colormap: str,
    filename: str = "legend.png",
    ncol: int = 1,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DPI,
) -> None:
    if not models:
        return
    _ensure_dir(outdir)
    labels = [_display_model_name(model) for model in models]
    unique_labels = list(dict.fromkeys(labels))
    column_count = min(ncol, len(unique_labels))
    row_count = int(np.ceil(len(unique_labels) / column_count))
    if column_count == 1:
        fig_width = 2.4
        fig_height = max(1.6, 0.26 * len(unique_labels) + 0.3)
    else:
        fig_width = max(5.0, 1.1 * column_count)
        fig_height = max(0.6, 0.32 * row_count + 0.3)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        handles = _legend_handles(unique_labels, colormap)
        fig.legend(
            handles=handles,
            labels=unique_labels,
            loc="center",
            ncol=column_count,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=1.0,
            handlelength=2.6,
            columnspacing=1.2,
        )
        stem = outdir / Path(filename).stem
        for fmt in formats:
            fig.savefig(
                stem.parent / f"{stem.name}.{fmt}",
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.03,
            )
        plt.close(fig)


def _plot_all(
    data: Dict,
    *,
    metrics: Iterable[str],
    datasets: Iterable[str],
    topics_list: Iterable[int],
    modes: Iterable[str],
    categories: Optional[List[str]],
    models: Optional[List[str]],
    outdir: Path,
    no_errorbar: bool,
    include_average: bool,
    ylim: Optional[Tuple[float, float]],
    colormap: str,
    label_variants: Sequence[Tuple[bool, bool]] = ((True, True),),
    figsize: Tuple[float, float] = CATEGORY_FIGSIZE,
    dpi: int = DPI,
    formats: Sequence[str] = DEFAULT_FORMATS,
    ytick_step: float = YTICK_STEP,
) -> None:
    def _draw(
        category: str,
        category_data: Dict[str, Dict[float, List[float]]],
        *,
        metric: str,
        dataset: str,
        topics: int,
        mode: str,
        errorbar: bool,
    ) -> None:
        for show_xlabel, show_ylabel in label_variants:
            _plot_category(
                metric=metric,
                dataset=dataset,
                topics=topics,
                mode=mode,
                category=category,
                category_data=category_data,
                outdir=outdir,
                models=models,
                no_errorbar=not errorbar,
                ylim=ylim,
                colormap=colormap,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                figsize=figsize,
                dpi=dpi,
                formats=formats,
                ytick_step=ytick_step,
            )

    for metric in metrics:
        metric_data = data.get(metric, {})
        for dataset in datasets:
            ds_data = metric_data.get(dataset, {})
            for topics in topics_list:
                topics_data = ds_data.get(topics, {})
                for mode in modes:
                    mode_data = topics_data.get(mode, {})
                    for category, category_data in mode_data.items():
                        if categories is not None and category not in categories:
                            continue
                        _draw(
                            category,
                            category_data,
                            metric=metric,
                            dataset=dataset,
                            topics=topics,
                            mode=mode,
                            errorbar=not no_errorbar,
                        )
                    if include_average:
                        average_data = _build_average_category_data(
                            mode_data,
                            categories=categories,
                        )
                        if average_data:
                            _draw(
                                "average",
                                average_data,
                                metric=metric,
                                dataset=dataset,
                                topics=topics,
                                mode=mode,
                                errorbar=False,
                            )


def _plot_grids(
    data: Dict,
    *,
    metrics: Iterable[str],
    datasets: Iterable[str],
    topics_list: Iterable[int],
    modes: Iterable[str],
    categories: Optional[List[str]],
    models: Optional[List[str]],
    outdir: Path,
    no_errorbar: bool,
    ylim: Optional[Tuple[float, float]],
    colormap: str,
    max_cols: int = GRID_NCOLS,
    legend_fontsize: float = GRID_LEGEND_FONTSIZE,
    dpi: int = DPI,
    formats: Sequence[str] = DEFAULT_FORMATS,
    ytick_step: float = YTICK_STEP,
) -> List[Path]:
    """One grid figure per (metric, dataset, topics, mode); returns the stems."""
    stems: List[Path] = []
    for metric in metrics:
        metric_data = data.get(metric, {})
        for dataset in datasets:
            ds_data = metric_data.get(dataset, {})
            for topics in topics_list:
                topics_data = ds_data.get(topics, {})
                for mode in modes:
                    mode_data = topics_data.get(mode, {})
                    if not mode_data:
                        continue
                    panel_categories = (
                        list(categories)
                        if categories is not None
                        else sorted(c for c in mode_data if c != "all")
                    )
                    stem = _plot_grid(
                        metric=metric,
                        dataset=dataset,
                        topics=topics,
                        mode=mode,
                        mode_data=mode_data,
                        categories=panel_categories,
                        outdir=outdir,
                        models=models,
                        no_errorbar=no_errorbar,
                        ylim=ylim,
                        colormap=colormap,
                        max_cols=max_cols,
                        legend_fontsize=legend_fontsize,
                        dpi=dpi,
                        formats=formats,
                        ytick_step=ytick_step,
                    )
                    if stem is not None:
                        stems.append(stem)
    return stems


def _paper_grid_stem(outdir: Path, dataset: str, topics: int) -> Path:
    return (
        outdir
        / f"{dataset}_{topics}topic_{PAPER_METRIC}_{PAPER_MODE}{GRID_FILENAME_SUFFIX}"
    )


def _plot_paper_figures(
    data: Dict,
    *,
    outdir: Path,
    datasets: Sequence[str],
    topics: Sequence[int],
    colormap: str = DEFAULT_COLORMAP,
    dpi: int = DPI,
    formats: Sequence[str] = DEFAULT_FORMATS,
    ytick_step: float = YTICK_STEP,
    models: Sequence[str] = PAPER_MODELS,
) -> List[Path]:
    """Draw the manuscript's sample-efficiency figures and return the grid stems.

    Per-category panels, the legend files and one grid per (dataset, topic
    count) are written with the fixed paper configuration above. Raises
    ``SystemExit`` when a grid that the paper needs was not produced, so that a
    stale figure is never carried into the manuscript unnoticed.
    """
    started = time.time()
    for dataset in datasets:
        categories = list(PAPER_DATASET_CATEGORIES[dataset])
        common = dict(
            metrics=[PAPER_METRIC],
            datasets=[dataset],
            topics_list=list(topics),
            modes=[PAPER_MODE],
            categories=categories,
            models=list(models),
            outdir=outdir,
            no_errorbar=False,
            ylim=None,
            colormap=colormap,
            dpi=dpi,
            formats=formats,
            ytick_step=ytick_step,
        )
        _plot_all(
            data,
            include_average=True,
            label_variants=ALL_LABEL_VARIANTS,
            **common,
        )
        _plot_grids(
            data,
            max_cols=PAPER_GRID_COLS,
            legend_fontsize=PAPER_GRID_LEGEND_FONTSIZE,
            **common,
        )

    if PAPER_AVERAGE_GRID:
        for topic in topics:
            dataset_blocks = []
            for dataset in datasets:
                mode_data = (
                    data.get(PAPER_METRIC, {})
                    .get(dataset, {})
                    .get(topic, {})
                    .get(PAPER_MODE, {})
                )
                if mode_data:
                    dataset_blocks.append(
                        (dataset, list(PAPER_DATASET_CATEGORIES[dataset]), mode_data)
                    )
            stem = _plot_grid_average(
                metric=PAPER_METRIC,
                topics=topic,
                mode=PAPER_MODE,
                dataset_blocks=dataset_blocks,
                outdir=outdir,
                models=list(models),
                colormap=colormap,
                legend_fontsize=PAPER_GRID_LEGEND_FONTSIZE,
                dpi=dpi,
                formats=formats,
                ytick_step=ytick_step,
            )
            if stem is None:
                raise SystemExit(
                    f"average grid for K={topic} was not produced (no scores loaded)"
                )
            print(f"[write] {stem}")

    legend_models = _collect_legend_models(
        data,
        metrics=[PAPER_METRIC],
        datasets=list(datasets),
        topics_list=list(topics),
        modes=[PAPER_MODE],
        categories=None,
        models=list(models),
        include_average=True,
    )
    _write_legend_figure(
        models=legend_models,
        outdir=outdir,
        colormap=colormap,
        formats=formats,
        dpi=dpi,
    )
    _write_legend_figure(
        models=legend_models,
        outdir=outdir,
        colormap=colormap,
        filename="legend_h.png",
        ncol=9,
        formats=formats,
        dpi=dpi,
    )

    stems = [
        _paper_grid_stem(outdir, dataset, topic)
        for dataset in datasets
        for topic in topics
    ]
    missing = [
        stem.parent / f"{stem.name}.{fmt}"
        for stem in stems
        for fmt in formats
        if not (stem.parent / f"{stem.name}.{fmt}").is_file()
        or (stem.parent / f"{stem.name}.{fmt}").stat().st_mtime < started - 1.0
    ]
    for stem in stems:
        print(f"plot_limited: paper figure {stem}.{'/'.join(formats)}")
    if missing:
        raise SystemExit(
            "plot_limited: --paper did not produce "
            + ", ".join(str(path) for path in missing)
            + "; check that results exist for every dataset and topic count."
        )
    return stems


ALL_LABEL_VARIANTS: Tuple[Tuple[bool, bool], ...] = (
    (True, True),
    (False, True),
    (True, False),
    (False, False),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot limited training results (accuracy and f1) as figures."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/classification",
        help="Base results directory (default: results/classification).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Output directory for figures (default: "
            f"results/classification/figures/limited; with --paper: {PAPER_OUTDIR})."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=["acc", "f1mac", "f1mic"],
        choices=["acc", "f1mac", "f1mic"],
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Datasets to plot (default: all found).",
    )
    parser.add_argument(
        "--topics",
        type=int,
        nargs="*",
        default=None,
        help="Topic counts to plot (default: all found).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=["ratio", "count"],
        choices=["ratio", "count"],
        help="Modes to plot.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Categories to plot (default: all found).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Models to plot (default: all found).",
    )
    parser.add_argument(
        "--no_errorbar",
        action="store_true",
        help="Disable error bars.",
    )
    parser.add_argument(
        "--include_average",
        action="store_true",
        help="Also plot the mean score across selected non-all categories.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Set a fixed y-axis range, for example --ylim 0 100.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default=DEFAULT_COLORMAP,
        help=(
            "Matplotlib colormap used to assign model colors mechanically "
            f"(default: {DEFAULT_COLORMAP})."
        ),
    )
    parser.add_argument(
        "--archive-history",
        action="store_true",
        help="Read every archived/legacy JSON file instead of only latest pointers.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help=(
            "Keep every score even when several runs share the same iteration "
            "and sampling repeat (default: keep only the newest started_at)."
        ),
    )
    parser.add_argument(
        "--legend-only",
        action="store_true",
        help="Only write legend.png and legend_h.png; skip category figures.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=list(DEFAULT_FORMATS),
        choices=list(SUPPORTED_FORMATS),
        help="Output formats (default: png pdf).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(CATEGORY_FIGSIZE),
        help=f"Panel size in inches (default: {CATEGORY_FIGSIZE[0]} {CATEGORY_FIGSIZE[1]}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help=f"Raster resolution for PNG output (default: {DPI}).",
    )
    parser.add_argument(
        "--ytick-step",
        type=float,
        default=YTICK_STEP,
        help=(
            "Spacing of y-axis major ticks in score points; panels with a narrow "
            f"range fall back to automatic integer ticks (default: {YTICK_STEP:g})."
        ),
    )
    parser.add_argument(
        "--no-xlabel",
        action="store_true",
        help="Omit the x-axis label (output filename gets a _nox suffix).",
    )
    parser.add_argument(
        "--no-ylabel",
        action="store_true",
        help="Omit the y-axis label (output filename gets a _noy suffix).",
    )
    parser.add_argument(
        "--label-variants",
        action="store_true",
        help=(
            "Write every axis-label combination in one run: the labeled panel "
            "plus _nox, _noy and _noxy variants for grid layouts."
        ),
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Draw the manuscript's sample-efficiency figures with the fixed "
            "configuration above (both datasets with their paper panel order, "
            f"K={'/'.join(str(k) for k in PAPER_TOPICS)}, the nine compared models, "
            f"{PAPER_GRID_COLS}-column grids with a "
            f"{PAPER_GRID_LEGEND_FONTSIZE:g} pt legend, written to {PAPER_OUTDIR}). "
            "Other selection options are ignored except --base_dir, --outdir, "
            "--datasets, --topics, --formats and --dpi; exits non-zero if a grid "
            "the paper needs is missing."
        ),
    )
    parser.add_argument(
        "--paper-combined",
        action="store_true",
        help=(
            "PROTOTYPE: additionally draw one full-line figure per topic count "
            "with both datasets' panels side by side "
            f"(combined_<K>topic_{PAPER_METRIC}_{PAPER_MODE}{GRID_FILENAME_SUFFIX}.<fmt> "
            f"in {PAPER_OUTDIR}); for evaluating a merged Figure 2+3 layout. "
            "Implies the fixed paper configuration; use together with --paper."
        ),
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help=(
            "Additionally write one multi-panel figure per dataset/topic count: "
            f"<dataset>_<K>topic_<metric>_<mode>{GRID_FILENAME_SUFFIX}.<fmt>, with the "
            "panels in --categories order, edge-only axis labels and a shared "
            "legend, sized for 1:1 placement at the manuscript text width. "
            "The average series is not part of the grid."
        ),
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=GRID_NCOLS,
        help=(
            "Maximum number of panel columns in --grid figures; fewer are used "
            f"when that fills the last row (default: {GRID_NCOLS})."
        ),
    )
    parser.add_argument(
        "--grid-legend-fontsize",
        type=float,
        default=GRID_LEGEND_FONTSIZE,
        help=(
            "Legend font size in points for --grid figures "
            f"(default: {GRID_LEGEND_FONTSIZE:g}; wide grids may use a larger size)."
        ),
    )
    parser.add_argument(
        "--vmf-assignment",
        dest="vmf_assignment",
        choices=sorted(PAPER_VMF_MODEL_BY_ASSIGNMENT),
        default=None,
        help=(
            "vSLDA document-topic estimator to draw (hard, soft or foldin); the "
            "baselines are read from every result file. With --paper the vSLDA "
            "series is the key of that estimator and, unless --outdir is given, "
            "the figures go to <paper outdir>_<assignment> for soft/foldin. "
            "Default: no filtering, as before."
        ),
    )
    args = parser.parse_args()

    ylim = tuple(args.ylim) if args.ylim is not None else None
    if ylim is not None and ylim[0] >= ylim[1]:
        raise SystemExit("--ylim requires MIN to be smaller than MAX.")
    if args.label_variants:
        label_variants: Tuple[Tuple[bool, bool], ...] = ALL_LABEL_VARIANTS
    else:
        label_variants = ((not args.no_xlabel, not args.no_ylabel),)
    figsize = (float(args.figsize[0]), float(args.figsize[1]))
    if figsize[0] <= 0 or figsize[1] <= 0:
        raise SystemExit("--figsize requires positive WIDTH and HEIGHT.")
    if args.grid_cols < 1:
        raise SystemExit("--grid-cols must be at least 1.")
    if args.grid_legend_fontsize <= 0:
        raise SystemExit("--grid-legend-fontsize must be positive.")

    base_dir = Path(args.base_dir)
    data = _load_scores(
        base_dir,
        archive_history=args.archive_history,
        keep_duplicates=args.keep_duplicates,
        vmf_assignment=args.vmf_assignment,
    )
    selected_models = paper_models(args.vmf_assignment or DEFAULT_VMF_ASSIGNMENT)
    if not data:
        raise SystemExit(f"No result files found under {base_dir}.")

    if args.paper or args.paper_combined:
        paper_datasets = [
            dataset
            for dataset in PAPER_DATASET_CATEGORIES
            if not args.datasets or dataset in args.datasets
        ]
        if not paper_datasets:
            raise SystemExit(
                "--paper covers "
                + ", ".join(PAPER_DATASET_CATEGORIES)
                + f"; --datasets {' '.join(args.datasets)} selects none of them."
            )
        if args.outdir:
            paper_outdir = Path(args.outdir)
        elif args.vmf_assignment in (None, DEFAULT_VMF_ASSIGNMENT):
            paper_outdir = Path(PAPER_OUTDIR)
        else:
            paper_outdir = Path(f"{PAPER_OUTDIR}_{args.vmf_assignment}")
        _ensure_dir(paper_outdir)
        if args.paper:
            _plot_paper_figures(
                data,
                outdir=paper_outdir,
                datasets=paper_datasets,
                topics=args.topics or PAPER_TOPICS,
                colormap=args.colormap,
                dpi=args.dpi,
                formats=args.formats,
                ytick_step=args.ytick_step,
                models=selected_models,
            )
        if args.paper_combined:
            metric_data = data.get(PAPER_METRIC, {})
            for topics in args.topics or PAPER_TOPICS:
                dataset_blocks = []
                for dataset in paper_datasets:
                    mode_data = (
                        metric_data.get(dataset, {}).get(topics, {}).get(PAPER_MODE, {})
                    )
                    if mode_data:
                        dataset_blocks.append(
                            (
                                dataset,
                                list(PAPER_DATASET_CATEGORIES[dataset]),
                                mode_data,
                            )
                        )
                stem = _plot_grid_combined(
                    metric=PAPER_METRIC,
                    topics=topics,
                    mode=PAPER_MODE,
                    dataset_blocks=dataset_blocks,
                    outdir=paper_outdir,
                    models=list(selected_models),
                    colormap=args.colormap,
                    legend_fontsize=PAPER_GRID_LEGEND_FONTSIZE,
                    dpi=args.dpi,
                    formats=args.formats,
                    ytick_step=args.ytick_step,
                )
                if stem is not None:
                    print(
                        f"plot_limited: combined prototype {stem}.{'/'.join(args.formats)}"
                    )
        return

    datasets = args.datasets or sorted(
        {ds for metric_data in data.values() for ds in metric_data.keys()}
    )
    topics_list = args.topics or sorted(
        {
            topic
            for metric_data in data.values()
            for ds_data in metric_data.values()
            for topic in ds_data.keys()
        }
    )

    outdir = Path(args.outdir or "results/classification/figures/limited")
    _ensure_dir(outdir)

    if not args.legend_only:
        _plot_all(
            data,
            metrics=args.metrics,
            datasets=datasets,
            topics_list=topics_list,
            modes=args.modes,
            categories=args.categories,
            models=args.models,
            outdir=outdir,
            no_errorbar=args.no_errorbar,
            include_average=args.include_average,
            ylim=ylim,
            colormap=args.colormap,
            label_variants=label_variants,
            figsize=figsize,
            dpi=args.dpi,
            formats=args.formats,
            ytick_step=args.ytick_step,
        )
        if args.grid:
            _plot_grids(
                data,
                metrics=args.metrics,
                datasets=datasets,
                topics_list=topics_list,
                modes=args.modes,
                categories=args.categories,
                models=args.models,
                outdir=outdir,
                no_errorbar=args.no_errorbar,
                ylim=ylim,
                colormap=args.colormap,
                max_cols=args.grid_cols,
                legend_fontsize=args.grid_legend_fontsize,
                dpi=args.dpi,
                formats=args.formats,
                ytick_step=args.ytick_step,
            )

    legend_models = _collect_legend_models(
        data,
        metrics=args.metrics,
        datasets=datasets,
        topics_list=topics_list,
        modes=args.modes,
        categories=args.categories,
        models=args.models,
        include_average=args.include_average,
    )
    _write_legend_figure(
        models=legend_models,
        outdir=outdir,
        colormap=args.colormap,
        formats=args.formats,
        dpi=args.dpi,
    )
    _write_legend_figure(
        models=legend_models,
        outdir=outdir,
        colormap=args.colormap,
        filename="legend_h.png",
        ncol=8,
        formats=args.formats,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

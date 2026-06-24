from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.core.artifacts import CURRENT_POINTER_FILENAME, load_json
from src.core.paths import resolve_project_path
from src.evaluation.reporting import read_evaluation_json

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import error handling
    raise SystemExit(
        "matplotlib is required to plot figures. "
        "Install it in your environment and rerun."
    ) from exc


LIMITED_FILE_RE = re.compile(r"^(acc|f1)_(.+)_(\d+)topic_(ratio|count)(.+)\.json$")
FULL_FILE_RE = re.compile(r"^(acc|f1)_(.+)_(\d+)topic\.json$")
MODEL_LABELS = {
    "Blei LDA": "LDA",
    "sentLDA": "SLDA",
    "Gaussian LDA": "GLDA",
    "MvTM": "vLDA",
    "ETM": "ETM",
    "Contextual TM": "CTM",
    "SenClu": "SenClu",
    "Sentence LDA": "GSLDA",
    "vMF Sentence LDA": "vSLDA(proposed)",
}
MODEL_ORDER = [
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
DEFAULT_COLORMAP = "tab10"
MODEL_COLOR_OVERRIDES = {
    "vLDA": "#17becf",
    "vSLDA": "#d62728",
    "vSLDA(proposed)": "#d62728",
}
METRIC_AXIS_LABELS = {
    "acc": "Accuracy",
    "f1mac": "Macro F1",
    "f1mic": "Micro F1",
}
MODE_AXIS_LABELS = {
    "ratio": "Training data used",
    "count": "Training examples",
}
LINEWIDTH = 1.8
MARKERSIZE = 5
MARKEREDGEWIDTH = 0.8
LINE_ALPHA = 0.95
ERROR_ALPHA = 0.08
CATEGORY_FIGSIZE = (4.8, 4.2)


def _read_json(path: Path) -> Dict:
    _meta, results = read_evaluation_json(path)
    if isinstance(results, dict) and isinstance(results.get("results"), dict):
        return results["results"]
    return results


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


def _load_scores(base_dir: Path, *, archive_history: bool = False) -> Dict:
    data: Dict = {}
    for path in _collect_files(base_dir, archive_history=archive_history):
        parsed = _parse_filename(path)
        if not parsed:
            continue
        metric_key, dataset, topics, mode, value = parsed
        results = _read_json(path)
        if metric_key == "acc":
            for category, model_scores in results.items():
                for model, score in model_scores.items():
                    _append_score(
                        data,
                        metric="acc",
                        dataset=dataset,
                        topics=topics,
                        mode=mode,
                        value=value,
                        category=category,
                        model=model,
                        score=score,
                    )
        else:
            for category, f1_payload in results.items():
                macro = f1_payload.get("macro", {})
                micro = f1_payload.get("micro", {})
                for model, score in macro.items():
                    _append_score(
                        data,
                        metric="f1mac",
                        dataset=dataset,
                        topics=topics,
                        mode=mode,
                        value=value,
                        category=category,
                        model=model,
                        score=score,
                    )
                for model, score in micro.items():
                    _append_score(
                        data,
                        metric="f1mic",
                        dataset=dataset,
                        topics=topics,
                        mode=mode,
                        value=value,
                        category=category,
                        model=model,
                        score=score,
                    )
    return data


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _base_model_name(name: str) -> str:
    normalized = re.sub(r"\s*\[(?:SVM|LogReg)\]\s*$", "", name.strip())
    while True:
        without_variant = re.sub(r"\s*\[[^\]]+\]\s*$", "", normalized)
        if without_variant == normalized:
            break
        normalized = without_variant
    normalized = re.sub(r"\s*\(soft\)\s*$", "", normalized)
    return normalized.strip()


def _display_model_name(name: str) -> str:
    return MODEL_LABELS.get(_base_model_name(name), _base_model_name(name))


def _model_sort_key(name: str) -> tuple[int, str]:
    display = _display_model_name(name)
    try:
        return (MODEL_ORDER.index(display), display)
    except ValueError:
        return (len(MODEL_ORDER), display)


def _stable_label_index(label: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(label))


def _model_color(label: str, colormap: str) -> object:
    if label in MODEL_COLOR_OVERRIDES:
        return MODEL_COLOR_OVERRIDES[label]

    try:
        color_index = MODEL_ORDER.index(label)
    except ValueError:
        color_index = len(MODEL_ORDER) + _stable_label_index(label)

    cmap = plt.get_cmap(colormap)
    color_count = getattr(cmap, "N", None)
    if isinstance(color_count, int) and color_count <= 20:
        return cmap(color_index % color_count)

    scale_count = max(len(MODEL_ORDER), 2)
    return cmap((color_index % scale_count) / (scale_count - 1))


def _metric_axis_label(metric: str) -> str:
    return METRIC_AXIS_LABELS.get(metric, metric)


def _mode_axis_label(mode: str) -> str:
    return MODE_AXIS_LABELS.get(mode, mode)


def _format_x_tick(value: float, mode: str) -> str:
    if mode == "ratio":
        return f"{value * 100:g}%"
    return f"{value:g}"


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
) -> None:
    values = set()
    for model_data in category_data.values():
        values.update(model_data.keys())
    if not values:
        return

    x_vals = sorted(values)
    fig, ax = plt.subplots(figsize=CATEGORY_FIGSIZE)
    x_plot = list(range(len(x_vals))) if mode == "ratio" else x_vals
    for model, model_data in sorted(
        category_data.items(), key=lambda item: _model_sort_key(item[0])
    ):
        if models is not None and model not in models:
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
        color = _model_color(label, colormap)
        ax.plot(
            x_plot,
            means,
            marker="o",
            label=label,
            color=color,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            markeredgewidth=MARKEREDGEWIDTH,
            alpha=LINE_ALPHA,
            zorder=2,
        )
        if not no_errorbar:
            lower = np.asarray(means) - np.asarray(stds)
            upper = np.asarray(means) + np.asarray(stds)
            ax.fill_between(
                x_plot,
                lower,
                upper,
                alpha=ERROR_ALPHA,
                color=color,
                zorder=1,
            )

    ax.set_xlabel(_mode_axis_label(mode))
    ax.set_ylabel(_metric_axis_label(metric))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_xticks(x_plot)
    ax.set_xticklabels([_format_x_tick(x, mode) for x in x_vals])
    fig.tight_layout()

    filename = f"{dataset}_{topics}topic_{metric}_{mode}_{category}.png"
    outpath = outdir / filename
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


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
                                legend_models.add(model)
                    if include_average:
                        average_data = _build_average_category_data(
                            mode_data,
                            categories=categories,
                        )
                        for model in average_data:
                            if models is None or model in models:
                                legend_models.add(model)
    return sorted(legend_models, key=lambda model: (*_model_sort_key(model), model))


def _write_legend_figure(
    *,
    models: List[str],
    outdir: Path,
    colormap: str,
    filename: str = "legend.png",
) -> None:
    if not models:
        return
    _ensure_dir(outdir)
    labels = [_display_model_name(model) for model in models]
    unique_labels = list(dict.fromkeys(labels))
    fig_width = 3.2
    fig_height = max(2.2, 0.34 * len(unique_labels) + 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=_model_color(label, colormap),
            label=label,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            markeredgewidth=MARKEREDGEWIDTH,
        )
        for label in unique_labels
    ]
    fig.legend(
        handles=handles,
        labels=unique_labels,
        loc="center",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=10,
    )
    fig.savefig(outdir / filename, dpi=150, bbox_inches="tight", pad_inches=0.05)
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
) -> None:
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
                        _plot_category(
                            metric=metric,
                            dataset=dataset,
                            topics=topics,
                            mode=mode,
                            category=category,
                            category_data=category_data,
                            outdir=outdir,
                            models=models,
                            no_errorbar=no_errorbar,
                            ylim=ylim,
                            colormap=colormap,
                        )
                    if include_average:
                        average_data = _build_average_category_data(
                            mode_data,
                            categories=categories,
                        )
                        if average_data:
                            _plot_category(
                                metric=metric,
                                dataset=dataset,
                                topics=topics,
                                mode=mode,
                                category="average",
                                category_data=average_data,
                                outdir=outdir,
                                models=models,
                                no_errorbar=True,
                                ylim=ylim,
                                colormap=colormap,
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
        default="results/classification/figures/limited",
        help="Output directory for figures.",
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
    args = parser.parse_args()

    ylim = tuple(args.ylim) if args.ylim is not None else None
    if ylim is not None and ylim[0] >= ylim[1]:
        raise SystemExit("--ylim requires MIN to be smaller than MAX.")

    base_dir = Path(args.base_dir)
    data = _load_scores(base_dir, archive_history=args.archive_history)
    if not data:
        raise SystemExit(f"No result files found under {base_dir}.")

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

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

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
    )


if __name__ == "__main__":
    main()

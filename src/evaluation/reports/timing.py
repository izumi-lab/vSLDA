"""Wall-clock and convergence summaries for the manuscript (paper TODO T12).

Reads the per-iteration traces that the trainers already record and writes

* ``timing_<dataset>_<data_run>_<encoder>_<K>topic.scores.json`` -- one sidecar per
  (dataset, data run, encoder, K), carrying the raw per-run values and the provenance
  of each cell. All aggregation happens on the paper side, as for every other table.
* ``--paper``: the two-panel convergence figure (one panel per model).

Two facts drive the design.

**The iteration is not the same unit across models.** One vSLDA iteration is a full
MCEM step (``gibbs_sweeps`` Gibbs sweeps, an M step and an alpha update); one GSLDA
iteration is a single collapsed Gibbs sweep. Per-iteration seconds are therefore not
comparable head-to-head, and the sidecar records ``gibbs_sweeps`` per run so the paper
side can report the per-sweep figure and, above all, the M=768/384 *ratio* -- which is
unit-free and is what tests the linear-vs-quadratic claim.

**Raw log-likelihoods are not comparable across models.** vMF is a density on the unit
sphere, the GSLDA posterior predictive is a Student-t density on R^M; the values also
scale with M within a model. The figure therefore gives each model its own panel and
its own likelihood axis, and shares only the wall-clock axis: the reader takes from it
*when* each trace flattens, never a comparison of likelihood levels. (The min-max
normalisation of Batmanghelich et al. (2016) Fig. 2, kept as ``normalize_trace`` for
reference, presumes monotone traces, which GSLDA's does not deliver on real data.)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from src.core.paths_roots import resolve_project_path

__all__ = [
    "TimingRun",
    "PANEL_TITLES",
    "PAPER_FIGURE_STEM",
    "collect_timing_runs",
    "build_timing_scores_payload",
    "draw_convergence_figure",
    "normalize_trace",
    "scores_sidecar_path",
    "summarize_timing",
    "write_timing_scores",
]

VSLDA_KEY = "vmf_sentence_lda"
GSLDA_KEY = "sentence_gaussianlda"

# model key -> label understood by src.evaluation.reports.model_style
MODEL_LABELS: Dict[str, str] = {
    VSLDA_KEY: "vSLDA (proposed)",
    GSLDA_KEY: "GSLDA",
}
MODEL_ORDER: tuple[str, ...] = (VSLDA_KEY, GSLDA_KEY)

PAPER_FIGURE_STEM = "convergence"

# Encoder variant -> embedding dimension. The sidecar records the dimension because the
# M=768/384 ratio is the number the manuscript's claim rests on.
ENCODER_DIMS: Dict[str, int] = {
    "minilm": 384,
    "mpnet": 768,
    "bge": 768,
}


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "-" for char in text).strip("-")


def _encoder_variant(raw: Any) -> str:
    """``"minilm_raw"``/``"minilm_norm"`` and ``"minilm"`` are the same encoder,
    differently normalised."""
    text = str(raw or "").strip().lower()
    for suffix in ("_raw", "_norm"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


@dataclass(frozen=True)
class TimingRun:
    """One measured run: its per-iteration trace plus the provenance of that trace."""

    model: str
    dataset: str
    data_run: str
    category: str
    num_topics: int
    seed: int
    encoder_variant: str
    encoder_model: str | None
    embedding_dim: int | None
    num_iterations: int
    gibbs_sweeps: int | None
    encode_batch_size: int | None
    iteration_sec: tuple[float, ...]
    avg_log_likelihood: tuple[float, ...]
    training_elapsed_sec: float | None
    encoding_sec: float | None
    archive_dir: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def cumulative_sec(self) -> tuple[float, ...]:
        total = 0.0
        out: List[float] = []
        for value in self.iteration_sec:
            total += value
            out.append(total)
        return tuple(out)

    def as_record(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "category": self.category,
            "seed": self.seed,
            "num_iterations": self.num_iterations,
            "gibbs_sweeps": self.gibbs_sweeps,
            "iteration_sec": list(self.iteration_sec),
            "cumulative_sec": list(self.cumulative_sec),
            "avg_log_likelihood": list(self.avg_log_likelihood),
            "training_elapsed_sec": self.training_elapsed_sec,
            "training_corpus_encoding_sec": self.encoding_sec,
            "archive_dir": self.archive_dir,
        }


# ---------------------------------------------------------------------------
# Reading the run artifacts
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_vslda(current: Mapping[str, Any], archive: Path) -> TimingRun | None:
    """vSLDA writes everything into ``metrics.json``."""
    metrics_path = archive / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = _load_json(metrics_path)
    diagnostics = metrics.get("iteration_diagnostics") or []
    if not diagnostics:
        return None

    encoder_config = current.get("encoder_config") or {}
    variant = _encoder_variant(
        current.get("embedding_variant") or encoder_config.get("embedding_variant")
    )
    return TimingRun(
        model=VSLDA_KEY,
        dataset=str(current.get("dataset")),
        data_run=str(current.get("data_run")),
        category=str(current.get("category")),
        num_topics=int(metrics.get("num_topics")),
        seed=_seed_from_display_key(current.get("display_key")),
        encoder_variant=variant,
        encoder_model=encoder_config.get("model_name"),
        embedding_dim=_embedding_dim(metrics.get("embedding_cache"), variant),
        num_iterations=len(diagnostics),
        gibbs_sweeps=_int_or_none(metrics.get("gibbs_sweeps")),
        encode_batch_size=_int_or_none(metrics.get("encoder_encode_batch_size"))
        or _int_or_none(encoder_config.get("encode_batch_size")),
        iteration_sec=tuple(
            float(item["iteration_elapsed_sec"]) for item in diagnostics
        ),
        avg_log_likelihood=tuple(
            float(item["avg_log_likelihood"]) for item in diagnostics
        ),
        training_elapsed_sec=_float_or_none(metrics.get("elapsed_sec")),
        encoding_sec=_float_or_none(metrics.get("training_corpus_encoding_sec")),
        archive_dir=str(current.get("archive_dir")),
        extra={
            "e_step_sec": [float(i.get("e_step_sec", 0.0)) for i in diagnostics],
            "m_step_sec": [float(i.get("m_step_sec", 0.0)) for i in diagnostics],
            "e_step_kernel_backend": metrics.get("e_step_kernel_backend"),
            "embedding_variant": _raw_variant(current),
            "strip_terminal_normalize": encoder_config.get("strip_terminal_normalize"),
            "num_documents": _int_or_none(
                (metrics.get("embedding_cache") or {}).get("num_documents")
            ),
            "total_sentences": _int_or_none(
                (metrics.get("embedding_cache") or {}).get("total_sentences")
            ),
        },
    )


def _read_gslda(current: Mapping[str, Any], archive: Path) -> TimingRun | None:
    """GSLDA writes into ``params/params.json``; the timing fields are recent."""
    params_path = archive / "params" / "params.json"
    metadata_path = archive / "metadata.json"
    if not params_path.is_file():
        return None
    params = _load_json(params_path)
    diagnostics = params.get("iteration_diagnostics") or []
    if not diagnostics:
        # Runs produced before the trainer persisted its timers carry only average_ll.
        return None
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
    baseline_params = metadata.get("baseline_params") or {}

    encoder_config = current.get("encoder_config") or {}
    variant = _encoder_variant(
        current.get("embedding_variant") or encoder_config.get("embedding_variant")
    )
    average_ll = [float(v) for v in (params.get("average_ll") or [])]
    return TimingRun(
        model=GSLDA_KEY,
        dataset=str(current.get("dataset")),
        data_run=str(current.get("data_run")),
        category=str(current.get("category")),
        num_topics=int(metadata.get("num_topics") or params.get("num_tables")),
        seed=_seed_from_display_key(current.get("display_key")),
        encoder_variant=variant,
        encoder_model=encoder_config.get("model_name")
        or baseline_params.get("encoder_model_name"),
        embedding_dim=ENCODER_DIMS.get(variant),
        num_iterations=len(diagnostics),
        # One GSLDA iteration IS one Gibbs sweep -- see the module docstring.
        gibbs_sweeps=1,
        encode_batch_size=_int_or_none(baseline_params.get("encode_batch_size"))
        or _int_or_none(encoder_config.get("encode_batch_size")),
        iteration_sec=tuple(
            float(item["iteration_elapsed_sec"]) for item in diagnostics
        ),
        avg_log_likelihood=tuple(average_ll[: len(diagnostics)]),
        training_elapsed_sec=_float_or_none(params.get("training_elapsed_sec")),
        encoding_sec=_float_or_none(params.get("training_corpus_encoding_sec")),
        archive_dir=str(current.get("archive_dir")),
        extra={
            "sampling_sec": [float(i.get("sampling_sec", 0.0)) for i in diagnostics],
            "prior_scale": params.get("prior_scale"),
            "embedding_variant": _raw_variant(current),
            "strip_terminal_normalize": encoder_config.get("strip_terminal_normalize"),
        },
    )


def _raw_variant(current: Mapping[str, Any]) -> str | None:
    """The variant as recorded (``"minilm_norm"``), normalisation suffix included."""
    encoder_config = current.get("encoder_config") or {}
    value = current.get("embedding_variant") or encoder_config.get("embedding_variant")
    return None if value is None else str(value)


def _seed_from_display_key(display_key: Any) -> int:
    """``"k20_it2_c1_minilm"`` -> ``2``. The seed is the run's ``--iteration``."""
    for part in str(display_key or "").split("_"):
        if part.startswith("it") and part[2:].isdigit():
            return int(part[2:])
    return -1


def _embedding_dim(embedding_cache: Any, variant: str) -> int | None:
    if isinstance(embedding_cache, Mapping):
        size = embedding_cache.get("embedding_size")
        if size:
            return int(size)
    return ENCODER_DIMS.get(variant)


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _resolve_archive_dir(archive_dir: str, *, root: Path) -> Path:
    """Locate a pointer's archive inside ``root``.

    A pointer records ``archive_dir`` relative to the results root it was
    written under (``results/experiments/...``). That root is not necessarily
    the one being summarized: an HPC tree rsync'ed back as ``results_hpc03``
    keeps the recorded ``results/`` prefix, so joining the value to the
    repository (or to ``root.parent``) would read the local copy of the same
    condition. Re-root the value on ``root`` instead, and only fall back to
    resolving it as recorded when that leaves nothing on disk.
    """
    recorded = Path(archive_dir)
    if recorded.is_absolute():
        return recorded
    parts = recorded.parts
    if len(parts) > 1:
        rerooted = root.joinpath(*parts[1:])
        if rerooted.is_dir():
            return rerooted
    fallback = resolve_project_path(recorded)
    if fallback.is_dir():
        return fallback
    return root.joinpath(*parts[1:]) if len(parts) > 1 else recorded


def collect_timing_runs(
    results_root: Path,
    *,
    dataset: str,
    data_run: str = "default",
) -> List[TimingRun]:
    """Read every measured run of ``dataset`` under ``results_root``.

    Runs whose trainer did not persist an iteration trace are skipped rather than
    guessed at, so a partially re-run tree cannot silently produce a short table.
    """
    root = Path(results_root)
    sources = (
        (VSLDA_KEY, root / "experiments" / dataset / data_run / VSLDA_KEY / "latest"),
        (GSLDA_KEY, root / "baselines" / dataset / data_run / GSLDA_KEY / "latest"),
    )
    readers = {VSLDA_KEY: _read_vslda, GSLDA_KEY: _read_gslda}

    runs: List[TimingRun] = []
    for model, latest_dir in sources:
        if not latest_dir.is_dir():
            continue
        for current_path in sorted(latest_dir.glob("*/*/CURRENT.json")):
            current = _load_json(current_path)
            archive = _resolve_archive_dir(str(current["archive_dir"]), root=root)
            run = readers[model](current, archive)
            if run is not None:
                runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# The *.scores.json sidecar
# ---------------------------------------------------------------------------


def scores_sidecar_path(
    output_dir: Path,
    *,
    dataset: str,
    data_run: str,
    encoder: str,
    num_topics: int,
) -> Path:
    return (
        Path(output_dir)
        / _slug(dataset)
        / _slug(encoder)
        / f"timing_{_slug(dataset)}_{_slug(data_run)}_{_slug(encoder)}_{num_topics}topic.scores.json"
    )


def build_timing_scores_payload(
    runs: Sequence[TimingRun],
    *,
    dataset: str,
    data_run: str,
    encoder: str,
    num_topics: int,
    environment: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Raw per-run values plus the provenance the paper side checks its captions against."""
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    provenance: Dict[str, Any] = {}
    for run in sorted(runs, key=lambda r: (MODEL_ORDER.index(r.model), r.seed)):
        by_model.setdefault(run.model, []).append(run.as_record())
        provenance.setdefault(
            run.model,
            {
                "encoder_model": run.encoder_model,
                # As recorded, suffix included: "minilm_norm" for the manuscript's
                # GSLDA, plain "minilm" for vSLDA (which L2-normalises itself).
                "embedding_variant": run.extra.get("embedding_variant")
                or run.encoder_variant,
                "encoder": run.encoder_variant,
                "strip_terminal_normalize": run.extra.get("strip_terminal_normalize"),
                "embedding_dim": run.embedding_dim,
                "encode_batch_size": run.encode_batch_size,
                "gibbs_sweeps": run.gibbs_sweeps,
                "num_iterations": run.num_iterations,
                "category": run.category,
                "prior_scale": run.extra.get("prior_scale"),
                # Recorded by vSLDA only; the size of the corpus both models encoded.
                "num_documents": run.extra.get("num_documents"),
                "total_sentences": run.extra.get("total_sentences"),
            },
        )

    return {
        "metric": "wall_clock",
        "dataset": dataset,
        "data_run": data_run,
        "encoder": encoder,
        "num_topics": num_topics,
        "seeds": sorted({run.seed for run in runs}),
        "models": [key for key in MODEL_ORDER if key in by_model],
        "model_labels": {k: MODEL_LABELS[k] for k in by_model},
        # One vSLDA iteration is an MCEM step, one GSLDA iteration is a Gibbs sweep.
        "iteration_unit": {
            VSLDA_KEY: "mcem_iteration",
            GSLDA_KEY: "gibbs_sweep",
        },
        "runs": by_model,
        "provenance": provenance,
        "environment": dict(environment or {}),
    }


def write_timing_scores(
    runs: Sequence[TimingRun],
    *,
    output_dir: Path,
    environment: Mapping[str, Any] | None = None,
) -> List[Path]:
    """One sidecar per (dataset, data run, encoder, K)."""
    groups: Dict[tuple[str, str, str, int], List[TimingRun]] = {}
    for run in runs:
        key = (run.dataset, run.data_run, run.encoder_variant, run.num_topics)
        groups.setdefault(key, []).append(run)

    written: List[Path] = []
    for (dataset, data_run, encoder, num_topics), members in sorted(groups.items()):
        payload = build_timing_scores_payload(
            members,
            dataset=dataset,
            data_run=data_run,
            encoder=encoder,
            num_topics=num_topics,
            environment=environment,
        )
        path = scores_sidecar_path(
            output_dir,
            dataset=dataset,
            data_run=data_run,
            encoder=encoder,
            num_topics=num_topics,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# The manuscript figure
# ---------------------------------------------------------------------------


def normalize_trace(values: Sequence[float]) -> List[float]:
    """Min-max normalise one run's log-likelihood trace to 0-100%.

    This is the normalisation of Batmanghelich et al. (2016) Fig. 2, and the reason
    is theirs: the raw values are not comparable between a vMF density on the sphere
    and a Student-t density on R^M, so only the shape -- how fast each scheme reaches
    its own plateau -- is read off the figure. It is applied per run, never pooled.
    """
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [math.nan] * len(values)
    low, high = min(finite), max(finite)
    if high <= low:
        return [100.0 if math.isfinite(v) else math.nan for v in values]
    return [
        100.0 * (v - low) / (high - low) if math.isfinite(v) else math.nan
        for v in values
    ]


PANEL_TITLES: Dict[str, str] = {
    VSLDA_KEY: "vSLDA (per MCEM iteration)",
    GSLDA_KEY: "GSLDA (per Gibbs sweep)",
}

# Figure width in inches: the manuscript's two-column-figure width (ISwA 4.87 in);
# TMLR centres the same figure on its 6.5 in text width.
CONVERGENCE_FIGURE_WIDTH = float(os.environ.get("PAPER_CONVERGENCE_WIDTH", "4.87"))


def draw_convergence_figure(
    runs: Sequence[TimingRun],
    *,
    output_dir: Path,
    stem: str = PAPER_FIGURE_STEM,
    mark_iteration: int | None = 10,
    formats: Sequence[str] | None = None,
    total_width: float | None = None,
) -> List[Path]:
    """Per-model average log-likelihood against cumulative training wall-clock.

    One panel per model, each with its own likelihood axis (the values live on
    different supports and are never compared); the wall-clock axis (log scale) is
    the only thing the panels share. Every run is drawn as its own thin line.
    ``mark_iteration`` draws a rule at the iteration count the manuscript actually
    trains for, so a reader can see whether the trace had already flattened by then.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # noqa: PLC0415 - after backend selection

    from src.evaluation.reports.grid_layout import (
        DEFAULT_FORMATS,
        DPI,
        GRID_MARGIN_BOTTOM,
        PAPER_RC,
        grid_geometry,
        panel_frame,
    )
    from src.evaluation.reports.model_style import model_style

    by_model: Dict[str, List[TimingRun]] = {}
    for run in runs:
        by_model.setdefault(run.model, []).append(run)
    models = [m for m in MODEL_ORDER if m in by_model]
    if not models:
        return []

    written: List[Path] = []
    with plt.rc_context(PAPER_RC):
        ncols = len(models)
        figsize, adjust, _legend_h = grid_geometry(
            1,
            ncols,
            legend_rows=0,
            panel_aspect=0.8,
            wspace=0.62,  # each panel carries its own y tick labels and y label
            # Room under the tick labels for the single shared x label below them.
            margin_bottom=GRID_MARGIN_BOTTOM + 0.16,
            total_width=total_width or CONVERGENCE_FIGURE_WIDTH,
        )
        # One shared time axis: the two models' traces then sit at their own place on
        # the same scale, which is what makes their speeds comparable at a glance.
        fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False, sharex=True)
        fig.subplots_adjust(**adjust)

        for ax, model in zip(axes[0], models):
            members = sorted(by_model[model], key=lambda r: r.seed)
            label = MODEL_LABELS[model]
            style = model_style(label, "tab10")
            kwargs = style.line_kwargs()
            kwargs.update(marker=".", markersize=2.5, linewidth=0.9)

            for run in members:
                ax.plot(
                    run.cumulative_sec,
                    run.avg_log_likelihood,
                    alpha=0.85,
                    **kwargs,
                )

            if mark_iteration:
                marks = [
                    r.cumulative_sec[mark_iteration - 1]
                    for r in members
                    if len(r.cumulative_sec) >= mark_iteration
                ]
                if marks:
                    ax.axvline(
                        sum(marks) / len(marks),
                        color="0.35",
                        linestyle=":",
                        linewidth=0.8,
                        zorder=1,
                    )

            ax.set_xscale("log")
            ax.set_title(PANEL_TITLES.get(model, label))
            ax.set_ylabel("Average log-likelihood")
            panel_frame(ax)

        # A single x label under the row; one per panel overflows the page width.
        fig.supxlabel(
            "Cumulative training time (s, log scale)",
            fontsize=plt.rcParams["axes.labelsize"],
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in formats or DEFAULT_FORMATS:
            path = output_dir / f"{stem}.{suffix}"
            # No bbox_inches="tight": grid_layout owns the page width.
            fig.savefig(path, dpi=DPI)
            written.append(path)
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def summarize_timing(
    *,
    results_root: Path = Path("results"),
    dataset: str = "20newsgroup_timing",
    data_run: str = "default",
    output_dir: Path | None = None,
    paper: bool = False,
    mark_iteration: int | None = 10,
    figure_encoder: str | None = None,
) -> Dict[str, Any]:
    """Collect the measured runs, write the sidecars and optionally the figure.

    ``figure_encoder`` names the encoder variant (``"minilm"``) whose runs the
    figure draws; by default it takes the runs with the longest traces, which is
    the encoder the extended-iteration phase was run with.
    """
    results_root = Path(results_root)
    output_dir = Path(output_dir or results_root / "timing" / "summaries")

    runs = collect_timing_runs(results_root, dataset=dataset, data_run=data_run)
    if not runs:
        raise FileNotFoundError(
            f"no timing runs with an iteration trace under {results_root} "
            f"for dataset={dataset!r}; run ./tmp_timing.sh first"
        )

    env_path = results_root / "experiments" / dataset / "timing_env.json"
    environment = _load_json(env_path) if env_path.is_file() else {}

    written = write_timing_scores(runs, output_dir=output_dir, environment=environment)
    figures: List[Path] = []
    if paper:
        # The convergence figure is a single setting.
        if figure_encoder:
            figure_runs = [r for r in runs if r.encoder_variant == figure_encoder]
            if not figure_runs:
                raise FileNotFoundError(
                    f"no timing runs for encoder {figure_encoder!r} under {results_root}"
                )
        else:
            longest = max(run.num_iterations for run in runs)
            figure_runs = [r for r in runs if r.num_iterations == longest]
        figures = draw_convergence_figure(
            figure_runs,
            output_dir=output_dir / "figures",
            mark_iteration=mark_iteration,
        )

    return {
        "runs": len(runs),
        "scores": [str(p) for p in written],
        "figures": [str(p) for p in figures],
        "environment": environment,
    }

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.evaluation.reports.timing import (
    GSLDA_KEY,
    VSLDA_KEY,
    build_timing_scores_payload,
    collect_timing_runs,
    draw_convergence_figure,
    normalize_trace,
    scores_sidecar_path,
    summarize_timing,
    write_timing_scores,
)

DATASET = "20newsgroup_timing"
DATA_RUN = "default"
CATEGORY = "computer"
K = 20


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(
    root: Path,
    *,
    model: str,
    encoder: str,
    dim: int,
    seed: int,
    iterations: int,
    per_iter: float,
    lls: list[float],
) -> None:
    """Write one run in the on-disk shape the real trainers produce."""
    results = root / "results"
    secs = [per_iter] * iterations
    if model == VSLDA_KEY:
        display_key = f"k{K}_it{seed}_c1_{encoder}"
        archive = (
            f"results/experiments/{DATASET}/{DATA_RUN}/{VSLDA_KEY}"
            f"/archive/d/{CATEGORY}/{display_key}/vmf_x"
        )
        latest = (
            results
            / f"experiments/{DATASET}/{DATA_RUN}/{VSLDA_KEY}/latest"
            / CATEGORY
            / display_key
            / "CURRENT.json"
        )
        variant = encoder
    else:
        display_key = f"k{K}_it{seed}_{encoder}_norm_psi0-0p1"
        archive = (
            f"results/baselines/{DATASET}/{DATA_RUN}/{GSLDA_KEY}"
            f"/archive/d/{CATEGORY}/{display_key}/base_x"
        )
        latest = (
            results
            / f"baselines/{DATASET}/{DATA_RUN}/{GSLDA_KEY}/latest"
            / CATEGORY
            / display_key
            / "CURRENT.json"
        )
        variant = f"{encoder}_norm"

    _write(
        latest,
        {
            "dataset": DATASET,
            "data_run": DATA_RUN,
            "category": CATEGORY,
            "display_key": display_key,
            "archive_dir": archive,
            "embedding_variant": variant,
            "encoder_config": {
                "model_name": f"sentence-transformers/all-{encoder}",
                "embedding_variant": encoder,
                "encode_batch_size": 128,
                "strip_terminal_normalize": model
                == VSLDA_KEY,  # GSLDA measured as `_norm`
            },
        },
    )

    diagnostics = [
        {"iteration": i, "iteration_elapsed_sec": secs[i]} for i in range(iterations)
    ]
    if model == VSLDA_KEY:
        for i, item in enumerate(diagnostics):
            item.update(
                avg_log_likelihood=lls[i], e_step_sec=secs[i] * 0.9, m_step_sec=0.01
            )
        _write(
            root / archive / "metrics.json",
            {
                "num_topics": K,
                "gibbs_sweeps": 20,
                "elapsed_sec": sum(secs),
                "training_corpus_encoding_sec": 12.0,
                "encoder_encode_batch_size": 128,
                "embedding_cache": {"embedding_size": dim},
                "iteration_diagnostics": diagnostics,
            },
        )
    else:
        for i, item in enumerate(diagnostics):
            item.update(sampling_sec=secs[i] * 0.98, avg_log_likelihood_sec=0.02)
        _write(
            root / archive / "metadata.json",
            {
                "num_topics": K,
                "baseline_params": {
                    "encoder_model_name": f"sentence-transformers/all-{encoder}",
                    "encode_batch_size": 128,
                },
            },
        )
        _write(
            root / archive / "params" / "params.json",
            {
                "average_ll": lls,
                "num_tables": K,
                "prior_scale": 0.1,
                "training_elapsed_sec": sum(secs),
                "training_corpus_encoding_sec": 12.0,
                "iteration_diagnostics": diagnostics,
            },
        )


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """vSLDA linear in M, GSLDA quadratic in M, three seeds each."""
    for encoder, dim in (("minilm", 384), ("mpnet", 768)):
        scale = dim / 384
        for seed in (0, 1, 2):
            _make_run(
                tmp_path,
                model=VSLDA_KEY,
                encoder=encoder,
                dim=dim,
                seed=seed,
                iterations=4,
                per_iter=2.0 * scale,
                lls=[600.0, 615.0, 620.0, 621.0],
            )
            _make_run(
                tmp_path,
                model=GSLDA_KEY,
                encoder=encoder,
                dim=dim,
                seed=seed,
                iterations=4,
                per_iter=10.0 * scale**2,
                lls=[-229.0, -246.0, -253.0, -255.0],
            )
    return tmp_path / "results"


class TestNormalizeTrace:
    def test_maps_endpoints_to_0_and_100(self):
        assert normalize_trace([1.0, 2.0, 3.0]) == [0.0, 50.0, 100.0]

    def test_decreasing_trace_starts_at_100(self):
        # GSLDA's recorded likelihood decreases; the normalisation must not hide it.
        assert normalize_trace([-229.0, -242.0, -255.0]) == [100.0, 50.0, 0.0]

    def test_constant_trace_does_not_divide_by_zero(self):
        assert normalize_trace([5.0, 5.0]) == [100.0, 100.0]

    def test_empty_and_non_finite(self):
        assert normalize_trace([]) == []
        assert all(math.isnan(v) for v in normalize_trace([math.nan]))

    def test_normalises_each_run_independently(self):
        # Pooling runs would let one model's scale leak into another's curve.
        assert normalize_trace([0.0, 10.0]) == normalize_trace([0.0, 1000.0])


class TestCollect:
    def test_reads_every_run(self, tree: Path):
        runs = collect_timing_runs(tree, dataset=DATASET)
        assert len(runs) == 12
        assert {r.model for r in runs} == {VSLDA_KEY, GSLDA_KEY}
        assert {r.seed for r in runs} == {0, 1, 2}

    def test_encoder_variant_strips_normalisation_suffix(self, tree: Path):
        # GSLDA records "minilm_norm"; it is the same encoder as vSLDA's "minilm",
        # and the two must land in the same sidecar.
        runs = collect_timing_runs(tree, dataset=DATASET)
        assert {r.encoder_variant for r in runs} == {"minilm", "mpnet"}

    def test_raw_variant_is_kept_for_provenance(self, tree: Path):
        # The paper side checks that the measured GSLDA is the manuscript's `_norm`
        # variant, so the suffix must survive next to the stripped encoder key.
        runs = collect_timing_runs(tree, dataset=DATASET)
        assert {r.extra["embedding_variant"] for r in runs if r.model == GSLDA_KEY} == {
            "minilm_norm",
            "mpnet_norm",
        }
        assert {r.extra["embedding_variant"] for r in runs if r.model == VSLDA_KEY} == {
            "minilm",
            "mpnet",
        }

    def test_records_the_iteration_unit_asymmetry(self, tree: Path):
        runs = collect_timing_runs(tree, dataset=DATASET)
        assert {r.gibbs_sweeps for r in runs if r.model == VSLDA_KEY} == {20}
        # One GSLDA iteration is one Gibbs sweep, not an MCEM step.
        assert {r.gibbs_sweeps for r in runs if r.model == GSLDA_KEY} == {1}

    def test_embedding_dim_per_encoder(self, tree: Path):
        runs = collect_timing_runs(tree, dataset=DATASET)
        dims = {(r.model, r.encoder_variant): r.embedding_dim for r in runs}
        assert dims[(VSLDA_KEY, "minilm")] == 384
        assert dims[(GSLDA_KEY, "mpnet")] == 768

    def test_cumulative_is_a_running_sum(self, tree: Path):
        run = next(iter(collect_timing_runs(tree, dataset=DATASET)))
        assert run.cumulative_sec[-1] == pytest.approx(sum(run.iteration_sec))

    def test_skips_runs_without_a_trace(self, tree: Path, tmp_path: Path):
        # A run whose trainer predates the timing fields must be dropped, not guessed at.
        target = next(
            (tree / f"baselines/{DATASET}/{DATA_RUN}/{GSLDA_KEY}/latest").glob(
                "*/*/CURRENT.json"
            )
        )
        archive = tmp_path / json.loads(target.read_text())["archive_dir"]
        params = json.loads((archive / "params" / "params.json").read_text())
        params.pop("iteration_diagnostics")
        _write(archive / "params" / "params.json", params)
        assert len(collect_timing_runs(tree, dataset=DATASET)) == 11

    def test_archive_is_read_from_the_summarized_root(self, tree: Path):
        # A tree rsync'ed back as results_hpc03 keeps the recorded "results/"
        # prefix. Resolving that prefix outside the summarized root read a
        # local copy of the same condition (here a decoy with other timings).
        root = tree.parent
        relocated = root / "results_hpc03"
        tree.rename(relocated)
        _make_run(
            root,
            model=VSLDA_KEY,
            encoder="minilm",
            dim=384,
            seed=0,
            iterations=4,
            per_iter=999.0,
            lls=[1.0, 2.0, 3.0, 4.0],
        )

        runs = collect_timing_runs(relocated, dataset=DATASET)

        assert len(runs) == 12
        assert not any(999.0 in run.iteration_sec for run in runs)

    def test_missing_dataset_yields_nothing(self, tree: Path):
        assert collect_timing_runs(tree, dataset="does_not_exist") == []


class TestScores:
    def test_one_sidecar_per_encoder_and_k(self, tree: Path, tmp_path: Path):
        runs = collect_timing_runs(tree, dataset=DATASET)
        written = write_timing_scores(runs, output_dir=tmp_path / "out")
        assert len(written) == 2
        assert all(p.name.endswith(".scores.json") for p in written)

    def test_payload_carries_raw_values_and_provenance(self, tree: Path):
        runs = [
            r
            for r in collect_timing_runs(tree, dataset=DATASET)
            if r.encoder_variant == "minilm"
        ]
        payload = build_timing_scores_payload(
            runs,
            dataset=DATASET,
            data_run=DATA_RUN,
            encoder="minilm",
            num_topics=K,
        )
        assert payload["models"] == [VSLDA_KEY, GSLDA_KEY]
        assert payload["seeds"] == [0, 1, 2]
        assert payload["iteration_unit"][GSLDA_KEY] == "gibbs_sweep"
        for model in payload["models"]:
            assert len(payload["runs"][model]) == 3
            prov = payload["provenance"][model]
            # The fields the paper-side caption is checked against.
            assert prov["embedding_dim"] == 384
            assert prov["encode_batch_size"] == 128
            assert prov["num_iterations"] == 4
            assert prov["encoder"] == "minilm"
        assert payload["provenance"][GSLDA_KEY]["embedding_variant"] == "minilm_norm"
        assert payload["provenance"][VSLDA_KEY]["embedding_variant"] == "minilm"

    def test_m_ratio_is_recoverable_from_the_sidecars(self, tree: Path, tmp_path: Path):
        """The one number the manuscript's linear-vs-quadratic claim rests on."""
        runs = collect_timing_runs(tree, dataset=DATASET)
        write_timing_scores(runs, output_dir=tmp_path / "out")
        per_iter: dict[tuple[str, str], float] = {}
        for encoder in ("minilm", "mpnet"):
            path = scores_sidecar_path(
                tmp_path / "out",
                dataset=DATASET,
                data_run=DATA_RUN,
                encoder=encoder,
                num_topics=K,
            )
            payload = json.loads(path.read_text())
            for model, records in payload["runs"].items():
                values = [
                    sum(r["iteration_sec"]) / len(r["iteration_sec"]) for r in records
                ]
                per_iter[(model, encoder)] = sum(values) / len(values)

        assert per_iter[(VSLDA_KEY, "mpnet")] / per_iter[
            (VSLDA_KEY, "minilm")
        ] == pytest.approx(2.0)
        assert per_iter[(GSLDA_KEY, "mpnet")] / per_iter[
            (GSLDA_KEY, "minilm")
        ] == pytest.approx(4.0)

    def test_sidecar_path_is_slugged(self, tmp_path: Path):
        path = scores_sidecar_path(
            tmp_path, dataset=DATASET, data_run=DATA_RUN, encoder="minilm", num_topics=K
        )
        assert path.name == (
            "timing_20newsgroup-timing_default_minilm_20topic.scores.json"
        )


class TestFigureAndEntryPoint:
    def test_draws_both_formats(self, tree: Path, tmp_path: Path):
        runs = collect_timing_runs(tree, dataset=DATASET)
        written = draw_convergence_figure(runs, output_dir=tmp_path / "fig")
        assert {p.suffix for p in written} == {".png", ".pdf"}
        assert all(p.stat().st_size > 0 for p in written)

    def test_no_runs_draws_nothing(self, tmp_path: Path):
        assert draw_convergence_figure([], output_dir=tmp_path / "fig") == []

    def test_one_panel_per_model_with_raw_likelihoods(
        self, tree: Path, tmp_path: Path, monkeypatch
    ):
        # The likelihoods live on different supports: each model gets its own panel
        # and axis, unnormalised, and only the time axis is shared.
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        captured = {}
        real_subplots = plt.subplots

        def spy(*args, **kwargs):
            fig, axes = real_subplots(*args, **kwargs)
            captured["fig"], captured["axes"] = fig, axes
            return fig, axes

        monkeypatch.setattr(plt, "subplots", spy)
        runs = [
            r
            for r in collect_timing_runs(tree, dataset=DATASET)
            if r.encoder_variant == "minilm"
        ]
        draw_convergence_figure(runs, output_dir=tmp_path / "fig")
        axes = captured["axes"][0]
        assert len(axes) == 2
        assert [ax.get_xscale() for ax in axes] == ["log", "log"]
        # Three seeds per panel; y data are the recorded values, not percentages.
        vslda_ax, gslda_ax = axes
        assert len(vslda_ax.get_lines()) >= 3 and len(gslda_ax.get_lines()) >= 3
        assert max(vslda_ax.get_lines()[0].get_ydata()) == pytest.approx(621.0)
        assert min(gslda_ax.get_lines()[0].get_ydata()) == pytest.approx(-255.0)
        assert "vSLDA" in vslda_ax.get_title() and "GSLDA" in gslda_ax.get_title()

    def test_summarize_writes_scores_and_figure(self, tree: Path, tmp_path: Path):
        report = summarize_timing(
            results_root=tree, dataset=DATASET, output_dir=tmp_path / "out", paper=True
        )
        assert report["runs"] == 12
        assert len(report["scores"]) == 2
        assert len(report["figures"]) == 2

    def test_summarize_figure_encoder_selects_runs(self, tree: Path, tmp_path: Path):
        report = summarize_timing(
            results_root=tree,
            dataset=DATASET,
            output_dir=tmp_path / "out",
            paper=True,
            figure_encoder="mpnet",
        )
        assert len(report["figures"]) == 2
        with pytest.raises(FileNotFoundError, match="bge"):
            summarize_timing(
                results_root=tree,
                dataset=DATASET,
                output_dir=tmp_path / "out2",
                paper=True,
                figure_encoder="bge",
            )

    def test_summarize_raises_when_nothing_measured(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="tmp_timing.sh"):
            summarize_timing(results_root=tmp_path / "results", dataset=DATASET)

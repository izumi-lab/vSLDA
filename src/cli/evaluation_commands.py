from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.baselines.params import (
    format_prior_scale_variant,
    normalize_covariance_type,
)
from src.core.paths import CLASSIFICATION_RESULTS_ROOT
from src.core.vmf_assignment import DEFAULT_VMF_ASSIGNMENT, normalize_vmf_assignment
from src.core.vmf_variant import normalize_vmf_parameter_variant
from src.evaluation.classification.config import (
    ALIGNMENT_MODES,
    DEFAULT_ALIGNMENT_MODE,
    DEFAULT_FEATURE_RESOLVE_MODE,
    FEATURE_RESOLVE_MODES,
)
from src.evaluation.reports.topic_sweep import (
    DEFAULT_CLASSIFICATION_METRICS as DEFAULT_TOPIC_SWEEP_CLASSIFICATION_METRICS,
)
from src.evaluation.reports.topic_sweep import (
    DEFAULT_CLASSIFICATION_ROOT as DEFAULT_TOPIC_SWEEP_CLASSIFICATION_ROOT,
)
from src.evaluation.reports.topic_sweep import (
    DEFAULT_COHERENCE_SUMMARY as DEFAULT_TOPIC_SWEEP_COHERENCE_SUMMARY,
)
from src.evaluation.reports.topic_sweep import (
    DEFAULT_MODELS as DEFAULT_TOPIC_SWEEP_MODELS,
)
from src.evaluation.reports.topic_sweep import (
    DEFAULT_OUT_ROOT as DEFAULT_TOPIC_SWEEP_OUT_ROOT,
)
from src.evaluation.reports.topic_sweep import (
    DEFAULT_WORD_BASED_METRICS as DEFAULT_TOPIC_SWEEP_WORD_BASED_METRICS,
)
from src.evaluation.word_based.summary import DEFAULT_COHERENCE_ROOT
from src.utils.random import DEFAULT_RANDOM_SEED

VMF_ASSIGNMENT_HELP = (
    "Document-topic estimator of the vMF Sentence LDA features: foldincounts (default; "
    "collapsed fold-in expected counts E[n_dk]/N_d for training and held-out documents, "
    "written at training time and by `evaluation vmf-foldin-theta`), foldin (the same "
    "with Dirichlet smoothing), hard (final-sweep counts / per-sentence posterior mean) "
    "or soft."
)


def _validate_vmf_assignment(value: str) -> str:
    try:
        return normalize_vmf_assignment(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--vmf-assignment") from exc


def register_evaluation_commands(evaluation_app: typer.Typer) -> None:
    @evaluation_app.command(
        "classify",
        help=(
            "Run classification and write outputs under "
            "results/classification/archive/<date>/<dataset>/<data_run>/all/<display_key>/<execution_id>/ "
            "and update results/classification/latest/<dataset>/<data_run>/all/<display_key>/CURRENT.json."
        ),
    )
    def classify(
        dataset: list[str] = typer.Option(["20newsgroup", "nyt"], "--dataset"),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        category: list[str] = typer.Option([], "--category"),
        topic: list[int] = typer.Option([10, 20], "--topic"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        classifier: list[str] = typer.Option(["svm"], "--classifier"),
        vmf_assignment: str = typer.Option(
            DEFAULT_VMF_ASSIGNMENT, help=VMF_ASSIGNMENT_HELP
        ),
        result_root: Path = typer.Option(CLASSIFICATION_RESULTS_ROOT, file_okay=False),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        seed: Optional[int] = typer.Option(42),
        alignment_mode: str = typer.Option(DEFAULT_ALIGNMENT_MODE, "--alignment-mode"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        model: list[str] = typer.Option([], "--model"),
        prior_scale: Optional[float] = typer.Option(None, "--prior-scale", min=0.0),
        covariance_type: Optional[str] = typer.Option(
            None,
            "--covariance-type",
            help=(
                "Covariance type of the sentence Gaussian LDA runs to use: full "
                "(default), diag, or spherical (alias iso)."
            ),
        ),
        vmf_variant: Optional[str] = typer.Option(
            None,
            "--vmf-variant",
            help=(
                "Hyperparameter-sweep label of the vMF Sentence LDA runs to use "
                "(e.g. kappa0-100, alpha0-0p1, zeta-40, b-16, t-5; src/core/vmf_variant.py). "
                "Unset selects the default runs and leaves every path unchanged."
            ),
        ),
        feature_resolve_mode: str = typer.Option(
            DEFAULT_FEATURE_RESOLVE_MODE,
            "--feature-resolve-mode",
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        if prior_scale is not None:
            try:
                format_prior_scale_variant(prior_scale)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--prior-scale") from exc
        if covariance_type is not None:
            try:
                covariance_type = normalize_covariance_type(covariance_type)
            except ValueError as exc:
                raise typer.BadParameter(
                    str(exc), param_hint="--covariance-type"
                ) from exc
        if vmf_variant is not None:
            try:
                vmf_variant = normalize_vmf_parameter_variant(vmf_variant)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--vmf-variant") from exc
        if alignment_mode not in ALIGNMENT_MODES:
            raise typer.BadParameter(
                f"alignment mode must be one of {', '.join(ALIGNMENT_MODES)}"
            )
        if feature_resolve_mode not in FEATURE_RESOLVE_MODES:
            raise typer.BadParameter(
                "feature resolve mode must be one of "
                f"{', '.join(FEATURE_RESOLVE_MODES)}"
            )
        register_builtin_tasks()
        run_task(
            "classification",
            iterations=iteration,
            datasets=dataset,
            data_runs=data_run,
            categories=list(category) or None,
            topics=topic,
            classifiers=classifier,
            vmf_assignment=_validate_vmf_assignment(vmf_assignment),
            result_root=result_root,
            target_column=target_column,
            label_schema=label_schema,
            seed=seed,
            alignment_mode=alignment_mode,
            embedding_variants=list(embedding_variant) or None,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=list(model) or None,
            prior_scale=prior_scale,
            covariance_type=covariance_type,
            vmf_variant=vmf_variant,
        )

    @evaluation_app.command(
        "classify-limited",
        help=(
            "Run limited-data classification and write outputs under "
            "results/classification/archive/<date>/<dataset>/<data_run>/all/<display_key>/<execution_id>/ "
            "and update results/classification/latest/<dataset>/<data_run>/all/<display_key>/CURRENT.json."
        ),
    )
    def classify_limited(
        dataset: list[str] = typer.Option(["20newsgroup", "nyt"], "--dataset"),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        category: list[str] = typer.Option([], "--category"),
        topic: list[int] = typer.Option([10, 20], "--topic"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        classifier: list[str] = typer.Option(["svm"], "--classifier"),
        ratio: Optional[float] = typer.Option(None, min=0.0, max=1.0),
        count: Optional[int] = typer.Option(None, min=0),
        vmf_assignment: str = typer.Option(
            DEFAULT_VMF_ASSIGNMENT, help=VMF_ASSIGNMENT_HELP
        ),
        result_root: Path = typer.Option(CLASSIFICATION_RESULTS_ROOT, file_okay=False),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        stratified: bool = typer.Option(True),
        seed: Optional[int] = typer.Option(42),
        sampling_repeat: list[int] = typer.Option(
            [],
            "--sampling-repeat",
            min=0,
            help=(
                "Limited-data sampling repeat index. Repeat this option to run "
                "multiple stratified train subsets per topic-model iteration."
            ),
        ),
        sampling_seed_stride: int = typer.Option(
            1000,
            "--sampling-seed-stride",
            min=1,
            help="Seed offset between limited-data sampling repeats.",
        ),
        sampling_max_attempts: int = typer.Option(
            1,
            "--sampling-max-attempts",
            min=1,
            help=(
                "Maximum seed attempts for a limited-data sample. Attempts after "
                "the first are used only when classifier training fails because "
                "feature alignment leaves fewer than two classes."
            ),
        ),
        sampling_retry_seed_stride: int = typer.Option(
            100000,
            "--sampling-retry-seed-stride",
            min=1,
            help="Seed offset between adaptive limited-data retry attempts.",
        ),
        alignment_mode: str = typer.Option(DEFAULT_ALIGNMENT_MODE, "--alignment-mode"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        model: list[str] = typer.Option([], "--model"),
        prior_scale: Optional[float] = typer.Option(None, "--prior-scale", min=0.0),
        covariance_type: Optional[str] = typer.Option(
            None,
            "--covariance-type",
            help=(
                "Covariance type of the sentence Gaussian LDA runs to use: full "
                "(default), diag, or spherical (alias iso)."
            ),
        ),
        vmf_variant: Optional[str] = typer.Option(
            None,
            "--vmf-variant",
            help=(
                "Hyperparameter-sweep label of the vMF Sentence LDA runs to use "
                "(e.g. kappa0-100, alpha0-0p1, zeta-40, b-16, t-5; src/core/vmf_variant.py). "
                "Unset selects the default runs and leaves every path unchanged."
            ),
        ),
        feature_resolve_mode: str = typer.Option(
            DEFAULT_FEATURE_RESOLVE_MODE,
            "--feature-resolve-mode",
        ),
    ) -> None:
        from src.cli.workflows import resolve_limited_classification_setting
        from src.evaluation.registry import register_builtin_tasks, run_task

        if prior_scale is not None:
            try:
                format_prior_scale_variant(prior_scale)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--prior-scale") from exc
        if covariance_type is not None:
            try:
                covariance_type = normalize_covariance_type(covariance_type)
            except ValueError as exc:
                raise typer.BadParameter(
                    str(exc), param_hint="--covariance-type"
                ) from exc
        if vmf_variant is not None:
            try:
                vmf_variant = normalize_vmf_parameter_variant(vmf_variant)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--vmf-variant") from exc
        try:
            mode, value = resolve_limited_classification_setting(
                ratio=ratio, count=count
            )
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
        if alignment_mode not in ALIGNMENT_MODES:
            raise typer.BadParameter(
                f"alignment mode must be one of {', '.join(ALIGNMENT_MODES)}"
            )
        if feature_resolve_mode not in FEATURE_RESOLVE_MODES:
            raise typer.BadParameter(
                "feature resolve mode must be one of "
                f"{', '.join(FEATURE_RESOLVE_MODES)}"
            )
        register_builtin_tasks()
        run_task(
            "classification_limited",
            mode=mode,
            value=value,
            result_root=result_root,
            iterations=iteration,
            datasets=dataset,
            data_runs=data_run,
            categories=list(category) or None,
            topics=topic,
            classifiers=classifier,
            vmf_assignment=_validate_vmf_assignment(vmf_assignment),
            target_column=target_column,
            label_schema=label_schema,
            stratified=stratified,
            seed=seed,
            sampling_repeats=list(sampling_repeat) or None,
            sampling_seed_stride=sampling_seed_stride,
            sampling_max_attempts=sampling_max_attempts,
            sampling_retry_seed_stride=sampling_retry_seed_stride,
            alignment_mode=alignment_mode,
            embedding_variants=list(embedding_variant) or None,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=list(model) or None,
            prior_scale=prior_scale,
            covariance_type=covariance_type,
            vmf_variant=vmf_variant,
        )

    @evaluation_app.command(
        "entropy-based-metrics",
        help=(
            "Compute entropy-based diagnostics on document-topic distributions "
            "(MALLET document_entropy / rank_1_docs and H(theta_d)) and write them under "
            "results/topic_analysis/entropy_based/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "with results/topic_analysis/entropy_based/latest/.../CURRENT.json pointers."
        ),
    )
    def entropy_based_metrics(
        model: list[str] = typer.Option(["vmf"], "--model"),
        dataset: str = typer.Option(...),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        topic: list[int] = typer.Option(..., "--topic"),
        category: list[str] = typer.Option(["all"], "--category"),
        split: str = typer.Option("test", "--split"),
        doc_topic_source: str = typer.Option(
            "auto",
            "--doc-topic-source",
            help=(
                "auto (default: foldincounts for vMF Sentence LDA, soft_proxy for the "
                "baselines; the resolved value is recorded per condition), soft_proxy "
                "(soft file, else mean of sentence posteriors, else hard), soft, hard, "
                "foldin or foldincounts (the collapsed fold-in theta / expected counts "
                "written at training time and by `evaluation vmf-foldin-theta`; vMF only)."
            ),
        ),
        diffuse_entropy_threshold: float = typer.Option(
            0.95,
            "--diffuse-entropy-threshold",
            min=0.0,
            max=1.0,
            help="A topic counts as diffuse when H(P(d|k))/ln D exceeds this value.",
        ),
        dead_rank1_threshold: float = typer.Option(
            0.01,
            "--dead-rank1-threshold",
            min=0.0,
            max=1.0,
            help="A topic counts as dead when its rank-1 document fraction is below this value.",
        ),
        embedding_variant: Optional[str] = typer.Option(None, "--embedding-variant"),
        encoder_model: Optional[str] = typer.Option(None, "--encoder-model"),
        word_embedding_variant: str = typer.Option(
            "googlenews300",
            "--word-embedding-variant",
            help="Result-path suffix for ETM / Gaussian LDA / MvTM conditions.",
        ),
        prior_scale: Optional[float] = typer.Option(
            None,
            "--prior-scale",
            min=0.0,
            help="Psi_0 prior scale used to disambiguate GaussianLDA-family conditions.",
        ),
        vmf_variant: Optional[str] = typer.Option(
            None,
            "--vmf-variant",
            help=(
                "Hyperparameter-sweep label of the vMF Sentence LDA runs to evaluate "
                "(e.g. kappa0-100; src/core/vmf_variant.py). Unset selects the default runs."
            ),
        ),
        out_root: Path = typer.Option(
            Path("results/topic_analysis/entropy_based"), file_okay=False
        ),
        save_per_iter_artifacts: bool = typer.Option(
            True, "--save-per-iter-artifacts/--no-save-per-iter-artifacts"
        ),
        skip_existing: bool = typer.Option(False, "--skip-existing"),
        condition_failure_policy: str = typer.Option(
            "fail-fast", "--condition-failure-policy"
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        if prior_scale is not None:
            try:
                format_prior_scale_variant(prior_scale)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--prior-scale") from exc
        register_builtin_tasks()
        summary_path = run_task(
            "entropy_based_metrics",
            models=model,
            dataset=dataset,
            data_runs=data_run,
            iterations=iteration,
            num_topics=topic,
            categories=category,
            split=split,
            doc_topic_source=doc_topic_source,
            diffuse_entropy_threshold=diffuse_entropy_threshold,
            dead_rank1_threshold=dead_rank1_threshold,
            embedding_variant=embedding_variant,
            encoder_model=encoder_model,
            word_embedding_variant=word_embedding_variant,
            prior_scale=prior_scale,
            vmf_variant=vmf_variant,
            out_root=out_root,
            save_per_iter_artifacts=save_per_iter_artifacts,
            skip_existing=skip_existing,
            condition_failure_policy=condition_failure_policy,
        )
        typer.echo(str(summary_path))

    @evaluation_app.command(
        "entropy-based-summary",
        help=(
            "Rebuild cross-condition entropy-based tables (CSV/JSON/LaTeX), the "
            "per-(dataset, encoder, K) *.scores.json sidecars and per-document "
            "entropy histograms from "
            "results/topic_analysis/entropy_based/latest/**/CURRENT.json."
        ),
    )
    def entropy_based_summary(
        out_root: Path = typer.Option(
            Path("results/topic_analysis/entropy_based"), "--out-root", file_okay=False
        ),
        output_dir: Optional[Path] = typer.Option(
            None, "--output-dir", file_okay=False
        ),
        dataset: list[str] = typer.Option([], "--dataset"),
        data_run: list[str] = typer.Option([], "--data-run"),
        model: list[str] = typer.Option([], "--model"),
        topic: list[int] = typer.Option([], "--topic"),
        split: Optional[str] = typer.Option(None, "--split"),
        doc_topic_source: Optional[str] = typer.Option(None, "--doc-topic-source"),
        exclude_category: list[str] = typer.Option(["all"], "--exclude-category"),
        tex: bool = typer.Option(True, "--tex/--no-tex"),
        plots: bool = typer.Option(True, "--plots/--no-plots"),
        paper: bool = typer.Option(
            False,
            "--paper/--no-paper",
            help=(
                "Also draw the manuscript's box-plot figure of the per-topic and "
                "per-document distributions (K=20, MiniLM) at the page width of the "
                "other paper grids, so it can be included at natural size. The paper "
                "repository's `make sync` passes this."
            ),
        ),
        paper_vmf_doc_topic_source: Optional[str] = typer.Option(
            None,
            "--paper-vmf-doc-topic-source",
            help=(
                "Draw the vMF boxes of the paper figure from the conditions computed "
                "under this doc-topic source (e.g. foldincounts) instead of "
                "--doc-topic-source; the tables and sidecars are unaffected."
            ),
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        summary_path = run_task(
            "entropy_based_summary",
            out_root=out_root,
            output_dir=output_dir,
            datasets=list(dataset) or None,
            data_runs=list(data_run) or None,
            models=list(model) or None,
            topics=list(topic) or None,
            split=split,
            doc_topic_source=doc_topic_source,
            exclude_categories=list(exclude_category),
            write_tex=tex,
            write_plots=plots,
            paper=paper,
            paper_vmf_doc_topic_source=paper_vmf_doc_topic_source,
        )
        typer.echo(str(summary_path))

    @evaluation_app.command(
        "summarize-timing",
        help=(
            "Wall-clock and convergence summaries for the manuscript (TODO T12): "
            "reads the per-iteration traces recorded by the vSLDA and GSLDA trainers "
            "and writes per-(dataset, encoder, K) timing *.scores.json sidecars. "
            "With --paper, also draws the two-panel convergence figure (each "
            "model's average log-likelihood against cumulative training "
            "wall-clock). Measure with ./tmp_timing.sh, which pins the thread "
            "pools and gives every run its own physical core."
        ),
    )
    def summarize_timing(
        results_root: Path = typer.Option(
            Path("results"), "--results-root", file_okay=False
        ),
        dataset: str = typer.Option("20newsgroup_timing", "--dataset"),
        data_run: str = typer.Option("default", "--data-run"),
        output_dir: Optional[Path] = typer.Option(
            None, "--output-dir", file_okay=False
        ),
        paper: bool = typer.Option(
            False,
            "--paper/--no-paper",
            help=(
                "Also draw the manuscript's convergence figure at the page width "
                "shared by the other paper figures."
            ),
        ),
        mark_iteration: Optional[int] = typer.Option(
            10,
            "--mark-iteration",
            help=(
                "Draw a rule at this iteration count, i.e. the budget the main "
                "experiments actually train for. Pass 0 to omit it."
            ),
        ),
        figure_encoder: Optional[str] = typer.Option(
            None,
            "--encoder",
            help=(
                "Encoder variant whose runs the figure draws (e.g. minilm). "
                "Default: the runs with the longest traces."
            ),
        ),
    ) -> None:
        from src.evaluation.reports.timing import summarize_timing as _summarize_timing

        report = _summarize_timing(
            results_root=results_root,
            dataset=dataset,
            data_run=data_run,
            output_dir=output_dir,
            paper=paper,
            mark_iteration=mark_iteration or None,
            figure_encoder=figure_encoder,
        )
        typer.echo(f"{report['runs']} run(s) summarized")
        for path in report["scores"] + report["figures"]:
            typer.echo(str(path))

    @evaluation_app.command(
        "topic-pair-metrics",
        help=(
            "All-topic-pair analysis of the sentence-level models (vmf, sentlda, "
            "sentence_gaussianlda) in the shared sentence-embedding space: the "
            "training sentences are re-encoded once per category (cached), the "
            "sentence-topic posteriors are re-estimated with the collapsed fold-in "
            "of the representative-word protocol, and per-topic concentration, "
            "centroid cosine, assignment confusion, fine-label divergence and the "
            "cross-model topic overlap are written under "
            "results/topic_analysis/topic_pairs/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "with results/topic_analysis/topic_pairs/latest/.../CURRENT.json pointers."
        ),
    )
    def topic_pair_metrics(
        model: list[str] = typer.Option(
            ["vmf", "sentlda", "sentence_gaussianlda"], "--model"
        ),
        dataset: str = typer.Option(...),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        topic: list[int] = typer.Option(..., "--topic"),
        category: list[str] = typer.Option(..., "--category"),
        split: str = typer.Option("train", "--split"),
        embedding_variant: Optional[str] = typer.Option(None, "--embedding-variant"),
        encoder_model: Optional[str] = typer.Option(None, "--encoder-model"),
        prior_scale: Optional[float] = typer.Option(
            None,
            "--prior-scale",
            min=0.0,
            help="Psi_0 prior scale that selects the Sentence Gaussian LDA runs (0.1 for the paper grid).",
        ),
        out_root: Path = typer.Option(
            Path("results/topic_analysis/topic_pairs"), "--out-root", file_okay=False
        ),
        cache_root: Optional[Path] = typer.Option(
            None,
            "--cache-root",
            file_okay=False,
            help="Sentence-embedding cache; default <out-root>/.cache/sentence_embeddings.",
        ),
        encoder_device: str = typer.Option("auto", "--encoder-device"),
        encode_batch_size: Optional[int] = typer.Option(
            None, "--encode-batch-size", min=1
        ),
        target_column: str = typer.Option("target_str", "--target-column"),
        foldin_burn_in: Optional[int] = typer.Option(None, "--foldin-burn-in", min=0),
        foldin_retained: Optional[int] = typer.Option(None, "--foldin-retained", min=1),
        foldin_seed: Optional[int] = typer.Option(None, "--foldin-seed"),
        model_reference: bool = typer.Option(
            True, "--model-reference/--no-model-reference"
        ),
        save_per_iter_artifacts: bool = typer.Option(
            True, "--save-per-iter-artifacts/--no-save-per-iter-artifacts"
        ),
        skip_existing: bool = typer.Option(False, "--skip-existing"),
        condition_failure_policy: str = typer.Option(
            "fail-fast", "--condition-failure-policy"
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task
        from src.evaluation.topic_pairs.metrics import foldin_config_from_options

        if prior_scale is not None:
            try:
                format_prior_scale_variant(prior_scale)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--prior-scale") from exc
        register_builtin_tasks()
        summary_path = run_task(
            "topic_pair_metrics",
            models=model,
            dataset=dataset,
            data_runs=data_run,
            iterations=iteration,
            num_topics=topic,
            categories=category,
            split=split,
            embedding_variant=embedding_variant,
            encoder_model=encoder_model,
            prior_scale=prior_scale,
            out_root=out_root,
            cache_root=(
                cache_root
                if cache_root is not None
                else out_root / ".cache" / "sentence_embeddings"
            ),
            encoder_device=encoder_device,
            encode_batch_size=encode_batch_size,
            target_column=target_column,
            foldin_config=foldin_config_from_options(
                burn_in=foldin_burn_in, retained=foldin_retained, seed=foldin_seed
            ),
            include_model_reference=model_reference,
            save_per_iter_artifacts=save_per_iter_artifacts,
            skip_existing=skip_existing,
            condition_failure_policy=condition_failure_policy,
        )
        typer.echo(str(summary_path))

    @evaluation_app.command(
        "vmf-foldin-theta",
        help=(
            "Collapsed fold-in document-topic distributions of the vMF Sentence LDA "
            "runs: for every selected run and split the sentences are re-encoded "
            "(cached, shared with topic-pair-metrics), the sentence-topic posteriors "
            "are re-estimated with the collapsed fold-in of the representative-word "
            "protocol under the frozen topic parameters, and "
            "theta_dk = (E[n_dk] + alpha_k) / (N_d + sum alpha) is written into the "
            "run directory as doc_topic_<split>_foldin.pkl, with the expected counts "
            "E[n_dk] in doc_topic_<split>_foldin_counts.pkl (and foldin_meta.json). "
            "--model mvtm does the same for the MvTM (vLDA) runs over word tokens, "
            "writing params/<category>_doc_topic_foldin*.pkl and infer/<category>_"
            "doc_topic_foldin*.pkl. Classification, entropy and the summaries read "
            "the files through --vmf-assignment / --doc-topic-source."
        ),
    )
    def vmf_foldin_theta(
        model: str = typer.Option(
            "vmf_sentence_lda",
            "--model",
            help="vmf_sentence_lda (default; sentence units) or mvtm (vLDA; word-token units).",
        ),
        chunk_docs: int = typer.Option(
            256,
            "--chunk-docs",
            min=1,
            help="mvtm only: documents per sampler call (bounds memory; part of the fingerprint).",
        ),
        dataset: list[str] = typer.Option(["20newsgroup", "nyt"], "--dataset"),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        category: list[str] = typer.Option([], "--category"),
        iteration: list[int] = typer.Option([], "--iteration"),
        topic: list[int] = typer.Option([], "--topic"),
        split: list[str] = typer.Option(["train", "test"], "--split"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        vmf_variant: list[str] = typer.Option(
            [],
            "--vmf-variant",
            help="Hyperparameter-sweep variants (e.g. t-1); by default only the main setting is selected.",
        ),
        include_vmf_variants: bool = typer.Option(
            False,
            "--include-vmf-variants",
            help="Also select every hyperparameter-sweep variant.",
        ),
        all_vmf_runs: bool = typer.Option(
            False,
            "--all-vmf-runs",
            help="Select every run of the datasets and data runs, ignoring the other filters.",
        ),
        cache_root: Optional[Path] = typer.Option(
            None,
            "--cache-root",
            file_okay=False,
            help="Sentence-embedding cache; default results/topic_analysis/topic_pairs/.cache/sentence_embeddings.",
        ),
        encoder_device: str = typer.Option("auto", "--encoder-device"),
        encode_batch_size: Optional[int] = typer.Option(
            None, "--encode-batch-size", min=1
        ),
        foldin_burn_in: Optional[int] = typer.Option(None, "--foldin-burn-in", min=0),
        foldin_retained: Optional[int] = typer.Option(None, "--foldin-retained", min=1),
        foldin_seed: Optional[int] = typer.Option(None, "--foldin-seed"),
        skip_existing: bool = typer.Option(
            True,
            "--skip-existing/--no-skip-existing",
            help="Skip a run/split whose artifact was written for the same settings and inputs.",
        ),
        write_sentence_posteriors: bool = typer.Option(
            False,
            "--write-sentence-posteriors",
            help="Also write sentence_topic_<split>_foldin.pkl (large).",
        ),
        update_pointer: bool = typer.Option(
            True,
            "--update-pointer/--no-update-pointer",
            help="Add the new artifact keys to the run's CURRENT.json.",
        ),
        condition_failure_policy: str = typer.Option(
            "fail-fast", "--condition-failure-policy"
        ),
        summary_path: Optional[Path] = typer.Option(
            None, "--summary-path", dir_okay=False
        ),
    ) -> None:
        from src.evaluation.foldin.runner import (
            CONDITION_FAILURE_POLICIES as FOLDIN_FAILURE_POLICIES,
        )
        from src.evaluation.foldin.runner import (
            normalize_foldin_model,
        )
        from src.evaluation.registry import register_builtin_tasks, run_task
        from src.evaluation.topic_pairs.metrics import (
            DEFAULT_CACHE_ROOT,
            foldin_config_from_options,
        )

        try:
            model = normalize_foldin_model(model)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--model") from exc
        for value in split:
            if value not in {"train", "test"}:
                raise typer.BadParameter(
                    f"split must be train or test, got {value!r}", param_hint="--split"
                )
        if condition_failure_policy not in FOLDIN_FAILURE_POLICIES:
            raise typer.BadParameter(
                "condition failure policy must be one of "
                f"{', '.join(FOLDIN_FAILURE_POLICIES)}",
                param_hint="--condition-failure-policy",
            )
        register_builtin_tasks()
        result_path = run_task(
            "vmf_foldin_theta",
            model=model,
            chunk_docs=chunk_docs,
            datasets=dataset,
            data_runs=data_run,
            categories=list(category) or None,
            iterations=list(iteration) or None,
            num_topics=list(topic) or None,
            splits=split,
            embedding_variants=list(embedding_variant) or None,
            vmf_variants=list(vmf_variant) or None,
            include_vmf_variants=include_vmf_variants,
            all_vmf_runs=all_vmf_runs,
            cache_root=cache_root if cache_root is not None else DEFAULT_CACHE_ROOT,
            encoder_device=encoder_device,
            encode_batch_size=encode_batch_size,
            foldin_config=foldin_config_from_options(
                burn_in=foldin_burn_in, retained=foldin_retained, seed=foldin_seed
            ),
            skip_existing=skip_existing,
            write_sentence_posteriors=write_sentence_posteriors,
            update_pointer=update_pointer,
            condition_failure_policy=condition_failure_policy,
            summary_path=summary_path,
        )
        typer.echo(str(result_path))

    @evaluation_app.command(
        "topic-pair-summary",
        help=(
            "Rebuild the cross-condition topic-pair review table and the "
            "per-(dataset, encoder, K) *.scores.json sidecars (per-topic arrays, "
            "K x K pair matrices, cross-model overlap, provenance; nothing "
            "aggregated) from results/topic_analysis/topic_pairs/latest/**/CURRENT.json."
        ),
    )
    def topic_pair_summary(
        out_root: Path = typer.Option(
            Path("results/topic_analysis/topic_pairs"), "--out-root", file_okay=False
        ),
        output_dir: Optional[Path] = typer.Option(
            None, "--output-dir", file_okay=False
        ),
        dataset: list[str] = typer.Option([], "--dataset"),
        data_run: list[str] = typer.Option([], "--data-run"),
        model: list[str] = typer.Option([], "--model"),
        topic: list[int] = typer.Option([], "--topic"),
        split: Optional[str] = typer.Option(None, "--split"),
        exclude_category: list[str] = typer.Option(["all"], "--exclude-category"),
        compact: bool = typer.Option(True, "--compact/--indent"),
        paper: bool = typer.Option(
            False,
            "--paper/--no-paper",
            help=(
                "Restrict the sidecars to the manuscript grid (20newsgroup + nyt, "
                "MiniLM, K in {10, 20, 30}, five runs, three models), fail when "
                "any cell of it is missing, and write the representative-word "
                "sidecars of the reference runs. The paper repository's "
                "`make sync` passes this."
            ),
        ),
        coherence_root: Path = typer.Option(
            Path("results/topic_analysis/coherence"),
            "--coherence-root",
            file_okay=False,
            help="Coherence results whose display words the reference-run sidecars carry.",
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        summary_path = run_task(
            "topic_pair_summary",
            out_root=out_root,
            output_dir=output_dir,
            datasets=list(dataset) or None,
            data_runs=list(data_run) or None,
            models=list(model) or None,
            topics=list(topic) or None,
            split=split,
            exclude_categories=list(exclude_category),
            paper=paper,
            compact=compact,
            coherence_root=coherence_root,
        )
        typer.echo(str(summary_path))

    @evaluation_app.command(
        "summarize-classification",
        help=(
            "Read classification outputs from "
            "results/classification/latest/<dataset>/<data_run>/all/<display_key>/CURRENT.json "
            "when available, otherwise fall back to legacy category-first directories. "
            "Use --resolve-mode strict to fail on ambiguous matches instead of picking the newest."
        ),
    )
    def summarize_classification(
        metric: str = typer.Option("acc"),
        dataset: str = typer.Option(...),
        data_run: str = typer.Option("default", "--data-run"),
        topic: int = typer.Option(...),
        iteration: list[int] = typer.Option([0], "--iteration"),
        classifier: list[str] = typer.Option([], "--classifier"),
        vmf_assignment: str = typer.Option(
            DEFAULT_VMF_ASSIGNMENT, help=VMF_ASSIGNMENT_HELP
        ),
        alignment_mode: str = typer.Option(DEFAULT_ALIGNMENT_MODE, "--alignment-mode"),
        resolve_mode: str = typer.Option("latest", "--resolve-mode"),
        result_root: Path = typer.Option(CLASSIFICATION_RESULTS_ROOT, file_okay=False),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        model: list[str] = typer.Option([], "--model"),
        exclude_category: list[str] = typer.Option([], "--exclude-category"),
        prior_scale: Optional[float] = typer.Option(None, "--prior-scale", min=0.0),
        covariance_type: Optional[str] = typer.Option(
            None,
            "--covariance-type",
            help=(
                "Covariance type of the sentence Gaussian LDA runs to use: full "
                "(default), diag, or spherical (alias iso)."
            ),
        ),
        vmf_variant: Optional[str] = typer.Option(
            None,
            "--vmf-variant",
            help=(
                "Hyperparameter-sweep label of the vMF Sentence LDA runs to use "
                "(e.g. kappa0-100, alpha0-0p1, zeta-40, b-16, t-5; src/core/vmf_variant.py). "
                "Unset selects the default runs and leaves every path unchanged."
            ),
        ),
        include_all_category: bool = typer.Option(False, "--include-all-category"),
        feature_resolve_mode: str = typer.Option(
            DEFAULT_FEATURE_RESOLVE_MODE,
            "--feature-resolve-mode",
        ),
        output_path: Optional[Path] = typer.Option(
            None,
            "--output-path",
            dir_okay=False,
            help="Write the rendered LaTeX table to this .tex file.",
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        if prior_scale is not None:
            try:
                format_prior_scale_variant(prior_scale)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--prior-scale") from exc
        if covariance_type is not None:
            try:
                covariance_type = normalize_covariance_type(covariance_type)
            except ValueError as exc:
                raise typer.BadParameter(
                    str(exc), param_hint="--covariance-type"
                ) from exc
        if vmf_variant is not None:
            try:
                vmf_variant = normalize_vmf_parameter_variant(vmf_variant)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--vmf-variant") from exc
        if resolve_mode not in {"latest", "strict"}:
            raise typer.BadParameter("resolve mode must be 'latest' or 'strict'")
        if alignment_mode not in ALIGNMENT_MODES:
            raise typer.BadParameter(
                f"alignment mode must be one of {', '.join(ALIGNMENT_MODES)}"
            )
        if feature_resolve_mode not in FEATURE_RESOLVE_MODES:
            raise typer.BadParameter(
                "feature resolve mode must be one of "
                f"{', '.join(FEATURE_RESOLVE_MODES)}"
            )
        register_builtin_tasks()
        run_task(
            "classification_summary",
            metric=metric,
            dataset=dataset,
            data_run=data_run,
            topics=topic,
            iterations=iteration,
            classifiers=list(classifier) or None,
            vmf_assignment=_validate_vmf_assignment(vmf_assignment),
            alignment_mode=alignment_mode,
            resolve_mode=resolve_mode,
            result_root=result_root,
            target_column=target_column,
            label_schema=label_schema,
            embedding_variants=list(embedding_variant) or None,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=list(model) or None,
            prior_scale=prior_scale,
            covariance_type=covariance_type,
            vmf_variant=vmf_variant,
            excluded_categories=list(exclude_category) or None,
            include_all_category=include_all_category,
            output_path=output_path,
        )

    @evaluation_app.command(
        "summarize-coherence",
        help=(
            "Aggregate word-based (topic coherence / diversity) runs from "
            "results/topic_analysis/coherence into per-(dataset, topic count, "
            "encoder) summaries. Writes <stem>.scores.json with the raw per-run "
            "values and provenance, which downstream table builders read, plus "
            ".tex/.runs.json/.runs.csv review artifacts. Pass --iteration to "
            "pin the expected runs instead of inferring them from the data."
        ),
    )
    def summarize_coherence(
        coherence_root: Path = typer.Option(
            DEFAULT_COHERENCE_ROOT, "--coherence-root", file_okay=False
        ),
        source: str = typer.Option("latest", "--source"),
        iteration: list[int] = typer.Option([], "--iteration"),
        dataset: list[str] = typer.Option([], "--dataset"),
        summary_root: Optional[Path] = typer.Option(
            None, "--summary-root", file_okay=False
        ),
        metric: list[str] = typer.Option([], "--metric"),
        table_digits: int = typer.Option(4, "--table-digits", min=0, max=12),
        include_all_category: bool = typer.Option(False, "--include-all-category"),
        strict: bool = typer.Option(False, "--strict"),
        word2vec: Optional[str] = typer.Option(None, "--word2vec"),
        prior_scale: Optional[float] = typer.Option(None, "--prior-scale", min=0.0),
        covariance_type: Optional[str] = typer.Option(
            None,
            "--covariance-type",
            help=(
                "Covariance type of the sentence Gaussian LDA runs to use: full "
                "(default), diag, or spherical (alias iso)."
            ),
        ),
        vmf_variant: Optional[str] = typer.Option(
            None,
            "--vmf-variant",
            help=(
                "Hyperparameter-sweep label of the vMF Sentence LDA runs to use "
                "(e.g. kappa0-100, alpha0-0p1, zeta-40, b-16, t-5; src/core/vmf_variant.py). "
                "Unset selects the default runs and leaves every path unchanged."
            ),
        ),
        coherence_reference_num_docs: Optional[int] = typer.Option(
            None, "--coherence-reference-num-docs", min=1
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        if source not in {"latest", "archive", "all"}:
            raise typer.BadParameter("source must be 'latest', 'archive' or 'all'")
        if prior_scale is not None:
            try:
                format_prior_scale_variant(prior_scale)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--prior-scale") from exc
        if covariance_type is not None:
            try:
                covariance_type = normalize_covariance_type(covariance_type)
            except ValueError as exc:
                raise typer.BadParameter(
                    str(exc), param_hint="--covariance-type"
                ) from exc
        if vmf_variant is not None:
            try:
                vmf_variant = normalize_vmf_parameter_variant(vmf_variant)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--vmf-variant") from exc
        register_builtin_tasks()
        summary_path = run_task(
            "word_based_summary",
            coherence_root=coherence_root,
            source_mode=source,
            iterations=list(iteration) or None,
            summary_root=summary_root,
            datasets=list(dataset) or None,
            metrics=list(metric) or None,
            table_digits=table_digits,
            include_all_category=include_all_category,
            strict=strict,
            word2vec=word2vec,
            prior_scale=prior_scale,
            covariance_type=covariance_type,
            vmf_variant=vmf_variant,
            coherence_reference_num_docs=coherence_reference_num_docs,
        )
        typer.echo(str(summary_path))

    @evaluation_app.command(
        "topic-sweep",
        help=(
            "Aggregate existing classification and coherence results into "
            "results/analysis/topic_sweep/<dataset>/topic_sweep.csv (plus a "
            "provenance sidecar, LaTeX tables and the K sweep figures). "
            "Nothing is recomputed; conditions that are absent are listed under "
            "'missing' in topic_sweep.json."
        ),
    )
    def topic_sweep(
        dataset: list[str] = typer.Option(..., "--dataset"),
        topics: list[int] = typer.Option(..., "--topics"),
        models: list[str] = typer.Option(list(DEFAULT_TOPIC_SWEEP_MODELS), "--models"),
        categories: Optional[list[str]] = typer.Option(None, "--categories"),
        include_all_category: bool = typer.Option(
            True, "--include-all-category/--no-include-all-category"
        ),
        data_run: str = typer.Option("default", "--data-run"),
        classifier: str = typer.Option("svm", "--classifier"),
        embedding_variant: str = typer.Option("minilm", "--embedding-variant"),
        vmf_assignment: str = typer.Option(
            DEFAULT_VMF_ASSIGNMENT,
            "--vmf-assignment",
            help=(
                "Classification sidecars to read (the <assignment> of "
                "acc_<dataset>_<run>_<classifier>_<encoder>_<assignment>_<K>topic.scores.json); "
                "hard, soft, foldin or foldincounts."
            ),
        ),
        baseline_vmf_assignment: Optional[str] = typer.Option(
            None,
            "--baseline-vmf-assignment",
            help=(
                "Read every model but vSLDA from the sidecars of this estimator "
                "(the fold-in sidecars carry vSLDA alone). Default: --vmf-assignment."
            ),
        ),
        metric: list[str] = typer.Option(
            list(DEFAULT_TOPIC_SWEEP_CLASSIFICATION_METRICS), "--metric"
        ),
        word_based_metric: list[str] = typer.Option(
            list(DEFAULT_TOPIC_SWEEP_WORD_BASED_METRICS), "--word-based-metric"
        ),
        classification_root: Path = typer.Option(
            DEFAULT_TOPIC_SWEEP_CLASSIFICATION_ROOT,
            "--classification-root",
            file_okay=False,
        ),
        coherence_summary: Path = typer.Option(
            DEFAULT_TOPIC_SWEEP_COHERENCE_SUMMARY,
            "--coherence-summary",
            dir_okay=False,
        ),
        out_root: Path = typer.Option(
            DEFAULT_TOPIC_SWEEP_OUT_ROOT, "--out-root", file_okay=False
        ),
        latex: bool = typer.Option(True, "--latex/--no-latex"),
        plot: bool = typer.Option(True, "--plot/--no-plot"),
        paper: bool = typer.Option(
            False,
            "--paper/--no-paper",
            help=(
                "Lay the figures out at the manuscript's page width, with the "
                "panel size and 8/9 pt type of the sample-efficiency figures, "
                "so they can be included at natural size. The paper repository's "
                "`make sync` passes this."
            ),
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "topic_sweep_summary",
            datasets=dataset,
            topics=topics,
            models=models,
            categories=categories or None,
            include_all_category=include_all_category,
            data_run=data_run,
            classifier=classifier,
            embedding_variant=embedding_variant,
            vmf_assignment=_validate_vmf_assignment(vmf_assignment),
            baseline_vmf_assignment=(
                None
                if baseline_vmf_assignment is None
                else _validate_vmf_assignment(baseline_vmf_assignment)
            ),
            classification_metrics=metric,
            word_based_metrics=word_based_metric,
            classification_root=classification_root,
            coherence_summary=coherence_summary,
            out_root=out_root,
            write_latex=latex,
            plot=plot,
            paper=paper,
        )

    @evaluation_app.command("list-tasks")
    def list_evaluation_tasks() -> None:
        from src.evaluation.registry import list_tasks, register_builtin_tasks

        register_builtin_tasks()
        typer.echo("task\toutput\trun_from_config\tdescription")
        for task in list_tasks():
            typer.echo(
                f"{task.name}\t{task.output_kind}\t"
                f"{'yes' if task.run_from_config_supported else 'no'}\t"
                f"{task.description}"
            )

    @evaluation_app.command(
        "run-from-config",
        help=(
            "Run supported evaluation task(s) from a comparison config. "
            "When --task is omitted, evaluation.tasks is used. "
            "Generated outputs follow each task's current layout; migrated tasks use "
            "latest/archive pointers."
        ),
    )
    def run_evaluation_from_config(
        config: Path = typer.Option(..., exists=True, dir_okay=False),
        task: Optional[str] = typer.Option(None),
        classifier: list[str] = typer.Option([], "--classifier"),
        vmf_assignment: str = typer.Option(
            DEFAULT_VMF_ASSIGNMENT, help=VMF_ASSIGNMENT_HELP
        ),
        result_root: Path = typer.Option(CLASSIFICATION_RESULTS_ROOT, file_okay=False),
        target_column: Optional[str] = typer.Option(None),
        label_schema: str = typer.Option("identity"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        feature_resolve_mode: Optional[str] = typer.Option(
            None,
            "--feature-resolve-mode",
        ),
    ) -> None:
        from src.cli.workflows import run_evaluation_from_config_workflow

        if (
            feature_resolve_mode is not None
            and feature_resolve_mode not in FEATURE_RESOLVE_MODES
        ):
            raise typer.BadParameter(
                "feature resolve mode must be one of "
                f"{', '.join(FEATURE_RESOLVE_MODES)}"
            )
        run_evaluation_from_config_workflow(
            config=config,
            task=task,
            classifiers=classifier,
            vmf_assignment=_validate_vmf_assignment(vmf_assignment),
            result_root=result_root,
            target_column=target_column,
            label_schema=label_schema,
            embedding_variants=list(embedding_variant) or None,
            feature_resolve_mode=feature_resolve_mode,
        )

    @evaluation_app.command(
        "geometry-based-metrics",
        help=(
            "Write geometry metrics under "
            "results/topic_analysis/geometry_based/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/topic_analysis/geometry_based/latest/.../CURRENT.json "
            "by default, or under --out-root when provided."
        ),
    )
    def geometry_based_metrics(
        model: list[str] = typer.Option(["vmf"], "--model"),
        dataset: str = typer.Option(...),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        topic: int = typer.Option(...),
        category: list[str] = typer.Option(["all"], "--category"),
        dup_threshold: float = typer.Option(0.90),
        embedding_variant: Optional[str] = typer.Option(None, "--embedding-variant"),
        encoder_model: Optional[str] = typer.Option(
            None,
            "--encoder-model",
            "--encoder_model",
            help=(
                "Resolve the embedding variant from an encoder model name, e.g. "
                "sentence-transformers/all-mpnet-base-v2 -> mpnet."
            ),
        ),
        out_root: Path = typer.Option(
            Path("results/topic_analysis/geometry_based"), file_okay=False
        ),
        save_per_iter_artifacts: bool = typer.Option(False),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "geometry_based_metrics",
            models=model,
            dataset=dataset,
            data_runs=data_run,
            iterations=iteration,
            num_topics=topic,
            categories=category,
            dup_threshold=dup_threshold,
            embedding_variant=embedding_variant,
            encoder_model=encoder_model,
            out_root=out_root,
            save_per_iter_artifacts=save_per_iter_artifacts,
        )

    @evaluation_app.command(
        "sentence-topic-inspection",
        help=(
            "Write inspection payloads under "
            "results/visualization/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/visualization/latest/.../CURRENT.json."
        ),
    )
    def sentence_topic_inspection(
        model: str = typer.Option("vmf_sentence_lda", "--model"),
        dataset: str = typer.Option(...),
        data_run: str = typer.Option("default", "--data-run"),
        category: list[str] = typer.Option(["all"], "--category"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        topic: list[int] = typer.Option(..., "--topic"),
        condition_id: Optional[str] = typer.Option(None, "--condition-id"),
        embedding_variant: Optional[str] = typer.Option(None, "--embedding-variant"),
        num_components: Optional[int] = typer.Option(None, "--num-components"),
        gaussian_condition_id: Optional[str] = typer.Option(
            None, "--gaussian-condition-id"
        ),
        gaussian_embedding_variant: Optional[str] = typer.Option(
            None, "--gaussian-embedding-variant"
        ),
        gaussian_num_components: Optional[int] = typer.Option(
            None, "--gaussian-num-components"
        ),
        topk: int = typer.Option(5),
        encoder: Optional[str] = typer.Option(None),
        split: str = typer.Option("train"),
        data_column: str = typer.Option("data"),
        target_column: str = typer.Option("target_str"),
        delimiter: str = typer.Option(" / "),
        language: str = typer.Option("english"),
        segmenter: str = typer.Option("delimiter"),
        seed: int = typer.Option(DEFAULT_RANDOM_SEED),
        gaussian_topk: bool = typer.Option(False),
        device: Optional[str] = typer.Option(None),
        encode_batch_size: int = typer.Option(64),
        max_points: int = typer.Option(2000),
        results_root: Path = typer.Option(Path("results"), file_okay=False),
        out_root: Path = typer.Option(Path("results/visualization"), file_okay=False),
        no_progress: bool = typer.Option(False),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "sentence_topic_inspection",
            model=model,
            dataset=dataset,
            data_run=data_run,
            categories=category,
            iterations=iteration,
            num_topics_list=topic,
            source_condition_id=condition_id,
            embedding_variant=embedding_variant,
            num_components=num_components,
            gaussian_condition_id=gaussian_condition_id,
            gaussian_embedding_variant=gaussian_embedding_variant,
            gaussian_num_components=gaussian_num_components,
            top_k=topk,
            encoder_model=encoder,
            split=split,
            data_column=data_column,
            target_column=target_column,
            delimiter=delimiter,
            language=language,
            segmenter=segmenter,
            seed=seed,
            gaussian_topk=gaussian_topk,
            device=device,
            encode_batch_size=encode_batch_size,
            show_progress=not no_progress,
            max_points=max_points,
            results_root=results_root,
            out_root=out_root,
        )

    @evaluation_app.command(
        "word-based-label-profile",
        help=(
            "Write label-profile outputs under "
            "results/topic_analysis/label_profile/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/topic_analysis/label_profile/latest/.../CURRENT.json."
        ),
    )
    def word_based_label_profile(
        model: str = typer.Option(...),
        dataset: str = typer.Option(...),
        data_run: str = typer.Option("default", "--data-run"),
        category: str = typer.Option("all"),
        split: str = typer.Option("train"),
        iteration: int = typer.Option(...),
        topic: int = typer.Option(...),
        top_n: int = typer.Option(5),
        sort_by: str = typer.Option("ratio"),
        pmi_eps: float = typer.Option(1e-12),
        min_docs_per_label: int = typer.Option(1),
        vmf_assignment: str = typer.Option("soft", help=VMF_ASSIGNMENT_HELP),
        results_root: Path = typer.Option(Path("results"), file_okay=False),
        data_column: str = typer.Option("data"),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        delimiter: str = typer.Option(" / "),
        out_json: Optional[Path] = typer.Option(None),
        out_csv: Optional[Path] = typer.Option(None),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "word_based_label_profile",
            model=model,
            dataset=dataset,
            data_run=data_run,
            category=category,
            split=split,
            iteration=iteration,
            num_topics=topic,
            top_n=top_n,
            sort_by=sort_by,
            pmi_eps=pmi_eps,
            min_docs_per_label=min_docs_per_label,
            vmf_assignment=_validate_vmf_assignment(vmf_assignment),
            results_root=results_root,
            data_column=data_column,
            target_column=target_column,
            label_schema=label_schema,
            delimiter=delimiter,
            out_json=out_json,
            out_csv=out_csv,
        )

    @evaluation_app.command("word-based-topic-word-table")
    def word_based_topic_word_table(
        profile_json: Path = typer.Option(...),
        topic_words_json: Path = typer.Option(...),
        iteration: Optional[int] = typer.Option(None),
        labels: Optional[list[str]] = typer.Option(None),
        max_topics_per_group: Optional[int] = typer.Option(None),
        topic_source: str = typer.Option("labels"),
        words_per_topic: int = typer.Option(10),
        include_score: bool = typer.Option(False),
        layout: str = typer.Option("horizontal"),
        table_width_scale: float = typer.Option(0.95),
        out_tex: Optional[Path] = typer.Option(None),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        result = run_task(
            "word_based_topic_word_table",
            profile_json=profile_json,
            topic_words_json=topic_words_json,
            iteration=iteration,
            labels=labels,
            max_topics_per_group=max_topics_per_group,
            topic_source=topic_source,
            words_per_topic=words_per_topic,
            include_score=include_score,
            layout=layout,
            table_width_scale=table_width_scale,
            out_tex=out_tex,
        )
        if isinstance(result, Path):
            typer.echo(str(result))
        else:
            typer.echo(str(result), nl=False)

    @evaluation_app.command(
        "word-based-metrics",
        help=(
            "Write topic-word metrics. With the default --out-root "
            "(results/topic_analysis/coherence) results use the legacy "
            "archive/<date>/exec_*/ layout with latest/CURRENT.json; with a "
            "custom --out-root they are written under "
            "<out_root>/<dataset>/<data_run>/<category>/<condition_id>/ "
            "and <out_root>/condition_index.json is updated."
        ),
    )
    def word_based_metrics(
        model: list[str] = typer.Option(["vmf"], "--model"),
        dataset: str = typer.Option(...),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        topic: list[int] = typer.Option(..., "--topic"),
        category: list[str] = typer.Option(["all"], "--category"),
        embedding_variant: Optional[str] = typer.Option("mpnet", "--embedding-variant"),
        out_root: Path = typer.Option(
            Path("results/topic_analysis/coherence"), file_okay=False
        ),
        covariance_type: Optional[str] = typer.Option(
            None,
            "--covariance-type",
            help=(
                "Covariance type of the sentence Gaussian LDA runs to evaluate: full "
                "(default), diag, or spherical (alias iso)."
            ),
        ),
        vmf_variant: Optional[str] = typer.Option(
            None,
            "--vmf-variant",
            help=(
                "Hyperparameter-sweep label of the vMF Sentence LDA runs to use "
                "(e.g. kappa0-100, alpha0-0p1, zeta-40, b-16, t-5; src/core/vmf_variant.py). "
                "Unset selects the default runs and leaves every path unchanged."
            ),
        ),
        coherence: list[str] = typer.Option(["c_v"], "--coherence"),
        coherence_topn: int = typer.Option(10),
        coherence_window_size: Optional[int] = typer.Option(None),
        coherence_min_window_count: Optional[int] = typer.Option(None),
        diversity_topn: int = typer.Option(25),
        topic_word_score_mode: str = typer.Option(
            "word_topic_npmi", "--topic-word-score-mode"
        ),
        gaussian_word2vec: str = typer.Option("word2vec-google-news-300"),
        coherence_split: str = typer.Option("train"),
        coherence_min_token_len: int = typer.Option(2),
        dict_no_below: int = typer.Option(3),
        dict_no_above: float = typer.Option(0.7),
        reference_min_df: int = typer.Option(
            0,
            "--reference-min-df",
            help=(
                "Drop evaluation-dictionary words below this reference-corpus "
                "document frequency. 0 disables the restriction."
            ),
        ),
        reference_max_df_ratio: float = typer.Option(
            1.0,
            "--reference-max-df-ratio",
            help=(
                "Drop evaluation-dictionary words at or above this fraction of "
                "reference-corpus documents. 1.0 disables the restriction."
            ),
        ),
        dict_exclude_tokens: str = typer.Option(
            "", "--dict-exclude-tokens", "--dict_exclude_tokens"
        ),
        dict_exclude_single_alpha: bool = typer.Option(False),
        dict_exclude_with_digit: bool = typer.Option(False),
        dict_exclude_hiragana_only: bool = typer.Option(False),
        posterior_num_chains: int = typer.Option(1),
        posterior_burn_in_sweeps: int = typer.Option(20),
        posterior_retained_samples: int = typer.Option(20),
        posterior_thinning: int = typer.Option(1),
        posterior_seed: int = typer.Option(0),
        posterior_backend: str = typer.Option("numba"),
        etm_theta_samples: int = typer.Option(100),
        etm_posterior_seed: int = typer.Option(0),
        npmi_min_expected_count: Optional[float] = typer.Option(None),
        coherence_reference: str = typer.Option("wikipedia"),
        coherence_reference_path: Optional[Path] = typer.Option(None),
        coherence_reference_format: str = typer.Option("tokenized_jsonl"),
        coherence_reference_max_docs: Optional[int] = typer.Option(None),
        coherence_reference_min_doc_tokens: int = typer.Option(1),
        coherence_reference_streaming: bool = typer.Option(False),
        coherence_count_backend: str = typer.Option(
            "numba", "--coherence-count-backend"
        ),
        coherence_count_workers: int = typer.Option(8, "--coherence-count-workers"),
        coherence_count_chunk_size: int = typer.Option(
            25000, "--coherence-count-chunk-size"
        ),
        reference_index_mode: str = typer.Option("off", "--reference-index-mode"),
        reference_index_root: Optional[Path] = typer.Option(
            None, "--reference-index-root"
        ),
        reference_count_max_pending: Optional[int] = typer.Option(
            None, "--reference-count-max-pending"
        ),
        checkpoint_mode: str = typer.Option("auto", "--checkpoint-mode"),
        checkpoint_root: Optional[Path] = typer.Option(None, "--checkpoint-root"),
        reference_count_cache_mode: str = typer.Option(
            "auto", "--reference-count-cache-mode"
        ),
        coherence_topic_word_workers: int = typer.Option(
            1, "--coherence-topic-word-workers"
        ),
        coherence_score_workers: int = typer.Option(1, "--coherence-score-workers"),
        skip_existing: bool = typer.Option(False),
        condition_failure_policy: str = typer.Option(
            "exclude-condition", "--condition-failure-policy"
        ),
        mvtm_empty_topic_policy: str = typer.Option(
            "exclude", "--mvtm-empty-topic-policy"
        ),
        language: str = typer.Option("english"),
        delimiter: str = typer.Option(" / "),
        ja_replace_num: bool = typer.Option(True),
        ja_dicdir: Optional[str] = typer.Option(None),
        ja_require_unidic: bool = typer.Option(True),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "word_based_metrics",
            covariance_type=covariance_type,
            vmf_variant=vmf_variant,
            models=model,
            dataset=dataset,
            data_runs=data_run,
            iterations=iteration,
            num_topics=topic[0] if len(topic) == 1 else topic,
            categories=category,
            embedding_variant=embedding_variant,
            out_root=out_root,
            coherence=coherence[0] if len(coherence) == 1 else coherence,
            coherence_topn=coherence_topn,
            coherence_window_size=coherence_window_size,
            coherence_min_window_count=coherence_min_window_count,
            diversity_topn=diversity_topn,
            topic_word_score_mode=topic_word_score_mode,
            gaussian_word2vec=gaussian_word2vec,
            coherence_split=coherence_split,
            coherence_min_token_len=coherence_min_token_len,
            dict_no_below=dict_no_below,
            dict_no_above=dict_no_above,
            reference_min_df=reference_min_df,
            reference_max_df_ratio=reference_max_df_ratio,
            dict_exclude_tokens=frozenset(
                token.strip()
                for token in dict_exclude_tokens.split(",")
                if token.strip()
            ),
            dict_exclude_single_alpha=dict_exclude_single_alpha,
            dict_exclude_with_digit=dict_exclude_with_digit,
            dict_exclude_hiragana_only=dict_exclude_hiragana_only,
            posterior_num_chains=posterior_num_chains,
            posterior_burn_in_sweeps=posterior_burn_in_sweeps,
            posterior_retained_samples=posterior_retained_samples,
            posterior_thinning=posterior_thinning,
            posterior_seed=posterior_seed,
            posterior_backend=posterior_backend,
            etm_theta_samples=etm_theta_samples,
            etm_posterior_seed=etm_posterior_seed,
            npmi_min_expected_count=npmi_min_expected_count,
            coherence_reference=coherence_reference,
            coherence_reference_path=coherence_reference_path,
            coherence_reference_format=coherence_reference_format,
            coherence_reference_max_docs=coherence_reference_max_docs,
            coherence_reference_min_doc_tokens=coherence_reference_min_doc_tokens,
            coherence_reference_streaming=coherence_reference_streaming,
            coherence_count_backend=coherence_count_backend,
            coherence_count_workers=coherence_count_workers,
            coherence_count_chunk_size=coherence_count_chunk_size,
            reference_index_mode=reference_index_mode,
            reference_index_root=reference_index_root,
            reference_count_max_pending=reference_count_max_pending,
            checkpoint_mode=checkpoint_mode,
            checkpoint_root=checkpoint_root,
            reference_count_cache_mode=reference_count_cache_mode,
            coherence_topic_word_workers=coherence_topic_word_workers,
            coherence_score_workers=coherence_score_workers,
            skip_existing=skip_existing,
            condition_failure_policy=condition_failure_policy,
            mvtm_empty_topic_policy=mvtm_empty_topic_policy,
            language=language,
            delimiter=delimiter,
            ja_replace_num=ja_replace_num,
            ja_dicdir=ja_dicdir,
            ja_require_unidic=ja_require_unidic,
        )

    @evaluation_app.command(
        "topic-count-diagnostics",
        help=(
            "Write diagnostics under "
            "results/topic_count_analysis/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/topic_count_analysis/latest/.../CURRENT.json."
        ),
    )
    def topic_count_diagnostics(
        dataset: str = typer.Option(...),
        topic: list[int] = typer.Option(..., "--topic"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        category: list[str] = typer.Option(["all"], "--category"),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        condition_id: Optional[str] = typer.Option(None, "--condition-id"),
        embedding_variant: Optional[str] = typer.Option(None, "--embedding-variant"),
        num_components: Optional[int] = typer.Option(None, "--num-components"),
        split: str = typer.Option("test"),
        eval_mode: str = typer.Option("predictive-soft-theta", "--eval-mode"),
        strict: bool = typer.Option(True),
        encoder: Optional[str] = typer.Option(None),
        device: Optional[str] = typer.Option(None),
        encode_batch_size: int = typer.Option(64),
        data_column: str = typer.Option("data"),
        target_column: str = typer.Option("target_str"),
        delimiter: str = typer.Option(" / "),
        language: str = typer.Option("english"),
        segmenter: str = typer.Option("delimiter"),
        no_progress: bool = typer.Option(False),
        results_root: Path = typer.Option(Path("results/experiments"), file_okay=False),
        out_root: Path = typer.Option(
            Path("results/topic_count_analysis"), file_okay=False
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "topic_count_diagnostics",
            dataset=dataset,
            iterations=iteration,
            topics=topic,
            categories=category,
            data_runs=data_run,
            source_condition_id=condition_id,
            embedding_variant=embedding_variant,
            num_components=num_components,
            split=split,
            eval_mode=eval_mode,
            strict=strict,
            encoder_model=encoder,
            device=device,
            encode_batch_size=encode_batch_size,
            show_progress=not no_progress,
            data_column=data_column,
            target_column=target_column,
            delimiter=delimiter,
            language=language,
            segmenter=segmenter,
            results_root=results_root,
            out_root=out_root,
        )

    @evaluation_app.command(
        "cross-model-pair-diagnostics",
        help=(
            "Write pair diagnostics under "
            "results/analysis/vmf_vs_baseline/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/analysis/vmf_vs_baseline/latest/.../CURRENT.json."
        ),
    )
    def cross_model_pair_diagnostics(
        dataset: str = typer.Option(...),
        data_run: str = typer.Option("default", "--data-run"),
        category: str = typer.Option("all"),
        iteration: int = typer.Option(0),
        topic: int = typer.Option(...),
        split: str = typer.Option("train"),
        baseline: str = typer.Option("bleilda"),
        vmf_assignment: str = typer.Option(
            DEFAULT_VMF_ASSIGNMENT, help=VMF_ASSIGNMENT_HELP
        ),
        vmf_condition_id: Optional[str] = typer.Option(None, "--vmf-condition-id"),
        vmf_embedding_variant: Optional[str] = typer.Option(
            None, "--vmf-embedding-variant"
        ),
        vmf_num_components: Optional[int] = typer.Option(None, "--vmf-num-components"),
        baseline_condition_id: Optional[str] = typer.Option(
            None, "--baseline-condition-id"
        ),
        baseline_embedding_variant: Optional[str] = typer.Option(
            None, "--baseline-embedding-variant"
        ),
        baseline_num_components: Optional[int] = typer.Option(
            None, "--baseline-num-components"
        ),
        k_neighbors: int = typer.Option(30),
        baseline_max: float = typer.Option(0.05),
        vmf_min: float = typer.Option(0.6),
        topn: int = typer.Option(10),
        unique_docs: bool = typer.Option(False),
        no_row_normalize: bool = typer.Option(False),
        dump_vectors: bool = typer.Option(False),
        seed: int = typer.Option(DEFAULT_RANDOM_SEED),
        results_root: Path = typer.Option(Path("results"), file_okay=False),
        out_root: Path = typer.Option(
            Path("results/analysis/vmf_vs_baseline"), file_okay=False
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "cross_model_pair_diagnostics",
            dataset=dataset,
            data_run=data_run,
            category=category,
            iteration=iteration,
            num_topics=topic,
            split=split,
            baseline=baseline,
            k_neighbors=k_neighbors,
            baseline_max=baseline_max,
            vmf_min=vmf_min,
            topn=topn,
            unique_docs=unique_docs,
            row_normalize=not no_row_normalize,
            dump_vectors=dump_vectors,
            seed=seed,
            vmf_assignment=_validate_vmf_assignment(vmf_assignment),
            vmf_condition_id=vmf_condition_id,
            vmf_embedding_variant=vmf_embedding_variant,
            vmf_num_components=vmf_num_components,
            baseline_condition_id=baseline_condition_id,
            baseline_embedding_variant=baseline_embedding_variant,
            baseline_num_components=baseline_num_components,
            results_root=results_root,
            out_root=out_root,
        )

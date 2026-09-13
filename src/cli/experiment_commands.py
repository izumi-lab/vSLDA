from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.baselines.params import (
    format_prior_scale_variant,
    normalize_covariance_type,
)
from src.cli.options import empty_to_none, sorted_unique_ints
from src.cli.workflows import DEFAULT_ALL_EXPERIMENT_CONFIGS


def register_experiment_commands(experiments_app: typer.Typer) -> None:
    @experiments_app.command(
        "run",
        help="Run experiment training/inference from a comparison config and persist artifacts only.",
    )
    def run_experiments(
        config: Path = typer.Option(..., exists=True, dir_okay=False),
        models: Optional[str] = typer.Option(None),
        seed: Optional[int] = typer.Option(None),
        seed_base: Optional[int] = typer.Option(None),
        num_workers: Optional[int] = typer.Option(
            None, "--num-workers", "--num_workers"
        ),
        vmf_soft_temp: Optional[float] = typer.Option(
            None, "--vmf-soft-temp", "--vmf_soft_temp"
        ),
        encoder_model: Optional[str] = typer.Option(
            None, "--encoder-model", "--encoder_model"
        ),
        strip_terminal_normalize: Optional[bool] = typer.Option(
            None,
            "--strip-terminal-normalize/--keep-terminal-normalize",
            help="Override encoder.strip_terminal_normalize.",
        ),
        encoder_device: Optional[str] = typer.Option(
            None,
            "--encoder-device",
            help=(
                "Override encoder.device (cuda, cpu, or auto = cuda when available). Not part "
                "of the condition fingerprint, so a GPU-less host trains the same conditions."
            ),
        ),
        prior_scale: Optional[float] = typer.Option(
            None,
            "--prior-scale",
            min=0.0,
            help="Override Psi_0 = prior_scale * I for GaussianLDA-family baselines.",
        ),
        word2vec: Optional[str] = typer.Option(
            None,
            "--word2vec",
            help=(
                "Override the pretrained word vectors for GaussianLDA / ETM / MvTM "
                "(e.g. word2vec-google-news-300). The result-path suffix follows the "
                "name, so glove-wiki-gigaword-100 writes _glove100 and "
                "word2vec-google-news-300 writes _googlenews300."
            ),
        ),
        covariance_type: Optional[str] = typer.Option(
            None,
            "--covariance-type",
            help=(
                "Override params.covariance_type of the sentence-level Gaussian LDA "
                "(sentence_gaussianlda): full (default), diag, or spherical (alias iso). "
                "Reduced types add _cov-diag / _cov-iso to the result-path suffix."
            ),
        ),
        kappa0: Optional[float] = typer.Option(
            None,
            "--kappa0",
            min=0.0,
            help=(
                "Override train.kappa_default (initial vMF concentration kappa_0) of the "
                "vMF Sentence LDA. A value that differs from the config adds kappa0-<v> to "
                "the result-path suffix (src/core/vmf_variant.py)."
            ),
        ),
        alpha0: Optional[float] = typer.Option(
            None,
            "--alpha0",
            min=0.0,
            help="Override train.alpha (initial symmetric Dirichlet alpha_0; config default 50/K). Suffix alpha0-<v>.",
        ),
        gibbs_sweeps: Optional[int] = typer.Option(
            None,
            "--gibbs-sweeps",
            min=1,
            help="Override train.gibbs_sweeps (Gibbs sweeps per E-step, zeta). Suffix zeta-<n>.",
        ),
        num_samples: Optional[int] = typer.Option(
            None,
            "--num-samples",
            min=1,
            help="Override train.num_samples (stored sweeps B, must be <= zeta). Suffix b-<n>.",
        ),
        num_iterations: Optional[int] = typer.Option(
            None,
            "--num-iterations",
            min=1,
            help="Override train.num_iterations (MCEM iterations T). Suffix t-<n>.",
        ),
        saem_burn_in: Optional[int] = typer.Option(
            None,
            "--saem-burn-in",
            min=0,
            help=(
                "Override train.saem_burn_in (SAEM burn-in T_0: outer iterations before the "
                "statistics are averaged and alpha is frozen; T_0 = T disables the averaging). "
                "Suffix t0-<n>."
            ),
        ),
        saem_decay: Optional[float] = typer.Option(
            None,
            "--saem-decay",
            help="Override train.saem_decay (SAEM step-size exponent a in (1/2, 1]). Suffix a-<v>.",
        ),
        category: list[str] = typer.Option([], "--category"),
        topic: list[int] = typer.Option([], "--topic"),
        iteration: list[int] = typer.Option([], "--iteration"),
    ) -> None:
        from src.cli.workflows import run_experiments_workflow

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
        run_experiments_workflow(
            config=config,
            models=models,
            seed=seed,
            seed_base=seed_base,
            num_workers=num_workers,
            vmf_soft_temp=vmf_soft_temp,
            categories=empty_to_none(category),
            topics=empty_to_none(topic),
            iterations=empty_to_none(iteration),
            encoder_model=encoder_model,
            strip_terminal_normalize=strip_terminal_normalize,
            encoder_device=encoder_device,
            prior_scale=prior_scale,
            word2vec=word2vec,
            covariance_type=covariance_type,
            kappa0=kappa0,
            alpha0=alpha0,
            gibbs_sweeps=gibbs_sweeps,
            num_samples=num_samples,
            num_iterations=num_iterations,
            saem_burn_in=saem_burn_in,
            saem_decay=saem_decay,
        )

    @experiments_app.command(
        "smoke",
        help="Run a smoke-sized experiment config and persist artifacts only.",
    )
    def run_smoke_experiments(
        config: Path = typer.Option(..., exists=True, dir_okay=False),
        models: Optional[str] = typer.Option("vmf_sentence_lda"),
        seed: int = typer.Option(42),
        num_workers: Optional[int] = typer.Option(
            None, "--num-workers", "--num_workers"
        ),
        category: list[str] = typer.Option([], "--category"),
        topic: list[int] = typer.Option([], "--topic"),
        iteration: list[int] = typer.Option([], "--iteration"),
    ) -> None:
        from src.cli.workflows import run_smoke_workflow

        run_smoke_workflow(
            config=config,
            models=models,
            seed=seed,
            num_workers=num_workers,
            category=category,
            topic=topic,
            iteration=iteration,
        )

    @experiments_app.command(
        "run-all",
        help="Run canonical experiment presets and optional category=all overrides without evaluation.",
    )
    def run_all(
        config: list[Path] = typer.Option(
            [Path(p) for p in DEFAULT_ALL_EXPERIMENT_CONFIGS],
            "--config",
        ),
        models: str = typer.Option(
            "vmf_sentence_lda,ctm,bleilda,sam,sam_tf,gaussianlda,etm,mvtm,senclu,"
            "sentence_gaussianlda,sentlda,spherical_kmeans,gaussian_kmeans,"
            "movmf,gaussian_mixture"
        ),
        seed_base: Optional[int] = typer.Option(None),
        num_workers: Optional[int] = typer.Option(
            None, "--num-workers", "--num_workers"
        ),
        vmf_soft_temp: Optional[float] = typer.Option(
            None, "--vmf-soft-temp", "--vmf_soft_temp"
        ),
        include_all_category_runs: bool = typer.Option(True),
        all_category_topic: list[int] = typer.Option([50], "--all-category-topic"),
        all_category_iteration: list[int] = typer.Option(
            [0, 1, 2, 3, 4], "--all-category-iteration"
        ),
    ) -> None:
        from src.cli.workflows import run_all_experiments_workflow

        run_all_experiments_workflow(
            configs=config,
            models=models,
            seed_base=seed_base,
            num_workers=num_workers,
            vmf_soft_temp=vmf_soft_temp,
            include_all_category_runs=include_all_category_runs,
            all_category_topics=all_category_topic,
            all_category_iterations=all_category_iteration,
        )

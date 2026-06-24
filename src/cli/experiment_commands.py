from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.cli.options import empty_to_none
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
        category: list[str] = typer.Option([], "--category"),
        topic: list[int] = typer.Option([], "--topic"),
        iteration: list[int] = typer.Option([], "--iteration"),
    ) -> None:
        from src.cli.workflows import run_experiments_workflow

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
            "vmf_sentence_lda,ctm,bleilda,gaussianlda,etm,mvtm,senclu,"
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

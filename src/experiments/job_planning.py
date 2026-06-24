from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from src.data.datasets import (
    DatasetRun,
    resolve_dataset_categories,
    resolve_dataset_runs,
)
from src.experiments.config import BaselineConfig, ComparisonConfig


@dataclass(frozen=True)
class CategoryJob:
    data_run_name: str
    train_csvs: Tuple[Path, ...]
    test_csvs: Tuple[Path, ...]
    fiscal_years: Tuple[int, ...] | None
    category: str
    targets: Sequence[str] | None
    num_topics: int
    iteration: int
    baselines: List[BaselineConfig]
    selected_models: set[str] | None
    seed: int | None
    seed_base: int | None
    parallelism: "ParallelismPlan"
    config: ComparisonConfig
    vmf_soft_temp: float


@dataclass(frozen=True)
class ParallelismPlan:
    requested_num_workers: int
    category_num_workers: int
    baseline_num_workers: int
    encoder_device: str
    run_vmf: bool
    uses_cuda: bool
    reason: str | None = None


def resolve_algorithm_variant(
    *,
    num_components: int,
    estimate_alpha: bool,
    alpha_update_every: int,
) -> str:
    parts = [f"components_{int(num_components)}"]
    if not estimate_alpha:
        parts.append("fixed_alpha")
    else:
        parts.append(f"estimate_alpha_every_{int(alpha_update_every)}")
    return "__".join(parts)


def resolve_effective_num_workers(
    *,
    requested_num_workers: int,
    encoder_device: str,
    selected_models: set[str] | None,
) -> int:
    return resolve_parallelism_plan(
        requested_num_workers=requested_num_workers,
        encoder_device=encoder_device,
        selected_models=selected_models,
    ).category_num_workers


def resolve_parallelism_plan(
    *,
    requested_num_workers: int,
    encoder_device: str,
    selected_models: set[str] | None,
) -> ParallelismPlan:
    if requested_num_workers < 1:
        raise ValueError("num_workers must be >= 1.")
    run_vmf = selected_models is None or "vmf_sentence_lda" in selected_models
    uses_cuda = run_vmf and str(encoder_device).lower().startswith("cuda")
    return ParallelismPlan(
        requested_num_workers=requested_num_workers,
        category_num_workers=requested_num_workers,
        baseline_num_workers=1,
        encoder_device=encoder_device,
        run_vmf=run_vmf,
        uses_cuda=uses_cuda,
    )


def build_jobs(
    cfg: ComparisonConfig,
    data_runs: Sequence[DatasetRun],
    iterations: Sequence[int],
    num_topics_list: Sequence[int],
    categories: Mapping[str, Sequence[str] | None],
    selected_models: set[str] | None,
    seed: int | None,
    seed_base: int | None,
    parallelism: ParallelismPlan,
    vmf_soft_temp: float,
) -> List[CategoryJob]:
    jobs: List[CategoryJob] = []
    for data_run in data_runs:
        for iteration in iterations:
            for num_topics in num_topics_list:
                for category, targets in categories.items():
                    jobs.append(
                        CategoryJob(
                            data_run_name=data_run.name,
                            train_csvs=data_run.train_csvs,
                            test_csvs=data_run.test_csvs,
                            fiscal_years=data_run.fiscal_years,
                            category=category,
                            targets=targets,
                            num_topics=num_topics,
                            iteration=iteration,
                            baselines=cfg.baselines,
                            selected_models=selected_models,
                            seed=seed,
                            seed_base=seed_base,
                            parallelism=parallelism,
                            config=cfg,
                            vmf_soft_temp=vmf_soft_temp,
                        )
                    )
    return jobs


def resolve_data_runs(cfg: ComparisonConfig) -> List[DatasetRun]:
    return resolve_dataset_runs(cfg)


def resolve_categories(
    cfg: ComparisonConfig, categories: Sequence[str] | None
) -> Dict[str, Sequence[str] | None]:
    return dict(resolve_dataset_categories(cfg, categories))

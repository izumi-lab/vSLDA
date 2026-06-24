from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.config import (
    ComparisonConfig,
    DatasetConfig,
    EncoderConfig,
    ExperimentConfig,
    TrainConfig,
)
from src.experiments.job_planning import (
    resolve_algorithm_variant,
    resolve_categories,
    resolve_effective_num_workers,
    resolve_parallelism_plan,
)


def _build_cfg() -> ComparisonConfig:
    dataset = DatasetConfig(
        name="dummy",
        train_csv=Path("train.csv"),
        test_csv=Path("test.csv"),
        categories={"science": ["sci.space"], "sports": ["rec.sport.baseball"]},
    )
    train = TrainConfig(num_topics=[10], num_iterations=3, alpha=None)
    encoder = EncoderConfig()
    experiments = ExperimentConfig(iterations=[0])
    return ComparisonConfig(
        dataset=dataset,
        train=train,
        encoder=encoder,
        experiments=experiments,
        baselines=[],
        output_root=Path("results/experiments/dummy"),
    )


def test_resolve_categories_uses_config_when_unset() -> None:
    cfg = _build_cfg()
    resolved = resolve_categories(cfg, None)
    assert resolved == cfg.dataset.categories


def test_resolve_categories_rejects_unknown_category() -> None:
    cfg = _build_cfg()
    with pytest.raises(ValueError):
        resolve_categories(cfg, ["missing"])


def test_resolve_algorithm_variant() -> None:
    assert resolve_algorithm_variant(
        num_components=1,
        estimate_alpha=False,
        alpha_update_every=1,
    ) == ("components_1__fixed_alpha")
    assert resolve_algorithm_variant(
        num_components=3,
        estimate_alpha=True,
        alpha_update_every=3,
    ) == ("components_3__estimate_alpha_every_3")


def test_resolve_effective_num_workers_keeps_requested_value_for_cuda_vmf() -> None:
    resolved = resolve_effective_num_workers(
        requested_num_workers=8,
        encoder_device="cuda",
        selected_models=None,
    )
    assert resolved == 8


def test_resolve_parallelism_plan_keeps_num_workers_semantics_consistent() -> None:
    plan = resolve_parallelism_plan(
        requested_num_workers=4,
        encoder_device="cpu",
        selected_models={"ctm"},
    )

    assert plan.requested_num_workers == 4
    assert plan.category_num_workers == 4
    assert plan.baseline_num_workers == 1
    assert plan.run_vmf is False
    assert plan.uses_cuda is False

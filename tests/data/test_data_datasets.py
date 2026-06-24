from __future__ import annotations

from pathlib import Path

import pytest

from src.core.errors import MissingDatasetError
from src.data.datasets import resolve_dataset_runs
from src.experiments.config import (
    ComparisonConfig,
    DatasetConfig,
    EncoderConfig,
    ExperimentConfig,
    TrainConfig,
)


def _build_cfg(tmp_path: Path, *, fiscal_year_mode: str) -> ComparisonConfig:
    by_fy_root = tmp_path / "by_fy"
    for fiscal_year in (2022, 2023):
        fiscal_year_dir = by_fy_root / f"fy{fiscal_year}"
        fiscal_year_dir.mkdir(parents=True)
        (fiscal_year_dir / "train.csv").write_text("data\ntrain\n", encoding="utf-8")
        (fiscal_year_dir / "test.csv").write_text("data\ntest\n", encoding="utf-8")

    dataset = DatasetConfig(
        name="yearly_dataset",
        train_csv=by_fy_root / "fy2022" / "train.csv",
        test_csv=by_fy_root / "fy2022" / "test.csv",
        categories={"all": None},
        by_fy_root=by_fy_root,
        fiscal_years=[2022, 2023, 2022],
        fiscal_year_mode=fiscal_year_mode,
    )
    return ComparisonConfig(
        dataset=dataset,
        train=TrainConfig(num_topics=[10], num_iterations=3, alpha=None),
        encoder=EncoderConfig(),
        experiments=ExperimentConfig(iterations=[0]),
        baselines=[],
        output_root=tmp_path / "results",
    )


def test_resolve_dataset_runs_concat_years_deduplicates_years(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path, fiscal_year_mode="concat_years")

    runs = resolve_dataset_runs(cfg)

    assert len(runs) == 1
    assert runs[0].name == "fy2022_2023"
    assert runs[0].fiscal_years == (2022, 2023)
    assert len(runs[0].train_csvs) == 2


def test_resolve_dataset_runs_per_year_builds_individual_runs(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path, fiscal_year_mode="per_year")

    runs = resolve_dataset_runs(cfg)

    assert [run.name for run in runs] == ["fy2022", "fy2023"]
    assert runs[0].fiscal_years == (2022,)
    assert runs[1].fiscal_years == (2023,)


def test_resolve_dataset_runs_requires_split_files(tmp_path: Path) -> None:
    dataset = DatasetConfig(
        name="yearly_dataset",
        train_csv=tmp_path / "train.csv",
        test_csv=tmp_path / "test.csv",
        categories={"all": None},
        by_fy_root=tmp_path / "missing",
        fiscal_years=[2024],
        fiscal_year_mode="per_year",
    )
    cfg = ComparisonConfig(
        dataset=dataset,
        train=TrainConfig(num_topics=[10], num_iterations=3, alpha=None),
        encoder=EncoderConfig(),
        experiments=ExperimentConfig(iterations=[0]),
        baselines=[],
        output_root=tmp_path / "results",
    )

    with pytest.raises(MissingDatasetError) as exc_info:
        resolve_dataset_runs(cfg)
    assert "fiscal_year=2024" in str(exc_info.value)

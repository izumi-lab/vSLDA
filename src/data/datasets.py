from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from src.core.errors import require_dataset_path

if TYPE_CHECKING:
    from src.experiments.config import ComparisonConfig


@dataclass(frozen=True)
class DatasetRun:
    name: str
    train_csvs: tuple[Path, ...]
    test_csvs: tuple[Path, ...]
    fiscal_years: tuple[int, ...] | None = None


def resolve_dataset_runs(cfg: ComparisonConfig) -> list[DatasetRun]:
    fiscal_years = cfg.dataset.fiscal_years
    if not fiscal_years:
        return [
            DatasetRun(
                name="default",
                train_csvs=(cfg.dataset.train_csv,),
                test_csvs=(cfg.dataset.test_csv,),
                fiscal_years=None,
            )
        ]

    if cfg.dataset.by_fy_root is None:
        raise ValueError(
            "dataset.by_fy_root must be set when dataset.fiscal_years is configured."
        )

    normalized_years: list[int] = []
    seen_years: set[int] = set()
    for year in fiscal_years:
        fiscal_year = int(year)
        if fiscal_year in seen_years:
            continue
        seen_years.add(fiscal_year)
        normalized_years.append(fiscal_year)

    year_to_paths: dict[int, tuple[Path, Path]] = {}
    for year in normalized_years:
        fiscal_year_dir = cfg.dataset.by_fy_root / f"fy{year}"
        train_csv = fiscal_year_dir / "train.csv"
        test_csv = fiscal_year_dir / "test.csv"
        require_dataset_path(
            train_csv,
            detail=(
                f"Fiscal-year train split was not found for dataset='{cfg.dataset.name}', "
                f"fiscal_year={year}."
            ),
        )
        require_dataset_path(
            test_csv,
            detail=(
                f"Fiscal-year test split was not found for dataset='{cfg.dataset.name}', "
                f"fiscal_year={year}."
            ),
        )
        year_to_paths[year] = (train_csv, test_csv)

    if cfg.dataset.fiscal_year_mode == "per_year":
        return [
            DatasetRun(
                name=f"fy{year}",
                train_csvs=(year_to_paths[year][0],),
                test_csvs=(year_to_paths[year][1],),
                fiscal_years=(year,),
            )
            for year in normalized_years
        ]

    if cfg.dataset.fiscal_year_mode == "concat_years":
        return [
            DatasetRun(
                name="fy" + "_".join(str(year) for year in normalized_years),
                train_csvs=tuple(year_to_paths[year][0] for year in normalized_years),
                test_csvs=tuple(year_to_paths[year][1] for year in normalized_years),
                fiscal_years=tuple(normalized_years),
            )
        ]

    raise ValueError(
        "dataset.fiscal_year_mode must be either 'per_year' or 'concat_years'."
    )


def resolve_dataset_categories(
    cfg: ComparisonConfig,
    categories: list[str] | tuple[str, ...] | None,
) -> Dict[str, list[str] | tuple[str, ...] | None]:
    if categories is None:
        return dict(cfg.dataset.categories)

    selected: Dict[str, list[str] | tuple[str, ...] | None] = {}
    for raw_name in categories:
        name = str(raw_name).strip()
        if not name:
            continue
        if name == "all":
            selected["all"] = None
            continue
        if name not in cfg.dataset.categories:
            available = sorted(cfg.dataset.categories.keys())
            raise ValueError(
                f"Unknown category '{name}'. Available categories: {available} and 'all'."
            )
        selected[name] = cfg.dataset.categories[name]

    if not selected:
        raise ValueError("At least one category must be selected.")
    return selected

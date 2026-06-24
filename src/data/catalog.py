from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.core.paths import DATA_ROOT


def _with_all(
    base: dict[str, list[str]],
    exclude: Optional[set[str]] = None,
) -> dict[str, list[str]]:
    out = dict(base)
    excluded = exclude or set()
    out["all"] = sorted(
        {label for labels in base.values() for label in labels if label not in excluded}
    )
    return out


DATASET_TARGETS: dict[str, dict[str, list[str]]] = {
    "20newsgroup": _with_all(
        {
            "computer": [
                "comp.graphics",
                "comp.os.ms-windows.misc",
                "comp.sys.ibm.pc.hardware",
                "comp.sys.mac.hardware",
                "comp.windows.x",
            ],
            "ride": ["rec.autos", "rec.motorcycles"],
            "sports": ["rec.sport.baseball", "rec.sport.hockey"],
            "science": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
            "religion": ["alt.atheism", "soc.religion.christian", "talk.religion.misc"],
            "politics": [
                "talk.politics.guns",
                "talk.politics.mideast",
                "talk.politics.misc",
            ],
        }
    ),
    "nyt": _with_all(
        {
            "arts": ["dance", "music", "movies", "television"],
            "business": [
                "economy",
                "energy_companies",
                "international_business",
                "stocks_and_bonds",
            ],
            "politics": [
                "abortion",
                "federal_budget",
                "gay_rights",
                "gun_control",
                "immigration",
                "law_enforcement",
                "military",
                "surveillance",
                "the_affordable_care_act",
            ],
            "sports": [
                "baseball",
                "basketball",
                "football",
                "golf",
                "hockey",
                "soccer",
                "tennis",
            ],
        }
    ),
}
DATASET_ALIASES: dict[str, str] = {}


def register_dataset_targets(
    name: str,
    targets: dict[str, list[str]],
    *,
    aliases: Sequence[str] = (),
) -> None:
    """Register builtin category targets for a dataset and its aliases."""
    canonical_name = str(name).strip()
    if not canonical_name:
        raise ValueError("dataset name must not be empty.")
    DATASET_TARGETS[canonical_name] = targets
    for alias in aliases:
        register_dataset_alias(alias, canonical_name)


def register_dataset_alias(alias: str, canonical_name: str) -> None:
    """Register a dataset alias that resolves to an existing canonical dataset."""
    normalized_alias = str(alias).strip()
    normalized_name = str(canonical_name).strip()
    if not normalized_alias:
        raise ValueError("dataset alias must not be empty.")
    if normalized_name not in DATASET_TARGETS:
        raise KeyError(
            f"Cannot register alias '{normalized_alias}' for unknown dataset '{normalized_name}'."
        )
    DATASET_TARGETS[normalized_alias] = DATASET_TARGETS[normalized_name]
    DATASET_ALIASES[normalized_alias] = normalized_name


def candidate_dataset_dirs(name: str) -> list[Path]:
    return [
        DATA_ROOT / name,
    ]


def resolve_dataset_dir(name: str) -> Path | None:
    for dataset_dir in candidate_dataset_dirs(name):
        if (dataset_dir / "train.csv").exists():
            return dataset_dir
    return None


def resolve_dataset_name(name: str) -> str | None:
    if name in DATASET_TARGETS:
        return name
    if name in DATASET_ALIASES:
        return name
    if resolve_dataset_dir(name) is not None:
        return name
    return None


@lru_cache(maxsize=None)
def get_dataset_targets(
    dataset: str,
    *,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> dict[str, list[str]] | None:
    resolved_name = resolve_dataset_name(dataset) or dataset
    if resolved_name in DATASET_TARGETS:
        return DATASET_TARGETS[resolved_name]

    dataset_dir = resolve_dataset_dir(resolved_name)
    if dataset_dir is None:
        return None

    train_csv = dataset_dir / "train.csv"
    try:
        frame = pd.read_csv(train_csv, usecols=[target_column])
    except Exception:
        return None

    labels = sorted(
        {
            str(value).strip()
            for value in frame[target_column].dropna().tolist()
            if str(value).strip()
        }
    )
    if not labels:
        return None
    return {"all": labels}


def has_builtin_category_mapping(dataset: str) -> bool:
    resolved_name = resolve_dataset_name(dataset) or dataset
    return resolved_name in DATASET_TARGETS


def infer_all_targets_from_csv(
    train_csv: str | Path,
    *,
    target_column: str = "target_str",
) -> list[str]:
    frame = pd.read_csv(train_csv)
    if target_column not in frame.columns:
        raise ValueError(f"target_column '{target_column}' not found in {train_csv}")
    labels = sorted(
        {
            str(value).strip()
            for value in frame[target_column].dropna().tolist()
            if str(value).strip()
        }
    )
    return labels


def resolve_category_targets(
    dataset: str,
    category: str,
    explicit_targets: Sequence[str] | None,
    *,
    target_column: str = "target_str",
    label_schema: str = "identity",
    train_csv: str | Path | None = None,
    has_labels: bool = True,
    allow_all_unfiltered: bool = False,
) -> list[str] | None:
    if not has_labels:
        return None
    if explicit_targets is not None:
        return [str(target) for target in explicit_targets]
    if category == "all" and allow_all_unfiltered:
        return None

    labels_by_category = get_dataset_targets(
        dataset,
        target_column=target_column,
        label_schema=label_schema,
    )
    if labels_by_category is not None and category in labels_by_category:
        return list(labels_by_category[category])

    if category == "all":
        if train_csv is None:
            raise KeyError(
                f"No target mapping for dataset='{dataset}', category='{category}'."
            )
        labels = infer_all_targets_from_csv(train_csv, target_column=target_column)
        if dataset.startswith("20newsgroup"):
            labels = [label for label in labels if label != "misc.forsale"]
        return labels

    raise KeyError(
        f"No target mapping for dataset='{dataset}', category='{category}'. "
        "Pass explicit targets for non-legacy datasets."
    )

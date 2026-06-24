from __future__ import annotations

from pathlib import Path

import src.data.catalog as catalog
from src.data.catalog import (
    get_dataset_targets,
    has_builtin_category_mapping,
    register_dataset_alias,
    register_dataset_targets,
    resolve_category_targets,
    resolve_dataset_dir,
    resolve_dataset_name,
)


def test_resolve_dataset_dir_finds_builtin_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data"
    dataset_root = data_root / "20newsgroup"
    dataset_root.mkdir(parents=True)
    (dataset_root / "train.csv").write_text(
        "data,target_str\nx,sci.space\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(catalog, "DATA_ROOT", data_root)

    dataset_dir = resolve_dataset_dir("20newsgroup")
    assert dataset_dir is not None
    assert dataset_dir == dataset_root or dataset_dir.name == "20newsgroup"


def test_get_dataset_targets_returns_builtin_categories() -> None:
    targets = get_dataset_targets("20newsgroup")
    assert targets is not None
    assert "computer" in targets
    assert "all" in targets


def test_get_dataset_targets_excludes_nyt_science() -> None:
    targets = get_dataset_targets("nyt")

    assert targets is not None
    assert "science" not in targets
    assert "cosmos" not in targets["all"]
    assert "environment" not in targets["all"]


def test_has_builtin_category_mapping_accepts_builtin_dataset() -> None:
    assert has_builtin_category_mapping("20newsgroup") is True


def test_resolve_category_targets_returns_none_for_unfiltered_all() -> None:
    assert (
        resolve_category_targets(
            "custom_dataset",
            "all",
            None,
            allow_all_unfiltered=True,
        )
        is None
    )


def test_resolve_category_targets_uses_builtin_mapping_for_named_category() -> None:
    targets = resolve_category_targets("20newsgroup", "science", None)
    assert targets is not None
    assert "sci.space" in targets


def test_register_dataset_targets_adds_builtin_mapping() -> None:
    register_dataset_targets(
        "dummy_dataset",
        {"all": ["a", "b"], "subset": ["a"]},
    )

    targets = get_dataset_targets("dummy_dataset")

    assert targets is not None
    assert targets["subset"] == ["a"]


def test_register_dataset_alias_reuses_canonical_targets() -> None:
    register_dataset_targets(
        "dummy_dataset_alias_base",
        {"all": ["a", "b"]},
    )
    register_dataset_alias("dummy_dataset_alias", "dummy_dataset_alias_base")

    assert resolve_dataset_name("dummy_dataset_alias") == "dummy_dataset_alias"
    assert get_dataset_targets("dummy_dataset_alias") == {"all": ["a", "b"]}

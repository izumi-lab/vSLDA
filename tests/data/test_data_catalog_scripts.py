from __future__ import annotations

from src.data.catalog import get_dataset_targets


def test_get_dataset_targets_returns_all_category_for_builtin_dataset() -> None:
    targets = get_dataset_targets("nyt")
    assert targets is not None
    assert "all" in targets
    assert "sports" in targets
    assert "science" not in targets

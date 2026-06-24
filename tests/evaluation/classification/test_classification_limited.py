from __future__ import annotations

import logging

from src.evaluation.classification.limited import _sample_indices


def test_sample_indices_with_full_ratio_returns_all_rows() -> None:
    indices, counts = _sample_indices(
        ["a", "a", "b"],
        train_ratio=1.0,
        train_count=None,
        stratified=True,
        seed=42,
    )
    assert indices == [0, 1, 2]
    assert counts == {"a": 2, "b": 1}


def test_sample_indices_with_zero_count_returns_empty_selection() -> None:
    indices, counts = _sample_indices(
        ["a", "b"],
        train_ratio=None,
        train_count=0,
        stratified=False,
        seed=42,
    )
    assert indices == []
    assert counts == {}


def test_sample_indices_logs_when_ratio_is_adjusted(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        indices, counts = _sample_indices(
            ["a", "a", "b", "b", "c", "c"],
            train_ratio=0.2,
            train_count=None,
            stratified=True,
            seed=42,
        )

    assert len(indices) == 3
    assert counts == {"a": 1, "b": 1, "c": 1}
    assert "adjusting to" in caplog.text

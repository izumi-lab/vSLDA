from __future__ import annotations

import pytest

from src.evaluation.classification.limited import (
    _resolve_sampling_repeats,
    _sampling_seed,
)


def test_sampling_seed_preserves_legacy_iteration_seed_for_missing_repeat() -> None:
    assert (
        _sampling_seed(
            42,
            2,
            sampling_repeat=None,
            sampling_seed_stride=1000,
        )
        == 44
    )
    assert (
        _sampling_seed(
            42,
            2,
            sampling_repeat=0,
            sampling_seed_stride=1000,
        )
        == 44
    )
    assert (
        _sampling_seed(
            42,
            2,
            sampling_repeat=3,
            sampling_seed_stride=1000,
        )
        == 3044
    )


def test_resolve_sampling_repeats_uses_none_for_legacy_single_run() -> None:
    assert _resolve_sampling_repeats(None) == [None]
    assert _resolve_sampling_repeats([]) == [None]
    assert _resolve_sampling_repeats([0, 1, 4]) == [0, 1, 4]


def test_resolve_sampling_repeats_rejects_duplicate_or_negative_values() -> None:
    with pytest.raises(ValueError, match="duplicate sampling_repeat"):
        _resolve_sampling_repeats([0, 0])

    with pytest.raises(ValueError, match="non-negative"):
        _resolve_sampling_repeats([-1])


def test_sampling_seed_rejects_invalid_stride() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _sampling_seed(
            42,
            0,
            sampling_repeat=0,
            sampling_seed_stride=0,
        )

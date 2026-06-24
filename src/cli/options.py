from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def empty_to_none(values: Sequence[T]) -> list[T] | None:
    normalized = list(values)
    return normalized or None


def sorted_unique_ints(values: Sequence[int]) -> list[int]:
    return sorted({int(value) for value in values})

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class ProgressReporter(Protocol):
    def wrap(
        self,
        iterable: Iterable[T],
        total: int | None = None,
        desc: str | None = None,
        **kwargs: Any,
    ) -> Iterable[T]: ...


class NullProgressReporter:
    def wrap(
        self,
        iterable: Iterable[T],
        total: int | None = None,
        desc: str | None = None,
        **kwargs: Any,
    ) -> Iterable[T]:
        _ = (total, desc, kwargs)
        return iterable


class TqdmProgressReporter:
    def wrap(
        self,
        iterable: Iterable[T],
        total: int | None = None,
        desc: str | None = None,
        **kwargs: Any,
    ) -> Iterable[T]:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, **kwargs)

from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from src.core.artifacts import load_text_lines

T = TypeVar("T")

_PROC_STATUS_PATH = Path("/proc/self/status")


@dataclass(frozen=True)
class MemorySnapshot:
    rss_mb: float | None
    peak_rss_mb: float | None


@dataclass(frozen=True)
class RuntimeMeasurement:
    elapsed_sec: float
    rss_mb_before: float | None
    rss_mb_after: float | None
    peak_rss_mb_before: float | None
    peak_rss_mb_after: float | None

    @property
    def peak_rss_mb_delta(self) -> float | None:
        if self.peak_rss_mb_before is None or self.peak_rss_mb_after is None:
            return None
        return max(0.0, self.peak_rss_mb_after - self.peak_rss_mb_before)


def _parse_proc_status_mb(field_name: str) -> float | None:
    if not _PROC_STATUS_PATH.exists():
        return None
    try:
        for line in load_text_lines(_PROC_STATUS_PATH):
            if not line.startswith(field_name):
                continue
            parts = line.split()
            if len(parts) < 2:
                return None
            return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def _peak_rss_from_resource_mb() -> float | None:
    try:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ValueError):
        return None

    # Linux reports KiB, macOS reports bytes.
    if peak <= 0.0:
        return None
    if peak > 1024.0 * 1024.0 * 16.0:
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def capture_memory_snapshot() -> MemorySnapshot:
    rss_mb = _parse_proc_status_mb("VmRSS:")
    peak_rss_mb = _parse_proc_status_mb("VmHWM:")
    if peak_rss_mb is None:
        peak_rss_mb = _peak_rss_from_resource_mb()
    return MemorySnapshot(
        rss_mb=rss_mb,
        peak_rss_mb=peak_rss_mb,
    )


def measure_runtime(callable_obj: Callable[[], T]) -> tuple[T, RuntimeMeasurement]:
    before = capture_memory_snapshot()
    start = time.perf_counter()
    result = callable_obj()
    elapsed_sec = time.perf_counter() - start
    after = capture_memory_snapshot()
    return result, RuntimeMeasurement(
        elapsed_sec=float(elapsed_sec),
        rss_mb_before=before.rss_mb,
        rss_mb_after=after.rss_mb,
        peak_rss_mb_before=before.peak_rss_mb,
        peak_rss_mb_after=after.peak_rss_mb,
    )

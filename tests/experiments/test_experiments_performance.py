from __future__ import annotations

from src.experiments.performance import MemorySnapshot, measure_runtime


def test_measure_runtime_returns_result_and_memory_delta(monkeypatch) -> None:
    snapshots = iter(
        [
            MemorySnapshot(rss_mb=10.0, peak_rss_mb=20.0),
            MemorySnapshot(rss_mb=12.0, peak_rss_mb=27.5),
        ]
    )

    monkeypatch.setattr(
        "src.experiments.performance.capture_memory_snapshot",
        lambda: next(snapshots),
    )

    result, measurement = measure_runtime(lambda: "ok")

    assert result == "ok"
    assert measurement.elapsed_sec >= 0.0
    assert measurement.rss_mb_before == 10.0
    assert measurement.rss_mb_after == 12.0
    assert measurement.peak_rss_mb_before == 20.0
    assert measurement.peak_rss_mb_after == 27.5
    assert measurement.peak_rss_mb_delta == 7.5

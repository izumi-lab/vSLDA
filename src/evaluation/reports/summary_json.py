from __future__ import annotations

from typing import Any


def build_metric_summary(
    *,
    metric_name: str,
    value: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "value": float(value),
        "metadata": {} if metadata is None else dict(metadata),
    }

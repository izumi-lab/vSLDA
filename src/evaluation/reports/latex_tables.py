from __future__ import annotations


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"

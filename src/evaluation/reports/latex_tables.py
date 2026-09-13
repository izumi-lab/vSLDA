"""Shared numeric and LaTeX formatting helpers for evaluation summaries.

Every summary writer (classification and word-based) computes mean/standard
deviation and marks the best/second-best cell here, so the rules exist once.
Downstream consumers (the paper repository) aggregate from the ``*.scores.json``
sidecars instead, and this module keeps the review tables consistent with them.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "format_mean_std",
    "format_pm",
    "latex_escape_text",
    "mean_std",
    "rank_and_mark",
]


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and population standard deviation (ddof=0) of ``values``."""
    if not values:
        raise ValueError("mean_std() requires at least one value")
    numbers = [float(value) for value in values]
    mean = sum(numbers) / len(numbers)
    variance = sum((number - mean) ** 2 for number in numbers) / len(numbers)
    return mean, math.sqrt(variance)


def format_pm(mean: float, std: float, digits: int = 3) -> str:
    """``mean ± std`` as the LaTeX fragment used by the review tables."""
    return f"{mean:.{digits}f}" r"~\ensuremath{\pm}~" f"{std:.{digits}f}"


def rank_and_mark(
    cells: Mapping[str, str],
    means: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, str]:
    """Bold the best cell and underline the second-best.

    Ranking uses the unrounded means and shares a rank between ties, so two
    equal values are both marked best and no cell is marked second. This is the
    same rule the paper applies when it re-aggregates the raw per-run values.
    """
    marked = dict(cells)
    finite = {model: value for model, value in means.items() if math.isfinite(value)}
    if not finite:
        return marked
    ordered = sorted(finite.values(), reverse=True)
    best = ordered[0]
    second = next(
        (
            value
            for value in ordered
            if not math.isclose(value, best, abs_tol=tolerance)
        ),
        None,
    )
    for model, value in finite.items():
        if model not in marked:
            continue
        if math.isclose(value, best, abs_tol=tolerance):
            marked[model] = r"\textbf{" + marked[model] + "}"
        elif second is not None and math.isclose(value, second, abs_tol=tolerance):
            marked[model] = r"\underline{" + marked[model] + "}"
    return marked


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape_text(value: Any) -> str:
    return "".join(_LATEX_ESCAPES.get(char, char) for char in str(value))

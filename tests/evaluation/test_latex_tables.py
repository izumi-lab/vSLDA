from __future__ import annotations

import pytest

from src.evaluation.reports.latex_tables import (
    format_pm,
    latex_escape_text,
    mean_std,
    rank_and_mark,
)


def test_mean_std_uses_population_standard_deviation() -> None:
    mean, std = mean_std([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    assert std == pytest.approx(1.118033988749895)


def test_mean_std_of_single_value_has_zero_dispersion() -> None:
    assert mean_std([42.0]) == (42.0, 0.0)


def test_mean_std_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        mean_std([])


def test_format_pm_renders_the_latex_fragment() -> None:
    assert format_pm(1.2345, 0.5, digits=2) == r"1.23~\ensuremath{\pm}~0.50"


def test_rank_and_mark_bolds_best_and_underlines_second() -> None:
    cells = {"a": "1.00", "b": "3.00", "c": "2.00"}
    means = {"a": 1.0, "b": 3.0, "c": 2.0}
    marked = rank_and_mark(cells, means)
    assert marked == {
        "a": "1.00",
        "b": r"\textbf{3.00}",
        "c": r"\underline{2.00}",
    }
    assert cells["b"] == "3.00"  # the input mapping is not mutated


def test_rank_and_mark_ignores_non_finite_means() -> None:
    # A NaN mean (an empty topic-word list, say) must not take the best rank
    # away from a real value; sorted() alone puts it wherever it happens to sit.
    cells = {"a": "0.31", "b": "nan", "c": "0.28"}
    means = {"a": 0.31, "b": float("nan"), "c": 0.28}
    marked = rank_and_mark(cells, means)
    assert marked == {
        "a": r"\textbf{0.31}",
        "b": "nan",
        "c": r"\underline{0.28}",
    }


def test_rank_and_mark_shares_a_rank_between_ties() -> None:
    # Tied bests are both bold, and the underline goes to the next distinct
    # value rather than being consumed by the tie.
    means = {"a": 3.0, "b": 3.0, "c": 1.0}
    marked = rank_and_mark({model: "x" for model in means}, means)
    assert marked["a"] == r"\textbf{x}"
    assert marked["b"] == r"\textbf{x}"
    assert marked["c"] == r"\underline{x}"


def test_rank_and_mark_ranks_on_unrounded_means() -> None:
    means = {"a": 1.004, "b": 1.001}
    marked = rank_and_mark({"a": "1.00", "b": "1.00"}, means)
    assert marked["a"] == r"\textbf{1.00}"
    assert marked["b"] == r"\underline{1.00}"


def test_rank_and_mark_ignores_models_without_a_mean() -> None:
    marked = rank_and_mark({"a": "-", "b": "2.00"}, {"b": 2.0})
    assert marked == {"a": "-", "b": r"\textbf{2.00}"}


def test_rank_and_mark_without_means_returns_the_cells() -> None:
    assert rank_and_mark({"a": "-"}, {}) == {"a": "-"}


def test_latex_escape_text_escapes_specials() -> None:
    assert latex_escape_text("a_b&c%") == r"a\_b\&c\%"

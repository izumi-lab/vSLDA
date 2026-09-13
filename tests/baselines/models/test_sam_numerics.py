"""Unit tests for the SAM numerical core.

The two finite-difference tests below are the load-bearing ones: they validate
the whole clean-room derivation (``S_d``, ``rho_d``, every partial derivative)
against an independently written closed-form ELBO.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import ive

from src.baselines.models import sam_numerics as numerics


def _problem(seed: int = 0, *, num_documents: int = 4, vocab: int = 7, topics: int = 3):
    rng = np.random.default_rng(seed)
    documents = rng.normal(size=(num_documents, vocab))
    documents /= np.linalg.norm(documents, axis=1, keepdims=True)
    mutilde = rng.normal(size=(vocab, topics))
    mutilde /= np.linalg.norm(mutilde, axis=0, keepdims=True)
    alphatilde = rng.uniform(0.5, 3.0, size=(num_documents, topics))
    mtilde = rng.normal(size=vocab)
    mtilde /= np.linalg.norm(mtilde)
    corpus_mean = rng.normal(size=vocab)
    corpus_mean /= np.linalg.norm(corpus_mean)
    hyper = numerics.SamHyperparameters(
        xi=12.0,
        kappa=30.0,
        kappa0=4.0,
        alpha=np.full(topics, 1.3),
        m=corpus_mean,
    )
    state = numerics.SamVariationalState(
        mutilde=mutilde, alphatilde=alphatilde, mtilde=mtilde
    )
    return documents, state, hyper


def _elbo(documents, state, hyper) -> float:
    stats = numerics.compute_sufficient_statistics(
        documents=documents, state=state, hyper=hyper
    )
    return numerics.compute_elbo(state=state, hyper=hyper, stats=stats)


# --------------------------------------------------------------------------- #
# vMF special functions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dimension", [3, 200, 20000])
@pytest.mark.parametrize("resultant", [1e-6, 1e-4, 0.3, 0.9, 0.999])
def test_mean_resultant_inverse_is_exact(dimension: int, resultant: float) -> None:
    concentration = numerics.inverse_vmf_mean_resultant_length(
        dimension=dimension, mean_resultant=resultant
    )
    recovered = numerics.vmf_mean_resultant_length(
        dimension=dimension, concentration=concentration
    )
    assert recovered == pytest.approx(resultant, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("dimension", [3, 50, 20000])
def test_mean_resultant_derivative_matches_finite_difference(dimension: int) -> None:
    concentration = 137.0
    analytic = numerics.vmf_mean_resultant_derivative(
        dimension=dimension, concentration=concentration
    )
    step = 1e-4
    numeric = (
        numerics.vmf_mean_resultant_length(
            dimension=dimension, concentration=concentration + step
        )
        - numerics.vmf_mean_resultant_length(
            dimension=dimension, concentration=concentration - step
        )
    ) / (2.0 * step)
    assert analytic == pytest.approx(numeric, rel=1e-6)
    assert analytic > 0.0


def test_mean_resultant_is_increasing_and_bounded() -> None:
    grid = np.logspace(-2, 5, 200)
    values = numerics.vmf_mean_resultant_length(dimension=500, concentration=grid)
    assert np.all(np.diff(values) > 0.0)
    assert np.all((values > 0.0) & (values < 1.0))
    assert numerics.vmf_mean_resultant_length(dimension=500, concentration=0.0) == 0.0


def test_inverse_matches_banerjee_estimator_in_high_dimension() -> None:
    dimension, resultant = 20000, 0.4
    ours = numerics.inverse_vmf_mean_resultant_length(
        dimension=dimension, mean_resultant=resultant
    )
    banerjee = (resultant * dimension - resultant**3) / (1.0 - resultant**2)
    assert ours == pytest.approx(banerjee, rel=1e-3)


@pytest.mark.parametrize(
    ("dimension", "tolerance"), [(10, 3e-1), (50, 6e-2), (200, 2e-2), (1000, 4e-3)]
)
def test_mean_resultant_approaches_exact_bessel_ratio(
    dimension: int, tolerance: float
) -> None:
    """The Amos truncation converges to the true ratio like ``O(1/V)``.

    It is deliberately not exact at small ``V``; what matters is that it is
    self-consistent (value / inverse / derivative / log-normalizer come from one
    formula) and accurate at the ``V ~ 1e4`` where SAM actually runs, where
    ``scipy.special.ive`` underflows to zero and is unusable.
    """

    compared = 0
    for concentration in (0.5, 5.0, 50.0, 500.0):
        numerator = ive(dimension / 2, concentration)
        denominator = ive(dimension / 2 - 1, concentration)
        if (
            not np.isfinite(numerator)
            or not np.isfinite(denominator)
            or denominator <= 0.0
        ):
            # ``ive`` has already failed at this order; that is the whole reason
            # the closed form exists, so there is nothing to compare against.
            continue
        exact = numerator / denominator
        approx = numerics.vmf_mean_resultant_length(
            dimension=dimension, concentration=concentration
        )
        assert abs(approx - exact) / exact < tolerance
        compared += 1
    assert compared > 0


def test_ive_is_unusable_at_production_scale() -> None:
    """Pins the reason the closed form exists at all."""

    assert ive(16552 / 2 - 1, 1500.0) == 0.0


def test_log_normalizer_derivative_is_minus_mean_resultant() -> None:
    dimension, concentration, step = 20000, 137.0, 1e-3
    numeric = (
        numerics.vmf_log_normalizer(
            dimension=dimension, concentration=concentration + step
        )
        - numerics.vmf_log_normalizer(
            dimension=dimension, concentration=concentration - step
        )
    ) / (2.0 * step)
    expected = -numerics.vmf_mean_resultant_length(
        dimension=dimension, concentration=concentration
    )
    assert numeric == pytest.approx(expected, rel=1e-3)


def test_dimension_below_three_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        numerics.vmf_mean_resultant_length(dimension=2, concentration=1.0)


# --------------------------------------------------------------------------- #
# Sufficient statistics
# --------------------------------------------------------------------------- #


def test_s_values_and_rho_match_brute_force_moments() -> None:
    documents, state, hyper = _problem()
    stats = numerics.compute_sufficient_statistics(
        documents=documents, state=state, hyper=hyper
    )
    resultant = stats.mean_resultant_xi
    for index, alphatilde in enumerate(state.alphatilde):
        total = alphatilde.sum()
        second = np.outer(alphatilde, alphatilde) / (total * (total + 1.0))
        np.fill_diagonal(
            second, alphatilde * (alphatilde + 1.0) / (total * (total + 1.0))
        )
        # E[phi_i^T phi_j] = A^2 mu_i^T mu_j off the diagonal, 1 on it.
        topic_moments = resultant**2 * (state.mutilde.T @ state.mutilde)
        np.fill_diagonal(topic_moments, 1.0)
        assert float((second * topic_moments).sum()) == pytest.approx(
            stats.s_values[index], rel=1e-12
        )
        expected_rho = (
            resultant
            * float(alphatilde @ (state.mutilde.T @ documents[index]))
            / (total * np.sqrt(stats.s_values[index]))
        )
        assert expected_rho == pytest.approx(stats.rho[index], rel=1e-12)


def test_sufficient_statistics_agree_for_sparse_and_dense_documents() -> None:
    documents, state, hyper = _problem()
    dense = numerics.compute_sufficient_statistics(
        documents=documents, state=state, hyper=hyper
    )
    sparse = numerics.compute_sufficient_statistics(
        documents=sp.csr_matrix(documents), state=state, hyper=hyper
    )
    assert np.allclose(dense.rho, sparse.rho, rtol=1e-12, atol=1e-14)
    assert np.allclose(dense.s_values, sparse.s_values, rtol=1e-12, atol=1e-14)
    assert np.allclose(
        numerics.mutilde_gradient(
            documents=documents, state=state, hyper=hyper, stats=dense
        ),
        numerics.mutilde_gradient(
            documents=sp.csr_matrix(documents), state=state, hyper=hyper, stats=sparse
        ),
        rtol=1e-12,
        atol=1e-14,
    )


def test_gradients_never_densify_the_document_matrix(monkeypatch) -> None:
    documents, state, hyper = _problem()
    sparse = sp.csr_matrix(documents)

    def _fail(*_args, **_kwargs):  # pragma: no cover - the assertion is the point
        raise AssertionError("the (V, D) document matrix must never be densified")

    monkeypatch.setattr(sp.csr_matrix, "toarray", _fail)
    monkeypatch.setattr(sp.csc_matrix, "toarray", _fail)
    stats = numerics.compute_sufficient_statistics(
        documents=sparse, state=state, hyper=hyper
    )
    numerics.alphatilde_gradient(state=state, hyper=hyper, stats=stats)
    numerics.mutilde_gradient(documents=sparse, state=state, hyper=hyper, stats=stats)


# --------------------------------------------------------------------------- #
# Gradients versus finite differences of the ELBO
# --------------------------------------------------------------------------- #


def test_alphatilde_gradient_matches_central_differences() -> None:
    documents, state, hyper = _problem()
    stats = numerics.compute_sufficient_statistics(
        documents=documents, state=state, hyper=hyper
    )
    analytic = numerics.alphatilde_gradient(state=state, hyper=hyper, stats=stats)

    step = 1e-6
    numeric = np.zeros_like(analytic)
    for document in range(analytic.shape[0]):
        for topic in range(analytic.shape[1]):
            forward = state.alphatilde.copy()
            forward[document, topic] += step
            backward = state.alphatilde.copy()
            backward[document, topic] -= step
            numeric[document, topic] = (
                _elbo(documents, state.with_alphatilde(forward), hyper)
                - _elbo(documents, state.with_alphatilde(backward), hyper)
            ) / (2.0 * step)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_mutilde_gradient_matches_directional_derivative_on_the_sphere() -> None:
    """Pins the resolution of the paper's inconsistent ``grad_mutilde S_d``.

    The extra ``2 (1 - A^2) mutilde_j`` term the paper prints is purely radial,
    so the tangent projection makes both readings agree.  This test would fail if
    someone "corrected" the gradient back to the printed form and dropped the
    projection.
    """

    documents, state, hyper = _problem(seed=3)
    stats = numerics.compute_sufficient_statistics(
        documents=documents, state=state, hyper=hyper
    )
    analytic = numerics.project_to_tangent(
        numerics.mutilde_gradient(
            documents=documents, state=state, hyper=hyper, stats=stats
        ),
        state.mutilde,
    )

    rng = np.random.default_rng(11)
    direction = numerics.project_to_tangent(
        rng.normal(size=state.mutilde.shape), state.mutilde
    )
    step = 1e-6
    numeric = (
        _elbo(
            documents,
            state.with_mutilde(
                numerics.retract_to_sphere(state.mutilde + step * direction)
            ),
            hyper,
        )
        - _elbo(
            documents,
            state.with_mutilde(
                numerics.retract_to_sphere(state.mutilde - step * direction)
            ),
            hyper,
        )
    ) / (2.0 * step)
    assert float(np.sum(analytic * direction)) == pytest.approx(numeric, rel=1e-6)


# --------------------------------------------------------------------------- #
# Closed forms and projections
# --------------------------------------------------------------------------- #


def test_closed_form_mtilde_is_the_normalized_prior_combination() -> None:
    documents, state, hyper = _problem()
    result = numerics.closed_form_mtilde(state=state, hyper=hyper)
    resultant = numerics.vmf_mean_resultant_length(
        dimension=state.vocabulary_size, concentration=hyper.xi
    )
    expected = hyper.kappa0 * hyper.m + resultant * hyper.xi * state.mutilde.sum(axis=1)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(result, expected)
    assert float(np.linalg.norm(result)) == pytest.approx(1.0)


def test_closed_form_mtilde_increases_the_bound() -> None:
    documents, state, hyper = _problem()
    before = _elbo(documents, state, hyper)
    updated = state.with_mtilde(numerics.closed_form_mtilde(state=state, hyper=hyper))
    assert _elbo(documents, updated, hyper) >= before


def test_projection_and_retraction() -> None:
    rng = np.random.default_rng(5)
    basis = rng.normal(size=(9, 4))
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    projected = numerics.project_to_tangent(rng.normal(size=(9, 4)), basis)
    assert np.allclose(np.einsum("vt,vt->t", projected, basis), 0.0, atol=1e-12)
    retracted = numerics.retract_to_sphere(rng.normal(size=(9, 4)))
    assert np.allclose(np.linalg.norm(retracted, axis=0), 1.0)
    with pytest.raises(ValueError, match="retract"):
        numerics.retract_to_sphere(np.zeros((9, 4)))

"""Behavioural tests for the SAM variational EM driver."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from src.baselines.models import sam_inference as inference
from src.baselines.params import parse_sam_params

# The synthetic corpora below are V ~ 30.  The production default resolves
# ``kappa`` from ``A_V(kappa) = 0.1``, which at V = 30 is ``kappa ~ 2.8`` -- a
# document likelihood so weak that the bound is maximized by collapsing every
# topic onto the corpus mean.  These tests exercise the optimizer, not the
# default, so they pin ``kappa`` to the paper's raw value, which at this V is a
# sharp ``A_V ~ 0.99``.
_SHARP = {"kappa": 1500.0}


def _params(options: dict | None = None):
    """``parse_sam_params`` with the concentration pinned for tiny vocabularies."""

    merged = dict(_SHARP)
    merged.update(options or {})
    return parse_sam_params(merged)


def _random_corpus(seed: int = 1, *, num_documents: int = 40, vocab: int = 60):
    rng = np.random.default_rng(seed)
    matrix = rng.random((num_documents, vocab))
    matrix[matrix < 0.7] = 0.0
    matrix[matrix.sum(axis=1) == 0.0, 0] = 1.0
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return sp.csr_matrix(matrix), matrix


def _planted_corpus(
    seed: int = 7, *, num_documents: int = 300, topics: int = 4, per_topic: int = 8
):
    """Disjoint-support topics, so the planted solution is identifiable."""

    rng = np.random.default_rng(seed)
    vocab = topics * per_topic
    directions = np.zeros((vocab, topics))
    for topic in range(topics):
        span = slice(topic * per_topic, (topic + 1) * per_topic)
        directions[span, topic] = rng.uniform(0.5, 1.5, size=per_topic)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    proportions = rng.dirichlet(np.full(topics, 0.3), size=num_documents)
    # Noiseless kappa -> infinity limit of the SAM likelihood, which avoids
    # needing a vMF sampler while still matching the generative story.
    documents = (directions @ proportions.T).T
    documents /= np.linalg.norm(documents, axis=1, keepdims=True)
    return sp.csr_matrix(documents), directions, proportions


def test_elbo_is_monotone() -> None:
    documents, _ = _random_corpus()
    params = _params({"num_iterations": 25, "tol": 1e-12})
    result = inference.fit_sam(
        documents=documents, num_topics=4, params=params, random_state=0
    )
    trace = np.asarray(result.elbo_trace)
    assert trace.size > 1
    # Every block is Armijo-guarded, so a decrease is a bug rather than noise.
    assert np.all(np.diff(trace) >= -1e-9)


def test_elbo_is_monotone_with_kappa_m_step() -> None:
    documents, _ = _random_corpus()
    params = _params({"num_iterations": 15, "tol": 1e-12, "optimize_kappa": True})
    result = inference.fit_sam(
        documents=documents, num_topics=4, params=params, random_state=0
    )
    assert np.all(np.diff(np.asarray(result.elbo_trace)) >= -1e-9)


def test_fit_preserves_the_variational_constraints() -> None:
    documents, _ = _random_corpus()
    params = _params({"num_iterations": 10})
    result = inference.fit_sam(
        documents=documents, num_topics=4, params=params, random_state=0
    )
    state = result.state
    assert np.allclose(np.linalg.norm(state.mutilde, axis=0), 1.0, atol=1e-10)
    assert float(np.linalg.norm(state.mtilde)) == pytest.approx(1.0, abs=1e-10)
    assert np.all(state.alphatilde > 0.0)
    assert np.all(np.isfinite(state.mutilde))
    assert np.all(np.isfinite(state.alphatilde))


def test_fit_is_deterministic_for_a_fixed_seed() -> None:
    documents, _ = _random_corpus()
    params = _params({"num_iterations": 8})
    first = inference.fit_sam(
        documents=documents, num_topics=4, params=params, random_state=42
    )
    second = inference.fit_sam(
        documents=documents, num_topics=4, params=params, random_state=42
    )
    assert np.array_equal(first.state.mutilde, second.state.mutilde)
    assert np.array_equal(first.state.alphatilde, second.state.alphatilde)


def test_sparse_and_dense_documents_agree() -> None:
    sparse, dense = _random_corpus()
    params = _params({"num_iterations": 5})
    from_sparse = inference.fit_sam(
        documents=sparse, num_topics=4, params=params, random_state=0
    )
    from_dense = inference.fit_sam(
        documents=dense, num_topics=4, params=params, random_state=0
    )
    # Sparse and dense matmuls sum in different orders; over several nonlinear
    # iterations that drifts well beyond machine epsilon.  Exact agreement of a
    # single statistics/gradient evaluation is pinned in test_sam_numerics.py.
    assert np.allclose(
        from_sparse.state.mutilde, from_dense.state.mutilde, rtol=1e-4, atol=1e-4
    )


def test_recovers_planted_topics_and_proportions() -> None:
    documents, directions, proportions = _planted_corpus()
    params = _params({"num_iterations": 80, "alpha_steps": 30, "mu_steps": 10})
    result = inference.fit_sam(
        documents=documents,
        num_topics=directions.shape[1],
        params=params,
        random_state=3,
    )
    similarity = np.abs(directions.T @ result.state.mutilde)
    rows, columns = linear_sum_assignment(-similarity)
    assert similarity[rows, columns].mean() > 0.99

    estimated = inference.document_topic_proportions(result.state.alphatilde)[
        :, columns
    ]
    assert spearmanr(proportions.ravel(), estimated.ravel()).statistic > 0.95


def test_weak_topic_posterior_collapses_topics_onto_the_corpus_mean() -> None:
    """Guards the default that matters most.

    ``E_q[phi_t] = A_V(xi) mutilde_t``, so a small ``A_V(xi)`` removes the topic
    directions from the likelihood and the bound is maximized by putting every
    topic on the corpus mean.  ``mean |cos|`` then equals the value a single
    shared direction would score, not the planted structure.  If someone lowers
    ``xi_mean_resultant`` back toward the prior-like values, this is the test
    that explains why recovery dies.
    """

    documents, directions, _ = _planted_corpus()
    params = _params(
        {"num_iterations": 40, "xi_mean_resultant": 0.05, "alpha_steps": 30}
    )
    result = inference.fit_sam(
        documents=documents,
        num_topics=directions.shape[1],
        params=params,
        random_state=3,
    )
    pairwise = np.abs(
        result.state.mutilde.T @ result.state.mutilde - np.eye(directions.shape[1])
    ).max()
    assert pairwise > 0.9  # every topic ended up on the same direction
    similarity = np.abs(directions.T @ result.state.mutilde)
    rows, columns = linear_sum_assignment(-similarity)
    assert similarity[rows, columns].mean() < 0.7


def test_fold_in_reproduces_training_proportions() -> None:
    documents, directions, _ = _planted_corpus()
    params = _params({"num_iterations": 80, "alpha_steps": 30, "mu_steps": 10})
    result = inference.fit_sam(
        documents=documents,
        num_topics=directions.shape[1],
        params=params,
        random_state=3,
    )
    trained = inference.document_topic_proportions(result.state.alphatilde)
    folded = inference.document_topic_proportions(
        inference.infer_alphatilde(
            documents=documents,
            state=result.state,
            hyper=result.hyper,
            params=params,
            random_state=11,
        )
    )
    assert np.abs(trained - folded).sum(axis=1).mean() < 0.15


def test_non_convergence_is_surfaced() -> None:
    documents, _ = _random_corpus()
    strict = _params({"num_iterations": 1, "require_convergence": True})
    with pytest.raises(RuntimeError, match="did not converge"):
        inference.fit_sam(
            documents=documents, num_topics=4, params=strict, random_state=0
        )

    lenient = _params({"num_iterations": 1})
    result = inference.fit_sam(
        documents=documents, num_topics=4, params=lenient, random_state=0
    )
    assert result.converged is False


def test_resolved_concentrations_are_reported() -> None:
    documents, _ = _random_corpus()
    params = _params({"num_iterations": 2})
    result = inference.fit_sam(
        documents=documents, num_topics=4, params=params, random_state=0
    )
    resolved = result.diagnostics["resolved_mean_resultants"]
    assert resolved["A_V(xi)"] == pytest.approx(params.xi_mean_resultant)
    assert resolved["A_V(kappa0)"] == pytest.approx(params.kappa0_mean_resultant)


def test_resolve_concentration_prefers_the_raw_value() -> None:
    assert inference.resolve_concentration(
        raw=1500.0, mean_resultant=0.5, dimension=1000, name="kappa"
    ) == pytest.approx(1500.0)
    resolved = inference.resolve_concentration(
        raw=None, mean_resultant=0.5, dimension=1000, name="xi"
    )
    from src.baselines.models.sam_numerics import vmf_mean_resultant_length

    assert vmf_mean_resultant_length(
        dimension=1000, concentration=resolved
    ) == pytest.approx(0.5)


def test_random_initialization_is_available_but_diagnostic_only() -> None:
    documents, _ = _random_corpus()
    params = _params({"num_iterations": 2, "init": "random"})
    result = inference.fit_sam(
        documents=documents, num_topics=4, params=params, random_state=0
    )
    assert np.allclose(np.linalg.norm(result.state.mutilde, axis=0), 1.0)


def test_corpus_mean_direction_rejects_a_degenerate_corpus() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        inference.corpus_mean_direction(sp.csr_matrix((3, 5)))


def test_document_topic_proportions_rows_sum_to_one() -> None:
    proportions = inference.document_topic_proportions(
        np.array([[1.0, 3.0], [2.0, 2.0]])
    )
    assert np.allclose(proportions.sum(axis=1), 1.0)
    assert np.allclose(proportions, [[0.25, 0.75], [0.5, 0.5]])


def test_convergence_is_judged_on_the_variable_part_of_the_bound() -> None:
    """Regression guard: ``tol`` used to be swamped by a constant.

    ``L`` carries ``D * log c_V(kappa)``, a constant while ``kappa`` is fixed. On
    20 Newsgroups that term is ``3.82e8`` while the rest of the bound spans only
    ``-7.8e6`` to ``-6.5e6``, so a relative tolerance on the raw value measured
    the constant and not the fit -- ``tol=1e-5`` stopped where the moving part
    had only reached ``5.8e-4``, and tightening ``tol`` barely moved the stopping
    point.
    """

    documents, _ = _random_corpus()
    params = _params({"num_iterations": 40, "tol": 1e-4})
    result = inference.fit_sam(
        documents=documents, num_topics=4, params=params, random_state=0
    )

    assert result.diagnostics["elbo_convergence_basis"] == "variable_part"
    variable = result.diagnostics["elbo_variable_trace"]
    assert len(variable) == len(result.elbo_trace)

    constant = float(result.diagnostics["elbo_constant_term"])
    assert all(
        raw == pytest.approx(part + constant, rel=1e-9)
        for raw, part in zip(result.elbo_trace, variable, strict=True)
    )
    # The variable part is what moves; it must be monotone too.
    assert np.all(np.diff(np.asarray(variable)) >= -1e-9)


def test_tightening_tol_actually_runs_longer() -> None:
    """The point of the fix: ``tol`` must control the stopping point."""

    documents, _ = _random_corpus()
    loose = inference.fit_sam(
        documents=documents,
        num_topics=4,
        params=_params({"num_iterations": 80, "tol": 1e-3}),
        random_state=0,
    )
    tight = inference.fit_sam(
        documents=documents,
        num_topics=4,
        params=_params({"num_iterations": 80, "tol": 1e-6}),
        random_state=0,
    )
    assert tight.iterations > loose.iterations


def test_elbo_variable_part_subtracts_the_document_normalizer() -> None:
    from src.baselines.models.sam_numerics import vmf_log_normalizer

    expected = 100.0 - 7 * float(
        vmf_log_normalizer(dimension=500, concentration=1500.0)
    )
    assert inference.elbo_variable_part(
        elbo=100.0, num_documents=7, dimension=500, kappa=1500.0
    ) == pytest.approx(expected)

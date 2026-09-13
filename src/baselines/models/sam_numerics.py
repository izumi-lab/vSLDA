"""Numerical core of the Spherical Admixture Model (SAM).

Clean-room implementation derived from the generative model (Section 3.1) and
the mean-field variational approximation (Section 3.2) of:

    Reisinger, Waters, Silverthorn and Mooney.
    "Spherical Topic Models." ICML 2010.

Every quantity below is re-derived from the paper's model definition; no
third-party SAM implementation was consulted while writing this module.

Notation (matching the paper, with array orientations noted):

``V``           vocabulary size; documents and topics live on ``S^{V-1}``
``T``           number of topics, ``D`` number of documents
``mutilde``     ``(V, T)`` variational topic directions, unit-norm columns
``alphatilde``  ``(D, T)`` variational Dirichlet parameters, strictly positive
``mtilde``      ``(V,)``   variational corpus mean direction, unit norm
``documents``   ``(D, V)`` L2-normalized document vectors (sparse or dense)

The module never materializes a ``(V, D)`` dense array.  Every per-document
quantity factors through ``(D, T)`` and ``(T, T)`` objects:

    G = mutilde.T @ mutilde                      (T, T)
    U = documents @ mutilde                      (D, T)
    alignment_d = alphatilde_d @ U_d           = (mutilde @ alphatilde_d)^T v_d
    quadratic_d = alphatilde_d @ G @ alphatilde_d = ||mutilde @ alphatilde_d||^2

which is what keeps the E step at ``O(D T^2)`` instead of ``O(D V T)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, gammaln, polygamma

__all__ = [
    "SamHyperparameters",
    "SamSufficientStatistics",
    "SamVariationalState",
    "alphatilde_gradient",
    "closed_form_mtilde",
    "compute_document_objectives",
    "compute_elbo",
    "compute_sufficient_statistics",
    "document_statistics",
    "inverse_vmf_mean_resultant_length",
    "mutilde_gradient",
    "project_to_tangent",
    "retract_to_sphere",
    "vmf_log_normalizer",
    "vmf_mean_resultant_derivative",
    "vmf_mean_resultant_length",
]

# Smallest variational Dirichlet parameter tolerated; also the lower clip used
# by the log-parameterized ascent in :mod:`src.baselines.models.sam_inference`.
MIN_ALPHATILDE = 1e-3
MAX_ALPHATILDE = 1e6


def _order(dimension: int) -> float:
    """Bessel order ``nu = V/2 - 1`` used by every vMF quantity."""

    value = int(dimension)
    if value < 3:
        raise ValueError("vMF dimension must be >= 3 for the SAM numerics.")
    return 0.5 * value - 1.0


def vmf_mean_resultant_length(
    *, dimension: int, concentration: np.ndarray | float
) -> np.ndarray | float:
    """Mean resultant length ``A_V(kappa) = I_{V/2}(kappa) / I_{V/2-1}(kappa)``.

    Uses the Amos/Abramowitz-Stegun continued-fraction truncation

        A_V(kappa) = kappa / (nu + sqrt(nu^2 + kappa^2)),   nu = V/2 - 1

    which the paper points to (cf. Elkan 2006) for stable evaluation in high
    dimension.  ``scipy.special.ive`` is unusable at ``V ~ 2e4`` because the
    Bessel order reaches ``1e4``; this closed form has no such limit and is
    exactly invertible and differentiable (see the two functions below).
    """

    nu = _order(dimension)
    kappa = np.asarray(concentration, dtype=np.float64)
    if np.any(kappa < 0.0) or not np.all(np.isfinite(kappa)):
        raise ValueError("vMF concentration must be finite and >= 0.")
    result = kappa / (nu + np.sqrt(nu * nu + kappa * kappa))
    return float(result) if np.isscalar(concentration) or result.ndim == 0 else result


def inverse_vmf_mean_resultant_length(
    *, dimension: int, mean_resultant: np.ndarray | float
) -> np.ndarray | float:
    """Exact algebraic inverse of :func:`vmf_mean_resultant_length`.

        A_V^{-1}(R) = 2 nu R / (1 - R^2)

    In the large-``V`` limit this reduces to the familiar Banerjee et al. (2005)
    estimator ``(R V - R^3) / (1 - R^2)``, which is an independent check that
    the branch is right.
    """

    nu = _order(dimension)
    resultant = np.asarray(mean_resultant, dtype=np.float64)
    if np.any(resultant < 0.0) or np.any(resultant >= 1.0):
        raise ValueError("mean resultant length must lie in [0, 1).")
    result = 2.0 * nu * resultant / (1.0 - resultant * resultant)
    return float(result) if np.isscalar(mean_resultant) or result.ndim == 0 else result


def vmf_mean_resultant_derivative(
    *, dimension: int, concentration: np.ndarray | float
) -> np.ndarray | float:
    """Exact derivative ``dA_V/dkappa = nu / (s (nu + s))`` with ``s = sqrt(nu^2+kappa^2)``.

    Strictly positive, so ``A_V`` is strictly increasing and the inverse above is
    single valued.
    """

    nu = _order(dimension)
    kappa = np.asarray(concentration, dtype=np.float64)
    scale = np.sqrt(nu * nu + kappa * kappa)
    result = nu / (scale * (nu + scale))
    return float(result) if np.isscalar(concentration) or result.ndim == 0 else result


def vmf_log_normalizer(
    *, dimension: int, concentration: np.ndarray | float
) -> np.ndarray | float:
    """``log c_V(kappa) = nu log kappa - (V/2) log(2 pi) - log I_nu(kappa)``.

    ``log I_nu`` uses the uniform asymptotic expansion (Abramowitz & Stegun
    9.7.7), which is the companion of the ``A_V`` truncation above: their
    derivative relation ``d log c_V / d kappa = -A_V(kappa)`` holds exactly up to
    the ``O(1/V)`` quarter-log term.  Using a matched pair here is what keeps the
    ELBO and its gradient consistent, and hence the optimizer monotone.
    """

    nu = _order(dimension)
    kappa = np.asarray(concentration, dtype=np.float64)
    safe = np.maximum(kappa, 1e-12)
    scale = np.sqrt(nu * nu + safe * safe)
    log_bessel = (
        scale
        + nu * np.log(safe / (nu + scale))
        - 0.5 * np.log(2.0 * np.pi)
        - 0.25 * np.log(nu * nu + safe * safe)
    )
    result = nu * np.log(safe) - 0.5 * dimension * np.log(2.0 * np.pi) - log_bessel
    return float(result) if np.isscalar(concentration) or result.ndim == 0 else result


@dataclass(frozen=True)
class SamHyperparameters:
    """Fixed model hyperparameters (the paper's ``xi, kappa, kappa0, alpha, m``)."""

    xi: float
    kappa: float
    kappa0: float
    alpha: np.ndarray  # (T,)
    m: np.ndarray  # (V,), unit norm

    def replace_kappa(self, kappa: float) -> "SamHyperparameters":
        return SamHyperparameters(
            xi=self.xi,
            kappa=float(kappa),
            kappa0=self.kappa0,
            alpha=self.alpha,
            m=self.m,
        )


@dataclass(frozen=True)
class SamVariationalState:
    """Free variational parameters."""

    mutilde: np.ndarray  # (V, T)
    alphatilde: np.ndarray  # (D, T)
    mtilde: np.ndarray  # (V,)

    @property
    def num_topics(self) -> int:
        return int(self.mutilde.shape[1])

    @property
    def vocabulary_size(self) -> int:
        return int(self.mutilde.shape[0])

    @property
    def num_documents(self) -> int:
        return int(self.alphatilde.shape[0])

    def with_alphatilde(self, alphatilde: np.ndarray) -> "SamVariationalState":
        return SamVariationalState(
            mutilde=self.mutilde, alphatilde=alphatilde, mtilde=self.mtilde
        )

    def with_mutilde(self, mutilde: np.ndarray) -> "SamVariationalState":
        return SamVariationalState(
            mutilde=mutilde, alphatilde=self.alphatilde, mtilde=self.mtilde
        )

    def with_mtilde(self, mtilde: np.ndarray) -> "SamVariationalState":
        return SamVariationalState(
            mutilde=self.mutilde, alphatilde=self.alphatilde, mtilde=mtilde
        )


@dataclass(frozen=True)
class SamSufficientStatistics:
    """Everything the E step needs, all in ``(D, T)`` / ``(T, T)`` space."""

    gram: np.ndarray  # (T, T)  mutilde^T mutilde
    projections: np.ndarray  # (D, T)  documents @ mutilde
    alpha_sum: np.ndarray  # (D,)
    alpha_sq_sum: np.ndarray  # (D,)
    quadratic: np.ndarray  # (D,)  ||mutilde @ alphatilde_d||^2
    s_values: np.ndarray  # (D,)  E_q[||phi theta_d||^2]
    alignment: np.ndarray  # (D,)  (mutilde @ alphatilde_d)^T v_d
    rho: np.ndarray  # (D,)  E_q[Avg(phi, theta_d)]^T v_d
    mean_resultant_xi: float  # A_V(xi)


def document_statistics(
    *,
    alphatilde: np.ndarray,
    projections: np.ndarray,
    gram: np.ndarray,
    mean_resultant_xi: float,
) -> tuple[np.ndarray, ...]:
    """Per-document ``S_d`` and ``rho_d`` given precomputed ``U`` and ``G``.

    Split out from :func:`compute_sufficient_statistics` because the alphatilde
    ascent evaluates trial points many times with ``mutilde`` unchanged, so
    ``projections`` and ``gram`` stay valid and a trial costs one ``(D, T)`` pass.

    Returns ``(alpha_sum, alpha_sq_sum, quadratic, s_values, alignment, rho)``.
    """

    resultant = float(mean_resultant_xi)
    alpha_sum = alphatilde.sum(axis=1)
    alpha_sq_sum = np.einsum("dt,dt->d", alphatilde, alphatilde)
    quadratic = np.einsum("dt,dt->d", alphatilde @ gram, alphatilde)

    # S_d = E_q[||phi theta_d||^2]; see the paper's expression above eq. (3).
    # The Dirichlet second moments give
    #   sum_i E[theta_i^2] + sum_{i != j} E[theta_i theta_j] A^2 mu_i^T mu_j
    # which collects into the numerator below.  Every term is non-negative and
    # alpha_sum > 0, so S_d > 0 by construction.
    numerator = (
        alpha_sum
        + (1.0 - resultant * resultant) * alpha_sq_sum
        + resultant * resultant * quadratic
    )
    denominator = alpha_sum * (alpha_sum + 1.0)
    s_values = numerator / denominator

    alignment = np.einsum("dt,dt->d", alphatilde, projections)
    rho = resultant * alignment / (alpha_sum * np.sqrt(s_values))
    return alpha_sum, alpha_sq_sum, quadratic, s_values, alignment, rho


def compute_sufficient_statistics(
    *,
    documents,
    state: SamVariationalState,
    hyper: SamHyperparameters,
) -> SamSufficientStatistics:
    """Build ``G``, ``U`` and every per-document quantity."""

    vocabulary_size = state.vocabulary_size
    resultant = float(
        vmf_mean_resultant_length(dimension=vocabulary_size, concentration=hyper.xi)
    )
    gram = state.mutilde.T @ state.mutilde
    projections = np.asarray(documents @ state.mutilde, dtype=np.float64)
    (
        alpha_sum,
        alpha_sq_sum,
        quadratic,
        s_values,
        alignment,
        rho,
    ) = document_statistics(
        alphatilde=state.alphatilde,
        projections=projections,
        gram=gram,
        mean_resultant_xi=resultant,
    )
    return SamSufficientStatistics(
        gram=gram,
        projections=projections,
        alpha_sum=alpha_sum,
        alpha_sq_sum=alpha_sq_sum,
        quadratic=quadratic,
        s_values=s_values,
        alignment=alignment,
        rho=rho,
        mean_resultant_xi=resultant,
    )


def _dirichlet_expected_logs(
    alphatilde: np.ndarray, alpha_sum: np.ndarray
) -> np.ndarray:
    """``E_q[log theta_{d,i}] = Psi(alphatilde_{d,i}) - Psi(alphatilde_{d,0})``."""

    return digamma(alphatilde) - digamma(alpha_sum)[:, None]


def compute_document_objectives(
    *,
    state: SamVariationalState,
    hyper: SamHyperparameters,
    stats: SamSufficientStatistics,
) -> np.ndarray:
    """Per-document ELBO contribution, up to terms independent of ``alphatilde``.

    Used by the per-document backtracking line search in the alphatilde block.
    Dropping ``log c_V(kappa)`` and the ``log Gamma(alpha_0) - sum log Gamma(alpha_i)``
    constant is safe because the search only compares values at fixed ``d``.
    """

    alphatilde = state.alphatilde
    expected_logs = _dirichlet_expected_logs(alphatilde, stats.alpha_sum)
    log_p = ((hyper.alpha - 1.0)[None, :] * expected_logs).sum(axis=1)
    log_q = (
        gammaln(stats.alpha_sum)
        - gammaln(alphatilde).sum(axis=1)
        + ((alphatilde - 1.0) * expected_logs).sum(axis=1)
    )
    return hyper.kappa * stats.rho + log_p - log_q


def compute_elbo(
    *,
    state: SamVariationalState,
    hyper: SamHyperparameters,
    stats: SamSufficientStatistics,
) -> float:
    """Variational lower bound (the paper's eq. (1)), in closed form.

    The paper prints only the gradients.  Writing ``L`` out term by term gives

        L = sum_d [ log c_V(kappa) + kappa rho_d ]
          + xi A_V(xi) sum_t [ A_V(kappa0) mutilde_t^T mtilde - 1 ]
          + sum_d E_q[log p(theta_d)] - sum_d E_q[log q(theta_d)]
          + kappa0 A_V(kappa0) [ mtilde^T m - 1 ]

    where ``log c_V(xi)`` and ``log c_V(kappa0)`` cancel between prior and
    posterior because ``q(phi_t)`` and ``q(mu)`` reuse the prior concentrations.
    Differentiating this expression reproduces the paper's ``dL/dalphatilde``,
    ``grad_mtilde L`` and ``grad_mutilde L`` term for term, which is what the
    finite-difference tests check.
    """

    vocabulary_size = state.vocabulary_size
    num_topics = state.num_topics
    num_documents = state.num_documents

    resultant_xi = stats.mean_resultant_xi
    resultant_kappa0 = float(
        vmf_mean_resultant_length(dimension=vocabulary_size, concentration=hyper.kappa0)
    )
    log_c_kappa = float(
        vmf_log_normalizer(dimension=vocabulary_size, concentration=hyper.kappa)
    )

    document_term = num_documents * log_c_kappa + hyper.kappa * float(stats.rho.sum())

    topic_alignment = float((state.mutilde.T @ state.mtilde).sum())
    topic_term = (
        hyper.xi
        * resultant_xi
        * (resultant_kappa0 * topic_alignment - float(num_topics))
    )

    corpus_term = (
        hyper.kappa0 * resultant_kappa0 * (float(state.mtilde @ hyper.m) - 1.0)
    )

    alphatilde = state.alphatilde
    expected_logs = _dirichlet_expected_logs(alphatilde, stats.alpha_sum)
    log_p_theta = num_documents * (
        float(gammaln(hyper.alpha.sum())) - float(gammaln(hyper.alpha).sum())
    ) + float(((hyper.alpha - 1.0)[None, :] * expected_logs).sum())
    log_q_theta = float(
        (
            gammaln(stats.alpha_sum)
            - gammaln(alphatilde).sum(axis=1)
            + ((alphatilde - 1.0) * expected_logs).sum(axis=1)
        ).sum()
    )

    return document_term + topic_term + corpus_term + log_p_theta - log_q_theta


def alphatilde_gradient(
    *,
    state: SamVariationalState,
    hyper: SamHyperparameters,
    stats: SamSufficientStatistics,
) -> np.ndarray:
    """``dL/dalphatilde``, shape ``(D, T)``.

    Reproduces the paper's

        dL/dalphatilde_{d,i} = kappa d rho_d / d alphatilde_{d,i}
                             + Psi'(alphatilde_0)(alphatilde_0 - alpha_0)
                             - Psi'(alphatilde_i)(alphatilde_i - alpha_i)

    with ``d rho_d / d alphatilde`` expanded through ``S_d`` exactly as in
    Section 3.2, but written entirely in ``(D, T)`` space.
    """

    resultant = stats.mean_resultant_xi
    alphatilde = state.alphatilde
    alpha_sum = stats.alpha_sum[:, None]
    s_values = stats.s_values[:, None]
    alignment = stats.alignment[:, None]
    denominator = (stats.alpha_sum * (stats.alpha_sum + 1.0))[:, None]

    gram_product = alphatilde @ stats.gram
    ds_dalpha = (
        1.0
        + 2.0 * (1.0 - resultant * resultant) * alphatilde
        + 2.0 * resultant * resultant * gram_product
    ) / denominator - ((2.0 * stats.alpha_sum + 1.0)[:, None] / denominator) * s_values

    sqrt_s = np.sqrt(s_values)
    drho_dalpha = (resultant / alpha_sum) * (
        (stats.projections - alignment / alpha_sum) / sqrt_s
        - alignment * ds_dalpha / (2.0 * s_values * sqrt_s)
    )

    alpha_total = float(hyper.alpha.sum())
    dirichlet = polygamma(1, stats.alpha_sum)[:, None] * (
        alpha_sum - alpha_total
    ) - polygamma(1, alphatilde) * (alphatilde - hyper.alpha[None, :])
    return hyper.kappa * drho_dalpha + dirichlet


def mutilde_gradient(
    *,
    documents,
    state: SamVariationalState,
    hyper: SamHyperparameters,
    stats: SamSufficientStatistics,
) -> np.ndarray:
    """``dL/dmutilde``, shape ``(V, T)``, without materializing ``(V, D)``.

    Only two terms of ``L`` depend on ``mutilde``.  Differentiating them gives

        grad_mutilde L = xi A_V(xi) A_V(kappa0) mtilde 1_T^T
                       + kappa [ documents^T @ C1 - mutilde @ (alphatilde^T @ C2) ]

    with ``C1``, ``C2`` the ``(D, T)`` coefficient matrices built below.

    Note on the paper: its printed ``grad_{mutilde_j} S_d`` carries an extra
    ``2 (1 - A^2) mutilde_j`` term that does not follow from the stated ``S_d``
    (whose middle sum has no ``mutilde`` dependence).  That term is purely
    radial in ``mutilde_j``, so it is annihilated by the tangent-space
    projection the sphere constraint requires; we therefore implement the exact
    gradient of our own ELBO and take a Riemannian step, which makes the two
    readings equivalent.  Do not "fix" this back to match the PDF.
    """

    resultant_xi = stats.mean_resultant_xi
    resultant_kappa0 = float(
        vmf_mean_resultant_length(
            dimension=state.vocabulary_size, concentration=hyper.kappa0
        )
    )

    alpha_sum = stats.alpha_sum[:, None]
    s_values = stats.s_values[:, None]
    sqrt_s = np.sqrt(s_values)
    alignment = stats.alignment[:, None]

    coefficient_linear = resultant_xi * state.alphatilde / (alpha_sum * sqrt_s)
    coefficient_quadratic = (
        resultant_xi**3
        * alignment
        * state.alphatilde
        / ((stats.alpha_sum**2 * (stats.alpha_sum + 1.0))[:, None] * s_values * sqrt_s)
    )

    likelihood = np.asarray(
        documents.T @ coefficient_linear, dtype=np.float64
    ) - state.mutilde @ (state.alphatilde.T @ coefficient_quadratic)

    prior = (
        hyper.xi
        * resultant_xi
        * resultant_kappa0
        * np.repeat(state.mtilde[:, None], state.num_topics, axis=1)
    )
    return prior + hyper.kappa * likelihood


def closed_form_mtilde(
    *, state: SamVariationalState, hyper: SamHyperparameters
) -> np.ndarray:
    """Maximizer of ``L`` over the unit-norm ``mtilde``.

    ``grad_mtilde L = kappa0 A_V(kappa0) m + xi A_V(xi) A_V(kappa0) sum_t mutilde_t``;
    the shared ``A_V(kappa0)`` cancels in the normalization, leaving the paper's

        mtilde ~ kappa0 m + A_V(xi) xi sum_t mutilde_t
    """

    resultant_xi = float(
        vmf_mean_resultant_length(
            dimension=state.vocabulary_size, concentration=hyper.xi
        )
    )
    raw = hyper.kappa0 * hyper.m + resultant_xi * hyper.xi * state.mutilde.sum(axis=1)
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("closed-form mtilde update produced a degenerate direction.")
    return raw / norm


def project_to_tangent(gradient: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the radial component of each column, so the step stays on the sphere."""

    radial = np.einsum("vt,vt->t", gradient, basis)
    return gradient - basis * radial[None, :]


def retract_to_sphere(matrix: np.ndarray) -> np.ndarray:
    """Renormalize columns to unit L2 norm."""

    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("cannot retract a zero or non-finite column to the sphere.")
    return matrix / norms[None, :]

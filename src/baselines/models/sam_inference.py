"""Variational EM driver for the Spherical Admixture Model (SAM).

Clean-room implementation following Reisinger et al., "Spherical Topic Models",
ICML 2010, Section 3.2.  The numerical primitives live in
:mod:`src.baselines.models.sam_numerics`; this module only orchestrates them.

Design notes that are not obvious from the paper:

* ``alphatilde`` is optimized in log space (``alphatilde = exp(ell)``).  This
  enforces positivity without an active set, and makes the step scale-free
  across documents whose ``alphatilde_0`` differ by orders of magnitude.
* The ``alphatilde`` objective separates over documents, so each document keeps
  its own step size and its own Armijo backtracking.  A joint quasi-Newton
  solver over ``D * T`` variables would spend its curvature model on
  cross-document couplings that do not exist.
* ``mutilde`` columns live on ``S^{V-1}``, so its ascent is Riemannian: project
  the gradient onto the tangent space, step, renormalize.
* ``mtilde`` has an exact closed-form maximizer, so it needs no step size.

Each block maximizes the same ELBO holding the others fixed, and every step is
Armijo-guarded, so the bound is non-decreasing by construction.  A decrease is a
bug, not a tolerance issue, and is raised rather than smoothed over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from src.baselines.models.sam_numerics import (
    MAX_ALPHATILDE,
    MIN_ALPHATILDE,
    SamHyperparameters,
    SamSufficientStatistics,
    SamVariationalState,
    alphatilde_gradient,
    closed_form_mtilde,
    compute_document_objectives,
    compute_elbo,
    compute_sufficient_statistics,
    document_statistics,
    inverse_vmf_mean_resultant_length,
    mutilde_gradient,
    project_to_tangent,
    retract_to_sphere,
    vmf_log_normalizer,
    vmf_mean_resultant_length,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SamFitResult",
    "corpus_mean_direction",
    "document_topic_proportions",
    "fit_sam",
    "infer_alphatilde",
    "elbo_variable_part",
    "initialize_state",
    "resolve_concentration",
]

_LOG_MIN = float(np.log(MIN_ALPHATILDE))
_LOG_MAX = float(np.log(MAX_ALPHATILDE))
_ARMIJO_C1 = 1e-4
_MAX_BACKTRACKS = 25
_ELBO_DECREASE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class SamFitResult:
    state: SamVariationalState
    hyper: SamHyperparameters
    elbo_trace: list[float]
    converged: bool
    iterations: int
    diagnostics: dict[str, object] = field(default_factory=dict)


def _dense_rows(documents, indices: np.ndarray) -> np.ndarray:
    rows = documents[indices]
    return np.asarray(rows.todense() if sp.issparse(rows) else rows, dtype=np.float64)


def corpus_mean_direction(documents) -> np.ndarray:
    """``m = normalize(sum_d v_d)``, the empirical corpus direction.

    ``m`` is deliberately never optimized: setting ``grad_m L = 0`` subject to
    ``||m|| = 1`` yields ``m ~ mtilde``, a degenerate solution that simply
    deletes the prior.  "Optimize every hyperparameter" is the wrong instinct
    here.
    """

    total = np.asarray(documents.sum(axis=0), dtype=np.float64).ravel()
    norm = float(np.linalg.norm(total))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("corpus mean direction is degenerate (empty corpus?).")
    return total / norm


def resolve_concentration(
    *,
    raw: float | None,
    mean_resultant: float,
    dimension: int,
    name: str,
) -> float:
    """Resolve a vMF concentration given either a raw value or a target ``A_V``.

    ``kappa`` and friends are only meaningful relative to ``V``: the paper's
    ``kappa = 1500`` is ``A_V ~ 0.09`` at its ``V = 16552`` and would mean
    something quite different at another vocabulary size.  Leaving the raw value
    ``None`` therefore selects the concentration whose mean resultant length is
    ``mean_resultant``, which transfers across corpora.
    """

    if raw is not None:
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"SAM {name} must be finite and > 0.")
        return value
    return float(
        inverse_vmf_mean_resultant_length(
            dimension=dimension, mean_resultant=float(mean_resultant)
        )
    )


def _statistics_for_alphatilde(
    *,
    alphatilde: np.ndarray,
    projections: np.ndarray,
    gram: np.ndarray,
    mean_resultant_xi: float,
) -> SamSufficientStatistics:
    (
        alpha_sum,
        alpha_sq_sum,
        quadratic,
        s_values,
        alignment,
        rho,
    ) = document_statistics(
        alphatilde=alphatilde,
        projections=projections,
        gram=gram,
        mean_resultant_xi=mean_resultant_xi,
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
        mean_resultant_xi=mean_resultant_xi,
    )


def _document_objective_values(
    *,
    alphatilde: np.ndarray,
    projections: np.ndarray,
    gram: np.ndarray,
    hyper: SamHyperparameters,
    mean_resultant_xi: float,
    template: SamVariationalState,
) -> tuple[np.ndarray, SamSufficientStatistics]:
    stats = _statistics_for_alphatilde(
        alphatilde=alphatilde,
        projections=projections,
        gram=gram,
        mean_resultant_xi=mean_resultant_xi,
    )
    values = compute_document_objectives(
        state=template.with_alphatilde(alphatilde), hyper=hyper, stats=stats
    )
    return values, stats


def ascend_alphatilde(
    *,
    state: SamVariationalState,
    hyper: SamHyperparameters,
    projections: np.ndarray,
    gram: np.ndarray,
    mean_resultant_xi: float,
    step_sizes: np.ndarray,
    max_steps: int,
    min_step_size: float = 1e-10,
    max_step_size: float = 1e4,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-document Armijo ascent on ``log alphatilde``.

    ``projections`` and ``gram`` are held fixed, so each trial point costs one
    ``(D, T)`` pass and no ``V``-dimensional work at all.
    """

    alphatilde = np.array(state.alphatilde, dtype=np.float64, copy=True)
    steps = np.array(step_sizes, dtype=np.float64, copy=True)

    for _ in range(int(max_steps)):
        log_alpha = np.log(alphatilde)
        values, stats = _document_objective_values(
            alphatilde=alphatilde,
            projections=projections,
            gram=gram,
            hyper=hyper,
            mean_resultant_xi=mean_resultant_xi,
            template=state,
        )
        gradient = alphatilde_gradient(
            state=state.with_alphatilde(alphatilde), hyper=hyper, stats=stats
        )
        # Chain rule for the log parameterization.
        log_gradient = gradient * alphatilde
        gradient_norms = np.sqrt(np.einsum("dt,dt->d", log_gradient, log_gradient))
        # Ascend along the unit-normalized direction so that ``steps`` measures an
        # actual displacement.  Raw gradient norms vary by orders of magnitude
        # across documents and across ``kappa`` settings, which would otherwise
        # burn the whole backtracking budget before the first accepted step.
        safe_norms = np.where(gradient_norms > 0.0, gradient_norms, 1.0)
        direction = log_gradient / safe_norms[:, None]

        best_log_alpha = log_alpha.copy()
        active = gradient_norms > 0.0
        trial_steps = steps.copy()
        accepted_steps = steps.copy()

        for _ in range(_MAX_BACKTRACKS):
            if not np.any(active):
                break
            candidate = best_log_alpha.copy()
            candidate[active] = np.clip(
                log_alpha[active] + trial_steps[active, None] * direction[active],
                _LOG_MIN,
                _LOG_MAX,
            )
            candidate_values, _ = _document_objective_values(
                alphatilde=np.exp(candidate),
                projections=projections,
                gram=gram,
                hyper=hyper,
                mean_resultant_xi=mean_resultant_xi,
                template=state,
            )
            improved = active & (
                candidate_values >= values + _ARMIJO_C1 * trial_steps * gradient_norms
            )
            best_log_alpha[improved] = candidate[improved]
            accepted_steps[improved] = trial_steps[improved] * 1.5
            active = active & ~improved
            trial_steps[active] *= 0.5

        accepted_steps[active] = trial_steps[active]
        steps = np.clip(accepted_steps, min_step_size, max_step_size)
        alphatilde = np.exp(best_log_alpha)

    return alphatilde, steps


def ascend_mutilde(
    *,
    documents,
    state: SamVariationalState,
    hyper: SamHyperparameters,
    step_size: float,
    max_steps: int,
    min_step_size: float = 1e-12,
    max_step_size: float = 1e4,
) -> tuple[np.ndarray, float]:
    """Riemannian Armijo ascent on the unit-norm columns of ``mutilde``."""

    mutilde = np.array(state.mutilde, dtype=np.float64, copy=True)
    step = float(step_size)

    for _ in range(int(max_steps)):
        current = state.with_mutilde(mutilde)
        stats = compute_sufficient_statistics(
            documents=documents, state=current, hyper=hyper
        )
        value = compute_elbo(state=current, hyper=hyper, stats=stats)
        gradient = project_to_tangent(
            mutilde_gradient(
                documents=documents, state=current, hyper=hyper, stats=stats
            ),
            mutilde,
        )
        gradient_norm = float(np.sqrt(np.einsum("vt,vt->", gradient, gradient)))
        if gradient_norm <= 0.0:
            break
        direction = gradient / gradient_norm

        trial = step
        accepted = False
        for _ in range(_MAX_BACKTRACKS):
            candidate = retract_to_sphere(mutilde + trial * direction)
            candidate_state = state.with_mutilde(candidate)
            candidate_stats = compute_sufficient_statistics(
                documents=documents, state=candidate_state, hyper=hyper
            )
            candidate_value = compute_elbo(
                state=candidate_state, hyper=hyper, stats=candidate_stats
            )
            if candidate_value >= value + _ARMIJO_C1 * trial * gradient_norm:
                mutilde = candidate
                step = min(trial * 1.5, max_step_size)
                accepted = True
                break
            trial *= 0.5
            if trial < min_step_size:
                break
        if not accepted:
            step = max(trial, min_step_size)
            break

    return mutilde, step


def initialize_state(
    *,
    documents,
    num_topics: int,
    hyper: SamHyperparameters,
    rng: np.random.Generator,
    init: str = "corpus_random",
) -> SamVariationalState:
    """Seed the variational parameters.

    ``corpus_random`` places each topic at a jittered document direction.  This
    matters: at ``V ~ 2e4`` a uniformly random direction on ``S^{V-1}`` is
    near-orthogonal to every document, so ``projections ~ 0``, ``rho ~ 0`` and
    the gradient vanishes.  ``random`` is kept for diagnostics only.
    """

    num_documents, vocabulary_size = documents.shape
    if init == "corpus_random":
        indices = rng.choice(
            num_documents, size=num_topics, replace=num_documents < num_topics
        )
        seeds = _dense_rows(documents, indices)
        seeds = seeds + 0.01 * rng.normal(size=seeds.shape)
        mutilde = retract_to_sphere(seeds.T)
    elif init == "random":
        mutilde = retract_to_sphere(rng.normal(size=(vocabulary_size, num_topics)))
    else:
        raise ValueError(f"unknown SAM initialization: {init!r}")

    jitter = rng.uniform(0.9, 1.1, size=(num_documents, num_topics))
    alphatilde = np.clip(
        (hyper.alpha[None, :] + 1.0) * jitter, MIN_ALPHATILDE, MAX_ALPHATILDE
    )
    return SamVariationalState(
        mutilde=mutilde, alphatilde=alphatilde, mtilde=np.array(hyper.m, copy=True)
    )


def elbo_variable_part(
    *, elbo: float, num_documents: int, dimension: int, kappa: float
) -> float:
    """The part of the bound that actually moves during optimization.

    ``L`` contains ``D * log c_V(kappa)``, which is a constant whenever ``kappa``
    is fixed -- and a very large one: at 20 Newsgroups (``D=10856, V=10914,
    kappa=1500``) it is ``3.82e8`` while the rest of the bound only spans
    ``-7.8e6`` to ``-6.5e6``.  Judging convergence on the full value therefore
    measures the constant, not the fit: a relative tolerance of ``1e-5`` on ``L``
    corresponds to roughly ``5.8e-4`` on the part that moves, so tightening
    ``tol`` has almost no effect on where the run stops.

    Subtracting the term makes ``tol`` mean what a reader expects.  It is
    subtracted at the current ``kappa`` each iteration so the quantity stays
    well defined when ``optimize_kappa`` is on.
    """

    log_normalizer = float(vmf_log_normalizer(dimension=dimension, concentration=kappa))
    return float(elbo) - float(num_documents) * log_normalizer


def _maybe_update_kappa(
    *, hyper: SamHyperparameters, stats: SamSufficientStatistics, dimension: int
) -> SamHyperparameters:
    """Closed-form ``kappa`` M step.

    ``dL/dkappa = -D A_V(kappa) + sum_d rho_d = 0`` gives
    ``kappa = A_V^{-1}(mean rho_d)``, using ``d log c_V / d kappa = -A_V``.
    """

    mean_rho = float(np.mean(stats.rho))
    if not np.isfinite(mean_rho) or mean_rho <= 0.0:
        return hyper
    resultant = float(np.clip(mean_rho, 1e-8, 1.0 - 1e-8))
    kappa = float(
        inverse_vmf_mean_resultant_length(dimension=dimension, mean_resultant=resultant)
    )
    return hyper.replace_kappa(kappa)


def fit_sam(
    *,
    documents,
    num_topics: int,
    params,
    random_state: int | None = None,
) -> SamFitResult:
    """Run variational EM to convergence (or to ``params.num_iterations``)."""

    num_documents, vocabulary_size = documents.shape
    alpha = np.full(int(num_topics), float(params.alpha), dtype=np.float64)
    corpus_mean = corpus_mean_direction(documents)
    hyper = SamHyperparameters(
        xi=resolve_concentration(
            raw=params.xi,
            mean_resultant=params.xi_mean_resultant,
            dimension=vocabulary_size,
            name="xi",
        ),
        kappa=resolve_concentration(
            raw=params.kappa,
            mean_resultant=params.kappa_mean_resultant,
            dimension=vocabulary_size,
            name="kappa",
        ),
        kappa0=resolve_concentration(
            raw=params.kappa0,
            mean_resultant=params.kappa0_mean_resultant,
            dimension=vocabulary_size,
            name="kappa0",
        ),
        alpha=alpha,
        m=corpus_mean,
    )
    resolved = {
        "A_V(kappa)": float(
            vmf_mean_resultant_length(
                dimension=vocabulary_size, concentration=hyper.kappa
            )
        ),
        "A_V(xi)": float(
            vmf_mean_resultant_length(dimension=vocabulary_size, concentration=hyper.xi)
        ),
        "A_V(kappa0)": float(
            vmf_mean_resultant_length(
                dimension=vocabulary_size, concentration=hyper.kappa0
            )
        ),
    }
    logger.info(
        "SAM fit: V=%d D=%d T=%d resolved mean resultants %s",
        vocabulary_size,
        num_documents,
        num_topics,
        resolved,
    )

    rng = np.random.default_rng(random_state)
    state = initialize_state(
        documents=documents,
        num_topics=int(num_topics),
        hyper=hyper,
        rng=rng,
        init=str(params.init),
    )

    alpha_steps = np.full(num_documents, float(params.alpha_step_size))
    mu_step = float(params.mu_step_size)
    elbo_trace: list[float] = []
    variable_trace: list[float] = []
    converged = False
    iterations = 0

    for iteration in range(int(params.num_iterations)):
        iterations = iteration + 1
        resultant_xi = float(
            vmf_mean_resultant_length(dimension=vocabulary_size, concentration=hyper.xi)
        )
        gram = state.mutilde.T @ state.mutilde
        projections = np.asarray(documents @ state.mutilde, dtype=np.float64)
        alphatilde, alpha_steps = ascend_alphatilde(
            state=state,
            hyper=hyper,
            projections=projections,
            gram=gram,
            mean_resultant_xi=resultant_xi,
            step_sizes=alpha_steps,
            max_steps=int(params.alpha_steps),
        )
        state = state.with_alphatilde(alphatilde)

        mutilde, mu_step = ascend_mutilde(
            documents=documents,
            state=state,
            hyper=hyper,
            step_size=mu_step,
            max_steps=int(params.mu_steps),
        )
        state = state.with_mutilde(mutilde)
        state = state.with_mtilde(closed_form_mtilde(state=state, hyper=hyper))

        stats = compute_sufficient_statistics(
            documents=documents, state=state, hyper=hyper
        )
        if bool(params.optimize_kappa):
            hyper = _maybe_update_kappa(
                hyper=hyper, stats=stats, dimension=vocabulary_size
            )
        value = float(compute_elbo(state=state, hyper=hyper, stats=stats))
        variable = elbo_variable_part(
            elbo=value,
            num_documents=num_documents,
            dimension=vocabulary_size,
            kappa=hyper.kappa,
        )

        if elbo_trace:
            previous = elbo_trace[-1]
            decrease = previous - value
            if decrease > _ELBO_DECREASE_TOLERANCE * max(1.0, abs(previous)):
                raise RuntimeError(
                    "SAM ELBO decreased from "
                    f"{previous!r} to {value!r} at iteration {iterations}; "
                    "this indicates a gradient or step-size bug, not a tolerance issue."
                )
            previous_variable = variable_trace[-1]
            # Judged on the variable part; see ``elbo_variable_part``.
            relative = abs(variable - previous_variable) / (
                abs(previous_variable) + 1e-12
            )
            elbo_trace.append(value)
            variable_trace.append(variable)
            if relative < float(params.tol):
                converged = True
                break
        else:
            elbo_trace.append(value)
            variable_trace.append(variable)

    if not converged:
        message = (
            f"SAM did not converge in {iterations} iterations "
            f"(tol={params.tol} on the variable part of the bound); "
            f"last ELBO={elbo_trace[-1] if elbo_trace else float('nan')} "
            f"(variable part {variable_trace[-1] if variable_trace else float('nan')})"
        )
        if bool(params.require_convergence):
            raise RuntimeError(message)
        logger.warning(message)

    return SamFitResult(
        state=state,
        hyper=hyper,
        elbo_trace=elbo_trace,
        converged=converged,
        iterations=iterations,
        diagnostics={
            "resolved_mean_resultants": resolved,
            "final_mu_step_size": float(mu_step),
            "median_alpha_step_size": float(np.median(alpha_steps)),
            "elbo_convergence_basis": "variable_part",
            "elbo_constant_term": float(value - variable),
            "elbo_variable_trace": [float(item) for item in variable_trace],
        },
    )


def infer_alphatilde(
    *,
    documents,
    state: SamVariationalState,
    hyper: SamHyperparameters,
    params,
    random_state: int | None = None,
) -> np.ndarray:
    """Fold-in: run only the ``alphatilde`` block with everything else frozen.

    Uses the identical objective as training, which is what makes the
    "fold-in reproduces training theta" test meaningful.
    """

    num_documents = documents.shape[0]
    vocabulary_size = state.vocabulary_size
    num_topics = state.num_topics
    rng = np.random.default_rng(random_state)

    resultant_xi = float(
        vmf_mean_resultant_length(dimension=vocabulary_size, concentration=hyper.xi)
    )
    gram = state.mutilde.T @ state.mutilde
    projections = np.asarray(documents @ state.mutilde, dtype=np.float64)

    jitter = rng.uniform(0.9, 1.1, size=(num_documents, num_topics))
    alphatilde = np.clip(
        (hyper.alpha[None, :] + 1.0) * jitter, MIN_ALPHATILDE, MAX_ALPHATILDE
    )
    seed_state = SamVariationalState(
        mutilde=state.mutilde, alphatilde=alphatilde, mtilde=state.mtilde
    )
    result, _ = ascend_alphatilde(
        state=seed_state,
        hyper=hyper,
        projections=projections,
        gram=gram,
        mean_resultant_xi=resultant_xi,
        step_sizes=np.full(num_documents, float(params.alpha_step_size)),
        max_steps=int(params.infer_num_iterations),
    )
    return result


def document_topic_proportions(alphatilde: np.ndarray) -> np.ndarray:
    """``E_q[theta_d] = alphatilde_d / alphatilde_{d,0}``, rows summing to one."""

    array = np.asarray(alphatilde, dtype=np.float64)
    totals = array.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0) or not np.all(np.isfinite(totals)):
        raise ValueError("alphatilde rows must be positive and finite.")
    return array / totals

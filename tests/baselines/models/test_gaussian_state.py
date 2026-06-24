from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.baselines.models.gaussian_state import (
    GaussianTrainerState,
    coerce_gaussian_trainer_state,
    snapshot_gaussian_trainer,
    validate_gaussian_trainer_state,
)


def _build_trainer() -> SimpleNamespace:
    return SimpleNamespace(
        alpha=0.1,
        num_tables=2,
        average_ll=[1.0, 2.0],
        prior=SimpleNamespace(kappa=0.2, mu=np.asarray([0.5, 0.5])),
        table_density_kernel_backend="numba",
        posterior_sampling_kernel_backend="numba",
        avg_ll_kernel_backend="python",
        table_counts=np.asarray([1, 2]),
        table_means=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        table_inverse_covariances=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        log_determinants=np.asarray([0.0, 0.1]),
        sum_table_customers=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        sum_squared_table_customers=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        table_cholesky_ltriangular_mat=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        table_counts_per_doc=np.asarray([[1, 0], [0, 1]]),
        training_corpus_preencoded=True,
        training_corpus_encoding_sec=0.25,
    )


def test_snapshot_gaussian_trainer_captures_repo_owned_state() -> None:
    state = snapshot_gaussian_trainer(_build_trainer(), include_prior_mu=True)

    assert isinstance(state, GaussianTrainerState)
    assert state.average_ll == (1.0, 2.0)
    assert state.prior_kappa == 0.2
    assert np.allclose(state.prior_mu, np.asarray([0.5, 0.5]))
    assert state.table_counts_per_doc.shape == (2, 2)
    assert state.table_density_kernel_backend == "numba"
    assert state.posterior_sampling_kernel_backend == "numba"
    assert state.avg_ll_kernel_backend == "python"
    assert state.training_corpus_preencoded is True
    assert state.training_corpus_encoding_sec == 0.25


def test_coerce_gaussian_trainer_state_accepts_existing_snapshot() -> None:
    state = snapshot_gaussian_trainer(_build_trainer(), include_prior_mu=True)

    coerced = coerce_gaussian_trainer_state(state, include_prior_mu=True)

    assert coerced is state


def test_validate_gaussian_trainer_state_rejects_incompatible_shapes() -> None:
    state = GaussianTrainerState(
        average_ll=(1.0,),
        alpha=0.1,
        num_tables=2,
        prior_kappa=0.2,
        table_counts=np.asarray([1]),
        table_means=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        table_inverse_covariances=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        log_determinants=np.asarray([0.0, 0.1]),
        sum_table_customers=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        sum_squared_table_customers=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        table_cholesky_ltriangular_mat=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
    )

    try:
        validate_gaussian_trainer_state(state)
    except ValueError as exc:
        assert "table_counts" in str(exc)
    else:
        raise AssertionError(
            "Expected invalid Gaussian trainer state to fail validation"
        )


def test_coerce_gaussian_trainer_state_requires_prior_mu_when_requested() -> None:
    state = snapshot_gaussian_trainer(_build_trainer(), include_prior_mu=False)

    try:
        coerce_gaussian_trainer_state(state, include_prior_mu=True)
    except ValueError as exc:
        assert "prior_mu" in str(exc)
    else:
        raise AssertionError("Expected missing prior_mu to fail validation")

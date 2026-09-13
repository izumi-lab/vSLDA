from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol

import numpy as np


class GaussianPriorLike(Protocol):
    kappa: float
    mu: Any
    scale_sigma: float
    nu: float


class GaussianTrainerLike(Protocol):
    average_ll: Any
    alpha: float
    num_tables: int
    prior: GaussianPriorLike
    table_density_kernel_backend: Any
    posterior_sampling_kernel_backend: Any
    avg_ll_kernel_backend: Any
    table_counts: Any
    table_means: Any
    table_inverse_covariances: Any
    log_determinants: Any
    sum_table_customers: Any
    sum_squared_table_customers: Any
    table_cholesky_ltriangular_mat: Any
    table_counts_per_doc: Any
    training_corpus_preencoded: Any
    training_corpus_encoding_sec: Any
    iteration_diagnostics: Any
    training_elapsed_sec: Any


@dataclass(frozen=True)
class GaussianTrainerState:
    average_ll: tuple[float, ...]
    alpha: float
    num_tables: int
    prior_kappa: float
    table_counts: Any
    table_means: Any
    table_inverse_covariances: Any
    log_determinants: Any
    sum_table_customers: Any
    sum_squared_table_customers: Any
    table_cholesky_ltriangular_mat: Any
    table_counts_per_doc: Any | None = None
    prior_mu: Any | None = None
    table_density_kernel_backend: str | None = None
    posterior_sampling_kernel_backend: str | None = None
    avg_ll_kernel_backend: str | None = None
    training_corpus_preencoded: bool | None = None
    training_corpus_encoding_sec: float | None = None
    prior_scale: float | None = None
    prior_nu: float | None = None
    iteration_diagnostics: tuple[dict[str, Any], ...] = ()
    training_elapsed_sec: float | None = None
    # Reduced-covariance (diag / spherical) sentence Gaussian LDA only. ``None`` for the
    # full-covariance trainers, whose artifacts are unchanged.
    covariance_type: str | None = None
    sum_squared_table_customers_diag: Any | None = None
    table_scaled_variances: Any | None = None


def is_reduced_covariance_state(trainer_state: "GaussianTrainerState") -> bool:
    return trainer_state.covariance_type in {"diag", "spherical"}


def validate_gaussian_trainer_state(
    trainer_state: GaussianTrainerState,
    *,
    require_prior_mu: bool = False,
) -> GaussianTrainerState:
    def _shape(array: Any) -> tuple[int, ...]:
        return tuple(np.asarray(array).shape)

    expected_num_tables = int(trainer_state.num_tables)
    fields_to_check = {
        "table_counts": trainer_state.table_counts,
        "table_means": trainer_state.table_means,
        "log_determinants": trainer_state.log_determinants,
        "sum_table_customers": trainer_state.sum_table_customers,
    }
    if is_reduced_covariance_state(trainer_state):
        fields_to_check["sum_squared_table_customers_diag"] = (
            trainer_state.sum_squared_table_customers_diag
        )
        fields_to_check["table_scaled_variances"] = trainer_state.table_scaled_variances
    else:
        fields_to_check["sum_squared_table_customers"] = (
            trainer_state.sum_squared_table_customers
        )
        fields_to_check["table_cholesky_ltriangular_mat"] = (
            trainer_state.table_cholesky_ltriangular_mat
        )
    for field_name, value in fields_to_check.items():
        if _shape(value)[0] != expected_num_tables:
            raise ValueError(
                f"Gaussian trainer state field `{field_name}` expected first dimension "
                f"{expected_num_tables}, got {_shape(value)}."
            )

    if (
        trainer_state.table_counts_per_doc is not None
        and _shape(trainer_state.table_counts_per_doc)[0] != expected_num_tables
    ):
        raise ValueError(
            "Gaussian trainer state field `table_counts_per_doc` has incompatible "
            f"shape {_shape(trainer_state.table_counts_per_doc)} for {expected_num_tables} tables."
        )
    if require_prior_mu and trainer_state.prior_mu is None:
        raise ValueError(
            "Gaussian trainer state requires `prior_mu`, but it was not captured."
        )
    return trainer_state


def snapshot_gaussian_trainer(
    trainer: GaussianTrainerLike,
    *,
    include_prior_mu: bool = False,
) -> GaussianTrainerState:
    return GaussianTrainerState(
        average_ll=tuple(float(value) for value in trainer.average_ll),
        alpha=float(trainer.alpha),
        num_tables=int(trainer.num_tables),
        prior_kappa=float(trainer.prior.kappa),
        table_counts=trainer.table_counts,
        table_means=trainer.table_means,
        table_inverse_covariances=trainer.table_inverse_covariances,
        log_determinants=trainer.log_determinants,
        sum_table_customers=trainer.sum_table_customers,
        sum_squared_table_customers=trainer.sum_squared_table_customers,
        table_cholesky_ltriangular_mat=trainer.table_cholesky_ltriangular_mat,
        table_counts_per_doc=getattr(trainer, "table_counts_per_doc", None),
        prior_mu=(getattr(trainer.prior, "mu", None) if include_prior_mu else None),
        table_density_kernel_backend=getattr(
            trainer, "table_density_kernel_backend", None
        ),
        posterior_sampling_kernel_backend=getattr(
            trainer, "posterior_sampling_kernel_backend", None
        ),
        avg_ll_kernel_backend=getattr(trainer, "avg_ll_kernel_backend", None),
        training_corpus_preencoded=getattr(trainer, "training_corpus_preencoded", None),
        training_corpus_encoding_sec=(
            None
            if getattr(trainer, "training_corpus_encoding_sec", None) is None
            else float(getattr(trainer, "training_corpus_encoding_sec"))
        ),
        prior_scale=(
            None
            if getattr(trainer.prior, "scale_sigma", None) is None
            else float(getattr(trainer.prior, "scale_sigma"))
        ),
        prior_nu=_trainer_prior_nu(trainer),
        iteration_diagnostics=tuple(
            asdict(item) if is_dataclass(item) else dict(item)
            for item in getattr(trainer, "iteration_diagnostics", ()) or ()
        ),
        training_elapsed_sec=(
            None
            if getattr(trainer, "training_elapsed_sec", None) is None
            else float(getattr(trainer, "training_elapsed_sec"))
        ),
        covariance_type=(
            None
            if getattr(trainer, "covariance_type", None) is None
            else str(getattr(trainer, "covariance_type"))
        ),
        sum_squared_table_customers_diag=getattr(
            trainer, "sum_squared_table_customers_diag", None
        ),
        table_scaled_variances=getattr(trainer, "table_scaled_variances", None),
    )


def _trainer_prior_nu(trainer: GaussianTrainerLike) -> float | None:
    # The reduced-covariance trainer keeps its own (smaller) prior degrees of freedom; the
    # full-covariance trainers report the prior's ``nu`` (= M).
    own = getattr(trainer, "prior_nu", None)
    if own is not None:
        return float(own)
    value = getattr(trainer.prior, "nu", None)
    return None if value is None else float(value)


def coerce_gaussian_trainer_state(
    trainer_or_state: GaussianTrainerState | GaussianTrainerLike,
    *,
    include_prior_mu: bool = False,
) -> GaussianTrainerState:
    if isinstance(trainer_or_state, GaussianTrainerState):
        return validate_gaussian_trainer_state(
            trainer_or_state,
            require_prior_mu=include_prior_mu,
        )
    return validate_gaussian_trainer_state(
        snapshot_gaussian_trainer(
            trainer_or_state,
            include_prior_mu=include_prior_mu,
        ),
        require_prior_mu=include_prior_mu,
    )

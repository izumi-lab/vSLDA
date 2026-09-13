"""Posterior bookkeeping for the reduced-covariance (diagonal / spherical) sentence Gaussian LDA.

The full-covariance model places a normal--inverse-Wishart prior on each topic,
``Sigma ~ IW(nu_0 = M, Psi_0 = lambda I)`` and ``mu | Sigma ~ N(mu_0, Sigma / kappa_0)``.
The reduced variants keep the mean side unchanged and replace the inverse-Wishart by a
scaled-inverse-chi-squared prior on the variances (Murphy 2007, "Conjugate Bayesian analysis
of the Gaussian distribution", NIX form):

* spherical: ``Sigma = sigma^2 I``, ``sigma^2 ~ Inv-chi^2(nu_0', lambda)``,
  posterior ``nu_n = nu_0' + n M``, ``nu_n sigma_n^2 = nu_0' lambda + sum_i ||x_i||^2
  + kappa_0 ||mu_0||^2 - kappa_n ||mu_n||^2``;
* diagonal: one such prior per dimension, ``nu_n = nu_0' + n``,
  ``nu_n sigma_{n,d}^2 = nu_0' lambda + sum_i x_{i,d}^2 + kappa_0 mu_{0,d}^2 - kappa_n mu_{n,d}^2``.

Both use ``nu_0' = nu_0 - M + 1 = 1`` (``reduced_prior_nu``), so that an empty topic has the same
predictive scale ``lambda (kappa_0 + 1) / kappa_0`` and the same degrees of freedom (1) as under
the full model: the variants change the covariance structure only, not the width of the prior.
The posterior predictive of a topic is a t distribution with ``nu_n`` degrees of freedom and
scale ``sigma_n^2 (kappa_n + 1) / kappa_n`` (multivariate for spherical, a product of univariate
t's for diagonal); ``log_determinants`` stores ``0.5 * log|S|`` of that scaled covariance, as the
full-covariance kernels do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.baselines.models.gaussian_reduced_numba import (
    GAUSSIAN_REDUCED_KERNEL_BACKEND,
    accumulate_reduced_log_likelihood_encoded_kernel,
    log_reduced_tdensity_single_kernel,
    log_reduced_tdensity_tables_kernel,
)
from src.baselines.params import (
    COVARIANCE_TYPES,
    is_reduced_covariance,
    normalize_covariance_type,
)

# Floor on the posterior sum of squares; mathematically it is >= nu_0' * lambda > 0, the floor
# only guards floating-point cancellation in ``S2 - kappa_n * mu_n^2``.
_VARIANCE_FLOOR = 1e-12


def reduced_prior_nu(embedding_size: int) -> float:
    """Degrees of freedom of the reduced prior: ``nu_0 - M + 1`` with the full model's ``nu_0 = M``."""
    del embedding_size
    return 1.0


@dataclass(frozen=True)
class ReducedTableParameters:
    table_means: np.ndarray
    nu: np.ndarray
    scaled_variances: np.ndarray
    log_determinants: np.ndarray


def compute_reduced_table_parameters(
    *,
    covariance_type: str,
    table_counts: np.ndarray,
    sum_table_customers: np.ndarray,
    sum_squared_table_customers_diag: np.ndarray,
    prior_mu: np.ndarray,
    kappa: float,
    prior_nu: float,
    prior_scale: float,
) -> ReducedTableParameters:
    """Vectorised posterior parameters of every table from its sufficient statistics."""
    covariance_type = normalize_covariance_type(covariance_type)
    if not is_reduced_covariance(covariance_type):
        raise ValueError(
            f"compute_reduced_table_parameters expects 'diag' or 'spherical', got {covariance_type!r}."
        )
    counts = np.asarray(table_counts, dtype=np.float64)
    sum_x = np.asarray(sum_table_customers, dtype=np.float64)
    sum_x2 = np.asarray(sum_squared_table_customers_diag, dtype=np.float64)
    mu0 = np.asarray(prior_mu, dtype=np.float64)
    kappa = float(kappa)
    prior_nu = float(prior_nu)
    prior_scale = float(prior_scale)
    embedding_size = int(sum_x.shape[1])

    k_n = kappa + counts
    means = (kappa * mu0[np.newaxis, :] + sum_x) / k_n[:, np.newaxis]
    if covariance_type == "spherical":
        nu = prior_nu + counts * embedding_size
        q = (
            prior_nu * prior_scale
            + sum_x2.sum(axis=1)
            + kappa * float(np.dot(mu0, mu0))
            - k_n * np.einsum("km,km->k", means, means)
        )
        q = np.maximum(q, _VARIANCE_FLOOR)
        scaled = (q / nu) * (k_n + 1.0) / k_n
        log_det = 0.5 * embedding_size * np.log(scaled)
    else:
        nu = prior_nu + counts
        q = (
            prior_nu * prior_scale
            + sum_x2
            + kappa * (mu0 * mu0)[np.newaxis, :]
            - k_n[:, np.newaxis] * means * means
        )
        q = np.maximum(q, _VARIANCE_FLOOR)
        scaled = (q / nu[:, np.newaxis]) * ((k_n + 1.0) / k_n)[:, np.newaxis]
        log_det = 0.5 * np.log(scaled).sum(axis=1)
    return ReducedTableParameters(
        table_means=means,
        nu=nu,
        scaled_variances=scaled,
        log_determinants=log_det,
    )


class ReducedCovarianceTables:
    """Sufficient statistics and predictive-density caches of the reduced variants.

    The trainer owns ``table_counts``, ``sum_table_customers``, ``table_means`` and
    ``log_determinants`` and updates the first two itself; this object owns the squared sums and
    the variance caches and rewrites the means / log-determinants of a table on ``refresh``.
    """

    def __init__(
        self,
        *,
        covariance_type: str,
        num_tables: int,
        embedding_size: int,
        prior_mu: np.ndarray,
        kappa: float,
        prior_nu: float,
        prior_scale: float,
        table_counts: np.ndarray,
        sum_table_customers: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
    ) -> None:
        self.covariance_type = normalize_covariance_type(covariance_type)
        if not is_reduced_covariance(self.covariance_type):
            raise ValueError(
                f"ReducedCovarianceTables expects 'diag' or 'spherical', got {covariance_type!r}."
            )
        self.spherical = self.covariance_type == "spherical"
        self.num_tables = int(num_tables)
        self.embedding_size = int(embedding_size)
        self.prior_mu = np.asarray(prior_mu, dtype=np.float64)
        self.kappa = float(kappa)
        self.prior_nu = float(prior_nu)
        self.prior_scale = float(prior_scale)
        self._prior_mu_sq = self.prior_mu * self.prior_mu
        self._prior_mu_norm_sq = float(np.dot(self.prior_mu, self.prior_mu))
        # Shared with the trainer (views, never reassigned).
        self.table_counts = table_counts
        self.sum_table_customers = sum_table_customers
        self.table_means = table_means
        self.log_determinants = log_determinants
        # Owned.
        self.sum_squared_table_customers_diag = np.zeros(
            (self.num_tables, self.embedding_size), dtype=np.float64
        )
        self.nu = np.zeros(self.num_tables, dtype=np.float64)
        self.scaled_variances = (
            np.zeros(self.num_tables, dtype=np.float64)
            if self.spherical
            else np.zeros((self.num_tables, self.embedding_size), dtype=np.float64)
        )
        self.kernel_backend = GAUSSIAN_REDUCED_KERNEL_BACKEND

    # -- sufficient statistics -------------------------------------------------

    def reset(self) -> None:
        self.sum_squared_table_customers_diag[:] = 0.0

    def add(self, table_id: int, encoding: np.ndarray) -> None:
        self.sum_squared_table_customers_diag[table_id] += encoding * encoding
        self.refresh(table_id)

    def remove(self, table_id: int, encoding: np.ndarray) -> None:
        self.sum_squared_table_customers_diag[table_id] -= encoding * encoding
        self.refresh(table_id)

    def refresh(self, table_id: int) -> None:
        count = float(self.table_counts[table_id])
        k_n = self.kappa + count
        mu_n = (self.kappa * self.prior_mu + self.sum_table_customers[table_id]) / k_n
        self.table_means[table_id] = mu_n
        prior_term = self.prior_nu * self.prior_scale
        if self.spherical:
            nu_n = self.prior_nu + count * self.embedding_size
            q = (
                prior_term
                + float(self.sum_squared_table_customers_diag[table_id].sum())
                + self.kappa * self._prior_mu_norm_sq
                - k_n * float(np.dot(mu_n, mu_n))
            )
            q = max(q, _VARIANCE_FLOOR)
            scaled = (q / nu_n) * (k_n + 1.0) / k_n
            self.nu[table_id] = nu_n
            self.scaled_variances[table_id] = scaled
            self.log_determinants[table_id] = (
                0.5 * self.embedding_size * float(np.log(scaled))
            )
        else:
            nu_n = self.prior_nu + count
            q = (
                prior_term
                + self.sum_squared_table_customers_diag[table_id]
                + self.kappa * self._prior_mu_sq
                - k_n * mu_n * mu_n
            )
            np.maximum(q, _VARIANCE_FLOOR, out=q)
            scaled = (q / nu_n) * ((k_n + 1.0) / k_n)
            self.nu[table_id] = nu_n
            self.scaled_variances[table_id] = scaled
            self.log_determinants[table_id] = 0.5 * float(np.log(scaled).sum())

    def refresh_all(self) -> None:
        params = compute_reduced_table_parameters(
            covariance_type=self.covariance_type,
            table_counts=self.table_counts,
            sum_table_customers=self.sum_table_customers,
            sum_squared_table_customers_diag=self.sum_squared_table_customers_diag,
            prior_mu=self.prior_mu,
            kappa=self.kappa,
            prior_nu=self.prior_nu,
            prior_scale=self.prior_scale,
        )
        self.table_means[:] = params.table_means
        self.nu[:] = params.nu
        self.scaled_variances[:] = params.scaled_variances
        self.log_determinants[:] = params.log_determinants

    def recompute_sufficient_statistics(
        self,
        encoded_corpus: Iterable[np.ndarray],
        table_assignments: Iterable[Sequence[int]],
    ) -> None:
        """Rebuild ``sum_squared_table_customers_diag`` from scratch (drains update drift)."""
        self.sum_squared_table_customers_diag[:] = 0.0
        for doc, tables in zip(encoded_corpus, table_assignments):
            doc_arr = np.asarray(doc, dtype=np.float64)
            for row, table_id in zip(doc_arr, tables):
                if int(table_id) < 0:
                    continue
                self.sum_squared_table_customers_diag[int(table_id)] += row * row
        self.refresh_all()

    # -- densities -------------------------------------------------------------

    def log_density_tables(self, x: np.ndarray) -> np.ndarray:
        return log_reduced_tdensity_tables_kernel(
            x,
            covariance_type=self.covariance_type,
            embedding_size=self.embedding_size,
            nu=self.nu,
            table_means=self.table_means,
            log_determinants=self.log_determinants,
            scaled_variances=self.scaled_variances,
        )

    def log_density(self, x: np.ndarray, table_id: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim > 1:
            return np.asarray(
                [self.log_density(row, table_id) for row in arr], dtype=np.float64
            )
        return np.float64(
            log_reduced_tdensity_single_kernel(
                arr,
                covariance_type=self.covariance_type,
                table_id=table_id,
                embedding_size=self.embedding_size,
                nu=self.nu,
                table_means=self.table_means,
                log_determinants=self.log_determinants,
                scaled_variances=self.scaled_variances,
            )
        )

    def avg_ll(
        self,
        encoded_corpus: Iterable[np.ndarray],
        table_assignments: Iterable[Sequence[int]],
    ) -> float:
        total_log_ll = 0.0
        total_items = 0
        for doc, tables in zip(encoded_corpus, table_assignments):
            doc_ll, doc_count = accumulate_reduced_log_likelihood_encoded_kernel(
                np.asarray(doc, dtype=np.float64),
                np.asarray(tables, dtype=np.int64),
                covariance_type=self.covariance_type,
                embedding_size=self.embedding_size,
                nu=self.nu,
                table_means=self.table_means,
                log_determinants=self.log_determinants,
                scaled_variances=self.scaled_variances,
            )
            total_log_ll += float(doc_ll)
            total_items += int(doc_count)
        if total_items == 0:
            return 0.0
        return total_log_ll / total_items


def log_reduced_tdensity_tables(
    x: np.ndarray,
    *,
    covariance_type: str,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> np.ndarray:
    return log_reduced_tdensity_tables_kernel(
        np.asarray(x, dtype=np.float64),
        covariance_type=normalize_covariance_type(covariance_type),
        embedding_size=embedding_size,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_variances=scaled_variances,
    )


def log_reduced_tdensity(
    x: np.ndarray,
    *,
    covariance_type: str,
    table_id: int,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    kwargs = dict(
        covariance_type=normalize_covariance_type(covariance_type),
        table_id=table_id,
        embedding_size=embedding_size,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_variances=scaled_variances,
    )
    if arr.ndim > 1:
        return np.asarray(
            [log_reduced_tdensity_single_kernel(row, **kwargs) for row in arr],
            dtype=np.float64,
        )
    return np.float64(log_reduced_tdensity_single_kernel(arr, **kwargs))


__all__ = [
    "COVARIANCE_TYPES",
    "GAUSSIAN_REDUCED_KERNEL_BACKEND",
    "ReducedCovarianceTables",
    "ReducedTableParameters",
    "compute_reduced_table_parameters",
    "is_reduced_covariance",
    "log_reduced_tdensity",
    "log_reduced_tdensity_tables",
    "normalize_covariance_type",
    "reduced_prior_nu",
]

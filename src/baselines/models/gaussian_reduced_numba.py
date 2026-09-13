"""Density kernels for the reduced-covariance (diagonal / spherical) sentence Gaussian LDA.

The full-covariance kernels in ``gaussian_numba.py`` take a ``(K, M, M)`` Cholesky factor per
table. The reduced variants keep one scaled variance per table (spherical, ``(K,)``) or per table
and dimension (diagonal, ``(K, M)``), so every density evaluation is ``O(M)``. The kernels share
the full kernels' conventions: ``nu`` holds the predictive degrees of freedom per table and
``log_determinants`` holds ``0.5 * log|S|`` of the scaled predictive covariance ``S``.

Spherical: ``S = s^2 I`` and the density is a multivariate t with ``nu`` degrees of freedom.
Diagonal: ``S = diag(s_1^2, ..., s_M^2)`` and the density is a product of univariate t's, each
with ``nu`` degrees of freedom.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency fallback
    njit = None
    NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure-python reference kernels
# ---------------------------------------------------------------------------


def _log_spherical_tdensity_single_python(
    x: np.ndarray,
    *,
    table_id: int,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> float:
    diff = (
        np.asarray(x, dtype=np.float64)
        - np.asarray(table_means, dtype=np.float64)[table_id]
    )
    val = float(np.dot(diff, diff)) / float(scaled_variances[table_id])
    nu_value = float(nu[table_id])
    return math.lgamma((nu_value + embedding_size) / 2.0) - (
        math.lgamma(nu_value / 2.0)
        + embedding_size / 2.0 * (math.log(nu_value) + math.log(math.pi))
        + float(log_determinants[table_id])
        + (nu_value + embedding_size) / 2.0 * math.log1p(val / nu_value)
    )


def _log_diag_tdensity_single_python(
    x: np.ndarray,
    *,
    table_id: int,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> float:
    diff = (
        np.asarray(x, dtype=np.float64)
        - np.asarray(table_means, dtype=np.float64)[table_id]
    )
    nu_value = float(nu[table_id])
    quad = float(
        np.sum(
            np.log1p(diff * diff / (nu_value * np.asarray(scaled_variances)[table_id]))
        )
    )
    return embedding_size * (
        math.lgamma((nu_value + 1.0) / 2.0)
        - math.lgamma(nu_value / 2.0)
        - 0.5 * (math.log(nu_value) + math.log(math.pi))
    ) - (float(log_determinants[table_id]) + (nu_value + 1.0) / 2.0 * quad)


def _log_reduced_tdensity_tables_python(
    x: np.ndarray,
    *,
    spherical: bool,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> np.ndarray:
    means = np.asarray(table_means, dtype=np.float64)
    output = np.empty(means.shape[0], dtype=np.float64)
    single = (
        _log_spherical_tdensity_single_python
        if spherical
        else _log_diag_tdensity_single_python
    )
    for table_id in range(means.shape[0]):
        output[table_id] = single(
            x,
            table_id=table_id,
            embedding_size=embedding_size,
            nu=nu,
            table_means=means,
            log_determinants=log_determinants,
            scaled_variances=scaled_variances,
        )
    return output


def _accumulate_reduced_log_likelihood_encoded_python(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    *,
    spherical: bool,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> tuple[float, int]:
    single = (
        _log_spherical_tdensity_single_python
        if spherical
        else _log_diag_tdensity_single_python
    )
    total_log_ll = 0.0
    total_items = 0
    limit = min(int(encoded_doc.shape[0]), int(assignments.shape[0]))
    for index in range(limit):
        table_id = int(assignments[index])
        if table_id < 0 or table_id >= int(table_means.shape[0]):
            continue
        total_log_ll += single(
            encoded_doc[index],
            table_id=table_id,
            embedding_size=embedding_size,
            nu=nu,
            table_means=table_means,
            log_determinants=log_determinants,
            scaled_variances=scaled_variances,
        )
        total_items += 1
    return total_log_ll, total_items


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _log_spherical_tdensity_single_numba(
        x: np.ndarray,
        table_id: int,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_variances: np.ndarray,
    ) -> float:
        val = 0.0
        for row in range(embedding_size):
            diff = float(x[row]) - float(table_means[table_id, row])
            val += diff * diff
        val /= float(scaled_variances[table_id])
        nu_value = float(nu[table_id])
        return math.lgamma((nu_value + embedding_size) / 2.0) - (
            math.lgamma(nu_value / 2.0)
            + embedding_size / 2.0 * (math.log(nu_value) + math.log(math.pi))
            + float(log_determinants[table_id])
            + (nu_value + embedding_size) / 2.0 * math.log1p(val / nu_value)
        )

    @njit(cache=True)
    def _log_diag_tdensity_single_numba(
        x: np.ndarray,
        table_id: int,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_variances: np.ndarray,
    ) -> float:
        nu_value = float(nu[table_id])
        quad = 0.0
        for row in range(embedding_size):
            diff = float(x[row]) - float(table_means[table_id, row])
            quad += math.log1p(
                diff * diff / (nu_value * float(scaled_variances[table_id, row]))
            )
        return embedding_size * (
            math.lgamma((nu_value + 1.0) / 2.0)
            - math.lgamma(nu_value / 2.0)
            - 0.5 * (math.log(nu_value) + math.log(math.pi))
        ) - (float(log_determinants[table_id]) + (nu_value + 1.0) / 2.0 * quad)

    @njit(cache=True)
    def _log_spherical_tdensity_tables_numba(
        x: np.ndarray,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_variances: np.ndarray,
    ) -> np.ndarray:
        num_tables = table_means.shape[0]
        output = np.empty(num_tables, dtype=np.float64)
        for table_id in range(num_tables):
            output[table_id] = _log_spherical_tdensity_single_numba(
                x,
                table_id,
                embedding_size,
                nu,
                table_means,
                log_determinants,
                scaled_variances,
            )
        return output

    @njit(cache=True)
    def _log_diag_tdensity_tables_numba(
        x: np.ndarray,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_variances: np.ndarray,
    ) -> np.ndarray:
        num_tables = table_means.shape[0]
        output = np.empty(num_tables, dtype=np.float64)
        for table_id in range(num_tables):
            output[table_id] = _log_diag_tdensity_single_numba(
                x,
                table_id,
                embedding_size,
                nu,
                table_means,
                log_determinants,
                scaled_variances,
            )
        return output

    @njit(cache=True)
    def _accumulate_spherical_log_likelihood_encoded_numba(
        encoded_doc: np.ndarray,
        assignments: np.ndarray,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_variances: np.ndarray,
    ) -> tuple[float, int]:
        total_log_ll = 0.0
        total_items = 0
        limit = min(encoded_doc.shape[0], assignments.shape[0])
        num_tables = table_means.shape[0]
        for index in range(limit):
            table_id = int(assignments[index])
            if table_id < 0 or table_id >= num_tables:
                continue
            total_log_ll += _log_spherical_tdensity_single_numba(
                encoded_doc[index],
                table_id,
                embedding_size,
                nu,
                table_means,
                log_determinants,
                scaled_variances,
            )
            total_items += 1
        return total_log_ll, total_items

    @njit(cache=True)
    def _accumulate_diag_log_likelihood_encoded_numba(
        encoded_doc: np.ndarray,
        assignments: np.ndarray,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_variances: np.ndarray,
    ) -> tuple[float, int]:
        total_log_ll = 0.0
        total_items = 0
        limit = min(encoded_doc.shape[0], assignments.shape[0])
        num_tables = table_means.shape[0]
        for index in range(limit):
            table_id = int(assignments[index])
            if table_id < 0 or table_id >= num_tables:
                continue
            total_log_ll += _log_diag_tdensity_single_numba(
                encoded_doc[index],
                table_id,
                embedding_size,
                nu,
                table_means,
                log_determinants,
                scaled_variances,
            )
            total_items += 1
        return total_log_ll, total_items


# ---------------------------------------------------------------------------
# Dispatch wrappers (python or numba)
# ---------------------------------------------------------------------------


def _is_spherical(covariance_type: str) -> bool:
    if covariance_type == "spherical":
        return True
    if covariance_type == "diag":
        return False
    raise ValueError(
        f"Reduced Gaussian kernels support 'diag' or 'spherical', got {covariance_type!r}."
    )


def log_reduced_tdensity_single_kernel(
    x: np.ndarray,
    *,
    covariance_type: str,
    table_id: int,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> float:
    spherical = _is_spherical(covariance_type)
    args = dict(
        table_id=int(table_id),
        embedding_size=int(embedding_size),
        nu=np.asarray(nu, dtype=np.float64),
        table_means=np.asarray(table_means, dtype=np.float64),
        log_determinants=np.asarray(log_determinants, dtype=np.float64),
        scaled_variances=np.asarray(scaled_variances, dtype=np.float64),
    )
    arr = np.asarray(x, dtype=np.float64)
    if NUMBA_AVAILABLE:
        kernel = (
            _log_spherical_tdensity_single_numba
            if spherical
            else _log_diag_tdensity_single_numba
        )
        return float(
            kernel(
                arr,
                args["table_id"],
                args["embedding_size"],
                args["nu"],
                args["table_means"],
                args["log_determinants"],
                args["scaled_variances"],
            )
        )
    single = (
        _log_spherical_tdensity_single_python
        if spherical
        else _log_diag_tdensity_single_python
    )
    return float(single(arr, **args))


def log_reduced_tdensity_tables_kernel(
    x: np.ndarray,
    *,
    covariance_type: str,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> np.ndarray:
    spherical = _is_spherical(covariance_type)
    arr = np.asarray(x, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)
    means = np.asarray(table_means, dtype=np.float64)
    log_det = np.asarray(log_determinants, dtype=np.float64)
    variances = np.asarray(scaled_variances, dtype=np.float64)
    if NUMBA_AVAILABLE:
        kernel = (
            _log_spherical_tdensity_tables_numba
            if spherical
            else _log_diag_tdensity_tables_numba
        )
        return kernel(arr, int(embedding_size), nu_arr, means, log_det, variances)
    return _log_reduced_tdensity_tables_python(
        arr,
        spherical=spherical,
        embedding_size=int(embedding_size),
        nu=nu_arr,
        table_means=means,
        log_determinants=log_det,
        scaled_variances=variances,
    )


def accumulate_reduced_log_likelihood_encoded_kernel(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    *,
    covariance_type: str,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_variances: np.ndarray,
) -> tuple[float, int]:
    spherical = _is_spherical(covariance_type)
    doc = np.asarray(encoded_doc, dtype=np.float64)
    assigned = np.asarray(assignments, dtype=np.int64)
    nu_arr = np.asarray(nu, dtype=np.float64)
    means = np.asarray(table_means, dtype=np.float64)
    log_det = np.asarray(log_determinants, dtype=np.float64)
    variances = np.asarray(scaled_variances, dtype=np.float64)
    if NUMBA_AVAILABLE:
        kernel = (
            _accumulate_spherical_log_likelihood_encoded_numba
            if spherical
            else _accumulate_diag_log_likelihood_encoded_numba
        )
        total, count = kernel(
            doc, assigned, int(embedding_size), nu_arr, means, log_det, variances
        )
        return float(total), int(count)
    return _accumulate_reduced_log_likelihood_encoded_python(
        doc,
        assigned,
        spherical=spherical,
        embedding_size=int(embedding_size),
        nu=nu_arr,
        table_means=means,
        log_determinants=log_det,
        scaled_variances=variances,
    )


GAUSSIAN_REDUCED_KERNEL_BACKEND = "numba" if NUMBA_AVAILABLE else "python"

__all__ = [
    "GAUSSIAN_REDUCED_KERNEL_BACKEND",
    "NUMBA_AVAILABLE",
    "accumulate_reduced_log_likelihood_encoded_kernel",
    "log_reduced_tdensity_single_kernel",
    "log_reduced_tdensity_tables_kernel",
]

from __future__ import annotations

import numpy as np

from src.baselines.models.gaussian_numba import (
    GAUSSIAN_AVG_LL_BACKEND,
    GAUSSIAN_POSTERIOR_SAMPLING_BACKEND,
    GAUSSIAN_TABLE_DENSITY_BACKEND,
    accumulate_gaussian_log_likelihood_encoded_kernel,
    accumulate_gaussian_log_likelihood_words_kernel,
    log_multivariate_tdensity_single_kernel,
    log_multivariate_tdensity_tables_kernel,
    sample_doc_topic_assignments_kernel,
    sample_topic_assignment_kernel,
)


def build_gaussian_nu(
    *,
    table_counts: np.ndarray,
    embedding_size: int,
) -> np.ndarray:
    prior_nu = float(embedding_size)
    return prior_nu + np.asarray(table_counts, dtype=np.float64) - embedding_size + 1.0


def build_scaled_cholesky(
    *,
    table_counts: np.ndarray,
    kappa: float,
    embedding_size: int,
    table_cholesky_ltriangular_mat: np.ndarray,
) -> np.ndarray:
    counts = np.asarray(table_counts, dtype=np.float64)
    chol = np.asarray(table_cholesky_ltriangular_mat, dtype=np.float64)
    prior_nu = float(embedding_size)
    k_n = float(kappa) + counts
    nu_n = prior_nu + counts
    scale_tdistrn = np.sqrt((k_n + 1.0) / (k_n * (nu_n - embedding_size + 1.0)))
    return scale_tdistrn[:, np.newaxis, np.newaxis] * chol


def log_multivariate_tdensity(
    x: np.ndarray,
    *,
    table_id: int,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim > 1:
        output = np.empty(arr.shape[0], dtype=np.float64)
        for index, row in enumerate(arr):
            output[index] = log_multivariate_tdensity_single_kernel(
                row,
                table_id=table_id,
                embedding_size=embedding_size,
                nu=nu,
                table_means=table_means,
                log_determinants=log_determinants,
                scaled_table_cholesky_ltriangular_mat=(
                    scaled_table_cholesky_ltriangular_mat
                ),
            )
        return output

    return np.float64(
        log_multivariate_tdensity_single_kernel(
            arr,
            table_id=table_id,
            embedding_size=embedding_size,
            nu=nu,
            table_means=table_means,
            log_determinants=log_determinants,
            scaled_table_cholesky_ltriangular_mat=(
                scaled_table_cholesky_ltriangular_mat
            ),
        )
    )


def log_multivariate_tdensity_tables(
    x: np.ndarray,
    *,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> np.ndarray:
    return log_multivariate_tdensity_tables_kernel(
        np.asarray(x, dtype=np.float64),
        embedding_size=embedding_size,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled_table_cholesky_ltriangular_mat,
    )


def sample_topic_assignment(
    counts: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    alpha: float,
    uniform: float,
) -> int:
    return sample_topic_assignment_kernel(
        counts=np.asarray(counts),
        log_likelihoods=np.asarray(log_likelihoods, dtype=np.float64),
        alpha=float(alpha),
        uniform=float(uniform),
    )


def sample_doc_topic_assignments(
    assignments: np.ndarray,
    counts: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    alpha: float,
    uniforms: np.ndarray,
) -> None:
    sample_doc_topic_assignments_kernel(
        assignments=np.asarray(assignments),
        counts=np.asarray(counts),
        log_likelihoods=np.asarray(log_likelihoods, dtype=np.float64),
        alpha=float(alpha),
        uniforms=np.asarray(uniforms, dtype=np.float64),
    )


def accumulate_gaussian_log_likelihood_words(
    doc_words: np.ndarray,
    assignments: np.ndarray,
    embeddings: np.ndarray,
    *,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> tuple[float, int]:
    return accumulate_gaussian_log_likelihood_words_kernel(
        doc_words=np.asarray(doc_words),
        assignments=np.asarray(assignments),
        embeddings=np.asarray(embeddings, dtype=np.float64),
        embedding_size=int(embedding_size),
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled_table_cholesky_ltriangular_mat,
    )


def accumulate_gaussian_log_likelihood_encoded(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    *,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> tuple[float, int]:
    return accumulate_gaussian_log_likelihood_encoded_kernel(
        encoded_doc=np.asarray(encoded_doc, dtype=np.float64),
        assignments=np.asarray(assignments),
        embedding_size=int(embedding_size),
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled_table_cholesky_ltriangular_mat,
    )


__all__ = [
    "GAUSSIAN_AVG_LL_BACKEND",
    "GAUSSIAN_POSTERIOR_SAMPLING_BACKEND",
    "GAUSSIAN_TABLE_DENSITY_BACKEND",
    "accumulate_gaussian_log_likelihood_encoded",
    "accumulate_gaussian_log_likelihood_words",
    "build_gaussian_nu",
    "build_scaled_cholesky",
    "log_multivariate_tdensity",
    "log_multivariate_tdensity_tables",
    "sample_doc_topic_assignments",
    "sample_topic_assignment",
]

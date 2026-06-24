from __future__ import annotations

import numpy as np

from src.baselines.models import gaussian_numba
from src.baselines.models.gaussian_numerics import (
    GAUSSIAN_AVG_LL_BACKEND,
    GAUSSIAN_POSTERIOR_SAMPLING_BACKEND,
    GAUSSIAN_TABLE_DENSITY_BACKEND,
    accumulate_gaussian_log_likelihood_encoded,
    accumulate_gaussian_log_likelihood_words,
    build_gaussian_nu,
    build_scaled_cholesky,
    log_multivariate_tdensity,
    log_multivariate_tdensity_tables,
    sample_doc_topic_assignments,
    sample_topic_assignment,
)


def test_gaussian_numerics_builds_shared_posterior_terms() -> None:
    counts = np.asarray([2.0, 4.0], dtype=np.float64)
    chol = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=np.float64,
    )

    nu = build_gaussian_nu(table_counts=counts, embedding_size=2)
    scaled = build_scaled_cholesky(
        table_counts=counts,
        kappa=0.1,
        embedding_size=2,
        table_cholesky_ltriangular_mat=chol,
    )

    assert np.allclose(nu, np.asarray([3.0, 5.0]))
    assert scaled.shape == chol.shape
    assert np.all(np.isfinite(scaled))


def test_gaussian_numerics_match_single_and_multi_table_density_calls() -> None:
    table_means = np.asarray([[0.5, 0.5], [0.2, 0.8]], dtype=np.float64)
    log_determinants = np.asarray([0.0, 0.0], dtype=np.float64)
    scaled = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    nu = np.asarray([3.0, 4.0], dtype=np.float64)
    x = np.asarray([1.0, 0.0], dtype=np.float64)

    all_scores = log_multivariate_tdensity_tables(
        x,
        embedding_size=2,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled,
    )
    single_score = log_multivariate_tdensity(
        x,
        table_id=0,
        embedding_size=2,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled,
    )

    assert all_scores.shape == (2,)
    assert np.isfinite(all_scores).all()
    assert np.isclose(single_score, all_scores[0])


def test_gaussian_numerics_backend_labels_are_known() -> None:
    assert GAUSSIAN_TABLE_DENSITY_BACKEND in {"python", "numba"}
    assert GAUSSIAN_POSTERIOR_SAMPLING_BACKEND in {"python", "numba"}
    assert GAUSSIAN_AVG_LL_BACKEND in {"python", "numba"}


def test_gaussian_numerics_samples_topic_from_log_posteriors() -> None:
    sampled = sample_topic_assignment(
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([0.0, -100.0], dtype=np.float64),
        alpha=0.1,
        uniform=0.5,
    )

    assert sampled == 0


def test_gaussian_numerics_samples_full_doc_assignments_in_place() -> None:
    assignments = np.asarray([0, 1], dtype=np.int32)
    counts = np.asarray([1, 1], dtype=np.int32)
    log_likelihoods = np.asarray(
        [
            [0.0, -100.0],
            [-100.0, 0.0],
        ],
        dtype=np.float64,
    )
    uniforms = np.asarray([0.5, 0.5], dtype=np.float64)

    sample_doc_topic_assignments(
        assignments,
        counts,
        log_likelihoods,
        alpha=0.1,
        uniforms=uniforms,
    )

    assert np.array_equal(assignments, np.asarray([0, 1], dtype=np.int32))
    assert np.array_equal(counts, np.asarray([1, 1], dtype=np.int32))


def test_gaussian_numerics_accumulate_encoded_and_indexed_docs_consistently() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assignments = np.asarray([0, 0], dtype=np.int64)
    table_means = np.asarray([[0.5, 0.5]], dtype=np.float64)
    log_determinants = np.asarray([0.0], dtype=np.float64)
    scaled = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    nu = np.asarray([3.0], dtype=np.float64)

    indexed_ll, indexed_count = accumulate_gaussian_log_likelihood_words(
        np.asarray([0, 1], dtype=np.int64),
        assignments,
        embeddings,
        embedding_size=2,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled,
    )
    encoded_ll, encoded_count = accumulate_gaussian_log_likelihood_encoded(
        embeddings,
        assignments,
        embedding_size=2,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled,
    )

    assert indexed_count == 2
    assert encoded_count == 2
    assert np.isclose(indexed_ll, encoded_ll)


def test_gaussian_numba_table_density_matches_python_reference() -> None:
    x = np.asarray([1.0, 0.0], dtype=np.float64)
    nu = np.asarray([3.0, 4.0], dtype=np.float64)
    table_means = np.asarray([[0.5, 0.5], [0.2, 0.8]], dtype=np.float64)
    log_determinants = np.asarray([0.0, 0.0], dtype=np.float64)
    scaled = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )

    expected = gaussian_numba._log_multivariate_tdensity_tables_python(
        x,
        embedding_size=2,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled,
    )
    actual = gaussian_numba.log_multivariate_tdensity_tables_kernel(
        x,
        embedding_size=2,
        nu=nu,
        table_means=table_means,
        log_determinants=log_determinants,
        scaled_table_cholesky_ltriangular_mat=scaled,
    )

    assert np.allclose(actual, expected)

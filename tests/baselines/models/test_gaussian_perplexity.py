from __future__ import annotations

import numpy as np

from src.baselines.models.gaussian_internal import perplexity
from src.baselines.models.gaussian_numerics import (
    accumulate_gaussian_log_likelihood_encoded,
    accumulate_gaussian_log_likelihood_words,
    build_gaussian_nu,
    build_scaled_cholesky,
)


class _Prior:
    def __init__(self, kappa: float, nu: float) -> None:
        self.kappa = float(kappa)
        self.nu = float(nu)


def test_gaussian_avg_ll_reuses_word_table_density_cache(monkeypatch) -> None:
    corpus = [[0, 0, 0]]
    table_assignments = [[0, 0, 0]]
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float64)
    table_means = np.asarray([[0.5, 0.5]], dtype=np.float64)
    table_cholesky = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    table_counts_per_doc = np.asarray([[3]], dtype=np.int32)
    prior = _Prior(kappa=0.1, nu=2.0)

    calls = {"count": 0}
    original = perplexity.log_multivariate_tdensity

    def _wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(perplexity, "log_multivariate_tdensity", _wrapped)

    avg_ll = perplexity.calculate_gaussianlda_avg_ll(
        corpus,
        table_assignments,
        embeddings,
        table_means,
        table_cholesky,
        prior,
        table_counts_per_doc,
    )

    assert np.isfinite(avg_ll)
    assert calls["count"] == 1


def test_gaussian_avg_ll_keeps_log_likelihood_sign() -> None:
    corpus = [[0, 1]]
    table_assignments = [[0, 0]]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    table_means = np.asarray([[0.5, 0.5]], dtype=np.float64)
    table_cholesky = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    table_counts_per_doc = np.asarray([[2]], dtype=np.int32)
    prior = _Prior(kappa=0.1, nu=2.0)

    table_counts = table_counts_per_doc.sum(axis=1)
    nu = build_gaussian_nu(table_counts=table_counts, embedding_size=2)
    scaled_cholesky = build_scaled_cholesky(
        table_counts=table_counts,
        kappa=prior.kappa,
        embedding_size=2,
        table_cholesky_ltriangular_mat=table_cholesky,
    )
    doc_ll, doc_count = accumulate_gaussian_log_likelihood_words(
        np.asarray(corpus[0], dtype=np.int64),
        np.asarray(table_assignments[0], dtype=np.int64),
        embeddings,
        embedding_size=2,
        nu=nu,
        table_means=table_means,
        log_determinants=perplexity.log_multivariate_log_determinants(scaled_cholesky),
        scaled_table_cholesky_ltriangular_mat=scaled_cholesky,
    )

    avg_ll = perplexity.calculate_gaussianlda_avg_ll(
        corpus,
        table_assignments,
        embeddings,
        table_means,
        table_cholesky,
        prior,
        table_counts_per_doc,
    )

    assert np.isclose(avg_ll, doc_ll / doc_count)


def test_sentence_gaussian_avg_ll_keeps_log_likelihood_sign() -> None:
    encoded_corpus = [
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    ]
    table_assignments = [[0, 0]]
    table_means = np.asarray([[0.5, 0.5]], dtype=np.float64)
    table_cholesky = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    table_counts_per_doc = np.asarray([[2]], dtype=np.int32)
    prior = _Prior(kappa=0.1, nu=2.0)

    table_counts = table_counts_per_doc.sum(axis=1)
    nu = build_gaussian_nu(table_counts=table_counts, embedding_size=2)
    scaled_cholesky = build_scaled_cholesky(
        table_counts=table_counts,
        kappa=prior.kappa,
        embedding_size=2,
        table_cholesky_ltriangular_mat=table_cholesky,
    )
    doc_ll, doc_count = accumulate_gaussian_log_likelihood_encoded(
        encoded_corpus[0],
        np.asarray(table_assignments[0], dtype=np.int64),
        embedding_size=2,
        nu=nu,
        table_means=table_means,
        log_determinants=perplexity.log_multivariate_log_determinants(scaled_cholesky),
        scaled_table_cholesky_ltriangular_mat=scaled_cholesky,
    )

    avg_ll = perplexity.calculate_sentence_gaussianlda_avg_ll_from_encoded(
        encoded_corpus,
        table_assignments,
        table_means,
        table_cholesky,
        prior,
        table_counts_per_doc,
    )

    assert np.isclose(avg_ll, doc_ll / doc_count)

from __future__ import annotations

import numpy as np

from src.baselines.models.gaussian_numerics import (
    accumulate_gaussian_log_likelihood_encoded,
    build_gaussian_nu,
    build_scaled_cholesky,
    log_multivariate_tdensity,
)


def calculate_gaussianlda_avg_ll(
    corpus,
    table_assignments,
    embeddings,
    table_means,
    table_cholesky_ltriangular_mat,
    prior,
    table_counts_per_doc,
):
    embedding_size = embeddings.shape[1]
    table_counts = table_counts_per_doc.sum(axis=1)
    nu = build_gaussian_nu(
        table_counts=table_counts,
        embedding_size=embedding_size,
    )
    scaled_choleskies = build_scaled_cholesky(
        table_counts=table_counts,
        kappa=prior.kappa,
        embedding_size=embedding_size,
        table_cholesky_ltriangular_mat=table_cholesky_ltriangular_mat,
    )
    log_det = np.asarray(
        log_multivariate_log_determinants(scaled_choleskies),
        dtype=np.float64,
    )
    log_density_cache: dict[tuple[int, int], float] = {}
    total_log_ll = 0.0
    total_words = 0
    for doc, tables in zip(corpus, table_assignments):
        doc_words = np.asarray(doc, dtype=np.int64)
        doc_tables = np.asarray(tables, dtype=np.int64)
        limit = min(int(doc_words.shape[0]), int(doc_tables.shape[0]))
        if limit <= 0:
            continue
        doc_ll = 0.0
        doc_count = 0
        miss_multiplicity: dict[tuple[int, int], int] = {}
        for index in range(limit):
            table_id = int(doc_tables[index])
            word_id = int(doc_words[index])
            cached = log_density_cache.get((table_id, word_id))
            if cached is None:
                key = (table_id, word_id)
                miss_multiplicity[key] = miss_multiplicity.get(key, 0) + 1
                continue
            doc_ll += cached
            doc_count += 1
        if miss_multiplicity:
            for table_id, word_id in miss_multiplicity:
                value = float(
                    log_multivariate_tdensity(
                        embeddings[word_id],
                        table_id=table_id,
                        embedding_size=embedding_size,
                        nu=nu,
                        table_means=table_means,
                        log_determinants=log_det,
                        scaled_table_cholesky_ltriangular_mat=scaled_choleskies,
                    )
                )
                log_density_cache[(table_id, word_id)] = value
                doc_ll += value * int(miss_multiplicity[(table_id, word_id)])
            doc_count += int(sum(miss_multiplicity.values()))
        total_log_ll += float(doc_ll)
        total_words += int(doc_count)

    return total_log_ll / total_words


def calculate_sentence_gaussianlda_avg_ll(
    corpus,
    table_assignments,
    embeddings,
    table_means,
    table_cholesky_ltriangular_mat,
    prior,
    table_counts_per_doc,
):
    embedding_size = embeddings.get_sentence_embedding_dimension()
    table_counts = table_counts_per_doc.sum(axis=1)
    nu = build_gaussian_nu(
        table_counts=table_counts,
        embedding_size=embedding_size,
    )
    scaled_choleskies = build_scaled_cholesky(
        table_counts=table_counts,
        kappa=prior.kappa,
        embedding_size=embedding_size,
        table_cholesky_ltriangular_mat=table_cholesky_ltriangular_mat,
    )
    log_det = np.asarray(
        log_multivariate_log_determinants(scaled_choleskies),
        dtype=np.float64,
    )

    total_log_ll = 0.0
    total_sentences = 0
    for doc, tables in zip(corpus, table_assignments):
        sentences = embeddings.encode(doc)
        doc_ll, doc_count = accumulate_gaussian_log_likelihood_encoded(
            np.asarray(sentences, dtype=np.float64),
            np.asarray(tables, dtype=np.int64),
            embedding_size=embedding_size,
            nu=nu,
            table_means=table_means,
            log_determinants=log_det,
            scaled_table_cholesky_ltriangular_mat=scaled_choleskies,
        )
        total_log_ll += float(doc_ll)
        total_sentences += int(doc_count)

    return total_log_ll / total_sentences


def calculate_sentence_gaussianlda_avg_ll_from_encoded(
    encoded_corpus,
    table_assignments,
    table_means,
    table_cholesky_ltriangular_mat,
    prior,
    table_counts_per_doc,
) -> float:
    embedding_size = table_means.shape[1]
    table_counts = table_counts_per_doc.sum(axis=1)
    nu = build_gaussian_nu(
        table_counts=table_counts,
        embedding_size=embedding_size,
    )
    scaled_choleskies = build_scaled_cholesky(
        table_counts=table_counts,
        kappa=prior.kappa,
        embedding_size=embedding_size,
        table_cholesky_ltriangular_mat=table_cholesky_ltriangular_mat,
    )
    log_det = np.asarray(
        log_multivariate_log_determinants(scaled_choleskies),
        dtype=np.float64,
    )

    total_log_ll = 0.0
    total_sentences = 0
    for doc, tables in zip(encoded_corpus, table_assignments):
        doc_ll, doc_count = accumulate_gaussian_log_likelihood_encoded(
            np.asarray(doc, dtype=np.float64),
            np.asarray(tables, dtype=np.int64),
            embedding_size=embedding_size,
            nu=nu,
            table_means=table_means,
            log_determinants=log_det,
            scaled_table_cholesky_ltriangular_mat=scaled_choleskies,
        )
        total_log_ll += float(doc_ll)
        total_sentences += int(doc_count)

    return total_log_ll / total_sentences


def log_multivariate_log_determinants(
    scaled_choleskies: np.ndarray,
) -> np.ndarray:
    log_det = np.zeros(scaled_choleskies.shape[0], dtype=np.float64)
    for table in range(scaled_choleskies.shape[0]):
        log_det[table] = np.sum(np.log(np.diagonal(scaled_choleskies[table])))
    return log_det

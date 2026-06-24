from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency fallback
    njit = None
    NUMBA_AVAILABLE = False


def _log_multivariate_tdensity_single_python(
    x: np.ndarray,
    *,
    table_id: int,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> float:
    arr = np.asarray(x, dtype=np.float64)
    means = np.asarray(table_means, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)
    log_det = np.asarray(log_determinants, dtype=np.float64)
    scaled_chol = np.asarray(
        scaled_table_cholesky_ltriangular_mat,
        dtype=np.float64,
    )

    rhs = arr - means[table_id]
    solved = np.empty(embedding_size, dtype=np.float64)
    for row in range(embedding_size):
        value = float(rhs[row])
        for col in range(row):
            value -= float(scaled_chol[table_id, row, col]) * float(solved[col])
        solved[row] = value / float(scaled_chol[table_id, row, row])

    val = float(np.sum(solved**2.0))
    nu_value = float(nu_arr[table_id])
    return math.lgamma((nu_value + embedding_size) / 2.0) - (
        math.lgamma(nu_value / 2.0)
        + embedding_size / 2.0 * (math.log(nu_value) + math.log(math.pi))
        + float(log_det[table_id])
        + (nu_value + embedding_size) / 2.0 * math.log1p(val / nu_value)
    )


def _log_multivariate_tdensity_tables_python(
    x: np.ndarray,
    *,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> np.ndarray:
    means = np.asarray(table_means, dtype=np.float64)
    output = np.empty(means.shape[0], dtype=np.float64)
    for table_id in range(means.shape[0]):
        output[table_id] = _log_multivariate_tdensity_single_python(
            x,
            table_id=table_id,
            embedding_size=embedding_size,
            nu=nu,
            table_means=means,
            log_determinants=log_determinants,
            scaled_table_cholesky_ltriangular_mat=(
                scaled_table_cholesky_ltriangular_mat
            ),
        )
    return output


def _sample_topic_assignment_python(
    counts: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    alpha: float,
    uniform: float,
) -> int:
    counts_arr = np.asarray(counts, dtype=np.float64)
    log_lik = np.asarray(log_likelihoods, dtype=np.float64)
    num_tables = int(counts_arr.shape[0])

    log_post = np.empty(num_tables, dtype=np.float64)
    max_log = -1.0e300
    for table_id in range(num_tables):
        value = math.log(float(counts_arr[table_id]) + float(alpha)) + float(
            log_lik[table_id]
        )
        log_post[table_id] = value
        if value > max_log:
            max_log = value

    post_sum = 0.0
    for table_id in range(num_tables):
        value = math.exp(log_post[table_id] - max_log)
        log_post[table_id] = value
        post_sum += value

    if (not math.isfinite(post_sum)) or post_sum <= 0.0:
        return min(int(uniform * num_tables), num_tables - 1)

    threshold = float(uniform) * post_sum
    cumsum = 0.0
    for table_id in range(num_tables):
        cumsum += float(log_post[table_id])
        if threshold <= cumsum:
            return table_id
    return num_tables - 1


def _sample_doc_topic_assignments_python(
    assignments: np.ndarray,
    counts: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    alpha: float,
    uniforms: np.ndarray,
) -> None:
    num_tables = int(counts.shape[0])
    limit = min(
        int(assignments.shape[0]),
        int(log_likelihoods.shape[0]),
        int(uniforms.shape[0]),
    )
    for index in range(limit):
        old_table_id = int(assignments[index])
        if 0 <= old_table_id < num_tables:
            counts[old_table_id] -= 1
        assignments[index] = _sample_topic_assignment_python(
            counts=counts,
            log_likelihoods=log_likelihoods[index],
            alpha=alpha,
            uniform=float(uniforms[index]),
        )
        counts[int(assignments[index])] += 1


def _accumulate_gaussian_log_likelihood_words_python(
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
    total_log_ll = 0.0
    total_words = 0
    limit = min(int(doc_words.shape[0]), int(assignments.shape[0]))
    for index in range(limit):
        table_id = int(assignments[index])
        if table_id < 0 or table_id >= int(table_means.shape[0]):
            continue
        word_id = int(doc_words[index])
        total_log_ll += _log_multivariate_tdensity_single_python(
            embeddings[word_id],
            table_id=table_id,
            embedding_size=embedding_size,
            nu=nu,
            table_means=table_means,
            log_determinants=log_determinants,
            scaled_table_cholesky_ltriangular_mat=(
                scaled_table_cholesky_ltriangular_mat
            ),
        )
        total_words += 1
    return total_log_ll, total_words


def _accumulate_gaussian_log_likelihood_encoded_python(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    *,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> tuple[float, int]:
    total_log_ll = 0.0
    total_items = 0
    limit = min(int(encoded_doc.shape[0]), int(assignments.shape[0]))
    for index in range(limit):
        table_id = int(assignments[index])
        if table_id < 0 or table_id >= int(table_means.shape[0]):
            continue
        total_log_ll += _log_multivariate_tdensity_single_python(
            encoded_doc[index],
            table_id=table_id,
            embedding_size=embedding_size,
            nu=nu,
            table_means=table_means,
            log_determinants=log_determinants,
            scaled_table_cholesky_ltriangular_mat=(
                scaled_table_cholesky_ltriangular_mat
            ),
        )
        total_items += 1
    return total_log_ll, total_items


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _log_multivariate_tdensity_single_numba(
        x: np.ndarray,
        table_id: int,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_table_cholesky_ltriangular_mat: np.ndarray,
    ) -> float:
        solved = np.empty(embedding_size, dtype=np.float64)
        for row in range(embedding_size):
            value = float(x[row]) - float(table_means[table_id, row])
            for col in range(row):
                value -= float(
                    scaled_table_cholesky_ltriangular_mat[table_id, row, col]
                ) * float(solved[col])
            solved[row] = value / float(
                scaled_table_cholesky_ltriangular_mat[table_id, row, row]
            )

        val = 0.0
        for row in range(embedding_size):
            val += float(solved[row]) * float(solved[row])
        nu_value = float(nu[table_id])
        return math.lgamma((nu_value + embedding_size) / 2.0) - (
            math.lgamma(nu_value / 2.0)
            + embedding_size / 2.0 * (math.log(nu_value) + math.log(math.pi))
            + float(log_determinants[table_id])
            + (nu_value + embedding_size) / 2.0 * math.log1p(val / nu_value)
        )

    @njit(cache=True)
    def _log_multivariate_tdensity_tables_numba(
        x: np.ndarray,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_table_cholesky_ltriangular_mat: np.ndarray,
    ) -> np.ndarray:
        num_tables = table_means.shape[0]
        output = np.empty(num_tables, dtype=np.float64)
        for table_id in range(num_tables):
            output[table_id] = _log_multivariate_tdensity_single_numba(
                x=x,
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

    @njit(cache=True)
    def _sample_topic_assignment_numba(
        counts: np.ndarray,
        log_likelihoods: np.ndarray,
        alpha: float,
        uniform: float,
    ) -> int:
        num_tables = counts.shape[0]
        log_post = np.empty(num_tables, dtype=np.float64)
        max_log = -1.0e300
        for table_id in range(num_tables):
            value = math.log(float(counts[table_id]) + float(alpha)) + float(
                log_likelihoods[table_id]
            )
            log_post[table_id] = value
            if value > max_log:
                max_log = value

        post_sum = 0.0
        for table_id in range(num_tables):
            value = math.exp(log_post[table_id] - max_log)
            log_post[table_id] = value
            post_sum += value

        if (not math.isfinite(post_sum)) or post_sum <= 0.0:
            return min(int(uniform * num_tables), num_tables - 1)

        threshold = float(uniform) * post_sum
        cumsum = 0.0
        new_table_id = num_tables - 1
        for table_id in range(num_tables):
            cumsum += float(log_post[table_id])
            if threshold <= cumsum:
                new_table_id = table_id
                break
        return new_table_id

    @njit(cache=True)
    def _sample_doc_topic_assignments_numba(
        assignments: np.ndarray,
        counts: np.ndarray,
        log_likelihoods: np.ndarray,
        alpha: float,
        uniforms: np.ndarray,
    ) -> None:
        num_tables = counts.shape[0]
        limit = assignments.shape[0]
        if log_likelihoods.shape[0] < limit:
            limit = log_likelihoods.shape[0]
        if uniforms.shape[0] < limit:
            limit = uniforms.shape[0]
        for index in range(limit):
            old_table_id = int(assignments[index])
            if 0 <= old_table_id < num_tables:
                counts[old_table_id] -= 1
            assignments[index] = _sample_topic_assignment_numba(
                counts=counts,
                log_likelihoods=log_likelihoods[index],
                alpha=alpha,
                uniform=float(uniforms[index]),
            )
            counts[int(assignments[index])] += 1

    @njit(cache=True)
    def _accumulate_gaussian_log_likelihood_words_numba(
        doc_words: np.ndarray,
        assignments: np.ndarray,
        embeddings: np.ndarray,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_table_cholesky_ltriangular_mat: np.ndarray,
    ) -> tuple[float, int]:
        total_log_ll = 0.0
        total_words = 0
        limit = doc_words.shape[0]
        if assignments.shape[0] < limit:
            limit = assignments.shape[0]
        num_tables = table_means.shape[0]
        for index in range(limit):
            table_id = int(assignments[index])
            if table_id < 0 or table_id >= num_tables:
                continue
            total_log_ll += _log_multivariate_tdensity_single_numba(
                x=embeddings[int(doc_words[index])],
                table_id=table_id,
                embedding_size=embedding_size,
                nu=nu,
                table_means=table_means,
                log_determinants=log_determinants,
                scaled_table_cholesky_ltriangular_mat=(
                    scaled_table_cholesky_ltriangular_mat
                ),
            )
            total_words += 1
        return total_log_ll, total_words

    @njit(cache=True)
    def _accumulate_gaussian_log_likelihood_encoded_numba(
        encoded_doc: np.ndarray,
        assignments: np.ndarray,
        embedding_size: int,
        nu: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        scaled_table_cholesky_ltriangular_mat: np.ndarray,
    ) -> tuple[float, int]:
        total_log_ll = 0.0
        total_items = 0
        limit = encoded_doc.shape[0]
        if assignments.shape[0] < limit:
            limit = assignments.shape[0]
        num_tables = table_means.shape[0]
        for index in range(limit):
            table_id = int(assignments[index])
            if table_id < 0 or table_id >= num_tables:
                continue
            total_log_ll += _log_multivariate_tdensity_single_numba(
                x=encoded_doc[index],
                table_id=table_id,
                embedding_size=embedding_size,
                nu=nu,
                table_means=table_means,
                log_determinants=log_determinants,
                scaled_table_cholesky_ltriangular_mat=(
                    scaled_table_cholesky_ltriangular_mat
                ),
            )
            total_items += 1
        return total_log_ll, total_items

else:
    _log_multivariate_tdensity_single_numba = None
    _log_multivariate_tdensity_tables_numba = None
    _sample_topic_assignment_numba = None
    _sample_doc_topic_assignments_numba = None
    _accumulate_gaussian_log_likelihood_words_numba = None
    _accumulate_gaussian_log_likelihood_encoded_numba = None


def log_multivariate_tdensity_single_kernel(
    x: np.ndarray,
    *,
    table_id: int,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> float:
    if NUMBA_AVAILABLE:
        return float(
            _log_multivariate_tdensity_single_numba(
                x=np.asarray(x, dtype=np.float64),
                table_id=int(table_id),
                embedding_size=int(embedding_size),
                nu=np.asarray(nu, dtype=np.float64),
                table_means=np.asarray(table_means, dtype=np.float64),
                log_determinants=np.asarray(log_determinants, dtype=np.float64),
                scaled_table_cholesky_ltriangular_mat=np.asarray(
                    scaled_table_cholesky_ltriangular_mat,
                    dtype=np.float64,
                ),
            )
        )
    return _log_multivariate_tdensity_single_python(
        x=np.asarray(x, dtype=np.float64),
        table_id=int(table_id),
        embedding_size=int(embedding_size),
        nu=np.asarray(nu, dtype=np.float64),
        table_means=np.asarray(table_means, dtype=np.float64),
        log_determinants=np.asarray(log_determinants, dtype=np.float64),
        scaled_table_cholesky_ltriangular_mat=np.asarray(
            scaled_table_cholesky_ltriangular_mat,
            dtype=np.float64,
        ),
    )


def log_multivariate_tdensity_tables_kernel(
    x: np.ndarray,
    *,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> np.ndarray:
    if NUMBA_AVAILABLE:
        return _log_multivariate_tdensity_tables_numba(
            x=np.asarray(x, dtype=np.float64),
            embedding_size=int(embedding_size),
            nu=np.asarray(nu, dtype=np.float64),
            table_means=np.asarray(table_means, dtype=np.float64),
            log_determinants=np.asarray(log_determinants, dtype=np.float64),
            scaled_table_cholesky_ltriangular_mat=np.asarray(
                scaled_table_cholesky_ltriangular_mat,
                dtype=np.float64,
            ),
        )
    return _log_multivariate_tdensity_tables_python(
        x=np.asarray(x, dtype=np.float64),
        embedding_size=int(embedding_size),
        nu=np.asarray(nu, dtype=np.float64),
        table_means=np.asarray(table_means, dtype=np.float64),
        log_determinants=np.asarray(log_determinants, dtype=np.float64),
        scaled_table_cholesky_ltriangular_mat=np.asarray(
            scaled_table_cholesky_ltriangular_mat,
            dtype=np.float64,
        ),
    )


def sample_topic_assignment_kernel(
    counts: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    alpha: float,
    uniform: float,
) -> int:
    if NUMBA_AVAILABLE:
        return int(
            _sample_topic_assignment_numba(
                counts=np.asarray(counts, dtype=np.float64),
                log_likelihoods=np.asarray(log_likelihoods, dtype=np.float64),
                alpha=float(alpha),
                uniform=float(uniform),
            )
        )
    return _sample_topic_assignment_python(
        counts=np.asarray(counts, dtype=np.float64),
        log_likelihoods=np.asarray(log_likelihoods, dtype=np.float64),
        alpha=float(alpha),
        uniform=float(uniform),
    )


def sample_doc_topic_assignments_kernel(
    assignments: np.ndarray,
    counts: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    alpha: float,
    uniforms: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _sample_doc_topic_assignments_numba(
            assignments=np.asarray(assignments, dtype=np.int32),
            counts=np.asarray(counts, dtype=np.int32),
            log_likelihoods=np.asarray(log_likelihoods, dtype=np.float64),
            alpha=float(alpha),
            uniforms=np.asarray(uniforms, dtype=np.float64),
        )
        return
    _sample_doc_topic_assignments_python(
        assignments=np.asarray(assignments, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.int32),
        log_likelihoods=np.asarray(log_likelihoods, dtype=np.float64),
        alpha=float(alpha),
        uniforms=np.asarray(uniforms, dtype=np.float64),
    )


def accumulate_gaussian_log_likelihood_words_kernel(
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
    if NUMBA_AVAILABLE:
        return _accumulate_gaussian_log_likelihood_words_numba(
            doc_words=np.asarray(doc_words, dtype=np.int64),
            assignments=np.asarray(assignments, dtype=np.int64),
            embeddings=np.asarray(embeddings, dtype=np.float64),
            embedding_size=int(embedding_size),
            nu=np.asarray(nu, dtype=np.float64),
            table_means=np.asarray(table_means, dtype=np.float64),
            log_determinants=np.asarray(log_determinants, dtype=np.float64),
            scaled_table_cholesky_ltriangular_mat=np.asarray(
                scaled_table_cholesky_ltriangular_mat,
                dtype=np.float64,
            ),
        )
    return _accumulate_gaussian_log_likelihood_words_python(
        doc_words=np.asarray(doc_words, dtype=np.int64),
        assignments=np.asarray(assignments, dtype=np.int64),
        embeddings=np.asarray(embeddings, dtype=np.float64),
        embedding_size=int(embedding_size),
        nu=np.asarray(nu, dtype=np.float64),
        table_means=np.asarray(table_means, dtype=np.float64),
        log_determinants=np.asarray(log_determinants, dtype=np.float64),
        scaled_table_cholesky_ltriangular_mat=np.asarray(
            scaled_table_cholesky_ltriangular_mat,
            dtype=np.float64,
        ),
    )


def accumulate_gaussian_log_likelihood_encoded_kernel(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    *,
    embedding_size: int,
    nu: np.ndarray,
    table_means: np.ndarray,
    log_determinants: np.ndarray,
    scaled_table_cholesky_ltriangular_mat: np.ndarray,
) -> tuple[float, int]:
    if NUMBA_AVAILABLE:
        return _accumulate_gaussian_log_likelihood_encoded_numba(
            encoded_doc=np.asarray(encoded_doc, dtype=np.float64),
            assignments=np.asarray(assignments, dtype=np.int64),
            embedding_size=int(embedding_size),
            nu=np.asarray(nu, dtype=np.float64),
            table_means=np.asarray(table_means, dtype=np.float64),
            log_determinants=np.asarray(log_determinants, dtype=np.float64),
            scaled_table_cholesky_ltriangular_mat=np.asarray(
                scaled_table_cholesky_ltriangular_mat,
                dtype=np.float64,
            ),
        )
    return _accumulate_gaussian_log_likelihood_encoded_python(
        encoded_doc=np.asarray(encoded_doc, dtype=np.float64),
        assignments=np.asarray(assignments, dtype=np.int64),
        embedding_size=int(embedding_size),
        nu=np.asarray(nu, dtype=np.float64),
        table_means=np.asarray(table_means, dtype=np.float64),
        log_determinants=np.asarray(log_determinants, dtype=np.float64),
        scaled_table_cholesky_ltriangular_mat=np.asarray(
            scaled_table_cholesky_ltriangular_mat,
            dtype=np.float64,
        ),
    )


GAUSSIAN_TABLE_DENSITY_BACKEND = "numba" if NUMBA_AVAILABLE else "python"
GAUSSIAN_POSTERIOR_SAMPLING_BACKEND = "numba" if NUMBA_AVAILABLE else "python"
GAUSSIAN_AVG_LL_BACKEND = "numba" if NUMBA_AVAILABLE else "python"

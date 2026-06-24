from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency fallback
    njit = None
    NUMBA_AVAILABLE = False


def _sample_doc_topic_assignments_python(
    assignments: np.ndarray,
    counts: np.ndarray,
    log_lik_doc: np.ndarray,
    alpha: np.ndarray,
    uniforms: np.ndarray,
) -> None:
    num_topics = int(counts.shape[0])
    for i in range(assignments.shape[0]):
        old_k = int(assignments[i])
        if 0 <= old_k < num_topics:
            counts[old_k] -= 1

        log_post = np.log(counts.astype(np.float64) + alpha) + log_lik_doc[i]
        log_post -= log_post.max()
        post = np.exp(log_post)
        post_sum = post.sum()
        if not np.isfinite(post_sum) or post_sum <= 0.0:
            new_k = min(int(uniforms[i] * num_topics), num_topics - 1)
        else:
            threshold = float(uniforms[i]) * float(post_sum)
            cumsum = 0.0
            new_k = num_topics - 1
            for k in range(num_topics):
                cumsum += float(post[k])
                if threshold <= cumsum:
                    new_k = k
                    break

        assignments[i] = new_k
        counts[new_k] += 1


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _sample_doc_topic_assignments_numba(
        assignments: np.ndarray,
        counts: np.ndarray,
        log_lik_doc: np.ndarray,
        alpha: np.ndarray,
        uniforms: np.ndarray,
    ) -> None:
        num_topics = counts.shape[0]
        log_post = np.empty(num_topics, dtype=np.float64)

        for i in range(assignments.shape[0]):
            old_k = int(assignments[i])
            if 0 <= old_k < num_topics:
                counts[old_k] -= 1

            max_log = -1.0e300
            for k in range(num_topics):
                value = math.log(float(counts[k]) + float(alpha[k])) + float(
                    log_lik_doc[i, k]
                )
                log_post[k] = value
                if value > max_log:
                    max_log = value

            post_sum = 0.0
            for k in range(num_topics):
                value = math.exp(log_post[k] - max_log)
                log_post[k] = value
                post_sum += value

            if (not math.isfinite(post_sum)) or post_sum <= 0.0:
                new_k = min(int(uniforms[i] * num_topics), num_topics - 1)
            else:
                threshold = float(uniforms[i]) * post_sum
                cumsum = 0.0
                new_k = num_topics - 1
                for k in range(num_topics):
                    cumsum += log_post[k]
                    if threshold <= cumsum:
                        new_k = k
                        break

            assignments[i] = new_k
            counts[new_k] += 1

else:
    _sample_doc_topic_assignments_numba = None


def sample_doc_topic_assignments(
    assignments: np.ndarray,
    counts: np.ndarray,
    log_lik_doc: np.ndarray,
    alpha: np.ndarray,
    uniforms: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _sample_doc_topic_assignments_numba(
            assignments=assignments,
            counts=counts,
            log_lik_doc=log_lik_doc,
            alpha=alpha,
            uniforms=uniforms,
        )
        return
    _sample_doc_topic_assignments_python(
        assignments=assignments,
        counts=counts,
        log_lik_doc=log_lik_doc,
        alpha=alpha,
        uniforms=uniforms,
    )


SAMPLE_DOC_TOPIC_ASSIGNMENTS_BACKEND = "numba" if NUMBA_AVAILABLE else "python"


def _accumulate_doc_assignment_statistics_python(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    log_mixture_weights: np.ndarray,
    scaled_component_means: np.ndarray,
    nk: np.ndarray,
    nk_comp: np.ndarray,
    r: np.ndarray,
) -> None:
    num_topics = int(nk.shape[0])
    num_components = int(nk_comp.shape[1])
    limit = min(int(encoded_doc.shape[0]), int(assignments.shape[0]))
    if num_components == 1:
        for i in range(limit):
            k = int(assignments[i])
            if k < 0 or k >= num_topics:
                continue
            nk[k] += 1.0
            nk_comp[k, 0] += 1.0
            r[k, 0] += encoded_doc[i]
        return

    for i in range(limit):
        k = int(assignments[i])
        if k < 0 or k >= num_topics:
            continue
        x = encoded_doc[i]
        log_resp = log_mixture_weights[k] + scaled_component_means[k] @ x
        log_resp -= log_resp.max()
        resp = np.exp(log_resp)
        resp_sum = float(resp.sum())
        if (not np.isfinite(resp_sum)) or resp_sum <= 0.0:
            resp = np.full(num_components, 1.0 / num_components, dtype=np.float64)
        else:
            resp /= resp_sum
        nk[k] += 1.0
        nk_comp[k] += resp
        r[k] += resp[:, None] * x[None, :]


def _accumulate_doc_average_log_likelihood_python(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    log_c_per_topic: np.ndarray,
    scaled_topic_means: np.ndarray,
) -> tuple[float, int]:
    num_topics = int(log_c_per_topic.shape[0])
    total_ll = 0.0
    total_count = 0
    limit = min(int(encoded_doc.shape[0]), int(assignments.shape[0]))
    for i in range(limit):
        k = int(assignments[i])
        if k < 0 or k >= num_topics:
            continue
        total_ll += float(log_c_per_topic[k] + scaled_topic_means[k] @ encoded_doc[i])
        total_count += 1
    return total_ll, total_count


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _accumulate_doc_assignment_statistics_numba(
        encoded_doc: np.ndarray,
        assignments: np.ndarray,
        log_mixture_weights: np.ndarray,
        scaled_component_means: np.ndarray,
        nk: np.ndarray,
        nk_comp: np.ndarray,
        r: np.ndarray,
    ) -> None:
        num_topics = nk.shape[0]
        num_components = nk_comp.shape[1]
        dim = r.shape[2]
        limit = encoded_doc.shape[0]
        if assignments.shape[0] < limit:
            limit = assignments.shape[0]

        if num_components == 1:
            for i in range(limit):
                k = int(assignments[i])
                if k < 0 or k >= num_topics:
                    continue
                nk[k] += 1.0
                nk_comp[k, 0] += 1.0
                for j in range(dim):
                    r[k, 0, j] += float(encoded_doc[i, j])
            return

        log_resp = np.empty(num_components, dtype=np.float64)
        for i in range(limit):
            k = int(assignments[i])
            if k < 0 or k >= num_topics:
                continue

            max_log = -1.0e300
            for c in range(num_components):
                value = float(log_mixture_weights[k, c])
                for j in range(dim):
                    value += float(scaled_component_means[k, c, j]) * float(
                        encoded_doc[i, j]
                    )
                log_resp[c] = value
                if value > max_log:
                    max_log = value

            resp_sum = 0.0
            for c in range(num_components):
                value = math.exp(log_resp[c] - max_log)
                log_resp[c] = value
                resp_sum += value

            nk[k] += 1.0
            if (not math.isfinite(resp_sum)) or resp_sum <= 0.0:
                uniform = 1.0 / num_components
                for c in range(num_components):
                    nk_comp[k, c] += uniform
                    for j in range(dim):
                        r[k, c, j] += uniform * float(encoded_doc[i, j])
                continue

            inv_resp_sum = 1.0 / resp_sum
            for c in range(num_components):
                weight = log_resp[c] * inv_resp_sum
                nk_comp[k, c] += weight
                for j in range(dim):
                    r[k, c, j] += weight * float(encoded_doc[i, j])

    @njit(cache=True)
    def _accumulate_doc_average_log_likelihood_numba(
        encoded_doc: np.ndarray,
        assignments: np.ndarray,
        log_c_per_topic: np.ndarray,
        scaled_topic_means: np.ndarray,
    ) -> tuple[float, int]:
        num_topics = log_c_per_topic.shape[0]
        dim = scaled_topic_means.shape[1]
        limit = encoded_doc.shape[0]
        if assignments.shape[0] < limit:
            limit = assignments.shape[0]

        total_ll = 0.0
        total_count = 0
        for i in range(limit):
            k = int(assignments[i])
            if k < 0 or k >= num_topics:
                continue
            value = float(log_c_per_topic[k])
            for j in range(dim):
                value += float(scaled_topic_means[k, j]) * float(encoded_doc[i, j])
            total_ll += value
            total_count += 1
        return total_ll, total_count

else:
    _accumulate_doc_assignment_statistics_numba = None
    _accumulate_doc_average_log_likelihood_numba = None


def accumulate_doc_assignment_statistics(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    log_mixture_weights: np.ndarray,
    scaled_component_means: np.ndarray,
    nk: np.ndarray,
    nk_comp: np.ndarray,
    r: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _accumulate_doc_assignment_statistics_numba(
            encoded_doc=encoded_doc,
            assignments=assignments,
            log_mixture_weights=log_mixture_weights,
            scaled_component_means=scaled_component_means,
            nk=nk,
            nk_comp=nk_comp,
            r=r,
        )
        return
    _accumulate_doc_assignment_statistics_python(
        encoded_doc=encoded_doc,
        assignments=assignments,
        log_mixture_weights=log_mixture_weights,
        scaled_component_means=scaled_component_means,
        nk=nk,
        nk_comp=nk_comp,
        r=r,
    )


def accumulate_doc_average_log_likelihood(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    log_c_per_topic: np.ndarray,
    scaled_topic_means: np.ndarray,
) -> tuple[float, int]:
    if NUMBA_AVAILABLE:
        return _accumulate_doc_average_log_likelihood_numba(
            encoded_doc=encoded_doc,
            assignments=assignments,
            log_c_per_topic=log_c_per_topic,
            scaled_topic_means=scaled_topic_means,
        )
    return _accumulate_doc_average_log_likelihood_python(
        encoded_doc=encoded_doc,
        assignments=assignments,
        log_c_per_topic=log_c_per_topic,
        scaled_topic_means=scaled_topic_means,
    )


ACCUMULATE_DOC_ASSIGNMENT_STATISTICS_BACKEND = "numba" if NUMBA_AVAILABLE else "python"
ACCUMULATE_DOC_AVG_LL_BACKEND = "numba" if NUMBA_AVAILABLE else "python"

from __future__ import annotations

from typing import Sequence

import numpy as np


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def vmf_log_density_all_topics(
    *,
    embeddings: np.ndarray,
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros((0, topic_means.shape[0]), dtype=float)

    x = normalize_rows(embeddings)
    _, embedding_dim = x.shape
    num_topics, topic_dim = topic_means.shape
    if embedding_dim != topic_dim:
        raise ValueError(
            f"Dim mismatch: embeddings {embedding_dim} vs topic_means {topic_dim}"
        )

    num_topics_components, num_components, component_dim = component_means.shape
    if num_topics_components != num_topics:
        raise ValueError(
            f"K mismatch: topic_means {num_topics} vs component_means {num_topics_components}"
        )
    if component_dim != embedding_dim:
        raise ValueError(
            f"Dim mismatch: embeddings {embedding_dim} vs component_means {component_dim}"
        )
    if mixture_weights.shape != (num_topics, num_components):
        raise ValueError(
            "mixture_weights shape "
            f"{mixture_weights.shape} does not match {(num_topics, num_components)}"
        )

    comp_flat = component_means.reshape(num_topics * num_components, embedding_dim)
    dots = (x @ comp_flat.T).reshape(x.shape[0], num_topics, num_components)
    scores = dots * kappa_per_topic[None, :, None]
    log_pi = np.log(mixture_weights + 1e-12)[None, :, :]
    log_comp = scores + log_pi
    max_log = log_comp.max(axis=2, keepdims=True)
    return (
        max_log + np.log(np.exp(log_comp - max_log).sum(axis=2, keepdims=True) + 1e-12)
    ).squeeze(-1)


def top_sentences_by_topic_vmf_loglik(
    *,
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    sentences: Sequence[str],
    embeddings: np.ndarray,
    top_k: int,
) -> dict[int, list[dict[str, float | str]]]:
    if embeddings.shape[0] == 0 or topic_means.shape[0] == 0:
        return {}

    log_scores = vmf_log_density_all_topics(
        embeddings=embeddings,
        topic_means=topic_means,
        kappa_per_topic=kappa_per_topic,
        mixture_weights=mixture_weights,
        component_means=component_means,
    )
    results: dict[int, list[dict[str, float | str]]] = {}
    for topic_idx in range(topic_means.shape[0]):
        top_idx = np.argsort(-log_scores[:, topic_idx])[:top_k]
        results[topic_idx] = [
            {
                "sentence": sentences[int(row_idx)],
                "score": float(log_scores[int(row_idx), topic_idx]),
            }
            for row_idx in top_idx
        ]
    return results


def gaussian_log_density_all_topics(
    *,
    embeddings: np.ndarray,
    means: np.ndarray,
    cholesky: np.ndarray,
    log_determinants: np.ndarray,
) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros((0, means.shape[0]), dtype=float)

    _, embedding_dim = embeddings.shape
    num_topics, means_dim = means.shape
    if embedding_dim != means_dim:
        raise ValueError(
            f"Dim mismatch: embeddings {embedding_dim} vs means {means_dim}"
        )
    if cholesky.shape != (num_topics, embedding_dim, embedding_dim):
        raise ValueError(
            f"Cholesky shape {cholesky.shape} does not match expected {(num_topics, embedding_dim, embedding_dim)}"
        )
    if log_determinants.shape[0] != num_topics:
        raise ValueError(
            f"log_determinants length {log_determinants.shape[0]} does not match K={num_topics}"
        )

    log_probs = np.zeros((embeddings.shape[0], num_topics), dtype=float)
    const = embedding_dim * np.log(2 * np.pi)
    for topic_idx in range(num_topics):
        diff = embeddings - means[topic_idx]
        solved = np.linalg.solve(cholesky[topic_idx], diff.T)
        maha = (solved**2).sum(axis=0)
        log_probs[:, topic_idx] = -0.5 * (maha + log_determinants[topic_idx] + const)
    return log_probs


def top_sentences_by_topic_gaussian_loglik(
    *,
    gaussian_means: np.ndarray,
    gaussian_cholesky: np.ndarray,
    gaussian_log_determinants: np.ndarray,
    sentences: Sequence[str],
    embeddings: np.ndarray,
    top_k: int,
) -> dict[int, list[dict[str, float | str]]]:
    if embeddings.shape[0] == 0 or gaussian_means.shape[0] == 0:
        return {}

    log_scores = gaussian_log_density_all_topics(
        embeddings=embeddings,
        means=gaussian_means,
        cholesky=gaussian_cholesky,
        log_determinants=gaussian_log_determinants,
    )
    results: dict[int, list[dict[str, float | str]]] = {}
    for topic_idx in range(gaussian_means.shape[0]):
        top_idx = np.argsort(-log_scores[:, topic_idx])[:top_k]
        results[topic_idx] = [
            {
                "sentence": sentences[int(row_idx)],
                "score": float(log_scores[int(row_idx), topic_idx]),
            }
            for row_idx in top_idx
        ]
    return results


def top_sentences_by_topic_log_score_matrix(
    *,
    log_scores: np.ndarray,
    sentences: Sequence[str],
    top_k: int,
) -> dict[int, list[dict[str, float | str]]]:
    if log_scores.shape[0] == 0 or log_scores.shape[1] == 0:
        return {}
    if log_scores.shape[0] != len(sentences):
        raise ValueError(
            f"log_scores rows {log_scores.shape[0]} do not match sentences {len(sentences)}"
        )
    results: dict[int, list[dict[str, float | str]]] = {}
    for topic_idx in range(log_scores.shape[1]):
        top_idx = np.argsort(-log_scores[:, topic_idx])[:top_k]
        results[topic_idx] = [
            {
                "sentence": sentences[int(row_idx)],
                "score": float(log_scores[int(row_idx), topic_idx]),
            }
            for row_idx in top_idx
        ]
    return results

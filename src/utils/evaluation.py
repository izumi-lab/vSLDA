from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.special import ive

from src.models.vmf_numba import accumulate_doc_average_log_likelihood


@dataclass(frozen=True)
class EvalResult:
    """Container for evaluation metrics of a probabilistic model."""

    avg_log_likelihood: Optional[float] = None
    perplexity: Optional[float] = None


def evaluate_from_history(average_ll_history: Sequence[float]) -> EvalResult:
    """Compute evaluation metrics from a history of average log-likelihoods.

    Args:
        average_ll_history: Sequence of average log-likelihood values
            (typically one per iteration).

    Returns:
        EvalResult with the latest average log-likelihood and its perplexity.
        If the history is empty, both fields are None.
    """
    if not average_ll_history:
        return EvalResult()

    avg_ll = float(average_ll_history[-1])
    perp = math.exp(-avg_ll)
    return EvalResult(avg_log_likelihood=avg_ll, perplexity=perp)


def evaluate_model(trainer: object) -> EvalResult:
    """Evaluate a trainer that exposes an `average_ll` history.

    The trainer is expected to have an attribute `average_ll` which is a
    sequence of average log-likelihood values (one per iteration).

    This keeps vMF / Gaussian trainers interoperable as long as they share
    this attribute.
    """
    average_ll_history = getattr(trainer, "average_ll", None)
    if average_ll_history is None:
        return EvalResult()

    return evaluate_from_history(average_ll_history)


def calculate_avg_ll_vmf(
    corpus: Sequence[Sequence[str]],
    topic_assignments: Sequence[Sequence[int]],
    encoder,
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
    transform_embeddings: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """Compute average vMF log-likelihood for a corpus (up to a constant).

    The log-likelihood for a single sentence vector x assigned to topic k is
    approximated by

        log p(x | k) = log C_M(κ_k) + κ_k μ_k^T x,

    ignoring the normalization constant of the vMF distribution.
    All sentence embeddings are L2-normalized so that ||x|| = 1.

    Args:
        corpus:
            Corpus of documents, each a sequence of sentences.
        topic_assignments:
            Topic index for each sentence in each document.
        encoder:
            Sentence encoder with an `encode(list[str]) -> np.ndarray` method
            (e.g. SentenceEncoder).
        topic_means:
            Array of shape (num_topics, embedding_dim) with unit-norm μ_k.
        kappa_per_topic:
            Array of shape (num_topics,) with concentration parameters κ_k.
        transform_embeddings:
            Optional pre-normalization transform applied to encoded embeddings.

    Returns:
        Average log-likelihood over all assigned sentences.
        If there are no sentences, returns float("nan").
    """
    d_dim = float(topic_means.shape[1])
    v = d_dim / 2.0 - 1.0
    kappa_safe = np.clip(np.asarray(kappa_per_topic, dtype=np.float64), 1e-12, None)
    ive_val = np.maximum(ive(v, kappa_safe), 1e-300)
    log_c = (
        v * np.log(kappa_safe)
        - (d_dim / 2.0) * math.log(2.0 * math.pi)
        - (np.log(ive_val) + kappa_safe)
    )

    total_ll = 0.0
    total_count = 0

    for doc, doc_topics in zip(corpus, topic_assignments):
        if not doc:
            continue

        # Encode and normalize this document
        enc = np.asarray(encoder.encode(list(doc)), dtype=np.float64)
        if enc.size == 0:
            continue
        if enc.ndim == 1:
            enc = enc.reshape(1, -1)
        if transform_embeddings is not None:
            enc = np.asarray(transform_embeddings(enc), dtype=np.float64)
            if enc.ndim == 1:
                enc = enc.reshape(1, -1)
        norms = np.linalg.norm(enc, axis=1, keepdims=True) + 1e-12
        enc = enc / norms

        for x, k in zip(enc, doc_topics):
            if k < 0:
                # Skip sentences that are temporarily "unassigned"
                continue
            total_ll += float(log_c[k] + kappa_per_topic[k] * np.dot(topic_means[k], x))
            total_count += 1

    if total_count == 0:
        return float("nan")

    return total_ll / total_count


def calculate_avg_ll_vmf_from_encoded(
    encoded_corpus: Sequence[np.ndarray],
    topic_assignments: Sequence[Sequence[int]],
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
) -> float:
    """Compute average vMF log-likelihood from pre-encoded unit vectors."""
    d_dim = float(topic_means.shape[1])
    v = d_dim / 2.0 - 1.0
    kappa_safe = np.clip(np.asarray(kappa_per_topic, dtype=np.float64), 1e-12, None)
    ive_val = np.maximum(ive(v, kappa_safe), 1e-300)
    log_c = (
        v * np.log(kappa_safe)
        - (d_dim / 2.0) * math.log(2.0 * math.pi)
        - (np.log(ive_val) + kappa_safe)
    )

    scaled_topic_means = np.asarray(kappa_per_topic, dtype=np.float64)[
        :, None
    ] * np.asarray(
        topic_means,
        dtype=np.float64,
    )

    total_ll = 0.0
    total_count = 0

    for enc, doc_topics in zip(encoded_corpus, topic_assignments):
        arr = np.asarray(enc)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        topics = np.asarray(doc_topics, dtype=np.int64)
        if topics.size == 0:
            continue
        doc_ll, doc_count = accumulate_doc_average_log_likelihood(
            encoded_doc=arr,
            assignments=topics,
            log_c_per_topic=log_c,
            scaled_topic_means=scaled_topic_means,
        )
        if doc_count <= 0:
            continue
        total_ll += float(doc_ll)
        total_count += int(doc_count)

    if total_count == 0:
        return float("nan")

    return total_ll / total_count

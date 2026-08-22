"""Model-specific likelihood adapters for frozen topic assignment.

Adapters return log likelihoods only.  They never update fitted global state and
never rank words; those operations belong to :mod:`topic_assignment`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.special import gammaln, ive


def lda_log_likelihood_by_type(phi: np.ndarray) -> np.ndarray:
    """Convert fitted LDA ``phi[k, w]`` to a ``[w, k]`` log table."""

    values = np.asarray(phi, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("LDA phi must be a non-empty topic-by-word matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("LDA phi must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("LDA phi rows must sum to one")
    with np.errstate(divide="ignore"):
        return np.log(values).T


def _log_vmf_normalizer(kappa: np.ndarray, dimension: int) -> np.ndarray:
    """log C_D(kappa), clamped exactly like the fitted vMF implementations.

    Mirrors ``VmfSentenceLda._log_vmf_normalization_const`` so post-hoc
    likelihoods match the training-time log densities bit for bit, including
    for degenerate small-kappa topics where ``ive`` underflows.
    """

    if dimension <= 1:
        raise ValueError("vMF dimension must be greater than one")
    concentration = np.asarray(kappa, dtype=np.float64)
    if np.any(concentration < 0.0) or not np.all(np.isfinite(concentration)):
        raise ValueError("vMF concentration must be finite and non-negative")
    order = dimension / 2.0 - 1.0
    kappa_safe = np.clip(concentration, 1e-12, None)
    scaled_bessel = ive(order, kappa_safe)
    if np.any(np.isnan(scaled_bessel)):
        raise ValueError("could not evaluate the vMF normalization constant")
    log_bessel = np.log(np.maximum(scaled_bessel, 1e-300)) + kappa_safe
    return (
        order * np.log(kappa_safe)
        - (dimension / 2.0) * np.log(2.0 * np.pi)
        - log_bessel
    )


def vmf_mixture_log_likelihood(
    observations: np.ndarray,
    *,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    kappa_per_topic: np.ndarray,
) -> np.ndarray:
    """Evaluate the normalized fitted per-topic mixture-of-vMF density."""

    x = np.asarray(observations, dtype=np.float64)
    means = np.asarray(component_means, dtype=np.float64)
    weights = np.asarray(mixture_weights, dtype=np.float64)
    kappa = np.asarray(kappa_per_topic, dtype=np.float64)
    if x.ndim != 2 or means.ndim != 3:
        raise ValueError("observations and component means must be 2D and 3D")
    num_topics, num_components, dimension = means.shape
    if x.shape[1] != dimension or weights.shape != (num_topics, num_components):
        raise ValueError("vMF observation, mean and weight shapes are inconsistent")
    if kappa.shape == (num_topics,):
        kappa = np.broadcast_to(kappa[:, None], (num_topics, num_components))
    elif kappa.shape != (num_topics, num_components):
        raise ValueError("vMF kappa must have shape (topics,) or (topics, components)")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("vMF mixture weights must be finite and non-negative")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("vMF mixture weights must sum to one by topic")
    x_norm = np.linalg.norm(x, axis=1)
    mean_norm = np.linalg.norm(means, axis=2)
    if not np.allclose(x_norm, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("vMF observations must be L2 normalized")
    if not np.allclose(mean_norm, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("vMF component means must be L2 normalized")
    log_components = (
        np.einsum("nd,kcd->nkc", x, kappa[:, :, None] * means, optimize=True)
        + _log_vmf_normalizer(kappa, dimension)[None, :, :]
    )
    with np.errstate(divide="ignore"):
        log_components += np.log(weights)[None, :, :]
    maximum = np.max(log_components, axis=2, keepdims=True)
    if np.any(np.isneginf(maximum)):
        raise ValueError("every vMF mixture component has zero weight")
    return maximum[..., 0] + np.log(np.exp(log_components - maximum).sum(axis=2))


def sentlda_log_likelihood_by_doc(
    sentence_bow_by_doc: Sequence[Sequence[Sequence[tuple[int, int]]]],
    *,
    topic_word_counts: np.ndarray,
    beta: float,
) -> list[np.ndarray]:
    """Frozen Dirichlet-multinomial sentence predictive probabilities."""

    counts = np.asarray(topic_word_counts, dtype=np.float64)
    if counts.ndim != 2 or np.any(counts < 0.0) or not np.all(np.isfinite(counts)):
        raise ValueError("sentLDA topic-word counts must be a non-negative matrix")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("sentLDA beta must be finite and positive")
    num_topics, vocab_size = counts.shape
    totals = counts.sum(axis=1)
    base = counts + float(beta)
    normalizer = totals + float(beta) * vocab_size
    result: list[np.ndarray] = []
    for doc_index, sentences in enumerate(sentence_bow_by_doc):
        likelihoods = np.empty((len(sentences), num_topics), dtype=np.float64)
        for sentence_index, bow in enumerate(sentences):
            length = 0.0
            score = np.zeros(num_topics, dtype=np.float64)
            for raw_word_id, raw_count in bow:
                word_id, word_count = int(raw_word_id), float(raw_count)
                if not 0 <= word_id < vocab_size or word_count < 0.0:
                    raise ValueError(
                        f"invalid sentence BoW at document {doc_index}, "
                        f"sentence {sentence_index}"
                    )
                length += word_count
                score += gammaln(base[:, word_id] + word_count) - gammaln(
                    base[:, word_id]
                )
            likelihoods[sentence_index] = (
                score + gammaln(normalizer) - gammaln(normalizer + length)
            )
        result.append(likelihoods)
    return result


def gaussian_log_likelihood(
    observations: np.ndarray,
    *,
    topic_log_density: Callable[[np.ndarray], np.ndarray],
    num_topics: int,
) -> np.ndarray:
    """Validate output from a frozen GaussianLDA predictive-density loader."""

    values = np.asarray(topic_log_density(np.asarray(observations)), dtype=np.float64)
    if values.ndim == 1 and np.asarray(observations).shape[0] == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape != (len(observations), num_topics):
        raise ValueError(
            "Gaussian predictive density must return (num_units, num_topics)"
        )
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Gaussian predictive density returned NaN or +inf")
    return values


def restrict_scores_to_evaluation_vocabulary(
    scores: np.ndarray,
    *,
    model_vocabulary: Sequence[str],
    evaluation_vocabulary: Sequence[str],
    require_coverage: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Map complete model scores before ranking, preserving V_eval ordering."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(model_vocabulary):
        raise ValueError("score matrix and model vocabulary are not aligned")
    model_id = {str(word): index for index, word in enumerate(model_vocabulary)}
    mapped = np.full((values.shape[0], len(evaluation_vocabulary)), -np.inf)
    covered = np.zeros(len(evaluation_vocabulary), dtype=bool)
    for eval_id, word in enumerate(evaluation_vocabulary):
        source_id = model_id.get(str(word))
        if source_id is not None:
            mapped[:, eval_id] = values[:, source_id]
            covered[eval_id] = True
    if require_coverage and not np.all(covered):
        missing = [str(evaluation_vocabulary[i]) for i in np.flatnonzero(~covered)[:10]]
        raise ValueError(f"model vocabulary does not cover V_eval; examples={missing}")
    return mapped, covered


def normalize_ctm_decoder_scores(
    scores: np.ndarray, *, already_probabilities: bool = True
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("CTM decoder scores must be a finite 2D matrix")
    if already_probabilities:
        if np.any(values < 0.0) or not np.allclose(
            values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6
        ):
            raise ValueError("CTM decoder probabilities must be normalized by topic")
        return values
    maximum = values.max(axis=1, keepdims=True)
    exponentiated = np.exp(values - maximum)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def sample_etm_theta(
    *, mu: np.ndarray, logsigma: np.ndarray, num_samples: int, seed: int
) -> np.ndarray:
    """Sample reproducible logistic-normal ETM document mixtures."""

    mean = np.asarray(mu, dtype=np.float64)
    log_variance = np.asarray(logsigma, dtype=np.float64)
    if mean.shape != log_variance.shape or mean.ndim != 2:
        raise ValueError("ETM mu and logsigma must be aligned (docs, topics) matrices")
    if (
        num_samples <= 0
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(log_variance))
    ):
        raise ValueError("invalid ETM posterior sampling inputs")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((num_samples,) + mean.shape)
    logits = mean[None, :, :] + np.exp(0.5 * log_variance)[None, :, :] * noise
    logits -= logits.max(axis=2, keepdims=True)
    theta = np.exp(logits)
    return theta / theta.sum(axis=2, keepdims=True)

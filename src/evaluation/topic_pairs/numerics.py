"""Pure numerics of the topic-pair analysis (no I/O).

Every quantity is computed from three aligned inputs: the L2-normalised
sentence embeddings ``X`` (S x D), the sentence-topic posteriors ``P``
(S x K, rows sum to one) and the fine-category label index of each sentence.
The procedure is the same for every model, so the per-topic concentration,
the centroid geometry, the assignment confusion and the label divergence are
comparable across models. Nothing here is aggregated across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KAPPA_ESTIMATOR = "banerjee_2005_approx"
# The vMF trainer clips the mean resultant length to this open interval before
# the closed-form estimate (src/models/vmf_sentence_lda.py).
R_BAR_EPS = 1e-6

PER_TOPIC_KEYS: tuple[str, ...] = (
    "mass_soft",
    "count_hard",
    "resultant_length_soft",
    "resultant_length_hard",
    "kappa_soft",
    "kappa_hard",
    "label_entropy_normalized",
    "label_dominant",
    "label_dominant_share",
)
PER_TOPIC_MATRIX_KEYS: tuple[str, ...] = ("label_mass",)
PER_PAIR_KEYS: tuple[str, ...] = (
    "centroid_cosine_soft",
    "centroid_cosine_hard",
    "assignment_confusion",
    "label_js_divergence",
)
MODEL_OWN_KEYS: tuple[str, ...] = ("kappa_model", "centroid_cosine_model")


def banerjee_kappa(
    r_bar: np.ndarray, dim: int, *, max_kappa: float | None = None
) -> np.ndarray:
    """Closed-form vMF concentration from the mean resultant length.

    ``kappa = r (d - r^2) / (1 - r^2)`` (Banerjee et al., 2005), with ``r``
    clipped to ``[R_BAR_EPS, 1 - R_BAR_EPS]`` exactly as the trainer does and
    optionally bounded by ``max_kappa``. Non-finite inputs stay NaN.
    """

    values = np.asarray(r_bar, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    clipped = np.clip(values[finite], R_BAR_EPS, 1.0 - R_BAR_EPS)
    numerator = clipped * (float(dim) - clipped**2)
    denominator = 1.0 - clipped**2
    kappa = numerator / (denominator + 1e-12)
    if max_kappa is not None:
        kappa = np.minimum(kappa, float(max_kappa))
    result[finite] = kappa
    return result


def row_normalize(rows: np.ndarray) -> np.ndarray:
    """Rows scaled to sum one; rows with zero mass become NaN."""

    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {values.shape}")
    sums = values.sum(axis=1, keepdims=True)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    nonzero = sums[:, 0] > 0.0
    result[nonzero] = values[nonzero] / sums[nonzero]
    return result


def normalized_entropy(rows: np.ndarray) -> np.ndarray:
    """Entropy of each row distribution divided by ``ln L``; NaN for empty rows."""

    probabilities = row_normalize(rows)
    num_labels = probabilities.shape[1]
    result = np.full(probabilities.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(probabilities).all(axis=1)
    if num_labels <= 1:
        result[valid] = 0.0
        return result
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            probabilities > 0.0, probabilities * np.log(probabilities), 0.0
        )
    entropy = -terms.sum(axis=1)
    result[valid] = entropy[valid] / np.log(float(num_labels))
    return result


def jensen_shannon_divergence_matrix(rows: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon divergence (nats) of the row distributions.

    ``JS(p, q) = KL(p || m)/2 + KL(q || m)/2`` with ``m = (p + q)/2``; rows
    without mass yield NaN entries.
    """

    probabilities = row_normalize(rows)
    num_rows = probabilities.shape[0]
    result = np.full((num_rows, num_rows), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(probabilities).all(axis=1))
    if valid.size == 0:
        return result
    p = probabilities[valid][:, None, :]
    q = probabilities[valid][None, :, :]
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0.0, p * (np.log(p) - np.log(m)), 0.0).sum(axis=2)
        kl_q = np.where(q > 0.0, q * (np.log(q) - np.log(m)), 0.0).sum(axis=2)
    divergence = 0.5 * kl_p + 0.5 * kl_q
    divergence = np.maximum(divergence, 0.0)
    result[np.ix_(valid, valid)] = np.where(
        np.eye(valid.size, dtype=bool), 0.0, divergence
    )
    return result


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm; zero rows become NaN rows."""

    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    nonzero = norms[:, 0] > 0.0
    result[nonzero] = values[nonzero] / norms[nonzero]
    return result


def cosine_matrix(unit_vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine of unit rows, clipped to [-1, 1]; NaN rows propagate."""

    values = np.asarray(unit_vectors, dtype=np.float64)
    result = values @ values.T
    result = np.clip(result, -1.0, 1.0)
    valid = np.isfinite(values).all(axis=1)
    result[np.ix_(valid, valid)] = np.where(
        np.eye(int(valid.sum()), dtype=bool), 1.0, result[np.ix_(valid, valid)]
    )
    return result


def offdiagonal_values(matrix: np.ndarray) -> np.ndarray:
    """Finite upper-triangular entries (j < k) of a square matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {values.shape}")
    upper = values[np.triu_indices(values.shape[0], k=1)]
    return upper[np.isfinite(upper)]


def _nan_stat(values: np.ndarray, reducer) -> float:
    return float(reducer(values)) if values.size else float("nan")


@dataclass(frozen=True)
class TopicPairResult:
    num_topics: int
    num_sentences: int
    embedding_dim: int
    num_labels: int
    per_topic: dict[str, np.ndarray]
    per_pair: dict[str, np.ndarray]

    @property
    def num_empty_topics(self) -> int:
        return int(np.count_nonzero(self.per_topic["mass_soft"] <= 0.0))

    def scalar_summary(self) -> dict[str, float]:
        """Per-run scalars for the review CSV and the sidecar ``scores`` block."""

        cosine = offdiagonal_values(self.per_pair["centroid_cosine_soft"])
        confusion = self.per_pair["assignment_confusion"]
        confusion_off = confusion[~np.eye(self.num_topics, dtype=bool)]
        confusion_off = confusion_off[np.isfinite(confusion_off)]
        entropy = self.per_topic["label_entropy_normalized"]
        entropy = entropy[np.isfinite(entropy)]
        return {
            "num_topics": float(self.num_topics),
            "num_sentences": float(self.num_sentences),
            "embedding_dim": float(self.embedding_dim),
            "num_empty_topics": float(self.num_empty_topics),
            "mean_offdiag_centroid_cosine_soft": _nan_stat(cosine, np.mean),
            "max_offdiag_centroid_cosine_soft": _nan_stat(cosine, np.max),
            "mean_offdiag_assignment_confusion": _nan_stat(confusion_off, np.mean),
            "mean_label_entropy_normalized": _nan_stat(entropy, np.mean),
        }


SCALAR_SUMMARY_KEYS: tuple[str, ...] = (
    "num_topics",
    "num_sentences",
    "embedding_dim",
    "num_empty_topics",
    "mean_offdiag_centroid_cosine_soft",
    "max_offdiag_centroid_cosine_soft",
    "mean_offdiag_assignment_confusion",
    "mean_label_entropy_normalized",
)


def _validate_inputs(
    embeddings_unit: np.ndarray, probs: np.ndarray, labels: np.ndarray, num_labels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(embeddings_unit, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or p.ndim != 2:
        raise ValueError("embeddings and posteriors must be matrices")
    if x.shape[0] != p.shape[0]:
        raise ValueError(
            f"embeddings ({x.shape[0]} rows) and posteriors ({p.shape[0]} rows) "
            "are not aligned"
        )
    if y.shape != (x.shape[0],):
        raise ValueError(
            f"labels must have one entry per sentence: {y.shape} vs {x.shape[0]}"
        )
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be integer label indices")
    if int(num_labels) <= 0:
        raise ValueError("num_labels must be positive")
    if y.size and (y.min() < 0 or y.max() >= int(num_labels)):
        raise ValueError("label index out of range")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        raise ValueError("embeddings and posteriors must be finite")
    if x.shape[0]:
        norms = np.linalg.norm(x, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
            raise ValueError("embeddings must be L2 normalised")
        if np.any(p < -1e-9) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("posterior rows must be non-negative and sum to one")
    return x, p, y.astype(np.int64)


def compute_topic_pair_metrics(
    embeddings_unit: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    num_labels: int,
    max_kappa: float | None = None,
) -> TopicPairResult:
    """Per-topic and per-pair quantities of one run in the shared space.

    Soft quantities weight every sentence by its posterior ``p(k | s)``; hard
    quantities use the arg-max assignment. Topics without mass yield NaN.
    """

    x, p, y = _validate_inputs(embeddings_unit, probs, labels, num_labels)
    num_sentences, dim = x.shape
    num_topics = p.shape[1]

    mass_soft = p.sum(axis=0)
    hard = p.argmax(axis=1) if num_sentences else np.empty(0, dtype=np.int64)
    count_hard = np.bincount(hard, minlength=num_topics).astype(np.float64)
    onehot_hard = np.zeros((num_sentences, num_topics), dtype=np.float64)
    if num_sentences:
        onehot_hard[np.arange(num_sentences), hard] = 1.0

    sums_soft = p.T @ x
    sums_hard = onehot_hard.T @ x
    with np.errstate(divide="ignore", invalid="ignore"):
        resultant_soft = np.where(
            mass_soft > 0.0, np.linalg.norm(sums_soft, axis=1) / mass_soft, np.nan
        )
        resultant_hard = np.where(
            count_hard > 0.0, np.linalg.norm(sums_hard, axis=1) / count_hard, np.nan
        )
    centroid_soft = unit_rows(sums_soft)
    centroid_hard = unit_rows(sums_hard)
    centroid_soft[mass_soft <= 0.0] = np.nan
    centroid_hard[count_hard <= 0.0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        confusion = (onehot_hard.T @ p) / count_hard[:, None]
    confusion[count_hard <= 0.0] = np.nan

    onehot_label = np.zeros((num_sentences, int(num_labels)), dtype=np.float64)
    if num_sentences:
        onehot_label[np.arange(num_sentences), y] = 1.0
    label_mass = p.T @ onehot_label
    label_entropy = normalized_entropy(label_mass)
    label_dominant = np.where(mass_soft > 0.0, label_mass.argmax(axis=1), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        label_dominant_share = np.where(
            mass_soft > 0.0, label_mass.max(axis=1) / label_mass.sum(axis=1), np.nan
        )

    per_topic = {
        "mass_soft": mass_soft,
        "count_hard": count_hard,
        "resultant_length_soft": resultant_soft,
        "resultant_length_hard": resultant_hard,
        "kappa_soft": banerjee_kappa(resultant_soft, dim, max_kappa=max_kappa),
        "kappa_hard": banerjee_kappa(resultant_hard, dim, max_kappa=max_kappa),
        "label_entropy_normalized": label_entropy,
        "label_dominant": label_dominant.astype(np.float64),
        "label_dominant_share": label_dominant_share,
        "label_mass": label_mass,
    }
    per_pair = {
        "centroid_cosine_soft": cosine_matrix(centroid_soft),
        "centroid_cosine_hard": cosine_matrix(centroid_hard),
        "assignment_confusion": confusion,
        "label_js_divergence": jensen_shannon_divergence_matrix(label_mass),
    }
    return TopicPairResult(
        num_topics=int(num_topics),
        num_sentences=int(num_sentences),
        embedding_dim=int(dim),
        num_labels=int(num_labels),
        per_topic=per_topic,
        per_pair=per_pair,
    )


def cross_model_overlap(probs_a: np.ndarray, probs_b: np.ndarray) -> np.ndarray:
    """Expected co-assignment mass ``N[a, b] = sum_s p_A(a | s) p_B(b | s)``.

    Both posterior matrices must describe the same sentences in the same
    order; the marginals recover each model's soft topic mass.
    """

    a = np.asarray(probs_a, dtype=np.float64)
    b = np.asarray(probs_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError(f"posteriors are not aligned: {a.shape} vs {b.shape}")
    return a.T @ b


def model_reference_metrics(
    topic_means: np.ndarray, kappa_model: np.ndarray
) -> dict[str, np.ndarray]:
    """The model's own centres and concentrations, for comparison with the
    empirical ones (vMF Sentence LDA only)."""

    means = unit_rows(topic_means)
    kappa = np.asarray(kappa_model, dtype=np.float64)
    if kappa.shape != (means.shape[0],):
        raise ValueError(
            f"kappa shape {kappa.shape} does not match {means.shape[0]} topics"
        )
    return {
        "kappa_model": kappa,
        "centroid_cosine_model": cosine_matrix(means),
    }


__all__ = [
    "KAPPA_ESTIMATOR",
    "MODEL_OWN_KEYS",
    "PER_PAIR_KEYS",
    "PER_TOPIC_KEYS",
    "PER_TOPIC_MATRIX_KEYS",
    "SCALAR_SUMMARY_KEYS",
    "TopicPairResult",
    "banerjee_kappa",
    "compute_topic_pair_metrics",
    "cosine_matrix",
    "cross_model_overlap",
    "jensen_shannon_divergence_matrix",
    "model_reference_metrics",
    "normalized_entropy",
    "offdiagonal_values",
    "row_normalize",
    "unit_rows",
]

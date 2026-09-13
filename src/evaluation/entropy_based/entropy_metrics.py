"""Pure numpy entropy-based metrics on document-topic distributions.

Three metrics are implemented, following the definitions used in prior work:

1. ``topic_doc_entropy`` — entropy of the distribution over documents given a
   topic, ``H(P(d|k))`` with ``P(d|k) = theta_dk / sum_d theta_dk``.  This is
   MALLET's ``document_entropy`` diagnostic (Boyd-Graber, Mimno, Newman 2014,
   *Care and Feeding of Topic Models*) and is equal to ``ln D`` minus the
   AlSumait et al. (2009) D-BGround divergence ``KL(theta^(k) || Uniform_D)``.
2. ``topic_rank1_doc_fraction`` — fraction of documents whose argmax topic is
   ``k`` (MALLET ``rank_1_docs``).
3. ``doc_topic_entropy`` — Shannon entropy of each document's topic
   distribution ``H(theta_d)``.

All entropies use the natural logarithm and ``0 * ln 0 = 0``.
Normalized variants divide by ``ln D`` (topic side) or ``ln K`` (document side).

Two derived per-model fractions summarize metrics 1 and 2 across topics and are
the headline values, because the plain minimum/maximum saturate as ``K`` grows:

- ``topic_diffuse_fraction`` — share of topics with
  ``topic_doc_entropy_normalized > diffuse_entropy_threshold`` (default 0.95),
  i.e. background topics spread almost uniformly across the corpus.
- ``topic_dead_fraction`` — share of topics with
  ``topic_rank1_doc_fraction < dead_rank1_threshold`` (default 0.01), i.e.
  topics that are essentially never a document's main topic. Empty topics count
  as dead.

Both use all ``K`` topics as the denominator, so they are directly comparable.
The thresholds are conventions of this repository, not of the cited papers;
report the per-topic distributions in ``iter<N>/topic_metrics.csv`` when the
exact cut-off matters.

Reading the numbers:

- ``topic_doc_entropy_normalized`` close to 1 for some topic means that topic is
  almost uniformly present in every document, i.e. a background topic. A value
  ``x`` corresponds to spreading the topic's mass as evenly as a uniform
  distribution over ``D ** x`` documents.
- ``topic_rank1_doc_fraction`` equal to 0 for some topic means that topic is
  never the main topic of any document. The K values sum to 1 by construction,
  so the equal-use reference value is ``1 / K``.
- ``doc_topic_entropy_normalized`` low means documents are focused. Its mean
  ``m`` corresponds to ``K ** m`` effective topics per document (geometric mean
  over documents). Very low values combined with many dead topics indicate
  collapse, not quality, so metric 3 must be read together with 1 and 2.

These are structural diagnostics of ``theta`` only; they say nothing about
whether the topics are semantically good. Read them next to
``word_based_metrics`` (coherence) and the classification results, never alone.
Per-topic values are persisted in ``iter<N>/topic_metrics.csv`` so topic-level
entropy can later be plotted against per-topic coherence.

Edge cases: zero-mass documents are excluded from document summaries and counted
in ``num_zero_mass_docs``; topics with zero mass get ``NaN`` entropy and are
counted in ``num_empty_topics``; ``K == 1`` and ``D == 1`` give ``NaN``
normalized values; argmax ties resolve to the first topic.

Considered but not implemented: effective numbers ``exp(H)`` (a monotone
transform of the normalized entropy), top-k mass, MALLET ``allocation_count`` /
``allocation_ratio``, topic-word entropy ``H(phi_k)``, Lund et al. (2019) topic
switch percent, and Koltcov's Renyi entropy for choosing K.

MALLET's ``document_entropy`` weights by token counts ``N_dk``; this module uses
the theta-normalized form of AlSumait et al. because token counts are not
available for every model in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

DEFAULT_DIFFUSE_ENTROPY_THRESHOLD = 0.95
DEFAULT_DEAD_RANK1_THRESHOLD = 0.01

SUMMARY_METRIC_KEYS: tuple[str, ...] = (
    "doc_topic_entropy_normalized_mean",
    "doc_topic_entropy_normalized_median",
    "topic_doc_entropy_normalized_mean",
    "topic_doc_entropy_normalized_max",
    "topic_diffuse_fraction",
    "topic_dead_fraction",
)

DOC_METRIC_COLUMNS: tuple[str, ...] = (
    "doc_topic_entropy",
    "doc_topic_entropy_normalized",
)
TOPIC_METRIC_COLUMNS: tuple[str, ...] = (
    "topic_doc_entropy",
    "topic_doc_entropy_normalized",
    "topic_rank1_doc_fraction",
)
TOPIC_FLAG_COLUMNS: tuple[str, ...] = ("is_diffuse", "is_dead", "empty")


def shannon_entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats along ``axis`` with ``0 * ln 0 = 0``."""
    arr = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(arr > 0.0, arr * np.log(arr), 0.0)
    return -np.sum(terms, axis=axis)


def normalize_rows(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalize ``theta``; zero-mass rows stay zero and are flagged invalid."""
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D document-topic matrix, got shape {arr.shape}")
    if np.any(arr < 0.0):
        raise ValueError("Document-topic matrix must be non-negative.")
    row_sums = arr.sum(axis=1)
    valid = row_sums > 0.0
    safe = np.where(valid, row_sums, 1.0)[:, None]
    return arr / safe, valid


def document_topic_entropy(theta: np.ndarray) -> dict[str, np.ndarray]:
    """Per-document ``H(theta_d)`` and its ``ln K`` normalization."""
    normalized, valid = normalize_rows(theta)
    num_topics = normalized.shape[1]
    entropy = shannon_entropy(normalized, axis=1)
    entropy = np.where(valid, entropy, np.nan)
    if num_topics > 1:
        entropy_normalized = entropy / np.log(num_topics)
    else:
        entropy_normalized = np.full_like(entropy, np.nan)
    return {
        "doc_topic_entropy": entropy,
        "doc_topic_entropy_normalized": entropy_normalized,
        "valid": valid,
    }


def topic_document_distribution(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``P(d|k)`` as a ``(K, D)`` matrix; topics with zero mass get NaN columns."""
    normalized, _valid = normalize_rows(theta)
    column_mass = normalized.sum(axis=0)
    empty = column_mass <= 0.0
    safe_mass = np.where(empty, 1.0, column_mass)
    distribution = (normalized / safe_mass[None, :]).T
    distribution[empty, :] = np.nan
    return distribution, empty


def topic_document_entropy(theta: np.ndarray) -> dict[str, np.ndarray]:
    """Per-topic ``H(P(d|k))`` (MALLET document_entropy) and its ``ln D`` normalization."""
    distribution, empty = topic_document_distribution(theta)
    num_documents = distribution.shape[1]
    entropy = np.array(
        [
            np.nan if empty[k] else float(shannon_entropy(distribution[k]))
            for k in range(distribution.shape[0])
        ],
        dtype=np.float64,
    )
    if num_documents > 1:
        entropy_normalized = entropy / np.log(num_documents)
    else:
        entropy_normalized = np.full_like(entropy, np.nan)
    return {
        "topic_doc_entropy": entropy,
        "topic_doc_entropy_normalized": entropy_normalized,
        "empty": empty,
    }


def topic_rank1_doc_fraction(theta: np.ndarray) -> np.ndarray:
    """Fraction of (valid) documents whose argmax topic is ``k`` (MALLET rank_1_docs)."""
    normalized, valid = normalize_rows(theta)
    num_topics = normalized.shape[1]
    num_valid = int(valid.sum())
    if num_valid == 0:
        return np.full(num_topics, np.nan, dtype=np.float64)
    argmax = np.argmax(normalized[valid], axis=1)
    counts = np.bincount(argmax, minlength=num_topics).astype(np.float64)
    return counts / float(num_valid)


@dataclass(frozen=True)
class EntropyMetricsResult:
    doc: dict[str, np.ndarray]
    topic: dict[str, np.ndarray]
    summary: dict[str, float]
    num_documents: int
    num_topics: int
    num_zero_mass_docs: int
    num_empty_topics: int
    diffuse_entropy_threshold: float = DEFAULT_DIFFUSE_ENTROPY_THRESHOLD
    dead_rank1_threshold: float = DEFAULT_DEAD_RANK1_THRESHOLD
    extra: dict[str, object] = field(default_factory=dict)


def _nan_stat(values: np.ndarray, func) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(func(finite))


def summarize_document_metrics(doc: Mapping[str, np.ndarray]) -> dict[str, float]:
    values = np.asarray(doc["doc_topic_entropy_normalized"], dtype=np.float64)
    raw = np.asarray(doc["doc_topic_entropy"], dtype=np.float64)
    return {
        "doc_topic_entropy_mean": _nan_stat(raw, np.mean),
        "doc_topic_entropy_normalized_mean": _nan_stat(values, np.mean),
        "doc_topic_entropy_normalized_std": _nan_stat(
            values, lambda v: np.std(v, ddof=0)
        ),
        "doc_topic_entropy_normalized_median": _nan_stat(values, np.median),
        "doc_topic_entropy_normalized_q10": _nan_stat(
            values, lambda v: np.quantile(v, 0.1)
        ),
        "doc_topic_entropy_normalized_q90": _nan_stat(
            values, lambda v: np.quantile(v, 0.9)
        ),
    }


def topic_flags(
    topic: Mapping[str, np.ndarray],
    *,
    diffuse_entropy_threshold: float = DEFAULT_DIFFUSE_ENTROPY_THRESHOLD,
    dead_rank1_threshold: float = DEFAULT_DEAD_RANK1_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Per-topic ``is_diffuse`` / ``is_dead`` booleans.

    A topic with no mass has ``NaN`` entropy (never diffuse) and zero rank-1
    fraction (always dead).
    """
    entropy_normalized = np.asarray(
        topic["topic_doc_entropy_normalized"], dtype=np.float64
    )
    rank1 = np.asarray(topic["topic_rank1_doc_fraction"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        is_diffuse = np.where(
            np.isfinite(entropy_normalized),
            entropy_normalized > float(diffuse_entropy_threshold),
            False,
        ).astype(bool)
        is_dead = np.where(
            np.isfinite(rank1), rank1 < float(dead_rank1_threshold), False
        ).astype(bool)
    return {"is_diffuse": is_diffuse, "is_dead": is_dead}


def summarize_topic_metrics(
    topic: Mapping[str, np.ndarray],
    *,
    diffuse_entropy_threshold: float = DEFAULT_DIFFUSE_ENTROPY_THRESHOLD,
    dead_rank1_threshold: float = DEFAULT_DEAD_RANK1_THRESHOLD,
) -> dict[str, float]:
    entropy_normalized = np.asarray(
        topic["topic_doc_entropy_normalized"], dtype=np.float64
    )
    entropy_raw = np.asarray(topic["topic_doc_entropy"], dtype=np.float64)
    rank1 = np.asarray(topic["topic_rank1_doc_fraction"], dtype=np.float64)
    flags = topic_flags(
        topic,
        diffuse_entropy_threshold=diffuse_entropy_threshold,
        dead_rank1_threshold=dead_rank1_threshold,
    )
    num_topics = int(entropy_normalized.size)
    return {
        "topic_doc_entropy_mean": _nan_stat(entropy_raw, np.mean),
        "topic_doc_entropy_normalized_mean": _nan_stat(entropy_normalized, np.mean),
        "topic_doc_entropy_normalized_std": _nan_stat(
            entropy_normalized, lambda v: np.std(v, ddof=0)
        ),
        "topic_doc_entropy_normalized_min": _nan_stat(entropy_normalized, np.min),
        "topic_doc_entropy_normalized_max": _nan_stat(entropy_normalized, np.max),
        "topic_rank1_doc_fraction_min": _nan_stat(rank1, np.min),
        "topic_rank1_doc_fraction_max": _nan_stat(rank1, np.max),
        "topic_rank1_doc_fraction_std": _nan_stat(rank1, lambda v: np.std(v, ddof=0)),
        "topic_diffuse_fraction": (
            float(flags["is_diffuse"].sum()) / num_topics
            if num_topics
            else float("nan")
        ),
        "topic_dead_fraction": (
            float(flags["is_dead"].sum()) / num_topics if num_topics else float("nan")
        ),
    }


def compute_entropy_metrics(
    theta: np.ndarray,
    *,
    diffuse_entropy_threshold: float = DEFAULT_DIFFUSE_ENTROPY_THRESHOLD,
    dead_rank1_threshold: float = DEFAULT_DEAD_RANK1_THRESHOLD,
) -> EntropyMetricsResult:
    """Compute all three metrics and their scalar summaries for one ``theta``."""
    arr = np.asarray(theta, dtype=np.float64)
    doc = document_topic_entropy(arr)
    topic_entropy = topic_document_entropy(arr)
    rank1 = topic_rank1_doc_fraction(arr)
    topic = {
        "topic_doc_entropy": topic_entropy["topic_doc_entropy"],
        "topic_doc_entropy_normalized": topic_entropy["topic_doc_entropy_normalized"],
        "topic_rank1_doc_fraction": rank1,
        "empty": topic_entropy["empty"],
    }
    topic.update(
        topic_flags(
            topic,
            diffuse_entropy_threshold=diffuse_entropy_threshold,
            dead_rank1_threshold=dead_rank1_threshold,
        )
    )
    summary: dict[str, float] = {}
    summary.update(summarize_document_metrics(doc))
    summary.update(
        summarize_topic_metrics(
            topic,
            diffuse_entropy_threshold=diffuse_entropy_threshold,
            dead_rank1_threshold=dead_rank1_threshold,
        )
    )
    return EntropyMetricsResult(
        doc=doc,
        topic=topic,
        summary=summary,
        num_documents=int(arr.shape[0]),
        num_topics=int(arr.shape[1]),
        num_zero_mass_docs=int((~doc["valid"]).sum()),
        num_empty_topics=int(topic_entropy["empty"].sum()),
        diffuse_entropy_threshold=float(diffuse_entropy_threshold),
        dead_rank1_threshold=float(dead_rank1_threshold),
    )


def aggregate_metrics(
    per_iter_summaries: Sequence[Mapping[str, float]],
    keys: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Mean/std (ddof=1) across iterations; a single value has std 0; NaN-aware."""
    if keys is None:
        seen: list[str] = []
        for summary in per_iter_summaries:
            for key in summary:
                if key not in seen:
                    seen.append(key)
        keys = seen
    aggregated: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray(
            [float(summary.get(key, np.nan)) for summary in per_iter_summaries],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            aggregated[key] = {"mean": float("nan"), "std": float("nan")}
        elif finite.size == 1:
            aggregated[key] = {"mean": float(finite[0]), "std": 0.0}
        else:
            aggregated[key] = {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)),
            }
    return aggregated

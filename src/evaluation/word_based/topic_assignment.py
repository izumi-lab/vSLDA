"""Topic assignment and topic-word artifact primitives.

The functions in this module deliberately do not know about a particular model's
on-disk layout.  Model adapters only have to provide frozen per-unit likelihoods;
all sampling, validation, expected-count construction and ranking is shared.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal, Sequence

import numpy as np

AssignmentUnitType = Literal["sentence", "token"]
PosteriorBackend = Literal["python", "numba"]

POSTHOC_POSTERIOR_KIND = "posthoc_document_mixture_collapsed_foldin_posterior_mean"
ETM_POSTERIOR_KIND = "variational_token_topic_posterior_mean"
WORD_TOPIC_NPMI_EPSILON = 1e-12


class ConditionEvaluationError(ValueError):
    """A data/model-specific evaluation failure that may be isolated safely."""


class EmptyTopicError(ConditionEvaluationError):
    """One or more fitted topics receive no evaluation-corpus mass."""

    def __init__(self, *, topic_ids: Sequence[int]):
        self.topic_ids = [int(topic_id) for topic_id in topic_ids]
        super().__init__(f"empty topics in expected counts: {self.topic_ids}")


class InsufficientTopicWordsError(ConditionEvaluationError):
    def __init__(self, *, topic_id: int, eligible_words: int, requested_topn: int):
        self.topic_id = int(topic_id)
        self.eligible_words = int(eligible_words)
        self.requested_topn = int(requested_topn)
        self.eligible_word_count = self.eligible_words
        self.required_topn = self.requested_topn
        super().__init__(
            f"topic {self.topic_id} has only {self.eligible_words} eligible words; "
            f"topn={self.requested_topn} was requested"
        )


class DegenerateTopicError(InsufficientTopicWordsError):
    """A topic with too few positive-mass words for the requested evaluation."""

    def __init__(
        self,
        *,
        topic_id: int,
        eligible_word_count: int,
        required_topn: int,
        eligible_words: Sequence[str],
        eligible_word_counts: Sequence[int],
        reason: str = "insufficient_positive_expected_counts",
    ):
        super().__init__(
            topic_id=topic_id,
            eligible_words=eligible_word_count,
            requested_topn=required_topn,
        )
        self.eligible_word_count = int(eligible_word_count)
        self.required_topn = int(required_topn)
        self.eligible_words = [str(word) for word in eligible_words]
        self.eligible_word_counts = np.asarray(eligible_word_counts, dtype=np.int64)
        self.reason = str(reason)
        self.args = (
            f"topic {self.topic_id} has only {self.eligible_word_count} positive-mass "
            f"words; topn={self.required_topn} was requested ({self.reason})",
        )


# Models whose evaluation/display words come from the shared collapsed fold-in
# protocol.  "vmf" is the CLI alias of "vmf_sentence_lda".
COLLAPSED_MODELS = {
    "bleilda",
    "vmf",
    "vmf_sentence_lda",
    "sentlda",
    "sentence_gaussianlda",
    "mvtm",
    "gaussianlda",
}


def resolve_topic_word_protocol(model: str) -> str:
    normalized = "vmf_sentence_lda" if model == "vmf" else str(model)
    if normalized in COLLAPSED_MODELS:
        return "collapsed_posthoc"
    if normalized == "etm":
        return "native_etm_beta_with_variational_responsibilities"
    if normalized == "ctm":
        return "native_ctm_decoder_with_token_responsibilities"
    raise ValueError(f"unsupported post-hoc topic-word model: {model!r}")


@dataclass(frozen=True)
class CollapsedFoldInConfig:
    num_chains: int = 1
    burn_in_sweeps: int = 20
    retained_samples: int = 20
    thinning: int = 1
    random_seed: int = 0
    backend: PosteriorBackend = "numba"

    def validate(self) -> None:
        for name in ("num_chains", "retained_samples", "thinning"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.burn_in_sweeps) < 0:
            raise ValueError("burn_in_sweeps must be non-negative")
        if self.backend not in {"python", "numba"}:
            raise ValueError("backend must be 'python' or 'numba'")


@dataclass(frozen=True)
class CollapsedPosteriorResult:
    posterior_mean_by_doc: list[np.ndarray]
    metadata: dict[str, object]


@dataclass(frozen=True)
class TopicWordStatistics:
    expected_counts: np.ndarray
    topic_counts: np.ndarray
    word_counts: np.ndarray
    total_count: float
    empty_topic_ids: tuple[int, ...] = ()


def fingerprint_jsonable(value: object) -> str:
    """Return a canonical SHA-256 fingerprint for JSON-compatible data."""

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _validate_alpha(alpha: np.ndarray) -> np.ndarray:
    result = np.asarray(alpha, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("alpha must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("alpha must contain only finite positive values")
    return result


def _resolve_log_likelihoods(
    *,
    num_topics: int,
    log_likelihood_by_doc: Sequence[np.ndarray] | None,
    log_likelihood_by_type: np.ndarray | None,
    type_ids_by_doc: Sequence[np.ndarray] | None,
) -> list[np.ndarray]:
    dense_supplied = log_likelihood_by_doc is not None
    typed_supplied = log_likelihood_by_type is not None or type_ids_by_doc is not None
    if dense_supplied == typed_supplied:
        raise ValueError(
            "provide exactly one likelihood representation: per-document matrices "
            "or a type table with per-document type IDs"
        )
    if dense_supplied:
        resolved = [
            np.asarray(item, dtype=np.float64) for item in log_likelihood_by_doc
        ]
    else:
        if log_likelihood_by_type is None or type_ids_by_doc is None:
            raise ValueError(
                "type likelihood table and type IDs must be provided together"
            )
        table = np.asarray(log_likelihood_by_type, dtype=np.float64)
        if table.ndim != 2 or table.shape[1] != num_topics:
            raise ValueError(
                "log_likelihood_by_type must have shape (num_types, num_topics)"
            )
        resolved = []
        for doc_index, raw_ids in enumerate(type_ids_by_doc):
            ids = np.asarray(raw_ids, dtype=np.int64)
            if ids.ndim != 1:
                raise ValueError(f"type IDs for document {doc_index} must be 1D")
            if ids.size and (ids.min() < 0 or ids.max() >= table.shape[0]):
                raise ValueError(f"type ID out of range in document {doc_index}")
            resolved.append(table[ids])
    for doc_index, values in enumerate(resolved):
        if values.ndim != 2 or values.shape[1] != num_topics:
            raise ValueError(
                f"likelihood matrix for document {doc_index} has shape {values.shape}; "
                f"expected (num_units, {num_topics})"
            )
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError(
                f"likelihood matrix for document {doc_index} contains NaN or +inf"
            )
        if values.shape[0] and np.any(np.all(np.isneginf(values), axis=1)):
            raise ValueError(
                f"all topic log likelihoods are -inf for an assignment unit in "
                f"document {doc_index}"
            )
    return resolved


def _init_assignments_kernel(
    log_likelihoods: np.ndarray,
    alpha: np.ndarray,
    uniforms: np.ndarray,
    assignments: np.ndarray,
) -> None:
    """Sample initial topics from alpha_k * p(x | k) with empty document counts."""

    num_units, num_topics = log_likelihoods.shape
    weights = np.empty(num_topics, dtype=np.float64)
    for i in range(num_units):
        maximum = -np.inf
        for k in range(num_topics):
            value = math.log(alpha[k]) + log_likelihoods[i, k]
            weights[k] = value
            if value > maximum:
                maximum = value
        total = 0.0
        for k in range(num_topics):
            value = math.exp(weights[k] - maximum)
            weights[k] = value
            total += value
        target = uniforms[i] * total
        cumulative = 0.0
        chosen = num_topics - 1
        for k in range(num_topics):
            cumulative += weights[k]
            if target < cumulative:
                chosen = k
                break
        assignments[i] = chosen


def _sweep_kernel(
    log_likelihoods: np.ndarray,
    alpha: np.ndarray,
    assignments: np.ndarray,
    counts: np.ndarray,
    uniforms: np.ndarray,
) -> None:
    """One collapsed Gibbs sweep: leave-one-out conditional + inverse-CDF draw."""

    num_units, num_topics = log_likelihoods.shape
    weights = np.empty(num_topics, dtype=np.float64)
    for i in range(num_units):
        old_topic = assignments[i]
        counts[old_topic] -= 1
        maximum = -np.inf
        for k in range(num_topics):
            value = math.log(counts[k] + alpha[k]) + log_likelihoods[i, k]
            weights[k] = value
            if value > maximum:
                maximum = value
        total = 0.0
        for k in range(num_topics):
            value = math.exp(weights[k] - maximum)
            weights[k] = value
            total += value
        target = uniforms[i] * total
        cumulative = 0.0
        chosen = num_topics - 1
        for k in range(num_topics):
            cumulative += weights[k]
            if target < cumulative:
                chosen = k
                break
        counts[chosen] += 1
        assignments[i] = chosen


def _accumulate_kernel(
    log_likelihoods: np.ndarray,
    alpha: np.ndarray,
    assignments: np.ndarray,
    counts: np.ndarray,
    posterior_sum: np.ndarray,
) -> None:
    """Accumulate the leave-one-out conditional in the retained state.  This
    Rao-Blackwellized estimate is less noisy than hard-state averaging."""

    num_units, num_topics = log_likelihoods.shape
    weights = np.empty(num_topics, dtype=np.float64)
    for i in range(num_units):
        current = assignments[i]
        counts[current] -= 1
        maximum = -np.inf
        for k in range(num_topics):
            value = math.log(counts[k] + alpha[k]) + log_likelihoods[i, k]
            weights[k] = value
            if value > maximum:
                maximum = value
        total = 0.0
        for k in range(num_topics):
            value = math.exp(weights[k] - maximum)
            weights[k] = value
            total += value
        for k in range(num_topics):
            posterior_sum[i, k] += weights[k] / total
        counts[current] += 1


_PYTHON_KERNELS = (_init_assignments_kernel, _sweep_kernel, _accumulate_kernel)
_NUMBA_KERNELS: tuple | None = None


def _resolve_kernels(backend: PosteriorBackend) -> tuple:
    """Return (init, sweep, accumulate) kernels for the requested backend.

    Both backends execute the same function bodies with the same externally
    supplied uniforms, so their results are bit-identical; ``numba`` compiles
    them with ``njit`` for throughput.
    """

    if backend == "python":
        return _PYTHON_KERNELS
    global _NUMBA_KERNELS
    if _NUMBA_KERNELS is None:
        try:
            from numba import njit
        except ImportError as error:
            raise RuntimeError(
                "backend='numba' requires the numba package; install numba or "
                "rerun with backend='python' (results are identical)"
            ) from error
        _NUMBA_KERNELS = tuple(njit(cache=True)(kernel) for kernel in _PYTHON_KERNELS)
    return _NUMBA_KERNELS


def _sample_document_chain(
    likelihoods: np.ndarray,
    alpha: np.ndarray,
    config: CollapsedFoldInConfig,
    rng: np.random.Generator,
    kernels: tuple,
) -> np.ndarray:
    init_kernel, sweep_kernel, accumulate_kernel = kernels
    num_units, num_topics = likelihoods.shape
    if num_units == 0:
        return np.empty((0, num_topics), dtype=np.float64)

    assignments = np.empty(num_units, dtype=np.int64)
    init_kernel(likelihoods, alpha, rng.random(num_units), assignments)
    counts = np.bincount(assignments, minlength=num_topics).astype(np.int64)

    posterior_sum = np.zeros((num_units, num_topics), dtype=np.float64)
    retained = 0
    final_sweep = config.burn_in_sweeps + config.retained_samples * config.thinning
    for sweep in range(1, final_sweep + 1):
        sweep_kernel(likelihoods, alpha, assignments, counts, rng.random(num_units))
        if (
            sweep > config.burn_in_sweeps
            and (sweep - config.burn_in_sweeps) % config.thinning == 0
        ):
            accumulate_kernel(likelihoods, alpha, assignments, counts, posterior_sum)
            retained += 1
    if retained != config.retained_samples:
        raise RuntimeError("internal retained-sample count mismatch")
    return posterior_sum / float(retained)


def validate_posterior_mean(
    posterior_mean_by_doc: Sequence[np.ndarray], *, num_topics: int
) -> None:
    for doc_index, raw in enumerate(posterior_mean_by_doc):
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != num_topics:
            raise ValueError(
                f"posterior for document {doc_index} must have width {num_topics}"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"posterior for document {doc_index} must be finite and non-negative"
            )
        if values.shape[0] and not np.allclose(
            values.sum(axis=1), 1.0, atol=1e-10, rtol=1e-10
        ):
            raise ValueError(
                f"posterior rows do not sum to one in document {doc_index}"
            )


def run_collapsed_fold_in(
    *,
    alpha: np.ndarray,
    assignment_unit_type: AssignmentUnitType,
    config: CollapsedFoldInConfig | None = None,
    log_likelihood_by_doc: Sequence[np.ndarray] | None = None,
    log_likelihood_by_type: np.ndarray | None = None,
    type_ids_by_doc: Sequence[np.ndarray] | None = None,
    source_condition_fingerprint: str | None = None,
    corpus_fingerprint: str | None = None,
    vocabulary_fingerprint: str | None = None,
    npmi_min_expected_count: float | None = None,
) -> CollapsedPosteriorResult:
    """Run frozen-global-parameter document-level collapsed fold-in.

    Uniform random draws are generated by the numpy ``Generator`` and consumed
    by shared inverse-CDF kernels, so ``backend='python'`` and
    ``backend='numba'`` return bit-identical posteriors.
    """

    if assignment_unit_type not in {"sentence", "token"}:
        raise ValueError("assignment_unit_type must be 'sentence' or 'token'")
    actual_config = config or CollapsedFoldInConfig()
    actual_config.validate()
    resolved_alpha = _validate_alpha(alpha)
    likelihoods = _resolve_log_likelihoods(
        num_topics=resolved_alpha.size,
        log_likelihood_by_doc=log_likelihood_by_doc,
        log_likelihood_by_type=log_likelihood_by_type,
        type_ids_by_doc=type_ids_by_doc,
    )
    kernels = _resolve_kernels(actual_config.backend)

    sums = [np.zeros_like(item, dtype=np.float64) for item in likelihoods]
    seed_sequence = np.random.SeedSequence(actual_config.random_seed)
    chain_seeds = seed_sequence.spawn(actual_config.num_chains)
    for chain_seed in chain_seeds:
        rng = np.random.default_rng(chain_seed)
        for doc_index, doc_likelihoods in enumerate(likelihoods):
            sums[doc_index] += _sample_document_chain(
                doc_likelihoods, resolved_alpha, actual_config, rng, kernels
            )
    posterior = [item / float(actual_config.num_chains) for item in sums]
    validate_posterior_mean(posterior, num_topics=resolved_alpha.size)
    metadata: dict[str, object] = {
        "posterior_kind": POSTHOC_POSTERIOR_KIND,
        "assignment_unit_type": assignment_unit_type,
        "global_topic_parameters": "frozen",
        **asdict(actual_config),
        "initialization": "seeded_alpha_likelihood_sample",
        "npmi_min_expected_count": npmi_min_expected_count,
        "source_condition_fingerprint": source_condition_fingerprint,
        "corpus_fingerprint": corpus_fingerprint,
        "vocabulary_fingerprint": vocabulary_fingerprint,
    }
    return CollapsedPosteriorResult(posterior, metadata)


def _accumulate_unit_word_counts(
    expected_counts: np.ndarray,
    unit_probabilities: np.ndarray,
    counts: object,
    *,
    vocab_size: int,
    doc_index: int,
    unit_index: int,
) -> None:
    """Add one unit's expected counts in O(nnz * K) without a dense vocab row."""

    if isinstance(counts, np.ndarray):
        row = np.asarray(counts, dtype=np.float64)
        if row.ndim != 1 or row.shape[0] != vocab_size:
            raise ValueError(
                f"word-count row at document {doc_index}, unit {unit_index} has "
                f"shape {row.shape}; expected ({vocab_size},)"
            )
        if not np.all(np.isfinite(row)) or np.any(row < 0.0):
            raise ValueError(
                f"invalid word-count row at document {doc_index}, unit {unit_index}"
            )
        word_ids = np.flatnonzero(row)
        if word_ids.size:
            expected_counts[:, word_ids] += (
                unit_probabilities[:, None] * row[word_ids][None, :]
            )
        return
    for word_id, count in counts:  # type: ignore[union-attr]
        word_id = int(word_id)
        count = float(count)
        if not 0 <= word_id < vocab_size or not np.isfinite(count) or count < 0.0:
            raise ValueError(
                f"invalid sparse word count at document {doc_index}, unit {unit_index}"
            )
        expected_counts[:, word_id] += unit_probabilities * count


def compute_expected_topic_word_counts(
    *,
    assignment_probabilities_by_doc: Sequence[np.ndarray],
    unit_word_counts_by_doc: Sequence[object],
    vocab_size: int,
    expected_covered_word_counts: np.ndarray | None = None,
    allow_empty_topics: bool = False,
) -> TopicWordStatistics:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if len(assignment_probabilities_by_doc) != len(unit_word_counts_by_doc):
        raise ValueError("posterior and word-count document counts differ")
    if not assignment_probabilities_by_doc:
        raise ValueError("at least one document is required")
    first = np.asarray(assignment_probabilities_by_doc[0], dtype=np.float64)
    if first.ndim != 2 or first.shape[1] == 0:
        raise ValueError("assignment probabilities must be two-dimensional")
    num_topics = first.shape[1]
    validate_posterior_mean(assignment_probabilities_by_doc, num_topics=num_topics)
    expected_counts = np.zeros((num_topics, vocab_size), dtype=np.float64)

    for doc_index, (raw_probabilities, raw_units) in enumerate(
        zip(assignment_probabilities_by_doc, unit_word_counts_by_doc)
    ):
        probabilities = np.asarray(raw_probabilities, dtype=np.float64)
        if isinstance(raw_units, np.ndarray) and raw_units.ndim == 2:
            if raw_units.shape != (probabilities.shape[0], vocab_size):
                raise ValueError(
                    f"unit count alignment mismatch in document {doc_index}"
                )
            unit_counts = np.asarray(raw_units, dtype=np.float64)
            if not np.all(np.isfinite(unit_counts)) or np.any(unit_counts < 0.0):
                raise ValueError("unit word counts must be finite and non-negative")
            expected_counts += probabilities.T @ unit_counts
            continue
        units = list(raw_units)  # type: ignore[arg-type]
        if len(units) != probabilities.shape[0]:
            raise ValueError(f"unit count alignment mismatch in document {doc_index}")
        for unit_index, raw_counts in enumerate(units):
            _accumulate_unit_word_counts(
                expected_counts,
                probabilities[unit_index],
                raw_counts,
                vocab_size=vocab_size,
                doc_index=doc_index,
                unit_index=unit_index,
            )

    word_counts = expected_counts.sum(axis=0)
    if expected_covered_word_counts is not None:
        expected = np.asarray(expected_covered_word_counts, dtype=np.float64)
        if expected.shape != (vocab_size,):
            raise ValueError("expected covered word counts have the wrong shape")
        if not np.allclose(word_counts, expected, atol=1e-8, rtol=1e-8):
            max_error = float(np.max(np.abs(word_counts - expected)))
            raise ValueError(
                "topic-word mass is not conserved on the covered token set; "
                f"maximum absolute error={max_error:.6g}"
            )
    topic_counts = expected_counts.sum(axis=1)
    empty = np.flatnonzero(topic_counts <= 0.0)
    if empty.size and not allow_empty_topics:
        raise EmptyTopicError(topic_ids=empty.tolist())
    return TopicWordStatistics(
        expected_counts=expected_counts,
        topic_counts=topic_counts,
        word_counts=word_counts,
        total_count=float(expected_counts.sum()),
        empty_topic_ids=tuple(int(topic_id) for topic_id in empty),
    )


def compute_coverage(
    *, covered_word_counts: np.ndarray, corpus_word_counts: np.ndarray
) -> dict[str, float | int]:
    covered = np.asarray(covered_word_counts, dtype=np.float64)
    corpus = np.asarray(corpus_word_counts, dtype=np.float64)
    if covered.shape != corpus.shape or covered.ndim != 1:
        raise ValueError("covered and corpus word counts must be aligned 1D arrays")
    if np.any(covered < 0.0) or np.any(corpus < 0.0) or np.any(covered > corpus + 1e-8):
        raise ValueError("covered counts must lie between zero and corpus counts")
    corpus_types = int(np.count_nonzero(corpus))
    covered_types = int(np.count_nonzero(covered))
    corpus_tokens = float(corpus.sum())
    covered_tokens = float(covered.sum())
    return {
        "vocabulary_size": int(corpus.size),
        "corpus_type_count": corpus_types,
        "covered_type_count": covered_types,
        "type_coverage": covered_types / corpus_types if corpus_types else 0.0,
        "corpus_token_count": corpus_tokens,
        "covered_token_count": covered_tokens,
        "token_coverage": covered_tokens / corpus_tokens if corpus_tokens else 0.0,
    }


def topic_word_probabilities(
    statistics: TopicWordStatistics, *, allow_empty_topics: bool = False
) -> np.ndarray:
    nonempty = statistics.topic_counts > 0.0
    if not np.all(nonempty) and not allow_empty_topics:
        raise ValueError("topic-word probabilities are undefined for empty topics")
    probabilities = np.full(statistics.expected_counts.shape, np.nan, dtype=np.float64)
    probabilities[nonempty] = (
        statistics.expected_counts[nonempty] / statistics.topic_counts[nonempty, None]
    )
    return probabilities


def word_topic_npmi(
    statistics: TopicWordStatistics,
    *,
    min_expected_count: float | None = None,
    epsilon: float = WORD_TOPIC_NPMI_EPSILON,
    allow_empty_topics: bool = False,
) -> np.ndarray:
    """Score word-topic pairs with epsilon-smoothed joint probabilities.

    Words with positive corpus marginal mass are eligible by default, including
    pairs whose expected joint count is zero.  An explicit ``min_expected_count``
    retains the stricter count filter for callers that want it.
    """
    counts = np.asarray(statistics.expected_counts, dtype=np.float64)
    if statistics.total_count <= 0.0:
        raise ValueError("word-topic NPMI requires positive total mass")
    nonempty = statistics.topic_counts > 0.0
    if not np.all(nonempty) and not allow_empty_topics:
        raise ValueError("word-topic NPMI is undefined for empty topics")
    threshold = 0.0 if min_expected_count is None else float(min_expected_count)
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError("min_expected_count must be finite and non-negative")
    smoothing = float(epsilon)
    if smoothing <= 0.0 or not np.isfinite(smoothing):
        raise ValueError("epsilon must be finite and positive")
    scores = np.full(counts.shape, np.nan, dtype=np.float64)
    eligible_words = statistics.word_counts > 0.0
    if min_expected_count is None:
        eligible = np.broadcast_to(eligible_words, counts.shape)
    else:
        eligible = (counts > threshold) & eligible_words[None, :]
    eligible = eligible & nonempty[:, None]
    if not np.any(eligible):
        return scores
    rows, columns = np.nonzero(eligible)
    selected_counts = counts[rows, columns]
    log_total = math.log(statistics.total_count)
    # Add smoothing before division so even the smallest positive float cannot
    # underflow while being normalized.
    smoothed_joint = (
        selected_counts + smoothing * statistics.total_count
    ) / statistics.total_count
    denominator = -np.log(smoothed_joint)
    if np.any(denominator <= 0.0):
        raise ValueError(
            "word-topic NPMI is undefined when smoothed p(w,k) is at least one"
        )
    pmi = (
        np.log(smoothed_joint)
        + 2.0 * log_total
        - np.log(statistics.topic_counts[rows])
        - np.log(statistics.word_counts[columns])
    )
    scores[rows, columns] = pmi / denominator
    return scores


def stable_top_word_indices(
    scores: np.ndarray,
    *,
    topn: int,
    eligible_mask: np.ndarray | None = None,
    empty_topic_ids: Sequence[int] = (),
) -> list[np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must be two-dimensional")
    if topn <= 0:
        raise ValueError("topn must be positive")
    if eligible_mask is None:
        eligible = np.isfinite(values)
    else:
        eligible = np.array(eligible_mask, dtype=bool, copy=True)
        if eligible.shape != values.shape:
            raise ValueError("eligible mask shape differs from scores")
        eligible &= np.isfinite(values)
    result: list[np.ndarray] = []
    allowed_empty = {int(topic_id) for topic_id in empty_topic_ids}
    invalid_empty = sorted(
        topic_id for topic_id in allowed_empty if not 0 <= topic_id < values.shape[0]
    )
    if invalid_empty:
        raise ValueError(f"empty topic IDs are out of range: {invalid_empty}")
    for topic_index in range(values.shape[0]):
        ids = np.flatnonzero(eligible[topic_index])
        if topic_index in allowed_empty and ids.size == 0:
            result.append(np.asarray([], dtype=np.int64))
            continue
        if ids.size < topn:
            raise InsufficientTopicWordsError(
                topic_id=topic_index,
                eligible_words=int(ids.size),
                requested_topn=topn,
            )
        order = np.argsort(-values[topic_index, ids], kind="stable")
        result.append(ids[order[:topn]])
    return result


def rank_topic_words(
    scores: np.ndarray,
    vocabulary: Sequence[str],
    *,
    topn: int,
    eligible_mask: np.ndarray | None = None,
    empty_topic_ids: Sequence[int] = (),
) -> list[list[tuple[str, float]]]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(vocabulary):
        raise ValueError("score matrix and vocabulary are not aligned")
    return [
        [
            (str(vocabulary[word_id]), float(values[topic_id, word_id]))
            for word_id in ids
        ]
        for topic_id, ids in enumerate(
            stable_top_word_indices(
                values,
                topn=topn,
                eligible_mask=eligible_mask,
                empty_topic_ids=empty_topic_ids,
            )
        )
    ]


def compute_etm_token_topic_posterior_mean(
    *,
    theta_samples: np.ndarray,
    beta: np.ndarray,
    corpus_bow: Sequence[Sequence[tuple[int, int]]],
) -> list[np.ndarray]:
    """Return one responsibility row per observed ETM token occurrence."""

    theta = np.asarray(theta_samples, dtype=np.float64)
    topic_word = np.asarray(beta, dtype=np.float64)
    if theta.ndim == 2:
        theta = theta[None, ...]
    if theta.ndim != 3 or topic_word.ndim != 2:
        raise ValueError("invalid theta or beta shape")
    if theta.shape[1] != len(corpus_bow) or theta.shape[2] != topic_word.shape[0]:
        raise ValueError("theta, beta and corpus are not aligned")
    posterior_by_doc: list[np.ndarray] = []
    for doc_index, bow in enumerate(corpus_bow):
        rows: list[np.ndarray] = []
        for raw_word_id, raw_count in bow:
            word_id = int(raw_word_id)
            count_float = float(raw_count)
            count = int(count_float)
            if (
                count_float != count
                or count < 0
                or not 0 <= word_id < topic_word.shape[1]
            ):
                raise ValueError(
                    "ETM token posterior requires non-negative integer BoW counts"
                )
            unnormalized = theta[:, doc_index, :] * topic_word[:, word_id][None, :]
            denominators = unnormalized.sum(axis=1)
            if np.any(denominators <= 0.0):
                raise ValueError(
                    f"ETM token responsibility is undefined for word ID {word_id}"
                )
            responsibility = (unnormalized / denominators[:, None]).mean(axis=0)
            rows.extend([responsibility.copy() for _ in range(count)])
        posterior_by_doc.append(
            np.vstack(rows)
            if rows
            else np.empty((0, topic_word.shape[0]), dtype=np.float64)
        )
    validate_posterior_mean(posterior_by_doc, num_topics=topic_word.shape[0])
    return posterior_by_doc


def compute_etm_expected_counts(
    *,
    theta_samples: np.ndarray,
    beta: np.ndarray,
    corpus_bow: Sequence[Sequence[tuple[int, int]]],
) -> tuple[TopicWordStatistics, dict[str, object]]:
    """Compute ETM token-topic responsibilities averaged over theta samples."""

    theta = np.asarray(theta_samples, dtype=np.float64)
    topic_word = np.asarray(beta, dtype=np.float64)
    if theta.ndim == 2:
        theta = theta[None, ...]
    if theta.ndim != 3:
        raise ValueError("theta_samples must have shape (samples, docs, topics)")
    num_samples, num_docs, num_topics = theta.shape
    if num_samples <= 0 or num_docs != len(corpus_bow):
        raise ValueError("theta samples and corpus documents are not aligned")
    if topic_word.ndim != 2 or topic_word.shape[0] != num_topics:
        raise ValueError("beta must have shape (topics, vocabulary)")
    if not np.all(np.isfinite(theta)) or np.any(theta < 0.0):
        raise ValueError("theta samples must be finite and non-negative")
    if not np.allclose(theta.sum(axis=2), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("theta samples must sum to one")
    if not np.all(np.isfinite(topic_word)) or np.any(topic_word < 0.0):
        raise ValueError("beta must be finite and non-negative")
    if not np.allclose(topic_word.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("beta rows must sum to one")

    vocab_size = topic_word.shape[1]
    expected_counts = np.zeros((num_topics, vocab_size), dtype=np.float64)
    observed_counts = np.zeros(vocab_size, dtype=np.float64)
    for doc_index, bow in enumerate(corpus_bow):
        for raw_word_id, raw_count in bow:
            word_id, count = int(raw_word_id), float(raw_count)
            if not 0 <= word_id < vocab_size or count < 0.0 or not np.isfinite(count):
                raise ValueError(f"invalid ETM BoW entry in document {doc_index}")
            unnormalized = theta[:, doc_index, :] * topic_word[:, word_id][None, :]
            denominators = unnormalized.sum(axis=1)
            if np.any(denominators <= 0.0):
                raise ValueError(
                    f"ETM token responsibility is undefined for word ID {word_id}"
                )
            responsibility = (unnormalized / denominators[:, None]).mean(axis=0)
            expected_counts[:, word_id] += count * responsibility
            observed_counts[word_id] += count
    stats = TopicWordStatistics(
        expected_counts=expected_counts,
        topic_counts=expected_counts.sum(axis=1),
        word_counts=expected_counts.sum(axis=0),
        total_count=float(expected_counts.sum()),
    )
    if not np.allclose(stats.word_counts, observed_counts, atol=1e-8, rtol=1e-8):
        raise RuntimeError("ETM responsibilities did not conserve token mass")
    empty = np.flatnonzero(stats.topic_counts <= 0.0)
    if empty.size:
        raise ValueError(f"empty ETM topics in expected counts: {empty.tolist()}")
    return stats, {
        "posterior_kind": ETM_POSTERIOR_KIND,
        "theta_samples": int(num_samples),
        "global_topic_parameters": "frozen",
    }


def build_topic_words_artifact(
    *,
    topic_words: Sequence[Sequence[tuple[str, float]]],
    topic_word_role: Literal["evaluation", "display"],
    score_mode: str,
    score_definition: str,
    source: str,
    consumers: Sequence[str],
    model: str,
    split: str,
    topn: int,
    vocabulary_fingerprint: str,
    condition_fingerprint: str,
) -> dict[str, object]:
    if topic_word_role not in {"evaluation", "display"}:
        raise ValueError("invalid topic-word role")
    return {
        "topic_word_role": topic_word_role,
        "score_mode": score_mode,
        "score_definition": score_definition,
        "source": source,
        "consumers": list(consumers),
        "model": model,
        "split": split,
        "topn": int(topn),
        "vocabulary_fingerprint": vocabulary_fingerprint,
        "condition_fingerprint": condition_fingerprint,
        "topics": [
            {
                "topic_id": topic_id,
                "words": [
                    {"word": str(word), "score": float(score)} for word, score in words
                ],
            }
            for topic_id, words in enumerate(topic_words)
        ],
    }

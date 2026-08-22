from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from src.baselines.models.gaussian_helpers import load_gaussianlda_model
from src.baselines.models.sentence_gaussian_helpers import (
    build_sentence_gaussian_encoder,
    load_sentence_gaussianlda_model,
)
from src.baselines.params import format_prior_scale_variant
from src.core.artifacts import (
    load_artifact_json,
    load_artifact_pickle,
)
from src.core.paths import resolve_baseline_condition_dir, resolve_vmf_experiment_dir
from src.data.preprocessing import PreprocessedDocument
from src.data.preprocessing_selection import (
    parse_raw_doc_indices,
    resolve_preprocessing_selection,
)
from src.evaluation.word_based.sentence_encoding import (
    ResolvedEncodeBatchSize,
    encode_sentence_corpus,
    resolve_topic_word_encode_batch_size,
)
from src.evaluation.word_based.topic_assignment import (
    CollapsedFoldInConfig,
    ConditionEvaluationError,
    DegenerateTopicError,
    TopicWordStatistics,
    compute_coverage,
    compute_etm_expected_counts,
    compute_etm_token_topic_posterior_mean,
    compute_expected_topic_word_counts,
    rank_topic_words,
    run_collapsed_fold_in,
    topic_word_probabilities,
    word_topic_npmi,
)
from src.evaluation.word_based.topic_word_audit import audit_model_artifacts
from src.evaluation.word_based.topic_word_model_adapters import (
    lda_log_likelihood_by_type,
    normalize_ctm_decoder_scores,
    restrict_scores_to_evaluation_vocabulary,
    sample_etm_theta,
    sentlda_log_likelihood_by_doc,
    vmf_mixture_log_likelihood,
)
from src.evaluation.word_based.topic_words import (
    TopicWords,
    TopicWordsResult,
    load_ctm_decoder_scores,
)
from src.utils.embedding_preprocess import EmbeddingPreprocessor
from src.utils.encoder import SentenceEncoder

if TYPE_CHECKING:
    from src.baselines.models.etm import EtmModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeTopicWords:
    evaluation: TopicWordsResult
    display_topic_words: TopicWords
    display_source: str
    display_score_mode: str
    protocol: str
    condition_dir: Path
    source_condition_fingerprint: str
    vocabulary_fingerprint: str
    corpus_fingerprint: str
    coverage: dict[str, float | int]
    posterior_mean_by_doc: list[np.ndarray] | None = None
    posterior_metadata: dict[str, object] | None = None
    expected_counts: np.ndarray | None = None
    execution_metadata: dict[str, object] | None = None
    empty_topic_ids: tuple[int, ...] = ()


TopicWordScoreMode = Literal["word_topic_npmi", "topic_word_probability"]
DEFAULT_TOPIC_WORD_SCORE_MODE: TopicWordScoreMode = "word_topic_npmi"
TOPIC_WORD_RANKING_SCHEMA_VERSION = 3
# Models whose result paths carry a prior-scale (Psi_0) parameter variant.
# Mirrors GAUSSIAN_PRIOR_SCALE_RUNNERS in src/baselines/adapter_runtime.py.
GAUSSIAN_PRIOR_SCALE_MODELS = {"gaussianlda", "sentence_gaussianlda", "gaussian"}


def select_metric_topic_words(
    runtime: RuntimeTopicWords,
    score_mode: str = DEFAULT_TOPIC_WORD_SCORE_MODE,
) -> TopicWordsResult:
    """Select the single ranking consumed by coherence and diversity."""

    if score_mode == "topic_word_probability":
        return runtime.evaluation
    if score_mode != "word_topic_npmi":
        raise ValueError(
            "topic_word_score_mode must be word_topic_npmi or " "topic_word_probability"
        )
    if runtime.display_score_mode != "word_topic_npmi":
        raise ValueError(
            f"protocol {runtime.protocol!r} does not provide word-topic NPMI words"
        )
    return TopicWordsResult(
        topic_words=runtime.display_topic_words,
        topic_word_source=runtime.display_source,
        score_mode=runtime.display_score_mode,
        score_definition=(
            "NPMI between word and topic from expected topic-word counts "
            "with joint-probability epsilon=1e-12"
        ),
    )


def _vocabulary(dictionary: Dictionary) -> list[str]:
    return [str(dictionary[index]) for index in range(len(dictionary))]


def _fingerprint(value: object) -> str:
    from src.evaluation.word_based.topic_assignment import fingerprint_jsonable

    return fingerprint_jsonable(value)


def _condition_metadata(condition_dir: Path) -> dict[str, Any]:
    path = condition_dir / "metadata.json"
    payload = load_artifact_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid condition metadata: {path}")
    return payload


def _condition_fingerprint(condition_dir: Path) -> str:
    metadata = _condition_metadata(condition_dir)
    value = metadata.get("condition_fingerprint")
    if value in {None, ""} and isinstance(metadata.get("axes"), dict):
        value = metadata.get("condition_id")
    if value in {None, ""}:
        raise ValueError(f"Condition fingerprint is missing: {condition_dir}")
    return str(value)


def _split_paths(*, model: str, condition_dir: Path, split: str) -> tuple[Path, Path]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if model == "vmf":
        preprocessed = condition_dir / (
            "train_preprocessed.pkl" if split == "train" else "test_preprocessed.pkl"
        )
        selection = condition_dir / "preprocessing_selection.json"
    else:
        split_dir = condition_dir / ("params" if split == "train" else "infer")
        preprocessed = split_dir / "preprocessed_corpus.pkl"
        selection = split_dir / "preprocessing_selection.json"
    if not preprocessed.exists():
        raise FileNotFoundError(f"Preprocessed corpus not found: {preprocessed}")
    if not selection.exists():
        raise FileNotFoundError(f"Preprocessing selection not found: {selection}")
    return preprocessed, selection


def _load_documents_and_ids(
    *, model: str, condition_dir: Path, split: str
) -> tuple[list[PreprocessedDocument], list[int]]:
    preprocessed_path, selection_path = _split_paths(
        model=model, condition_dir=condition_dir, split=split
    )
    raw_documents = load_artifact_pickle(preprocessed_path)
    if not isinstance(raw_documents, list) or not all(
        isinstance(document, PreprocessedDocument) for document in raw_documents
    ):
        raise ValueError(f"Invalid preprocessed corpus: {preprocessed_path}")
    payload = load_artifact_json(selection_path)
    selection = resolve_preprocessing_selection(
        payload, split=split, selection_path=selection_path
    )
    raw_ids = parse_raw_doc_indices(
        selection, selection_path=selection_path, split=split
    )
    if len(raw_ids) != len(raw_documents):
        raise ValueError(
            "Preprocessing selection/document mismatch: "
            f"model={model} split={split} selection={selection_path} "
            f"IDs={len(raw_ids)} corpus={preprocessed_path} "
            f"documents={len(raw_documents)}"
        )
    return raw_documents, raw_ids


def _corpus_word_counts(
    documents: Sequence[PreprocessedDocument], evaluation_id: dict[str, int]
) -> np.ndarray:
    result = np.zeros(len(evaluation_id), dtype=np.float64)
    for document in documents:
        for token in document.document_tokens:
            word_id = evaluation_id.get(token)
            if word_id is not None:
                result[word_id] += 1.0
    return result


def _sentence_word_counts(
    documents: Sequence[PreprocessedDocument],
    evaluation_id: dict[str, int],
    *,
    supported_words: set[str] | None = None,
) -> tuple[list[list[list[tuple[int, int]]]], np.ndarray]:
    counts_by_doc: list[list[list[tuple[int, int]]]] = []
    covered = np.zeros(len(evaluation_id), dtype=np.float64)
    for document in documents:
        doc_counts: list[list[tuple[int, int]]] = []
        for sentence in document.sentences_tokenized:
            counts: dict[int, int] = {}
            for token in sentence:
                if supported_words is not None and token not in supported_words:
                    continue
                word_id = evaluation_id.get(token)
                if word_id is None:
                    continue
                counts[word_id] = counts.get(word_id, 0) + 1
                covered[word_id] += 1.0
            doc_counts.append(sorted(counts.items()))
        counts_by_doc.append(doc_counts)
    return counts_by_doc, covered


def _token_units(
    documents: Sequence[PreprocessedDocument],
    evaluation_id: dict[str, int],
    supported_words: set[str],
) -> tuple[list[np.ndarray], list[list[list[tuple[int, int]]]], np.ndarray, list[str]]:
    covered_words = [word for word in evaluation_id if word in supported_words]
    compact_id = {word: index for index, word in enumerate(covered_words)}
    type_ids_by_doc: list[np.ndarray] = []
    unit_counts_by_doc: list[list[list[tuple[int, int]]]] = []
    covered = np.zeros(len(evaluation_id), dtype=np.float64)
    for document in documents:
        type_ids: list[int] = []
        unit_counts: list[list[tuple[int, int]]] = []
        for token in document.document_tokens:
            eval_id = evaluation_id.get(token)
            local_id = compact_id.get(token)
            if eval_id is None or local_id is None:
                continue
            type_ids.append(local_id)
            unit_counts.append([(eval_id, 1)])
            covered[eval_id] += 1.0
        type_ids_by_doc.append(np.asarray(type_ids, dtype=np.int64))
        unit_counts_by_doc.append(unit_counts)
    return type_ids_by_doc, unit_counts_by_doc, covered, covered_words


def _alpha_vector(alpha: object, num_topics: int) -> np.ndarray:
    values = np.asarray(alpha, dtype=np.float64)
    if values.ndim == 0:
        return np.full(num_topics, float(values), dtype=np.float64)
    if values.shape != (num_topics,):
        raise ValueError(
            f"Invalid alpha shape {values.shape}; expected ({num_topics},)"
        )
    return values


def _statistics_from_counts(counts: np.ndarray) -> TopicWordStatistics:
    values = np.asarray(counts, dtype=np.float64)
    return TopicWordStatistics(
        expected_counts=values,
        topic_counts=values.sum(axis=1),
        word_counts=values.sum(axis=0),
        total_count=float(values.sum()),
    )


def _raise_for_degenerate_topics(
    counts: np.ndarray,
    *,
    vocabulary: Sequence[str],
    required_topn: int,
    eligible_mask: np.ndarray | None = None,
    allow_zero_joint_counts: bool = False,
    empty_topic_ids: Sequence[int] = (),
) -> None:
    values = np.asarray(counts, dtype=np.float64)
    if allow_zero_joint_counts:
        # Epsilon-smoothed word-topic NPMI can rank a zero-joint-count pair as
        # long as the word has positive marginal mass in the covered corpus.
        eligible = np.broadcast_to(values.sum(axis=0) > 0.0, values.shape).copy()
    else:
        eligible = values > 0.0
    if eligible_mask is not None:
        mask = np.asarray(eligible_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("degenerate-topic eligibility mask is not aligned")
        eligible &= mask
    eligible_word_counts = np.count_nonzero(eligible, axis=1)
    allowed_empty = {int(topic_id) for topic_id in empty_topic_ids}
    invalid = np.asarray(
        [
            topic_id
            for topic_id in np.flatnonzero(eligible_word_counts < required_topn)
            if int(topic_id) not in allowed_empty
        ],
        dtype=np.int64,
    )
    if not invalid.size:
        return
    topic_id = int(invalid[0])
    word_ids = np.flatnonzero(eligible[topic_id])[:10]
    raise DegenerateTopicError(
        topic_id=topic_id,
        eligible_word_count=int(eligible_word_counts[topic_id]),
        required_topn=int(required_topn),
        eligible_words=[str(vocabulary[word_id]) for word_id in word_ids],
        eligible_word_counts=eligible_word_counts,
    )


def _collapsed_result(
    *,
    model: str,
    condition_dir: Path,
    split: str,
    vocabulary: list[str],
    documents: list[PreprocessedDocument],
    raw_ids: list[int],
    alpha: np.ndarray,
    assignment_unit_type: str,
    unit_word_counts_by_doc: Sequence[object],
    covered_word_counts: np.ndarray,
    corpus_word_counts: np.ndarray,
    config: CollapsedFoldInConfig,
    topn: int,
    log_likelihood_by_doc: Sequence[np.ndarray] | None = None,
    log_likelihood_by_type: np.ndarray | None = None,
    type_ids_by_doc: Sequence[np.ndarray] | None = None,
    npmi_min_expected_count: float | None = None,
    execution_metadata: dict[str, object] | None = None,
    topic_words_started_at: float | None = None,
) -> RuntimeTopicWords:
    vocabulary_fingerprint = _fingerprint({"ordered_vocabulary": vocabulary})
    corpus_fingerprint = _fingerprint({"raw_document_ids": raw_ids, "split": split})
    condition_fingerprint = _condition_fingerprint(condition_dir)
    foldin_started_at = time.perf_counter()
    posterior = run_collapsed_fold_in(
        alpha=alpha,
        assignment_unit_type=assignment_unit_type,  # type: ignore[arg-type]
        config=config,
        log_likelihood_by_doc=log_likelihood_by_doc,
        log_likelihood_by_type=log_likelihood_by_type,
        type_ids_by_doc=type_ids_by_doc,
        source_condition_fingerprint=condition_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        vocabulary_fingerprint=vocabulary_fingerprint,
        npmi_min_expected_count=npmi_min_expected_count,
    )
    collapsed_foldin_sec = time.perf_counter() - foldin_started_at
    ranking_started_at = time.perf_counter()
    statistics = compute_expected_topic_word_counts(
        assignment_probabilities_by_doc=posterior.posterior_mean_by_doc,
        unit_word_counts_by_doc=unit_word_counts_by_doc,
        vocab_size=len(vocabulary),
        expected_covered_word_counts=covered_word_counts,
        allow_empty_topics=True,
    )
    _raise_for_degenerate_topics(
        statistics.expected_counts,
        vocabulary=vocabulary,
        required_topn=topn,
        allow_zero_joint_counts=True,
        empty_topic_ids=statistics.empty_topic_ids,
    )
    evaluation_words = rank_topic_words(
        topic_word_probabilities(statistics, allow_empty_topics=True),
        vocabulary,
        topn=topn,
        empty_topic_ids=statistics.empty_topic_ids,
    )
    display_words = rank_topic_words(
        word_topic_npmi(
            statistics,
            min_expected_count=npmi_min_expected_count,
            allow_empty_topics=True,
        ),
        vocabulary,
        topn=topn,
        empty_topic_ids=statistics.empty_topic_ids,
    )
    resolved_execution_metadata = (
        None if execution_metadata is None else dict(execution_metadata)
    )
    if resolved_execution_metadata is not None:
        resolved_execution_metadata["collapsed_foldin_sec"] = float(
            collapsed_foldin_sec
        )
        resolved_execution_metadata["topic_word_ranking_sec"] = float(
            time.perf_counter() - ranking_started_at
        )
        if topic_words_started_at is not None:
            resolved_execution_metadata["topic_words_total_sec"] = float(
                time.perf_counter() - topic_words_started_at
            )
    return RuntimeTopicWords(
        evaluation=TopicWordsResult(
            topic_words=evaluation_words,
            topic_word_source="posthoc_expected_p_w_given_topic",
            score_mode="topic_word_probability",
            score_definition="p(w|topic) from expected assignment counts",
        ),
        display_topic_words=display_words,
        display_source="posthoc_word_topic_npmi",
        display_score_mode="word_topic_npmi",
        protocol="collapsed_posthoc",
        condition_dir=condition_dir,
        source_condition_fingerprint=condition_fingerprint,
        vocabulary_fingerprint=vocabulary_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        coverage=compute_coverage(
            covered_word_counts=covered_word_counts,
            corpus_word_counts=corpus_word_counts,
        ),
        posterior_mean_by_doc=posterior.posterior_mean_by_doc,
        posterior_metadata=posterior.metadata,
        expected_counts=statistics.expected_counts,
        execution_metadata=resolved_execution_metadata,
        empty_topic_ids=statistics.empty_topic_ids,
    )


def _validate_sentence_alignment(
    documents: Sequence[PreprocessedDocument], *, model: str
) -> None:
    """Sentence likelihood rows come from joined sentences while word counts
    come from tokenized sentences; the two views must line up per document."""

    for doc_index, document in enumerate(documents):
        joined = len(document.sentences_joined)
        tokenized = len(document.sentences_tokenized)
        if joined != tokenized:
            raise ValueError(
                f"{model} document {doc_index} has {joined} joined sentences "
                f"but {tokenized} tokenized sentences"
            )


def _vmf_embedding_preprocessor(
    condition_dir: Path, params: dict[str, Any]
) -> EmbeddingPreprocessor:
    """Rebuild the training-time embedding transform from saved artifacts.

    The transform mode comes from params.json; artifact presence must match it
    exactly.  Applying whatever files happen to exist would silently fall back
    to untransformed embeddings, which is forbidden.
    """

    mode = str(params.get("pre_normalize_transform", "none")).lower()
    preprocessor = EmbeddingPreprocessor(mode=mode)
    mean_path = condition_dir / "embedding_transform_mean.pkl"
    whitening_path = condition_dir / "embedding_transform_whitening_matrix.pkl"
    if mode == "none":
        for path in (mean_path, whitening_path):
            if path.exists():
                raise ValueError(
                    "pre_normalize_transform='none' but a transform artifact "
                    f"exists: {path}"
                )
        return preprocessor
    if not mean_path.exists():
        raise FileNotFoundError(
            f"pre_normalize_transform={mode!r} requires {mean_path}"
        )
    preprocessor.mean_ = np.asarray(load_artifact_pickle(mean_path), dtype=np.float64)
    if mode == "whitening":
        if not whitening_path.exists():
            raise FileNotFoundError(
                f"pre_normalize_transform='whitening' requires {whitening_path}"
            )
        preprocessor.whitening_matrix_ = np.asarray(
            load_artifact_pickle(whitening_path), dtype=np.float64
        )
    elif whitening_path.exists():
        raise ValueError(
            f"pre_normalize_transform={mode!r} does not use {whitening_path}"
        )
    preprocessor._fitted = True
    return preprocessor


def _reject_unfittable_encoder(encoder: Any, *, model: str) -> None:
    """Refuse encoders that cannot be reproduced from persisted artifacts.

    Training fits uSIF on the training sentences (``fit_encoder_on_sentences``),
    but its fitted state -- word probabilities and principal components -- is not
    written to ``metadata.json``.  A rebuilt encoder would therefore either raise
    inside ``encode`` or, if fitted here on the evaluation split, produce
    embeddings that do not match the fitted topic parameters.  Fail per condition
    so ``--condition-failure-policy`` can isolate it.
    """

    if not bool(getattr(encoder, "requires_fit", False)):
        return
    raise ConditionEvaluationError(
        f"{model}: the fitted encoder backend requires a fit whose state is not "
        "persisted in metadata.json, so post-hoc topic assignment cannot "
        "reproduce the training embeddings"
    )


def _encoder_from_metadata(
    metadata: dict[str, Any],
    *,
    device: str,
    encode_batch_size_override: int | None,
) -> tuple[SentenceEncoder, ResolvedEncodeBatchSize]:
    config = metadata.get("encoder_config")
    if not isinstance(config, dict) or not config.get("model_name"):
        raise ValueError("encoder_config.model_name is missing from model metadata")
    batch_size = resolve_topic_word_encode_batch_size(
        encoder_config=config,
        override=encode_batch_size_override,
    )
    kwargs: dict[str, Any] = {
        "model_name": str(config["model_name"]),
        "device": device,
        "encode_batch_size": batch_size.value,
        "strip_terminal_normalize": bool(config.get("strip_terminal_normalize", True)),
    }
    for key in (
        "backend",
        "pooling",
        "encode_prefix",
        "encode_prompt",
        "encode_prompt_name",
        "model_kwargs",
        "tokenizer_kwargs",
        "normalize_embeddings",
        "truncate_dim",
    ):
        if config.get(key) is not None:
            kwargs[key] = config[key]
    encoder = SentenceEncoder(**kwargs)
    _reject_unfittable_encoder(encoder, model="vmf")
    return encoder, batch_size


def _load_bleilda(
    *,
    condition_dir: Path,
    split: str,
    dictionary: Dictionary,
    config: CollapsedFoldInConfig,
    topn: int,
    npmi_min_expected_count: float | None,
) -> RuntimeTopicWords:
    documents, raw_ids = _load_documents_and_ids(
        model="bleilda", condition_dir=condition_dir, split=split
    )
    vocabulary = _vocabulary(dictionary)
    evaluation_id = {word: index for index, word in enumerate(vocabulary)}
    lda = LdaModel.load((condition_dir / "params/model.gensim").as_posix())
    phi = np.asarray(lda.get_topics(), dtype=np.float64)
    model_vocabulary = [str(lda.id2word[index]) for index in range(phi.shape[1])]
    model_id = {word: index for index, word in enumerate(model_vocabulary)}
    type_ids, unit_counts, covered, covered_words = _token_units(
        documents, evaluation_id, set(model_vocabulary)
    )
    likelihood = lda_log_likelihood_by_type(phi)[
        np.asarray([model_id[word] for word in covered_words], dtype=np.int64)
    ]
    return _collapsed_result(
        model="bleilda",
        condition_dir=condition_dir,
        split=split,
        vocabulary=vocabulary,
        documents=documents,
        raw_ids=raw_ids,
        alpha=_alpha_vector(lda.alpha, phi.shape[0]),
        assignment_unit_type="token",
        unit_word_counts_by_doc=unit_counts,
        covered_word_counts=covered,
        corpus_word_counts=_corpus_word_counts(documents, evaluation_id),
        config=config,
        topn=topn,
        log_likelihood_by_type=likelihood,
        type_ids_by_doc=type_ids,
        npmi_min_expected_count=npmi_min_expected_count,
    )


def _load_sentlda(
    *,
    condition_dir: Path,
    split: str,
    dictionary: Dictionary,
    config: CollapsedFoldInConfig,
    topn: int,
    npmi_min_expected_count: float | None,
) -> RuntimeTopicWords:
    documents, raw_ids = _load_documents_and_ids(
        model="sentlda", condition_dir=condition_dir, split=split
    )
    state = load_artifact_pickle(condition_dir / "params/model_state.pkl")
    vocabulary = _vocabulary(dictionary)
    evaluation_id = {word: index for index, word in enumerate(vocabulary)}
    model_vocabulary = set(str(word) for word in state.vocabulary)
    model_sentence_bow: list[list[list[tuple[int, int]]]] = []
    for document in documents:
        doc: list[list[tuple[int, int]]] = []
        for sentence in document.sentences_tokenized:
            counts: dict[int, int] = {}
            for token in sentence:
                word_id = state.vocabulary.get(token)
                if word_id is not None:
                    counts[int(word_id)] = counts.get(int(word_id), 0) + 1
            doc.append(sorted(counts.items()))
        model_sentence_bow.append(doc)
    likelihoods = sentlda_log_likelihood_by_doc(
        model_sentence_bow,
        topic_word_counts=state.topic_word_counts,
        beta=float(state.beta),
    )
    unit_counts, covered = _sentence_word_counts(
        documents, evaluation_id, supported_words=model_vocabulary
    )
    return _collapsed_result(
        model="sentlda",
        condition_dir=condition_dir,
        split=split,
        vocabulary=vocabulary,
        documents=documents,
        raw_ids=raw_ids,
        alpha=_alpha_vector(state.alpha, int(state.num_topics)),
        assignment_unit_type="sentence",
        unit_word_counts_by_doc=unit_counts,
        covered_word_counts=covered,
        corpus_word_counts=_corpus_word_counts(documents, evaluation_id),
        config=config,
        topn=topn,
        log_likelihood_by_doc=likelihoods,
        npmi_min_expected_count=npmi_min_expected_count,
    )


def _load_vmf(
    *,
    condition_dir: Path,
    split: str,
    dictionary: Dictionary,
    config: CollapsedFoldInConfig,
    topn: int,
    npmi_min_expected_count: float | None,
    encoder_device: str,
    encoder_device_requested: str,
    encoder_batch_size_override: int | None,
) -> RuntimeTopicWords:
    topic_words_started_at = time.perf_counter()
    documents_started_at = time.perf_counter()
    documents, raw_ids = _load_documents_and_ids(
        model="vmf", condition_dir=condition_dir, split=split
    )
    documents_load_sec = time.perf_counter() - documents_started_at
    _validate_sentence_alignment(documents, model="vmf")
    metadata = _condition_metadata(condition_dir)
    encoder_started_at = time.perf_counter()
    encoder, batch_size = _encoder_from_metadata(
        metadata,
        device=encoder_device,
        encode_batch_size_override=encoder_batch_size_override,
    )
    encoder_load_sec = time.perf_counter() - encoder_started_at
    params = load_artifact_json(condition_dir / "params.json")
    preprocessor = _vmf_embedding_preprocessor(condition_dir, params)
    likelihoods: list[np.ndarray] = []
    weights = np.asarray(
        load_artifact_pickle(condition_dir / "mixture_weights.pkl"), dtype=np.float64
    )
    means = np.asarray(
        load_artifact_pickle(condition_dir / "component_means.pkl"), dtype=np.float64
    )
    kappa = np.asarray(
        load_artifact_pickle(condition_dir / "kappa_per_topic.pkl"), dtype=np.float64
    )
    encoding_started_at = time.perf_counter()
    encoded_corpus = encode_sentence_corpus(
        encoder=encoder,
        documents=documents,
        batch_size=batch_size.value,
        show_progress_bar=False,
    )
    sentence_encoding_sec = time.perf_counter() - encoding_started_at
    likelihood_started_at = time.perf_counter()
    for encoded_view in encoded_corpus.iter_documents():
        encoded = np.asarray(encoded_view, dtype=np.float64)
        encoded = preprocessor.transform(encoded)
        if encoded.shape[0]:
            norms = np.linalg.norm(encoded, axis=1, keepdims=True)
            if np.any(norms <= 0.0):
                raise ValueError("vMF encoder produced a zero vector")
            encoded = encoded / norms
        likelihoods.append(
            vmf_mixture_log_likelihood(
                encoded,
                mixture_weights=weights,
                component_means=means,
                kappa_per_topic=kappa,
            )
        )
    topic_likelihood_sec = time.perf_counter() - likelihood_started_at
    execution_metadata: dict[str, object] = {
        "requested_device": encoder_device_requested,
        "effective_device": encoder_device,
        "encode_batch_size": batch_size.value,
        "encode_batch_size_source": batch_size.source,
        "training_encode_batch_size": batch_size.training_value,
        "num_documents": encoded_corpus.num_documents,
        "total_sentences": encoded_corpus.total_sentences,
        "embedding_dim": encoded_corpus.embedding_dim,
        "documents_load_sec": float(documents_load_sec),
        "encoder_load_sec": float(encoder_load_sec),
        "sentence_encoding_sec": float(sentence_encoding_sec),
        "topic_likelihood_sec": float(topic_likelihood_sec),
    }
    del encoded_corpus
    vocabulary = _vocabulary(dictionary)
    evaluation_id = {word: index for index, word in enumerate(vocabulary)}
    unit_counts, covered = _sentence_word_counts(documents, evaluation_id)
    result = _collapsed_result(
        model="vmf",
        condition_dir=condition_dir,
        split=split,
        vocabulary=vocabulary,
        documents=documents,
        raw_ids=raw_ids,
        alpha=_alpha_vector(params["alpha"], means.shape[0]),
        assignment_unit_type="sentence",
        unit_word_counts_by_doc=unit_counts,
        covered_word_counts=covered,
        corpus_word_counts=_corpus_word_counts(documents, evaluation_id),
        config=config,
        topn=topn,
        log_likelihood_by_doc=likelihoods,
        npmi_min_expected_count=npmi_min_expected_count,
        execution_metadata=execution_metadata,
        topic_words_started_at=topic_words_started_at,
    )
    assert result.execution_metadata is not None
    logger.info(
        "topic-word encoding model=vmf condition=%s requested_device=%s "
        "effective_device=%s batch_size=%s batch_source=%s documents=%s "
        "sentences=%s embedding_dim=%s documents_load_sec=%.3f "
        "encoder_load_sec=%.3f encoding_sec=%.3f likelihood_sec=%.3f "
        "foldin_sec=%.3f ranking_sec=%.3f total_sec=%.3f",
        condition_dir,
        encoder_device_requested,
        encoder_device,
        batch_size.value,
        batch_size.source,
        len(documents),
        execution_metadata["total_sentences"],
        execution_metadata["embedding_dim"],
        documents_load_sec,
        encoder_load_sec,
        sentence_encoding_sec,
        topic_likelihood_sec,
        float(result.execution_metadata["collapsed_foldin_sec"]),
        float(result.execution_metadata["topic_word_ranking_sec"]),
        float(result.execution_metadata["topic_words_total_sec"]),
    )
    return result


def _load_sentence_gaussian(
    *,
    condition_dir: Path,
    split: str,
    dictionary: Dictionary,
    config: CollapsedFoldInConfig,
    topn: int,
    npmi_min_expected_count: float | None,
    encoder_device: str,
    encoder_device_requested: str,
    encoder_batch_size_override: int | None,
) -> RuntimeTopicWords:
    topic_words_started_at = time.perf_counter()
    documents_started_at = time.perf_counter()
    documents, raw_ids = _load_documents_and_ids(
        model="sentence_gaussianlda", condition_dir=condition_dir, split=split
    )
    documents_load_sec = time.perf_counter() - documents_started_at
    _validate_sentence_alignment(documents, model="sentence_gaussianlda")
    metadata = _condition_metadata(condition_dir)
    encoder_config = metadata.get("encoder_config")
    if not isinstance(encoder_config, dict) or not encoder_config.get("model_name"):
        raise ValueError("Sentence GaussianLDA encoder metadata is missing")
    batch_size = resolve_topic_word_encode_batch_size(
        encoder_config=encoder_config,
        override=encoder_batch_size_override,
    )
    encoder_started_at = time.perf_counter()
    encoder = build_sentence_gaussian_encoder(
        str(encoder_config["model_name"]),
        device=encoder_device,
        encode_prefix=encoder_config.get("encode_prefix"),
        backend=str(encoder_config.get("backend", "auto")),
        pooling=encoder_config.get("pooling"),
        encode_prompt=encoder_config.get("encode_prompt"),
        encode_prompt_name=encoder_config.get("encode_prompt_name"),
        encode_batch_size=batch_size.value,
        model_kwargs=encoder_config.get("model_kwargs"),
        tokenizer_kwargs=encoder_config.get("tokenizer_kwargs"),
        normalize_embeddings=encoder_config.get("normalize_embeddings"),
        truncate_dim=encoder_config.get("truncate_dim"),
        strip_terminal_normalize=bool(
            encoder_config.get("strip_terminal_normalize", True)
        ),
    )
    _reject_unfittable_encoder(encoder, model="sentence_gaussianlda")
    encoder_load_sec = time.perf_counter() - encoder_started_at
    persisted = load_sentence_gaussianlda_model(
        param_dir=condition_dir / "params", encoder=encoder
    )
    likelihoods: list[np.ndarray] = []
    encoding_started_at = time.perf_counter()
    encoded_corpus = encode_sentence_corpus(
        encoder=encoder,
        documents=documents,
        batch_size=batch_size.value,
        show_progress_bar=False,
    )
    sentence_encoding_sec = time.perf_counter() - encoding_started_at
    likelihood_started_at = time.perf_counter()
    for encoded_view in encoded_corpus.iter_documents():
        encoded = np.asarray(encoded_view, dtype=np.float64)
        if encoded.size == 0:
            likelihoods.append(
                np.empty((0, persisted.model.num_tables), dtype=np.float64)
            )
            continue
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        likelihoods.append(
            np.vstack(
                [
                    persisted.model.log_multivariate_tdensity_tables(row)
                    for row in encoded
                ]
            )
        )
    topic_likelihood_sec = time.perf_counter() - likelihood_started_at
    execution_metadata = {
        "requested_device": encoder_device_requested,
        "effective_device": encoder_device,
        "encode_batch_size": batch_size.value,
        "encode_batch_size_source": batch_size.source,
        "training_encode_batch_size": batch_size.training_value,
        "num_documents": encoded_corpus.num_documents,
        "total_sentences": encoded_corpus.total_sentences,
        "embedding_dim": encoded_corpus.embedding_dim,
        "documents_load_sec": float(documents_load_sec),
        "encoder_load_sec": float(encoder_load_sec),
        "sentence_encoding_sec": float(sentence_encoding_sec),
        "topic_likelihood_sec": float(topic_likelihood_sec),
    }
    del encoded_corpus
    vocabulary = _vocabulary(dictionary)
    evaluation_id = {word: index for index, word in enumerate(vocabulary)}
    unit_counts, covered = _sentence_word_counts(documents, evaluation_id)
    result = _collapsed_result(
        model="sentence_gaussianlda",
        condition_dir=condition_dir,
        split=split,
        vocabulary=vocabulary,
        documents=documents,
        raw_ids=raw_ids,
        alpha=_alpha_vector(persisted.model.alpha, persisted.model.num_tables),
        assignment_unit_type="sentence",
        unit_word_counts_by_doc=unit_counts,
        covered_word_counts=covered,
        corpus_word_counts=_corpus_word_counts(documents, evaluation_id),
        config=config,
        topn=topn,
        log_likelihood_by_doc=likelihoods,
        npmi_min_expected_count=npmi_min_expected_count,
        execution_metadata=execution_metadata,
        topic_words_started_at=topic_words_started_at,
    )
    assert result.execution_metadata is not None
    logger.info(
        "topic-word encoding model=sentence_gaussianlda condition=%s "
        "requested_device=%s effective_device=%s batch_size=%s "
        "batch_source=%s documents=%s sentences=%s embedding_dim=%s "
        "documents_load_sec=%.3f encoder_load_sec=%.3f encoding_sec=%.3f "
        "likelihood_sec=%.3f foldin_sec=%.3f ranking_sec=%.3f total_sec=%.3f",
        condition_dir,
        encoder_device_requested,
        encoder_device,
        batch_size.value,
        batch_size.source,
        len(documents),
        execution_metadata["total_sentences"],
        execution_metadata["embedding_dim"],
        documents_load_sec,
        encoder_load_sec,
        sentence_encoding_sec,
        topic_likelihood_sec,
        float(result.execution_metadata["collapsed_foldin_sec"]),
        float(result.execution_metadata["topic_word_ranking_sec"]),
        float(result.execution_metadata["topic_words_total_sec"]),
    )
    return result


def _word_vector_condition(
    *,
    model: str,
    condition_dir: Path,
    split: str,
    dictionary: Dictionary,
    config: CollapsedFoldInConfig,
    topn: int,
    npmi_min_expected_count: float | None,
) -> RuntimeTopicWords:
    documents, raw_ids = _load_documents_and_ids(
        model=model, condition_dir=condition_dir, split=split
    )
    vocabulary = _vocabulary(dictionary)
    evaluation_id = {word: index for index, word in enumerate(vocabulary)}
    metadata = _condition_metadata(condition_dir)
    baseline_params = metadata.get("baseline_params")
    if not isinstance(baseline_params, dict) or not baseline_params.get("word2vec"):
        raise ValueError("Saved word-vector source is missing from model metadata")
    saved_word2vec = str(baseline_params["word2vec"])
    cache_dir = baseline_params.get("wikientvec_cache_dir")
    if model == "gaussianlda":
        persisted = load_gaussianlda_model(
            param_dir=condition_dir / "params",
            word2vec=saved_word2vec,
            wikientvec_cache_dir=cache_dir,
        )
        supported = set(persisted.vocab)
        type_ids, unit_counts, covered, covered_words = _token_units(
            documents, evaluation_id, supported
        )
        model_id = {word: index for index, word in enumerate(persisted.vocab)}
        embeddings = persisted.embeddings[
            np.asarray([model_id[word] for word in covered_words], dtype=np.int64)
        ]
        likelihood = np.column_stack(
            [
                persisted.model.log_multivariate_tdensity(embeddings, topic)
                for topic in range(persisted.model.num_tables)
            ]
        )
        num_topics = persisted.model.num_tables
        alpha = persisted.model.alpha
    else:
        from src.baselines.models.gaussian_helpers import load_gaussian_word_vectors

        vectors = load_gaussian_word_vectors(
            saved_word2vec,
            param_dir=condition_dir / "params",
            wikientvec_cache_dir=cache_dir,
        )
        supported = set(str(word) for word in vectors.key_to_index)
        type_ids, unit_counts, covered, covered_words = _token_units(
            documents, evaluation_id, supported
        )
        vector_rows = np.asarray([vectors[word] for word in covered_words])
        params = load_artifact_json(condition_dir / "params/params.json")
        weights = load_artifact_pickle(condition_dir / "params/mixture_weights.pkl")
        means = load_artifact_pickle(condition_dir / "params/component_means.pkl")
        kappa = load_artifact_pickle(condition_dir / "params/kappa_per_topic.pkl")
        likelihood = vmf_mixture_log_likelihood(
            vector_rows / np.linalg.norm(vector_rows, axis=1, keepdims=True),
            mixture_weights=weights,
            component_means=means,
            kappa_per_topic=kappa,
        )
        num_topics = likelihood.shape[1]
        alpha = params["alpha"]
    return _collapsed_result(
        model=model,
        condition_dir=condition_dir,
        split=split,
        vocabulary=vocabulary,
        documents=documents,
        raw_ids=raw_ids,
        alpha=_alpha_vector(alpha, num_topics),
        assignment_unit_type="token",
        unit_word_counts_by_doc=unit_counts,
        covered_word_counts=covered,
        corpus_word_counts=_corpus_word_counts(documents, evaluation_id),
        config=config,
        topn=topn,
        log_likelihood_by_type=likelihood,
        type_ids_by_doc=type_ids,
        npmi_min_expected_count=npmi_min_expected_count,
    )


def _load_etm(
    *,
    condition_dir: Path,
    split: str,
    dictionary: Dictionary,
    topn: int,
    theta_samples: int,
    posterior_seed: int,
    npmi_min_expected_count: float | None,
) -> RuntimeTopicWords:
    import torch

    from src.baselines.models.etm import EtmModel

    documents, raw_ids = _load_documents_and_ids(
        model="etm", condition_dir=condition_dir, split=split
    )
    params = load_artifact_json(condition_dir / "params/params.json")
    baseline_params = params["baseline_params"]
    embeddings = np.asarray(
        load_artifact_pickle(condition_dir / "params/embeddings.pkl"), dtype=np.float32
    )
    model = EtmModel(
        embeddings=embeddings,
        num_topics=int(params["num_topics"]),
        hidden_size=int(baseline_params["t_hidden_size"]),
        theta_act=str(baseline_params["theta_act"]),
        enc_drop=float(baseline_params["enc_drop"]),
    )
    state = torch.load(
        condition_dir / "params/model_state.pt", map_location="cpu", weights_only=True
    )
    model = _restore_etm_model(model, state)
    vocabulary_payload = load_artifact_json(condition_dir / "params/vocabulary.json")
    if isinstance(vocabulary_payload, dict):
        model_vocabulary = [
            word
            for word, _ in sorted(
                ((str(word), int(index)) for word, index in vocabulary_payload.items()),
                key=lambda item: item[1],
            )
        ]
    else:
        model_vocabulary = [str(word) for word in vocabulary_payload]
    model_id = {word: index for index, word in enumerate(model_vocabulary)}
    bow = np.zeros((len(documents), len(model_vocabulary)), dtype=np.float32)
    corpus_bow: list[list[tuple[int, int]]] = []
    for doc_index, document in enumerate(documents):
        counts: dict[int, int] = {}
        for token in document.document_tokens:
            word_id = model_id.get(token)
            if word_id is not None:
                counts[word_id] = counts.get(word_id, 0) + 1
        for word_id, count in counts.items():
            bow[doc_index, word_id] = count
        corpus_bow.append(sorted(counts.items()))
    normalized = bow.copy()
    if bool(baseline_params.get("bow_norm", True)):
        totals = normalized.sum(axis=1, keepdims=True)
        totals[totals == 0.0] = 1.0
        normalized /= totals
    with torch.no_grad():
        mu, logsigma = model.encode(torch.as_tensor(normalized))
        beta_model = model.get_beta().cpu().numpy().astype(np.float64)
    saved_beta = np.asarray(
        load_artifact_pickle(condition_dir / "params/topic_word_scores.pkl"),
        dtype=np.float64,
    )
    _validate_etm_checkpoint_beta(saved_beta=saved_beta, checkpoint_beta=beta_model)
    beta_sums = beta_model.sum(axis=1, keepdims=True)
    if np.any(beta_sums <= 0.0) or not np.all(np.isfinite(beta_model)):
        raise ValueError("ETM beta must be finite with positive row mass")
    beta_model = beta_model / beta_sums
    theta = sample_etm_theta(
        mu=mu.cpu().numpy(),
        logsigma=logsigma.cpu().numpy(),
        num_samples=theta_samples,
        seed=posterior_seed,
    )
    model_stats, metadata = compute_etm_expected_counts(
        theta_samples=theta, beta=beta_model, corpus_bow=corpus_bow
    )
    evaluation_vocabulary = _vocabulary(dictionary)
    mapped_beta, covered_types = restrict_scores_to_evaluation_vocabulary(
        beta_model,
        model_vocabulary=model_vocabulary,
        evaluation_vocabulary=evaluation_vocabulary,
    )
    eval_counts = np.zeros(
        (beta_model.shape[0], len(evaluation_vocabulary)), dtype=np.float64
    )
    eval_id = {word: index for index, word in enumerate(evaluation_vocabulary)}
    for source_id, word in enumerate(model_vocabulary):
        target_id = eval_id.get(word)
        if target_id is not None:
            eval_counts[:, target_id] = model_stats.expected_counts[:, source_id]
    stats = _statistics_from_counts(eval_counts)
    empty = np.flatnonzero(stats.topic_counts <= 0.0)
    if empty.size:
        raise ValueError(f"ETM topics empty after V_eval mapping: {empty.tolist()}")
    _raise_for_degenerate_topics(
        stats.expected_counts,
        vocabulary=evaluation_vocabulary,
        required_topn=topn,
        eligible_mask=np.broadcast_to(covered_types, stats.expected_counts.shape),
        allow_zero_joint_counts=True,
    )
    evaluation_words = rank_topic_words(
        mapped_beta,
        evaluation_vocabulary,
        topn=topn,
        eligible_mask=np.broadcast_to(covered_types, mapped_beta.shape),
    )
    display_words = rank_topic_words(
        word_topic_npmi(stats, min_expected_count=npmi_min_expected_count),
        evaluation_vocabulary,
        topn=topn,
    )
    corpus_counts = _corpus_word_counts(documents, eval_id)
    condition_fingerprint = _condition_fingerprint(condition_dir)
    vocabulary_fingerprint = _fingerprint({"ordered_vocabulary": evaluation_vocabulary})
    corpus_fingerprint = _fingerprint({"raw_document_ids": raw_ids, "split": split})
    metadata.update(
        {
            "posterior_seed": posterior_seed,
            "condition_fingerprint": condition_fingerprint,
            "vocabulary_fingerprint": vocabulary_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
        }
    )
    return RuntimeTopicWords(
        evaluation=TopicWordsResult(
            topic_words=evaluation_words,
            topic_word_source="native_etm_beta",
            score_mode="topic_word_probability",
            score_definition="fitted ETM beta",
        ),
        display_topic_words=display_words,
        display_source="variational_word_topic_npmi",
        display_score_mode="word_topic_npmi",
        protocol="native_etm_beta_with_variational_responsibilities",
        condition_dir=condition_dir,
        source_condition_fingerprint=condition_fingerprint,
        vocabulary_fingerprint=vocabulary_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        coverage=compute_coverage(
            covered_word_counts=stats.word_counts,
            corpus_word_counts=corpus_counts,
        ),
        posterior_mean_by_doc=compute_etm_token_topic_posterior_mean(
            theta_samples=theta, beta=beta_model, corpus_bow=corpus_bow
        ),
        posterior_metadata=metadata,
        expected_counts=stats.expected_counts,
    )


def _restore_etm_model(model: EtmModel, state: dict[str, Any]) -> EtmModel:
    """Restore an evaluation-only ETM, including models initialized on meta."""
    model.load_state_dict(state, strict=True, assign=True)
    meta_tensors = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.is_meta
    ]
    if meta_tensors:
        raise RuntimeError(
            "ETM checkpoint loading left meta tensors: "
            + ", ".join(sorted(meta_tensors))
        )
    return model.to("cpu").eval()


def _validate_etm_checkpoint_beta(
    *, saved_beta: np.ndarray, checkpoint_beta: np.ndarray
) -> None:
    # Training persists beta on CUDA, while evaluation restores the checkpoint
    # on CPU. Softmax over a large vocabulary can differ by a few tens of
    # micro-units across those backends, so allow that numerical drift while
    # still rejecting materially different checkpoints.
    if saved_beta.shape != checkpoint_beta.shape or not np.allclose(
        saved_beta,
        checkpoint_beta,
        atol=5e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "ETM checkpoint beta does not match saved topic_word_scores.pkl"
        )


def _load_ctm(
    *,
    condition_dir: Path,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    dictionary: Dictionary,
    topn: int,
    embedding_variant: str | None,
    split: str,
    npmi_min_expected_count: float | None = None,
) -> RuntimeTopicWords:
    documents, raw_ids = _load_documents_and_ids(
        model="ctm", condition_dir=condition_dir, split=split
    )
    decoder_scores, model_vocabulary = load_ctm_decoder_scores(
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
    )
    probabilities = normalize_ctm_decoder_scores(
        decoder_scores, already_probabilities=True
    )
    vocabulary = _vocabulary(dictionary)
    mapped, covered_types = restrict_scores_to_evaluation_vocabulary(
        probabilities,
        model_vocabulary=model_vocabulary,
        evaluation_vocabulary=vocabulary,
    )
    _raise_for_degenerate_topics(
        mapped,
        vocabulary=vocabulary,
        required_topn=topn,
        eligible_mask=np.broadcast_to(covered_types, mapped.shape),
    )
    words = rank_topic_words(
        mapped,
        vocabulary,
        topn=topn,
        eligible_mask=np.broadcast_to(covered_types, mapped.shape),
    )
    eval_id = {word: index for index, word in enumerate(vocabulary)}
    corpus_counts = _corpus_word_counts(documents, eval_id)
    covered_counts = corpus_counts * covered_types
    theta_path = (
        condition_dir / "params" / "ctm.pkl"
        if split == "train"
        else condition_dir / "infer" / f"{category}.pkl"
    )
    theta = np.asarray(load_artifact_pickle(theta_path), dtype=np.float64)
    if theta.ndim != 2 or theta.shape != (len(documents), num_topics):
        raise ValueError(
            "CTM document-topic distribution is not aligned with documents: "
            f"theta={theta.shape} documents={len(documents)} topics={num_topics}"
        )
    if not np.all(np.isfinite(theta)) or np.any(theta < 0.0):
        raise ValueError(
            "CTM document-topic distribution must be finite and non-negative"
        )
    theta_sums = theta.sum(axis=1, keepdims=True)
    if np.any(theta_sums <= 0.0):
        raise ValueError("CTM document-topic distribution contains an empty document")
    theta = theta / theta_sums
    model_id = {word: index for index, word in enumerate(model_vocabulary)}
    model_corpus_bow: list[list[tuple[int, int]]] = []
    for document in documents:
        counts: dict[int, int] = {}
        for token in document.document_tokens:
            word_id = model_id.get(token)
            if word_id is not None:
                counts[word_id] = counts.get(word_id, 0) + 1
        model_corpus_bow.append(sorted(counts.items()))
    model_stats, posterior_metadata = compute_etm_expected_counts(
        theta_samples=theta,
        beta=probabilities,
        corpus_bow=model_corpus_bow,
    )
    eval_counts = np.zeros((num_topics, len(vocabulary)), dtype=np.float64)
    for source_id, word in enumerate(model_vocabulary):
        target_id = eval_id.get(word)
        if target_id is not None:
            eval_counts[:, target_id] = model_stats.expected_counts[:, source_id]
    stats = _statistics_from_counts(eval_counts)
    _raise_for_degenerate_topics(
        stats.expected_counts,
        vocabulary=vocabulary,
        required_topn=topn,
        eligible_mask=np.broadcast_to(covered_types, stats.expected_counts.shape),
        allow_zero_joint_counts=True,
    )
    npmi_words = rank_topic_words(
        word_topic_npmi(stats, min_expected_count=npmi_min_expected_count),
        vocabulary,
        topn=topn,
    )
    vocabulary_fingerprint = _fingerprint({"ordered_vocabulary": vocabulary})
    corpus_fingerprint = _fingerprint({"raw_document_ids": raw_ids, "split": split})
    posterior_metadata.update(
        {
            "posterior_kind": "ctm_document_mixture_token_responsibility",
            "theta_source": str(theta_path),
            "condition_fingerprint": _condition_fingerprint(condition_dir),
            "vocabulary_fingerprint": vocabulary_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
        }
    )
    return RuntimeTopicWords(
        evaluation=TopicWordsResult(
            topic_words=words,
            topic_word_source="native_ctm_decoder_topic_word_distribution",
            score_mode="decoder_topic_word_probability",
            score_definition="fitted CTM decoder topic-word distribution",
        ),
        display_topic_words=npmi_words,
        display_source="ctm_variational_word_topic_npmi",
        display_score_mode="word_topic_npmi",
        protocol="native_ctm_decoder_with_token_responsibilities",
        condition_dir=condition_dir,
        source_condition_fingerprint=_condition_fingerprint(condition_dir),
        vocabulary_fingerprint=vocabulary_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        coverage=compute_coverage(
            covered_word_counts=covered_counts,
            corpus_word_counts=corpus_counts,
        ),
        posterior_metadata=posterior_metadata,
        expected_counts=stats.expected_counts,
    )


def resolve_runtime_topic_words(
    *,
    model: str,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    dictionary: Dictionary,
    topn: int,
    embedding_variant: str | None,
    posterior_config: CollapsedFoldInConfig,
    etm_theta_samples: int,
    etm_posterior_seed: int,
    npmi_min_expected_count: float | None = None,
    encoder_device: str = "cpu",
    encoder_device_requested: str | None = None,
    encoder_batch_size_override: int | None = None,
    prior_scale: float | None = None,
) -> RuntimeTopicWords:
    requested_encoder_device = encoder_device_requested or encoder_device
    normalized_model = "vmf" if model == "vmf_sentence_lda" else model
    # Only the gaussian family stores a prior-scale path variant; vMF experiment
    # dirs are not keyed on it.
    parameter_variant = (
        format_prior_scale_variant(prior_scale)
        if prior_scale is not None and normalized_model in GAUSSIAN_PRIOR_SCALE_MODELS
        else None
    )
    if normalized_model == "vmf":
        condition_dir = resolve_vmf_experiment_dir(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            run_name=data_run,
            embedding_variant=embedding_variant,
        )
    else:
        condition_dir = resolve_baseline_condition_dir(
            model=normalized_model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            embedding_variant=embedding_variant,
            parameter_variant=parameter_variant,
        )
    audit = audit_model_artifacts(model=normalized_model, condition_dir=condition_dir)
    if not audit.ok:
        problems = [" | ".join(group) for group in audit.missing_groups]
        problems.extend(getattr(audit, "inconsistencies", ()))
        raise FileNotFoundError(
            f"Artifact audit failed for {normalized_model} under {condition_dir}: "
            + ", ".join(problems)
        )
    if normalized_model == "bleilda":
        return _load_bleilda(
            condition_dir=condition_dir,
            split=split,
            dictionary=dictionary,
            config=posterior_config,
            topn=topn,
            npmi_min_expected_count=npmi_min_expected_count,
        )
    if normalized_model == "sentlda":
        return _load_sentlda(
            condition_dir=condition_dir,
            split=split,
            dictionary=dictionary,
            config=posterior_config,
            topn=topn,
            npmi_min_expected_count=npmi_min_expected_count,
        )
    if normalized_model == "vmf":
        return _load_vmf(
            condition_dir=condition_dir,
            split=split,
            dictionary=dictionary,
            config=posterior_config,
            topn=topn,
            npmi_min_expected_count=npmi_min_expected_count,
            encoder_device=encoder_device,
            encoder_device_requested=requested_encoder_device,
            encoder_batch_size_override=encoder_batch_size_override,
        )
    if normalized_model == "sentence_gaussianlda":
        return _load_sentence_gaussian(
            condition_dir=condition_dir,
            split=split,
            dictionary=dictionary,
            config=posterior_config,
            topn=topn,
            npmi_min_expected_count=npmi_min_expected_count,
            encoder_device=encoder_device,
            encoder_device_requested=requested_encoder_device,
            encoder_batch_size_override=encoder_batch_size_override,
        )
    if normalized_model in {"mvtm", "gaussianlda"}:
        return _word_vector_condition(
            model=normalized_model,
            condition_dir=condition_dir,
            split=split,
            dictionary=dictionary,
            config=posterior_config,
            topn=topn,
            npmi_min_expected_count=npmi_min_expected_count,
        )
    if normalized_model == "etm":
        return _load_etm(
            condition_dir=condition_dir,
            split=split,
            dictionary=dictionary,
            topn=topn,
            theta_samples=etm_theta_samples,
            posterior_seed=etm_posterior_seed,
            npmi_min_expected_count=npmi_min_expected_count,
        )
    if normalized_model == "ctm":
        return _load_ctm(
            condition_dir=condition_dir,
            dataset=dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            dictionary=dictionary,
            topn=topn,
            embedding_variant=embedding_variant,
            split=split,
            npmi_min_expected_count=npmi_min_expected_count,
        )
    raise ValueError(f"Unsupported runtime topic-word model: {model}")

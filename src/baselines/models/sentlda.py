from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import load_preprocessed_documents
from src.baselines.models.sentlda_numba import (
    build_sentence_topic_log_factors_infer,
    build_sentence_topic_log_factors_train,
    build_sentence_topic_soft_infer,
    build_sentence_topic_soft_train,
    resolve_sentlda_backend,
    run_sentlda_infer_iteration,
    run_sentlda_train_iteration,
)
from src.baselines.params import SentLdaParams
from src.core.artifacts import (
    PREPROCESSING_SELECTION_FILENAME,
    PickleArtifactSpec,
    save_json,
    save_split_jsons,
    save_split_pickles,
)
from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    filter_selected_corpus_by_vocabulary,
    select_modelable_documents,
)
from src.utils.logging import get_logger, get_progress_bar


@dataclass(frozen=True)
class SentLdaModelState:
    num_topics: int
    vocab_size: int
    alpha: float
    beta: float
    vocabulary: dict[str, int]
    topic_word_counts: np.ndarray
    topic_total_words: np.ndarray
    doc_sentence_topic_counts: np.ndarray


@dataclass(frozen=True)
class SentLdaTrainResult:
    model_state: SentLdaModelState
    train_doc_topic: np.ndarray
    train_sentence_topic_soft: list[np.ndarray]
    train_sentence_topic_loglik: list[np.ndarray]
    train_sentence_topic_logprior: list[np.ndarray]
    train_preprocessed: list[PreprocessedDocument]
    train_sentence_topic_assignments: list[np.ndarray]
    train_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class SentLdaInferResult:
    test_doc_topic: np.ndarray
    test_sentence_topic_soft: list[np.ndarray]
    test_sentence_topic_loglik: list[np.ndarray]
    test_sentence_topic_logprior: list[np.ndarray]
    test_preprocessed: list[PreprocessedDocument]
    test_sentence_topic_assignments: list[np.ndarray]
    test_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class _CorpusBundle:
    preprocessed: list[PreprocessedDocument]
    sentence_word_ids_flat: np.ndarray
    sentence_offsets: np.ndarray
    doc_offsets: np.ndarray
    sentence_doc_ids: np.ndarray
    sentence_unique_word_ids_flat: np.ndarray
    sentence_unique_offsets: np.ndarray
    sentence_word_counts_flat: np.ndarray
    sentence_lengths: np.ndarray
    selection: SelectedCorpus

    @property
    def num_docs(self) -> int:
        return int(self.doc_offsets.size) - 1

    @property
    def num_sentences(self) -> int:
        return int(self.sentence_lengths.size)


LOG = get_logger("sentLDA")


def _load_documents(
    *,
    csv_paths: Sequence[str],
    targets: Sequence[str] | None,
    text_column: str,
    target_column: str | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
    use_legacy: bool,
) -> list[PreprocessedDocument]:
    return load_preprocessed_documents(
        csv_paths=csv_paths,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
        delimiter=" / " if use_legacy else delimiter,
        language=language,
        segmenter="delimiter" if use_legacy else segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )


def _build_vocabulary(documents: Sequence[PreprocessedDocument]) -> dict[str, int]:
    vocabulary: dict[str, int] = {}
    selection = select_modelable_documents(documents)
    for doc in selection.documents:
        for sentence in doc.sentences_tokenized:
            for token in sentence:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
    return vocabulary


def _encode_corpus(
    documents: Sequence[PreprocessedDocument],
    vocabulary: dict[str, int],
) -> _CorpusBundle:
    selection = filter_selected_corpus_by_vocabulary(
        select_modelable_documents(documents),
        set(vocabulary),
    )
    sentence_word_ids_flat: list[int] = []
    sentence_offsets = [0]
    doc_offsets = [0]
    sentence_doc_ids: list[int] = []
    sentence_unique_word_ids_flat: list[int] = []
    sentence_unique_offsets = [0]
    sentence_word_counts_flat: list[int] = []
    sentence_lengths: list[int] = []
    filtered_preprocessed: list[PreprocessedDocument] = []

    for doc in selection.documents:
        kept_raw: list[str] = []
        kept_tokens: list[list[str]] = []
        kept_doc_tokens: list[str] = []
        kept_sentence_count = 0

        for raw_sentence, tokenized_sentence in zip(
            doc.sentences_raw, doc.sentences_tokenized
        ):
            word_ids = [
                vocabulary[token] for token in tokenized_sentence if token in vocabulary
            ]
            if not word_ids:
                continue
            encoded = np.asarray(word_ids, dtype=np.int32)
            unique_word_ids, counts = np.unique(encoded, return_counts=True)

            sentence_word_ids_flat.extend(encoded.tolist())
            sentence_offsets.append(len(sentence_word_ids_flat))
            sentence_unique_word_ids_flat.extend(
                unique_word_ids.astype(np.int32).tolist()
            )
            sentence_word_counts_flat.extend(counts.astype(np.int32).tolist())
            sentence_unique_offsets.append(len(sentence_unique_word_ids_flat))
            sentence_lengths.append(int(encoded.size))
            sentence_doc_ids.append(len(filtered_preprocessed))
            kept_raw.append(raw_sentence)
            kept_tokens.append(list(tokenized_sentence))
            kept_doc_tokens.extend(tokenized_sentence)
            kept_sentence_count += 1

        if kept_sentence_count == 0:
            continue

        filtered_preprocessed.append(
            PreprocessedDocument(
                raw_text=doc.raw_text,
                sentences_raw=kept_raw,
                sentences_tokenized=kept_tokens,
                sentences_joined=[" ".join(tokens) for tokens in kept_tokens],
                document_tokens=kept_doc_tokens,
            )
        )
        doc_offsets.append(doc_offsets[-1] + kept_sentence_count)

    return _CorpusBundle(
        preprocessed=filtered_preprocessed,
        sentence_word_ids_flat=np.asarray(sentence_word_ids_flat, dtype=np.int32),
        sentence_offsets=np.asarray(sentence_offsets, dtype=np.int32),
        doc_offsets=np.asarray(doc_offsets, dtype=np.int32),
        sentence_doc_ids=np.asarray(sentence_doc_ids, dtype=np.int32),
        sentence_unique_word_ids_flat=np.asarray(
            sentence_unique_word_ids_flat,
            dtype=np.int32,
        ),
        sentence_unique_offsets=np.asarray(sentence_unique_offsets, dtype=np.int32),
        sentence_word_counts_flat=np.asarray(sentence_word_counts_flat, dtype=np.int32),
        sentence_lengths=np.asarray(sentence_lengths, dtype=np.int32),
        selection=selection,
    )


def _group_sentence_values_by_doc(
    values: np.ndarray,
    doc_offsets: np.ndarray,
) -> list[np.ndarray]:
    grouped: list[np.ndarray] = []
    for doc_index in range(int(doc_offsets.size) - 1):
        sentence_start = int(doc_offsets[doc_index])
        sentence_end = int(doc_offsets[doc_index + 1])
        grouped.append(np.asarray(values[sentence_start:sentence_end]).copy())
    return grouped


def _update_topic_word_counts_for_sentence(
    *,
    bundle: _CorpusBundle,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    sentence_index: int,
    topic: int,
    delta: int,
) -> None:
    topic_total_words[topic] += int(delta) * int(
        bundle.sentence_lengths[sentence_index]
    )
    unique_start = int(bundle.sentence_unique_offsets[sentence_index])
    unique_end = int(bundle.sentence_unique_offsets[sentence_index + 1])
    for flat_index in range(unique_start, unique_end):
        word_id = int(bundle.sentence_unique_word_ids_flat[flat_index])
        count = int(bundle.sentence_word_counts_flat[flat_index])
        topic_word_counts[topic, word_id] += int(delta) * count


def _initialize_train_state(
    *,
    bundle: _CorpusBundle,
    num_topics: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assignments = rng.integers(
        num_topics,
        size=bundle.num_sentences,
        dtype=np.int32,
    )
    topic_word_counts = np.zeros((num_topics, vocab_size), dtype=np.int64)
    topic_total_words = np.zeros(num_topics, dtype=np.int64)
    doc_topic_counts = np.zeros((bundle.num_docs, num_topics), dtype=np.int64)

    for doc_index in range(bundle.num_docs):
        sentence_start = int(bundle.doc_offsets[doc_index])
        sentence_end = int(bundle.doc_offsets[doc_index + 1])
        for sentence_index in range(sentence_start, sentence_end):
            topic = int(assignments[sentence_index])
            doc_topic_counts[doc_index, topic] += 1
            _update_topic_word_counts_for_sentence(
                bundle=bundle,
                topic_word_counts=topic_word_counts,
                topic_total_words=topic_total_words,
                sentence_index=sentence_index,
                topic=topic,
                delta=1,
            )

    return topic_word_counts, topic_total_words, doc_topic_counts, assignments


def _initialize_infer_state(
    *,
    bundle: _CorpusBundle,
    num_topics: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    assignments = rng.integers(
        num_topics,
        size=bundle.num_sentences,
        dtype=np.int32,
    )
    doc_topic_counts = np.zeros((bundle.num_docs, num_topics), dtype=np.int64)

    for doc_index in range(bundle.num_docs):
        sentence_start = int(bundle.doc_offsets[doc_index])
        sentence_end = int(bundle.doc_offsets[doc_index + 1])
        for sentence_index in range(sentence_start, sentence_end):
            doc_topic_counts[doc_index, int(assignments[sentence_index])] += 1

    return doc_topic_counts, assignments


def _fit_sentlda(
    *,
    bundle: _CorpusBundle,
    num_topics: int,
    alpha: float,
    beta: float,
    vocab_size: int,
    num_iterations: int,
    random_state: int,
    backend: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    rng = np.random.default_rng(random_state)
    (
        topic_word_counts,
        topic_total_words,
        doc_topic_counts,
        assignments_flat,
    ) = _initialize_train_state(
        bundle=bundle,
        num_topics=num_topics,
        vocab_size=vocab_size,
        rng=rng,
    )

    for _ in get_progress_bar(
        range(num_iterations),
        desc="sentLDA train",
        leave=False,
    ):
        run_sentlda_train_iteration(
            doc_offsets=bundle.doc_offsets,
            sentence_lengths=bundle.sentence_lengths,
            sentence_unique_offsets=bundle.sentence_unique_offsets,
            sentence_unique_word_ids_flat=bundle.sentence_unique_word_ids_flat,
            sentence_word_counts_flat=bundle.sentence_word_counts_flat,
            topic_word_counts=topic_word_counts,
            topic_total_words=topic_total_words,
            doc_topic_counts=doc_topic_counts,
            assignments=assignments_flat,
            alpha=alpha,
            beta=beta,
            vocab_size=vocab_size,
            uniforms=rng.random(bundle.num_sentences),
            backend=backend,
        )

    sentence_topic_soft_flat = build_sentence_topic_soft_train(
        doc_offsets=bundle.doc_offsets,
        sentence_lengths=bundle.sentence_lengths,
        sentence_unique_offsets=bundle.sentence_unique_offsets,
        sentence_unique_word_ids_flat=bundle.sentence_unique_word_ids_flat,
        sentence_word_counts_flat=bundle.sentence_word_counts_flat,
        topic_word_counts=topic_word_counts,
        topic_total_words=topic_total_words,
        doc_topic_counts=doc_topic_counts,
        assignments=assignments_flat,
        alpha=alpha,
        beta=beta,
        vocab_size=vocab_size,
        backend=backend,
    )
    log_prior_flat, log_likelihood_flat = build_sentence_topic_log_factors_train(
        doc_offsets=bundle.doc_offsets,
        sentence_lengths=bundle.sentence_lengths,
        sentence_unique_offsets=bundle.sentence_unique_offsets,
        sentence_unique_word_ids_flat=bundle.sentence_unique_word_ids_flat,
        sentence_word_counts_flat=bundle.sentence_word_counts_flat,
        topic_word_counts=topic_word_counts,
        topic_total_words=topic_total_words,
        doc_topic_counts=doc_topic_counts,
        assignments=assignments_flat,
        alpha=alpha,
        beta=beta,
        vocab_size=vocab_size,
        backend=backend,
    )

    return (
        topic_word_counts,
        topic_total_words,
        doc_topic_counts,
        _group_sentence_values_by_doc(assignments_flat, bundle.doc_offsets),
        _group_sentence_values_by_doc(sentence_topic_soft_flat, bundle.doc_offsets),
        _group_sentence_values_by_doc(log_likelihood_flat, bundle.doc_offsets),
        _group_sentence_values_by_doc(log_prior_flat, bundle.doc_offsets),
    )


def _infer_sentlda(
    *,
    bundle: _CorpusBundle,
    num_topics: int,
    alpha: float,
    beta: float,
    vocab_size: int,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    num_iterations: int,
    random_state: int,
    backend: str,
) -> tuple[
    np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
]:
    rng = np.random.default_rng(random_state)
    doc_topic_counts, assignments_flat = _initialize_infer_state(
        bundle=bundle,
        num_topics=num_topics,
        rng=rng,
    )

    for _ in get_progress_bar(
        range(num_iterations),
        desc="sentLDA infer",
        leave=False,
    ):
        run_sentlda_infer_iteration(
            doc_offsets=bundle.doc_offsets,
            sentence_lengths=bundle.sentence_lengths,
            sentence_unique_offsets=bundle.sentence_unique_offsets,
            sentence_unique_word_ids_flat=bundle.sentence_unique_word_ids_flat,
            sentence_word_counts_flat=bundle.sentence_word_counts_flat,
            topic_word_counts=topic_word_counts,
            topic_total_words=topic_total_words,
            doc_topic_counts=doc_topic_counts,
            assignments=assignments_flat,
            alpha=alpha,
            beta=beta,
            vocab_size=vocab_size,
            uniforms=rng.random(bundle.num_sentences),
            backend=backend,
        )

    sentence_topic_soft_flat = build_sentence_topic_soft_infer(
        doc_offsets=bundle.doc_offsets,
        sentence_lengths=bundle.sentence_lengths,
        sentence_unique_offsets=bundle.sentence_unique_offsets,
        sentence_unique_word_ids_flat=bundle.sentence_unique_word_ids_flat,
        sentence_word_counts_flat=bundle.sentence_word_counts_flat,
        topic_word_counts=topic_word_counts,
        topic_total_words=topic_total_words,
        doc_topic_counts=doc_topic_counts,
        assignments=assignments_flat,
        alpha=alpha,
        beta=beta,
        vocab_size=vocab_size,
        backend=backend,
    )
    log_prior_flat, log_likelihood_flat = build_sentence_topic_log_factors_infer(
        doc_offsets=bundle.doc_offsets,
        sentence_lengths=bundle.sentence_lengths,
        sentence_unique_offsets=bundle.sentence_unique_offsets,
        sentence_unique_word_ids_flat=bundle.sentence_unique_word_ids_flat,
        sentence_word_counts_flat=bundle.sentence_word_counts_flat,
        topic_word_counts=topic_word_counts,
        topic_total_words=topic_total_words,
        doc_topic_counts=doc_topic_counts,
        assignments=assignments_flat,
        alpha=alpha,
        beta=beta,
        vocab_size=vocab_size,
        backend=backend,
    )

    return (
        doc_topic_counts,
        _group_sentence_values_by_doc(assignments_flat, bundle.doc_offsets),
        _group_sentence_values_by_doc(sentence_topic_soft_flat, bundle.doc_offsets),
        _group_sentence_values_by_doc(log_likelihood_flat, bundle.doc_offsets),
        _group_sentence_values_by_doc(log_prior_flat, bundle.doc_offsets),
    )


def train_sentlda(
    *,
    train_csvs: Sequence[str],
    targets: Sequence[str] | None,
    text_column: str,
    target_column: str | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
    num_topics: int,
    params: SentLdaParams,
    train_dir: Path,
    use_legacy: bool,
) -> SentLdaTrainResult:
    _ = train_dir
    backend = resolve_sentlda_backend(params.backend)
    LOG.info("Loading sentLDA training documents")
    train_preprocessed_raw = _load_documents(
        csv_paths=train_csvs,
        targets=targets,
        text_column=text_column,
        target_column=target_column,
        delimiter=delimiter,
        language=language,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
        use_legacy=use_legacy,
    )
    vocabulary = _build_vocabulary(train_preprocessed_raw)
    if not vocabulary:
        raise ValueError("Tokenization produced an empty vocabulary for sentLDA.")

    train_bundle = _encode_corpus(train_preprocessed_raw, vocabulary)
    if train_bundle.num_sentences == 0:
        raise ValueError("No non-empty sentences available for sentLDA training.")
    LOG.info(
        "sentLDA training corpus prepared: docs=%d, sentences=%d, vocab=%d, backend=%s",
        train_bundle.num_docs,
        train_bundle.num_sentences,
        len(vocabulary),
        backend,
    )

    alpha = float(params.alpha) if params.alpha is not None else 1.0 / float(num_topics)
    beta = float(params.beta) if params.beta is not None else 1.0 / float(num_topics)

    (
        topic_word_counts,
        topic_total_words,
        doc_topic_counts,
        assignments,
        sentence_topic_soft,
        sentence_topic_loglik,
        sentence_topic_logprior,
    ) = _fit_sentlda(
        bundle=train_bundle,
        num_topics=num_topics,
        alpha=alpha,
        beta=beta,
        vocab_size=len(vocabulary),
        num_iterations=params.num_iterations,
        random_state=params.random_state,
        backend=backend,
    )

    model_state = SentLdaModelState(
        num_topics=num_topics,
        vocab_size=len(vocabulary),
        alpha=alpha,
        beta=beta,
        vocabulary=vocabulary,
        topic_word_counts=topic_word_counts,
        topic_total_words=topic_total_words,
        doc_sentence_topic_counts=doc_topic_counts,
    )
    return SentLdaTrainResult(
        model_state=model_state,
        train_doc_topic=doc_topic_counts.astype(np.float64),
        train_sentence_topic_soft=sentence_topic_soft,
        train_sentence_topic_loglik=sentence_topic_loglik,
        train_sentence_topic_logprior=sentence_topic_logprior,
        train_preprocessed=train_bundle.preprocessed,
        train_sentence_topic_assignments=assignments,
        train_selection=train_bundle.selection,
    )


def infer_sentlda(
    *,
    train_result: SentLdaTrainResult,
    test_csvs: Sequence[str],
    targets: Sequence[str] | None,
    text_column: str,
    target_column: str | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
    num_topics: int,
    params: SentLdaParams,
    use_legacy: bool,
) -> SentLdaInferResult:
    _ = num_topics
    backend = resolve_sentlda_backend(params.backend)
    LOG.info("Loading sentLDA inference documents")
    test_preprocessed_raw = _load_documents(
        csv_paths=test_csvs,
        targets=targets,
        text_column=text_column,
        target_column=target_column,
        delimiter=delimiter,
        language=language,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
        use_legacy=use_legacy,
    )
    test_bundle = _encode_corpus(
        test_preprocessed_raw,
        train_result.model_state.vocabulary,
    )
    LOG.info(
        "sentLDA inference corpus prepared: docs=%d, sentences=%d, backend=%s",
        test_bundle.num_docs,
        test_bundle.num_sentences,
        backend,
    )

    (
        doc_topic_counts,
        assignments,
        sentence_topic_soft,
        sentence_topic_loglik,
        sentence_topic_logprior,
    ) = _infer_sentlda(
        bundle=test_bundle,
        num_topics=train_result.model_state.num_topics,
        alpha=train_result.model_state.alpha,
        beta=train_result.model_state.beta,
        vocab_size=train_result.model_state.vocab_size,
        topic_word_counts=train_result.model_state.topic_word_counts,
        topic_total_words=train_result.model_state.topic_total_words,
        num_iterations=params.infer_num_iterations,
        random_state=params.random_state,
        backend=backend,
    )
    return SentLdaInferResult(
        test_doc_topic=doc_topic_counts.astype(np.float64),
        test_sentence_topic_soft=sentence_topic_soft,
        test_sentence_topic_loglik=sentence_topic_loglik,
        test_sentence_topic_logprior=sentence_topic_logprior,
        test_preprocessed=test_bundle.preprocessed,
        test_sentence_topic_assignments=assignments,
        test_selection=test_bundle.selection,
    )


def persist_sentlda_run(
    *,
    train_result: SentLdaTrainResult,
    infer_result: SentLdaInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    phi = train_result.model_state.topic_word_counts.astype(np.float64)
    row_sums = phi.sum(axis=1, keepdims=True)
    bad_rows = row_sums.squeeze(-1) <= 0.0
    row_sums[bad_rows] = 1.0
    phi = phi / row_sums

    vocab_path = train_dir / "vocabulary.json"
    params_path = train_dir / "params.json"

    save_json(train_result.model_state.vocabulary, vocab_path)
    save_json(
        {
            "num_topics": train_result.model_state.num_topics,
            "vocab_size": train_result.model_state.vocab_size,
            "alpha": train_result.model_state.alpha,
            "beta": train_result.model_state.beta,
            "sentence_topic_score_schema": 1,
            "sentence_topic_soft_definition": "softmax(logprior + loglik)",
            "sentence_topic_loglik_definition": "log p(sentence|topic, rest)",
            "sentence_topic_logprior_definition": "log(doc_topic_count + alpha)",
        },
        params_path,
    )

    saved = save_split_pickles(
        [
            PickleArtifactSpec(
                name="train_path",
                filename="table_counts_per_doc.pkl",
                payload=train_result.train_doc_topic,
                split="train",
            ),
            PickleArtifactSpec(
                name="infer_path",
                filename=f"{category}.pkl",
                payload=infer_result.test_doc_topic,
                split="infer",
            ),
            PickleArtifactSpec(
                name="train_sentence_topic_soft",
                filename=f"{category}_sentence_topic_soft.pkl",
                payload=train_result.train_sentence_topic_soft,
                split="train",
            ),
            PickleArtifactSpec(
                name="test_sentence_topic_soft",
                filename=f"{category}_sentence_topic_soft.pkl",
                payload=infer_result.test_sentence_topic_soft,
                split="infer",
            ),
            PickleArtifactSpec(
                name="train_sentence_topic_loglik",
                filename=f"{category}_sentence_topic_loglik.pkl",
                payload=train_result.train_sentence_topic_loglik,
                split="train",
            ),
            PickleArtifactSpec(
                name="test_sentence_topic_loglik",
                filename=f"{category}_sentence_topic_loglik.pkl",
                payload=infer_result.test_sentence_topic_loglik,
                split="infer",
            ),
            PickleArtifactSpec(
                name="train_sentence_topic_logprior",
                filename=f"{category}_sentence_topic_logprior.pkl",
                payload=train_result.train_sentence_topic_logprior,
                split="train",
            ),
            PickleArtifactSpec(
                name="test_sentence_topic_logprior",
                filename=f"{category}_sentence_topic_logprior.pkl",
                payload=infer_result.test_sentence_topic_logprior,
                split="infer",
            ),
            PickleArtifactSpec(
                name="train_preprocessed",
                filename="preprocessed_corpus.pkl",
                payload=train_result.train_preprocessed,
                split="train",
            ),
            PickleArtifactSpec(
                name="infer_preprocessed",
                filename="preprocessed_corpus.pkl",
                payload=infer_result.test_preprocessed,
                split="infer",
            ),
            PickleArtifactSpec(
                name="train_sentence_topic_assignments",
                filename="sentence_topic_assignments.pkl",
                payload=train_result.train_sentence_topic_assignments,
                split="train",
            ),
            PickleArtifactSpec(
                name="infer_sentence_topic_assignments",
                filename="sentence_topic_assignments.pkl",
                payload=infer_result.test_sentence_topic_assignments,
                split="infer",
            ),
            PickleArtifactSpec(
                name="phi",
                filename="topic_word_distribution.pkl",
                payload=phi,
                split="train",
            ),
            PickleArtifactSpec(
                name="model_state",
                filename="model_state.pkl",
                payload=train_result.model_state,
                split="train",
            ),
        ],
        train_dir=train_dir,
        infer_dir=infer_dir,
    )
    train_selection = train_result.train_selection or select_modelable_documents(
        train_result.train_preprocessed
    )
    test_selection = infer_result.test_selection or select_modelable_documents(
        infer_result.test_preprocessed
    )
    selection_saved = save_split_jsons(
        {
            "train_preprocessing_selection": (
                train_selection.to_json_dict(),
                PREPROCESSING_SELECTION_FILENAME,
                "train",
            ),
            "infer_preprocessing_selection": (
                test_selection.to_json_dict(),
                PREPROCESSING_SELECTION_FILENAME,
                "infer",
            ),
        },
        train_dir=train_dir,
        infer_dir=infer_dir,
    )

    return BaselineArtifacts(
        train_path=saved["train_path"],
        infer_path=saved["infer_path"],
        extras={
            "vocabulary": vocab_path,
            "params_json": params_path,
            "model_state": saved["model_state"],
            "phi": saved["phi"],
            "train_sentence_topic_soft": saved["train_sentence_topic_soft"],
            "test_sentence_topic_soft": saved["test_sentence_topic_soft"],
            "train_sentence_topic_loglik": saved["train_sentence_topic_loglik"],
            "test_sentence_topic_loglik": saved["test_sentence_topic_loglik"],
            "train_sentence_topic_logprior": saved["train_sentence_topic_logprior"],
            "test_sentence_topic_logprior": saved["test_sentence_topic_logprior"],
            "train_preprocessed": saved["train_preprocessed"],
            "infer_preprocessed": saved["infer_preprocessed"],
            "train_preprocessing_selection": selection_saved[
                "train_preprocessing_selection"
            ],
            "infer_preprocessing_selection": selection_saved[
                "infer_preprocessing_selection"
            ],
            "train_sentence_topic_assignments": saved[
                "train_sentence_topic_assignments"
            ],
            "infer_sentence_topic_assignments": saved[
                "infer_sentence_topic_assignments"
            ],
        },
    )

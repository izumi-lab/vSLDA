from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gensim
import numpy as np

from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import (
    load_preprocessed_documents,
    load_preprocessed_documents_with_indices,
)
from src.baselines.models.gaussian_helpers import (
    GaussianLdaScorer,
    build_local_word2vec,
    load_word_vectors,
    should_use_external_vectors,
    to_index_docs,
)
from src.baselines.models.gaussian_persistence import persist_gaussian_family_run
from src.baselines.models.gaussian_state import (
    GaussianTrainerState,
    snapshot_gaussian_trainer,
)
from src.baselines.params import GaussianLdaParams
from src.core.artifacts import (
    PREPROCESSING_SELECTION_FILENAME,
    PickleArtifactSpec,
    save_split_jsons,
)
from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    filter_selected_corpus_by_vocabulary,
    select_modelable_documents,
)


@dataclass(frozen=True)
class GaussianLdaTrainResult:
    trainer_state: GaussianTrainerState
    model: Any
    train_doc_topic: np.ndarray
    local_word_vectors: gensim.models.KeyedVectors | None
    train_preprocessed: list[PreprocessedDocument]
    train_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class GaussianLdaInferResult:
    test_doc_topic: np.ndarray
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class PreparedWordVectorCorpus:
    index_docs: list[list[int]]
    token_docs: list[list[str]]
    embeddings: np.ndarray
    vocab: list[str]
    key_vectors: gensim.models.KeyedVectors
    local_word_vectors: gensim.models.KeyedVectors | None
    preprocessed: list[PreprocessedDocument]
    selection: SelectedCorpus


def prepare_word_vector_corpus(
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
    word2vec: str,
    wikientvec_cache_dir: str | None,
    local_word_vectors: gensim.models.KeyedVectors | None = None,
    empty_error_message: str = "No valid tokenized docs available for GaussianLDA.",
) -> PreparedWordVectorCorpus:
    if getattr(load_preprocessed_documents, "__module__", "").startswith(
        "src.baselines."
    ):
        documents, raw_indices = load_preprocessed_documents_with_indices(
            csv_paths=csv_paths,
            text_column=text_column,
            target_column=target_column,
            targets=targets,
            delimiter=delimiter,
            language=language,
            segmenter=segmenter,
            tokenizer=tokenizer,
            ja_replace_num=ja_replace_num,
            ja_stopwords_path=ja_stopwords_path,
            ja_dicdir=ja_dicdir,
            ja_require_unidic=ja_require_unidic,
        )
    else:
        documents = load_preprocessed_documents(
            csv_paths=csv_paths,
            text_column=text_column,
            target_column=target_column,
            targets=targets,
            delimiter=delimiter,
            language=language,
            segmenter=segmenter,
            tokenizer=tokenizer,
            ja_replace_num=ja_replace_num,
            ja_stopwords_path=ja_stopwords_path,
            ja_dicdir=ja_dicdir,
            ja_require_unidic=ja_require_unidic,
        )
        raw_indices = list(range(len(documents)))
    _ = use_legacy

    base_selection = select_modelable_documents(
        documents,
        raw_doc_indices=raw_indices,
    )
    token_docs = [doc.document_tokens for doc in base_selection.documents]
    if not token_docs:
        raise ValueError(empty_error_message)

    kv: gensim.models.KeyedVectors
    built_local = local_word_vectors
    if built_local is not None:
        kv = built_local
    elif should_use_external_vectors(word2vec):
        kv = load_word_vectors(
            word2vec,
            wikientvec_cache_dir=wikientvec_cache_dir,
        )
    else:
        kv = build_local_word2vec(token_docs)
        built_local = kv

    vocab_dict = kv.key_to_index
    selection = filter_selected_corpus_by_vocabulary(
        base_selection,
        {str(token) for token in vocab_dict},
    )
    if not selection.documents:
        raise ValueError(empty_error_message)
    token_docs_in_vocab = [doc.document_tokens for doc in selection.documents]
    index_docs = to_index_docs(token_docs_in_vocab, vocab_dict)
    return PreparedWordVectorCorpus(
        index_docs=index_docs,
        token_docs=token_docs_in_vocab,
        embeddings=np.asarray(kv.vectors),
        vocab=list(vocab_dict.keys()),
        key_vectors=kv,
        local_word_vectors=built_local,
        preprocessed=selection.documents,
        selection=selection,
    )


def _prepare_docs(
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
    word2vec: str,
    wikientvec_cache_dir: str | None,
    local_word_vectors: gensim.models.KeyedVectors | None = None,
) -> tuple[
    list[list[int]],
    np.ndarray,
    list[str],
    gensim.models.KeyedVectors | None,
    list[PreprocessedDocument],
]:
    prepared = prepare_word_vector_corpus(
        csv_paths=csv_paths,
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
        word2vec=word2vec,
        wikientvec_cache_dir=wikientvec_cache_dir,
        local_word_vectors=local_word_vectors,
    )
    return (
        prepared.index_docs,
        prepared.embeddings,
        prepared.vocab,
        prepared.local_word_vectors,
        prepared.preprocessed,
    )


def train_gaussianlda(
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
    params: GaussianLdaParams,
    train_dir: Path,
    use_legacy: bool,
) -> GaussianLdaTrainResult:
    from src.baselines.models.gaussian_trainer import GaussianLDATrainer

    _ = train_dir
    if getattr(_prepare_docs, "__module__", "") == __name__:
        prepared = prepare_word_vector_corpus(
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
            word2vec=params.word2vec,
            wikientvec_cache_dir=params.wikientvec_cache_dir,
        )
        corpus = prepared.index_docs
        embeddings = prepared.embeddings
        vocab = prepared.vocab
        local_kv = prepared.local_word_vectors
        train_preprocessed = prepared.preprocessed
        train_selection = prepared.selection
    else:
        corpus, embeddings, vocab, local_kv, train_preprocessed = _prepare_docs(
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
            word2vec=params.word2vec,
            wikientvec_cache_dir=params.wikientvec_cache_dir,
        )
        train_selection = select_modelable_documents(train_preprocessed)
    trainer = GaussianLDATrainer(
        corpus,
        embeddings,
        vocab,
        num_topics,
        1.0 / float(num_topics),
        save_path=None,
    )
    trainer.sample(params.num_iterations)
    trainer_state = snapshot_gaussian_trainer(trainer)
    model = GaussianLdaScorer(
        embeddings=embeddings,
        vocab=vocab,
        num_tables=trainer_state.num_tables,
        alpha=trainer_state.alpha,
        kappa=trainer_state.prior_kappa,
        table_counts=trainer_state.table_counts,
        table_means=trainer_state.table_means,
        log_determinants=trainer_state.log_determinants,
        table_cholesky_ltriangular_mat=trainer_state.table_cholesky_ltriangular_mat,
    )
    return GaussianLdaTrainResult(
        trainer_state=trainer_state,
        model=model,
        train_doc_topic=np.asarray(trainer_state.table_counts_per_doc.T, dtype=float),
        local_word_vectors=local_kv,
        train_preprocessed=train_preprocessed,
        train_selection=train_selection,
    )


def infer_gaussianlda(
    *,
    train_result: GaussianLdaTrainResult,
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
    params: GaussianLdaParams,
    use_legacy: bool,
) -> GaussianLdaInferResult:
    if getattr(_prepare_docs, "__module__", "") == __name__:
        prepared = prepare_word_vector_corpus(
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
            word2vec=params.word2vec,
            wikientvec_cache_dir=params.wikientvec_cache_dir,
            local_word_vectors=train_result.local_word_vectors,
        )
        corpus = prepared.index_docs
        test_preprocessed = prepared.preprocessed
        test_selection = prepared.selection
    else:
        corpus, _embeddings, _vocab, _local_kv, test_preprocessed = _prepare_docs(
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
            word2vec=params.word2vec,
            wikientvec_cache_dir=params.wikientvec_cache_dir,
            local_word_vectors=train_result.local_word_vectors,
        )
        test_selection = select_modelable_documents(test_preprocessed)
    output = np.zeros((len(corpus), num_topics), dtype=float)
    for row_index, doc in enumerate(corpus):
        topics = train_result.model.sample(doc, params.num_iterations)
        for topic_index in topics:
            output[row_index, topic_index] += 1.0
    return GaussianLdaInferResult(
        test_doc_topic=output,
        test_preprocessed=test_preprocessed,
        test_selection=test_selection,
    )


def persist_gaussianlda_run(
    *,
    train_result: GaussianLdaTrainResult,
    infer_result: GaussianLdaInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    artifacts = persist_gaussian_family_run(
        trainer=train_result.trainer_state,
        train_doc_topic=train_result.train_doc_topic,
        infer_doc_topic=infer_result.test_doc_topic,
        train_dir=train_dir,
        infer_dir=infer_dir,
        category=category,
        local_word_vectors=train_result.local_word_vectors,
        additional_specs=[
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
        ],
        extra_saved_artifact_names=["train_preprocessed", "infer_preprocessed"],
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
    artifacts.extras.update(selection_saved)
    return artifacts

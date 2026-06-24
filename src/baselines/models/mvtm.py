from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import gensim
import numpy as np

from src.baselines.contracts import BaselineArtifacts
from src.baselines.models.gaussianlda import (
    PreparedWordVectorCorpus,
    prepare_word_vector_corpus,
)
from src.baselines.params import MvTMParams
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
    select_modelable_documents,
)
from src.models.vmf_sentence_lda import VMFLDATrainer


class WordVectorEncoder:
    """Adapter that lets VMFLDATrainer observe tokens as word vectors."""

    def __init__(self, vectors: gensim.models.KeyedVectors) -> None:
        self.vectors = vectors

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        rows = [
            np.asarray(self.vectors[str(token)], dtype=np.float64)
            for token in tokens
            if str(token) in self.vectors.key_to_index
        ]
        if not rows:
            return np.zeros((0, self.get_sentence_embedding_dimension()), dtype=float)
        return np.vstack(rows).astype(np.float64, copy=False)

    def get_sentence_embedding_dimension(self) -> int:
        return int(self.vectors.vector_size)


@dataclass(frozen=True)
class MvTMTrainResult:
    trainer: VMFLDATrainer
    params: MvTMParams
    train_doc_topic: np.ndarray
    train_doc_topic_soft: np.ndarray
    local_word_vectors: gensim.models.KeyedVectors | None
    word_vectors: gensim.models.KeyedVectors
    vocab: list[str]
    train_preprocessed: list[PreprocessedDocument]
    topic_words: list[list[tuple[str, float]]]
    resolved_alpha: float
    train_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class MvTMInferResult:
    test_doc_topic: np.ndarray
    test_doc_topic_soft: np.ndarray
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None


def _prepare_mvtm_corpus(
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
    params: MvTMParams,
    local_word_vectors: gensim.models.KeyedVectors | None = None,
) -> PreparedWordVectorCorpus:
    return prepare_word_vector_corpus(
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
        word2vec=params.word2vec,
        wikientvec_cache_dir=params.wikientvec_cache_dir,
        local_word_vectors=local_word_vectors,
        empty_error_message="No valid tokenized docs available for MvTM.",
    )


def _build_algorithm_variant(params: MvTMParams) -> str:
    alpha_part = (
        "alpha_auto_inverse_k" if params.alpha is None else f"alpha={params.alpha}"
    )
    alpha_update_part = (
        "fixed_alpha"
        if not params.estimate_alpha
        else f"estimate_alpha_every_{params.alpha_update_every}"
    )
    return "__".join(
        [
            f"components_{params.num_components}",
            alpha_part,
            alpha_update_part,
        ]
    )


def _top_topic_words(
    *,
    trainer: VMFLDATrainer,
    vectors: gensim.models.KeyedVectors,
    topn: int = 20,
) -> list[list[tuple[str, float]]]:
    vocab = [str(word) for word in vectors.key_to_index.keys()]
    raw = np.asarray(vectors.vectors, dtype=np.float64)
    if raw.size == 0:
        return [[] for _ in range(trainer.num_topics)]
    norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
    scores = trainer.log_vmf_density_matrix(raw / norms)
    topic_words: list[list[tuple[str, float]]] = []
    for topic_index in range(trainer.num_topics):
        row = np.asarray(scores[:, topic_index], dtype=float)
        top_ids = np.argsort(-row, kind="stable")[:topn]
        topic_words.append(
            [(vocab[word_id], float(row[word_id])) for word_id in top_ids]
        )
    return topic_words


def train_mvtm(
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
    params: MvTMParams,
    train_dir: Path,
    use_legacy: bool,
) -> MvTMTrainResult:
    _ = train_dir
    prepared = _prepare_mvtm_corpus(
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
        params=params,
    )
    resolved_alpha = (
        1.0 / float(num_topics) if params.alpha is None else float(params.alpha)
    )
    trainer = VMFLDATrainer(
        corpus=prepared.token_docs,
        encoder=WordVectorEncoder(prepared.key_vectors),
        num_topics=num_topics,
        alpha=resolved_alpha,
        kappa=params.kappa_default,
        num_components=params.num_components,
        pre_normalize_transform="none",
        algorithm_variant=_build_algorithm_variant(params),
        save_path=None,
    )
    trainer.sample(
        params.num_iterations,
        num_sweeps=params.gibbs_sweeps,
        num_samples=params.num_samples,
        estimate_alpha=params.estimate_alpha,
        alpha_update_every=params.alpha_update_every,
        alpha_max_iter=params.alpha_max_iter,
        alpha_tol=params.alpha_tol,
        avg_log_likelihood_every=params.avg_log_likelihood_every,
        invariant_check_every=params.invariant_check_every,
    )
    train_inference = trainer.infer_encoded_corpus_topic_outputs(
        trainer.encoded_corpus,
        temperature=params.soft_temperature,
        include_document_posteriors=True,
    )
    if train_inference.document_posteriors is None:
        raise RuntimeError("MvTM train document posteriors were not produced.")
    return MvTMTrainResult(
        trainer=trainer,
        params=params,
        train_doc_topic=trainer.get_document_topic_distribution(),
        train_doc_topic_soft=train_inference.document_posteriors,
        local_word_vectors=prepared.local_word_vectors,
        word_vectors=prepared.key_vectors,
        vocab=prepared.vocab,
        train_preprocessed=prepared.preprocessed,
        topic_words=_top_topic_words(trainer=trainer, vectors=prepared.key_vectors),
        resolved_alpha=resolved_alpha,
        train_selection=prepared.selection,
    )


def infer_mvtm(
    *,
    train_result: MvTMTrainResult,
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
    params: MvTMParams,
    use_legacy: bool,
) -> MvTMInferResult:
    _ = num_topics
    prepared = _prepare_mvtm_corpus(
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
        params=params,
        local_word_vectors=train_result.word_vectors,
    )
    outputs = train_result.trainer.infer_corpus_topic_outputs(
        prepared.token_docs,
        temperature=params.soft_temperature,
        include_counts=True,
        include_document_posteriors=True,
    )
    if outputs.counts is None:
        raise RuntimeError("MvTM test counts were not produced.")
    if outputs.document_posteriors is None:
        raise RuntimeError("MvTM test document posteriors were not produced.")
    return MvTMInferResult(
        test_doc_topic=np.asarray(outputs.counts, dtype=float),
        test_doc_topic_soft=outputs.document_posteriors,
        test_preprocessed=prepared.preprocessed,
        test_selection=prepared.selection,
    )


def _params_payload(
    *,
    train_result: MvTMTrainResult,
) -> dict[str, object]:
    trainer = train_result.trainer
    payload: dict[str, object] = {
        "average_ll": list(trainer.average_ll),
        "iteration_diagnostics": [
            asdict(item) for item in trainer.iteration_diagnostics
        ],
        "alpha": np.asarray(trainer.alpha, dtype=float).tolist(),
        "resolved_alpha": float(train_result.resolved_alpha),
        "num_topics": int(trainer.num_topics),
        "num_components": int(trainer.num_components),
        "kappa_default": float(trainer.kappa_default),
        "algorithm_variant": trainer.algorithm_variant,
        "e_step_kernel_backend": trainer.e_step_kernel_backend,
        "m_step_statistics_kernel_backend": trainer.m_step_statistics_kernel_backend,
        "avg_ll_kernel_backend": trainer.avg_ll_kernel_backend,
    }
    baseline_params = dict(train_result.params.__dict__)
    baseline_params["word2vec"] = str(baseline_params["word2vec"])
    payload["baseline_params"] = baseline_params
    return payload


def persist_mvtm_run(
    *,
    train_result: MvTMTrainResult,
    infer_result: MvTMInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    trainer = train_result.trainer
    params_path = train_dir / "params.json"
    save_json(_params_payload(train_result=train_result), params_path)
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
                name="train_doc_topic_soft",
                filename="doc_topic_train_soft.pkl",
                payload=train_result.train_doc_topic_soft,
                split="train",
            ),
            PickleArtifactSpec(
                name="test_doc_topic_soft",
                filename=f"{category}_doc_topic_soft.pkl",
                payload=infer_result.test_doc_topic_soft,
                split="infer",
            ),
            PickleArtifactSpec(
                name="topic_counts",
                filename="topic_counts.pkl",
                payload=trainer.topic_counts,
                split="train",
            ),
            PickleArtifactSpec(
                name="topic_counts_per_doc",
                filename="topic_counts_per_doc.pkl",
                payload=trainer.topic_counts_per_doc.T,
                split="train",
            ),
            PickleArtifactSpec(
                name="topic_means",
                filename="topic_means.pkl",
                payload=trainer.topic_means,
                split="train",
            ),
            PickleArtifactSpec(
                name="sum_topic_vectors",
                filename="sum_topic_vectors.pkl",
                payload=trainer.sum_topic_vectors,
                split="train",
            ),
            PickleArtifactSpec(
                name="kappa_per_topic",
                filename="kappa_per_topic.pkl",
                payload=trainer.kappa_per_topic,
                split="train",
            ),
            PickleArtifactSpec(
                name="mixture_weights",
                filename="mixture_weights.pkl",
                payload=trainer.mixture_weights,
                split="train",
            ),
            PickleArtifactSpec(
                name="component_means",
                filename="component_means.pkl",
                payload=trainer.component_means,
                split="train",
            ),
            PickleArtifactSpec(
                name="topic_words",
                filename="topic_words.pkl",
                payload=train_result.topic_words,
                split="train",
            ),
            PickleArtifactSpec(
                name="vocab",
                filename="vocab.pkl",
                payload=train_result.vocab,
                split="train",
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
    extras: dict[str, Path] = {
        "params_json": params_path,
        "train_doc_topic_soft": saved["train_doc_topic_soft"],
        "test_doc_topic_soft": saved["test_doc_topic_soft"],
        "topic_words": saved["topic_words"],
        "vocab": saved["vocab"],
        "train_preprocessed": saved["train_preprocessed"],
        "infer_preprocessed": saved["infer_preprocessed"],
        "train_preprocessing_selection": selection_saved[
            "train_preprocessing_selection"
        ],
        "infer_preprocessing_selection": selection_saved[
            "infer_preprocessing_selection"
        ],
    }
    if train_result.local_word_vectors is not None:
        kv_path = train_dir / "local_word2vec.kv"
        train_result.local_word_vectors.save(kv_path.as_posix())
        extras["local_word2vec"] = kv_path
    return BaselineArtifacts(
        train_path=saved["train_path"],
        infer_path=saved["infer_path"],
        extras=extras,
    )

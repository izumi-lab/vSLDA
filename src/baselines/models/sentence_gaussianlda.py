from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import load_preprocessed_documents
from src.baselines.models.gaussian_persistence import persist_gaussian_family_run
from src.baselines.models.gaussian_state import (
    GaussianTrainerState,
    snapshot_gaussian_trainer,
)
from src.baselines.models.sentence_gaussian_helpers import (
    SentenceGaussianLdaModel,
    build_sentence_gaussian_encoder,
)
from src.baselines.params import SentenceGaussianLdaParams
from src.core.artifacts import (
    PREPROCESSING_SELECTION_FILENAME,
    PickleArtifactSpec,
    save_split_jsons,
)
from src.core.contracts import TopicModelOutput
from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    select_modelable_documents,
)
from src.utils.encoder_inputs import (
    fit_encoder_on_sentences,
    sentence_corpus_for_encoder,
)


@dataclass(frozen=True)
class SentenceGaussianLdaTrainResult:
    trainer_state: GaussianTrainerState
    model: Any
    train_doc_topic: np.ndarray
    train_sentence_topic_soft: list[np.ndarray]
    train_preprocessed: list[PreprocessedDocument]
    train_selection: SelectedCorpus | None = None

    def to_output(self) -> TopicModelOutput:
        return TopicModelOutput(
            doc_topic=self.train_doc_topic,
            sentence_topic=self.train_sentence_topic_soft,
            topic_embeddings=np.asarray(self.trainer_state.table_means),
            metadata={
                "model_name": "sentence_gaussian_lda",
                "num_topics": int(self.trainer_state.num_tables),
            },
        )


@dataclass(frozen=True)
class SentenceGaussianLdaInferResult:
    test_doc_topic: np.ndarray
    test_sentence_topic_soft: list[np.ndarray]
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None

    def to_output(self) -> TopicModelOutput:
        return TopicModelOutput(
            doc_topic=self.test_doc_topic,
            sentence_topic=self.test_sentence_topic_soft,
            metadata={"model_name": "sentence_gaussian_lda"},
        )


def _sentence_topic_soft(
    *,
    corpus: Sequence[Sequence[str]],
    model: SentenceGaussianLdaModel,
    num_topics: int,
    batch_size: int,
    soft_temperature: float,
    show_progress_bar: bool,
) -> list[np.ndarray]:
    sentence_topic_soft: list[np.ndarray] = []
    for doc in corpus:
        doc_embeddings = np.asarray(
            model.encoder.encode(
                list(doc),
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
        )
        if doc_embeddings.size == 0:
            sentence_topic_soft.append(np.zeros((0, num_topics), dtype=np.float32))
            continue
        sent_loglik = np.zeros((doc_embeddings.shape[0], num_topics), dtype=np.float64)
        for sentence_index, embedding in enumerate(doc_embeddings):
            sent_loglik[sentence_index] = model.log_multivariate_tdensity_tables(
                embedding
            )
        if soft_temperature != 1.0:
            sent_loglik = sent_loglik / soft_temperature
        sent_loglik -= sent_loglik.max(axis=1, keepdims=True)
        sent_probs = np.exp(sent_loglik)
        row_sums = sent_probs.sum(axis=1, keepdims=True)
        bad_rows = (~np.isfinite(row_sums)) | (row_sums <= 0.0)
        if np.any(bad_rows):
            sent_probs[bad_rows[:, 0]] = 1.0 / float(num_topics)
            row_sums = sent_probs.sum(axis=1, keepdims=True)
        sent_probs /= row_sums
        sentence_topic_soft.append(sent_probs.astype(np.float32))
    return sentence_topic_soft


def train_sentence_gaussianlda(
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
    encoder_device: str,
    params: SentenceGaussianLdaParams,
    train_dir: Path,
    use_legacy: bool,
) -> SentenceGaussianLdaTrainResult:
    from src.baselines.models.sentence_gaussian_trainer import GaussianLDATrainer

    _ = train_dir
    encoder = build_sentence_gaussian_encoder(
        params.encoder_model_name,
        device=encoder_device,
        encode_prefix=params.encode_prefix,
        backend=params.encoder_backend,
        pooling=params.pooling,
        encode_prompt=params.encode_prompt,
        encode_prompt_name=params.encode_prompt_name,
        encode_batch_size=params.encode_batch_size,
        model_kwargs=params.model_kwargs,
        tokenizer_kwargs=params.tokenizer_kwargs,
        normalize_embeddings=params.normalize_embeddings,
        truncate_dim=params.truncate_dim,
        strip_terminal_normalize=params.strip_terminal_normalize,
    )
    train_preprocessed = load_preprocessed_documents(
        csv_paths=train_csvs,
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
    train_selection = select_modelable_documents(train_preprocessed)
    train_preprocessed = train_selection.documents
    fit_encoder_on_sentences(encoder, train_preprocessed)
    corpus = sentence_corpus_for_encoder(train_preprocessed, encoder)
    trainer = GaussianLDATrainer(
        corpus,
        encoder,
        num_topics,
        0.1,
        0.1,
        save_path=None,
        preencode_corpus=params.preencode_corpus,
    )
    trainer.sample(params.num_iterations)
    trainer_state = snapshot_gaussian_trainer(trainer, include_prior_mu=True)
    model = SentenceGaussianLdaModel(
        prior_mu=np.asarray(trainer_state.prior_mu, dtype=np.float64),
        encoder=encoder,
        num_tables=trainer_state.num_tables,
        alpha=trainer_state.alpha,
        kappa=trainer_state.prior_kappa,
        table_counts=trainer_state.table_counts,
        table_means=trainer_state.table_means,
        log_determinants=trainer_state.log_determinants,
        table_cholesky_ltriangular_mat=trainer_state.table_cholesky_ltriangular_mat,
    )
    return SentenceGaussianLdaTrainResult(
        trainer_state=trainer_state,
        model=model,
        train_doc_topic=np.asarray(trainer_state.table_counts_per_doc.T, dtype=float),
        train_sentence_topic_soft=_sentence_topic_soft(
            corpus=corpus,
            model=model,
            num_topics=num_topics,
            batch_size=params.encode_batch_size,
            soft_temperature=params.soft_temperature,
            show_progress_bar=False,
        ),
        train_preprocessed=train_preprocessed,
        train_selection=train_selection,
    )


def infer_sentence_gaussianlda(
    *,
    train_result: SentenceGaussianLdaTrainResult,
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
    params: SentenceGaussianLdaParams,
    use_legacy: bool,
) -> SentenceGaussianLdaInferResult:
    test_preprocessed = load_preprocessed_documents(
        csv_paths=test_csvs,
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
    test_selection = select_modelable_documents(test_preprocessed)
    test_preprocessed = test_selection.documents
    corpus = sentence_corpus_for_encoder(test_preprocessed, train_result.model.encoder)
    output = np.zeros((len(corpus), num_topics), dtype=float)
    sentence_topic_soft = _sentence_topic_soft(
        corpus=corpus,
        model=train_result.model,
        num_topics=num_topics,
        batch_size=params.encode_batch_size,
        soft_temperature=params.soft_temperature,
        show_progress_bar=params.preencode_corpus,
    )
    for row_index, sent_probs in enumerate(sentence_topic_soft):
        if sent_probs.size == 0:
            continue
        if params.preencode_corpus:
            # Re-encode the matching sentences in a single batch for Gibbs inference.
            encoded_doc = np.asarray(
                train_result.model.encoder.encode(
                    list(corpus[row_index]),
                    batch_size=params.encode_batch_size,
                    show_progress_bar=False,
                )
            )
            topics = train_result.model.sample(encoded_doc, params.num_gibbs_iters)
        else:
            topics = train_result.model.sample(
                list(corpus[row_index]), params.num_gibbs_iters
            )
        for topic_index in topics:
            output[row_index, topic_index] += 1.0
    return SentenceGaussianLdaInferResult(
        test_doc_topic=output,
        test_sentence_topic_soft=sentence_topic_soft,
        test_preprocessed=test_preprocessed,
        test_selection=test_selection,
    )


def persist_sentence_gaussianlda_run(
    *,
    train_result: SentenceGaussianLdaTrainResult,
    infer_result: SentenceGaussianLdaInferResult,
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
        additional_specs=[
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
                name="prior_mu",
                filename="prior_mu.pkl",
                payload=np.asarray(
                    train_result.trainer_state.prior_mu, dtype=np.float64
                ),
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
        extra_saved_artifact_names=[
            "train_sentence_topic_soft",
            "test_sentence_topic_soft",
            "train_preprocessed",
            "infer_preprocessed",
        ],
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

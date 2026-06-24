from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import load_preprocessed_documents
from src.baselines.models.senclu_internal import SenClu
from src.baselines.params import SenCluParams
from src.core.artifacts import (
    PREPROCESSING_SELECTION_FILENAME,
    PickleArtifactSpec,
    save_split_jsons,
    save_split_pickles,
)
from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    select_modelable_documents,
)
from src.utils.encoder_inputs import flatten_sentence_tokens


@dataclass(frozen=True)
class SenCluTrainResult:
    train_doc_topic: np.ndarray
    test_doc_topic: np.ndarray
    train_sentence_topic_soft: list[np.ndarray]
    test_sentence_topic_soft: list[np.ndarray]
    train_preprocessed: list[PreprocessedDocument]
    test_preprocessed: list[PreprocessedDocument]
    train_selection: SelectedCorpus | None = None
    test_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class SenCluInferResult:
    test_doc_topic: np.ndarray

    test_sentence_topic_soft: list[np.ndarray]
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None


def train_senclu(
    *,
    train_csvs: Sequence[str],
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
    encoder_device: str,
    params: SenCluParams,
    train_dir: Path,
    use_legacy: bool,
) -> SenCluTrainResult:
    _ = train_dir
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
    train_selection = select_modelable_documents(train_preprocessed)
    test_selection = select_modelable_documents(test_preprocessed)
    train_preprocessed = train_selection.documents
    test_preprocessed = test_selection.documents
    use_usif = (
        str(params.encoder_model_name).strip().lower() == "usif"
        or str(params.encoder_backend).strip().lower() == "usif"
    )
    if use_usif:
        train_corpus = [
            [" ".join(tokens) for tokens in doc.sentences_tokenized]
            for doc in train_preprocessed
            if doc.sentences_raw
        ]
        test_corpus = [
            [" ".join(tokens) for tokens in doc.sentences_tokenized]
            for doc in test_preprocessed
            if doc.sentences_raw
        ]
    else:
        train_corpus = [
            doc.sentences_raw for doc in train_preprocessed if doc.sentences_raw
        ]
        test_corpus = [
            doc.sentences_raw for doc in test_preprocessed if doc.sentences_raw
        ]

    trainer = SenClu(
        device=encoder_device,
        encoder_model_name=params.encoder_model_name,
        encode_prefix=params.encode_prefix,
        encoder_backend=params.encoder_backend,
        pooling=params.pooling,
        encode_prompt=params.encode_prompt,
        encode_prompt_name=params.encode_prompt_name,
        encode_batch_size=params.encode_batch_size,
        model_kwargs=params.model_kwargs,
        tokenizer_kwargs=params.tokenizer_kwargs,
        normalize_embeddings=params.normalize_embeddings,
        truncate_dim=params.truncate_dim,
        encoder_fit_tokenized=flatten_sentence_tokens(train_preprocessed),
    )
    train_doc_topic, test_doc_topic = trainer.fit_transform(
        train_docs=train_corpus,
        test_docs=test_corpus,
        nTopics=num_topics,
        alpha=params.alpha,
        nEpochs=params.num_epochs,
        loadAndStoreInFolder=params.embedding_cache_dir,
        verbose=params.verbose,
    )
    sentence_topic_soft = trainer.get_sentence_topic_distribution_soft(
        temperature=params.soft_temperature
    )
    train_n = len(train_corpus)
    return SenCluTrainResult(
        train_doc_topic=np.asarray(train_doc_topic, dtype=float),
        test_doc_topic=np.asarray(test_doc_topic, dtype=float),
        train_sentence_topic_soft=[
            np.asarray(item, dtype=float) for item in sentence_topic_soft[:train_n]
        ],
        test_sentence_topic_soft=[
            np.asarray(item, dtype=float) for item in sentence_topic_soft[train_n:]
        ],
        train_preprocessed=train_preprocessed,
        test_preprocessed=test_preprocessed,
        train_selection=train_selection,
        test_selection=test_selection,
    )


def infer_senclu(*, train_result: SenCluTrainResult) -> SenCluInferResult:
    return SenCluInferResult(
        test_doc_topic=np.asarray(train_result.test_doc_topic, dtype=float),
        test_sentence_topic_soft=[
            np.asarray(item, dtype=float)
            for item in train_result.test_sentence_topic_soft
        ],
        test_preprocessed=list(train_result.test_preprocessed),
        test_selection=train_result.test_selection,
    )


def persist_senclu_run(
    *,
    train_result: SenCluTrainResult,
    infer_result: SenCluInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    pickle_specs = [
        PickleArtifactSpec(
            name="train_path",
            filename=f"{category}.pkl",
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
    ]
    pickle_specs.extend(
        [
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
        ]
    )
    saved = save_split_pickles(
        pickle_specs,
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
    extras = {
        "train_sentence_topic_soft": saved["train_sentence_topic_soft"],
        "test_sentence_topic_soft": saved["test_sentence_topic_soft"],
        "train_preprocessed": saved["train_preprocessed"],
        "infer_preprocessed": saved["infer_preprocessed"],
        "train_preprocessing_selection": selection_saved[
            "train_preprocessing_selection"
        ],
        "infer_preprocessing_selection": selection_saved[
            "infer_preprocessing_selection"
        ],
    }
    return BaselineArtifacts(
        train_path=saved["train_path"],
        infer_path=saved["infer_path"],
        extras=extras,
    )

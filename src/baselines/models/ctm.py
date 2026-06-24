from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import load_preprocessed_documents
from src.baselines.params import CtmParams
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
from src.utils.encoder import SentenceEncoder
from src.utils.encoder_inputs import encode_documents, fit_encoder_on_documents

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CombinedTM: Any | None = None
TopicModelDataPreparation: Any | None = None


def _load_ctm_dependencies() -> tuple[Any, Any]:
    global CombinedTM, TopicModelDataPreparation
    if CombinedTM is not None and TopicModelDataPreparation is not None:
        return CombinedTM, TopicModelDataPreparation
    try:
        from contextualized_topic_models.models.ctm import CombinedTM as _CombinedTM
        from contextualized_topic_models.utils.data_preparation import (
            TopicModelDataPreparation as _TopicModelDataPreparation,
        )
    except ImportError as exc:
        raise RuntimeError(
            "CTM baselines require ML dependencies. "
            "Install them with: poetry install --with ml"
        ) from exc
    if CombinedTM is None:
        CombinedTM = _CombinedTM
    if TopicModelDataPreparation is None:
        TopicModelDataPreparation = _TopicModelDataPreparation
    return CombinedTM, TopicModelDataPreparation


@dataclass(frozen=True)
class CtmTrainResult:
    model: CombinedTM
    topic_preparation: TopicModelDataPreparation
    encoder: SentenceEncoder | None
    train_doc_topic: np.ndarray
    model_dir: Path
    train_preprocessed: list[PreprocessedDocument]
    train_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class CtmInferResult:
    test_doc_topic: np.ndarray
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None


def _load_preprocessed_documents(
    *,
    csv_paths: Sequence[str],
    text_column: str,
    target_column: str | None,
    targets: Sequence[str] | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
) -> list[PreprocessedDocument]:
    return load_preprocessed_documents(
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


def _build_ctm_model_dir(train_dir: Path, num_topics: int) -> Path:
    topic_prior_variance = 1 - (1.0 / float(num_topics))
    name = (
        "contextualized_topic_model_"
        f"nc_{num_topics}_tpm_0.0_tpv_{topic_prior_variance}"
        "_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99"
    )
    return train_dir / name


def _apply_contextual_prefix(
    docs: Sequence[str],
    prefix: str | None,
) -> list[str]:
    if not prefix:
        return list(docs)
    return [text if text.startswith(prefix) else f"{prefix}{text}" for text in docs]


def _contextual_size_from_dataset(training_dataset: object) -> int:
    contextual = getattr(training_dataset, "X_contextual")
    return int(contextual.shape[1])


def _build_contextual_encoder(
    *,
    params: CtmParams,
    encoder_device: str,
) -> SentenceEncoder:
    return SentenceEncoder(
        params.contextual_model_name,
        device=encoder_device,
        encode_prefix=params.contextual_encode_prefix,
        backend=params.encoder_backend,
        pooling=params.pooling,
        encode_prompt=params.encode_prompt,
        encode_prompt_name=params.encode_prompt_name,
        encode_batch_size=params.encode_batch_size,
        model_kwargs=params.model_kwargs,
        tokenizer_kwargs=params.tokenizer_kwargs,
        normalize_embeddings=params.normalize_embeddings,
        truncate_dim=params.truncate_dim,
    )


def train_ctm(
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
    params: CtmParams,
    train_dir: Path,
    use_legacy: bool,
    encoder_device: str = "auto",
) -> CtmTrainResult:
    ctm_cls, topic_preparation_cls = _load_ctm_dependencies()
    train_preprocessed = _load_preprocessed_documents(
        csv_paths=train_csvs,
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
    _ = use_legacy
    train_selection = select_modelable_documents(train_preprocessed)
    train_preprocessed = train_selection.documents
    usable_docs = [
        doc for doc in train_preprocessed if doc.document_tokens and doc.contextual_text
    ]
    preproc_docs = [doc.lexical_text for doc in usable_docs]
    unpreproc_docs = _apply_contextual_prefix(
        [doc.contextual_text for doc in usable_docs],
        params.contextual_encode_prefix,
    )
    encoder: SentenceEncoder | None = None
    custom_embeddings: np.ndarray | None = None
    if params.use_custom_embeddings:
        encoder = _build_contextual_encoder(
            params=params,
            encoder_device=encoder_device,
        )
        fit_encoder_on_documents(encoder, usable_docs)
        if getattr(encoder, "accepts_tokenized", False):
            custom_embeddings = encode_documents(encoder, usable_docs)
        else:
            custom_embeddings = encoder.encode(unpreproc_docs)
        tp = topic_preparation_cls(params.contextual_model_name)
    else:
        tp = topic_preparation_cls(params.contextual_model_name)
    fit_kwargs = {
        "text_for_contextual": unpreproc_docs,
        "text_for_bow": preproc_docs,
    }
    if custom_embeddings is not None:
        fit_kwargs["custom_embeddings"] = custom_embeddings
    training_dataset = tp.fit(**fit_kwargs)

    batch_size = max(1, min(params.batch_size_cap, len(preproc_docs)))
    ctm = ctm_cls(
        bow_size=len(tp.vocab),
        contextual_size=_contextual_size_from_dataset(training_dataset),
        n_components=num_topics,
        num_epochs=params.num_epochs,
        num_data_loader_workers=0,
        batch_size=batch_size,
    )
    ctm.fit(training_dataset)
    train_doc_topic = np.asarray(
        ctm.get_doc_topic_distribution(training_dataset, params.num_samples),
        dtype=float,
    )
    return CtmTrainResult(
        model=ctm,
        topic_preparation=tp,
        encoder=encoder,
        train_doc_topic=train_doc_topic,
        model_dir=_build_ctm_model_dir(train_dir, num_topics),
        train_preprocessed=train_preprocessed,
        train_selection=train_selection,
    )


def infer_ctm(
    *,
    train_result: CtmTrainResult,
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
    params: CtmParams,
    use_legacy: bool,
) -> CtmInferResult:
    test_preprocessed = _load_preprocessed_documents(
        csv_paths=test_csvs,
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
    _ = use_legacy
    test_selection = select_modelable_documents(test_preprocessed)
    test_preprocessed = test_selection.documents
    usable_docs = [
        doc for doc in test_preprocessed if doc.document_tokens and doc.contextual_text
    ]
    preproc_docs = [doc.lexical_text for doc in usable_docs]
    unpreproc_docs = _apply_contextual_prefix(
        [doc.contextual_text for doc in usable_docs],
        params.contextual_encode_prefix,
    )
    custom_embeddings = None
    if params.use_custom_embeddings:
        if train_result.encoder is None:
            raise ValueError("CTM custom embeddings requested but encoder is missing.")
        if getattr(train_result.encoder, "accepts_tokenized", False):
            custom_embeddings = encode_documents(train_result.encoder, usable_docs)
        else:
            custom_embeddings = train_result.encoder.encode(unpreproc_docs)
    transform_kwargs = {
        "text_for_contextual": unpreproc_docs,
        "text_for_bow": preproc_docs,
    }
    if custom_embeddings is not None:
        transform_kwargs["custom_embeddings"] = custom_embeddings
    test_set = train_result.topic_preparation.transform(**transform_kwargs)
    test_doc_topic = np.asarray(
        train_result.model.get_doc_topic_distribution(
            test_set, n_samples=params.num_samples
        ),
        dtype=float,
    )
    if test_doc_topic.ndim != 2 or test_doc_topic.shape[1] != num_topics:
        raise ValueError(
            f"Unexpected CTM infer shape: {test_doc_topic.shape} for {num_topics} topics."
        )
    return CtmInferResult(
        test_doc_topic=test_doc_topic,
        test_preprocessed=test_preprocessed,
        test_selection=test_selection,
    )


def persist_ctm_run(
    *,
    train_result: CtmTrainResult,
    infer_result: CtmInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    train_result.model.save(models_dir=train_dir)
    saved = save_split_pickles(
        [
            PickleArtifactSpec(
                name="train_path",
                filename="ctm.pkl",
                payload=train_result.train_doc_topic,
                split="train",
            ),
            PickleArtifactSpec(
                name="topic_preparation",
                filename="tp.pkl",
                payload=train_result.topic_preparation,
                split="train",
            ),
            PickleArtifactSpec(
                name="infer_path",
                filename=f"{category}.pkl",
                payload=infer_result.test_doc_topic,
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
            "topic_preparation": saved["topic_preparation"],
            "model_dir": train_result.model_dir,
            "train_preprocessed": saved["train_preprocessed"],
            "infer_preprocessed": saved["infer_preprocessed"],
            "train_preprocessing_selection": selection_saved[
                "train_preprocessing_selection"
            ],
            "infer_preprocessing_selection": selection_saved[
                "infer_preprocessing_selection"
            ],
        },
    )

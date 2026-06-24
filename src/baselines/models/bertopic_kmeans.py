from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib_cache"))

import numpy as np
import pandas as pd

from src.baselines.contracts import BaselineArtifacts
from src.baselines.params import BertopicKMeansParams
from src.core.artifacts import (
    PREPROCESSING_SELECTION_FILENAME,
    PickleArtifactSpec,
    save_pickle,
    save_split_jsons,
    save_split_pickles,
)
from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    preprocess_document,
    select_modelable_documents,
)
from src.utils.encoder import SentenceEncoder
from src.utils.encoder_inputs import encode_documents, fit_encoder_on_documents

TopicWords = list[list[tuple[str, float]]]


@dataclass(frozen=True)
class BertopicKMeansTrainResult:
    model: object
    umap_model: object
    kmeans_model: object
    train_doc_topic: np.ndarray
    test_doc_topic: np.ndarray
    train_preprocessed: list[PreprocessedDocument]
    test_preprocessed: list[PreprocessedDocument]
    topic_words: TopicWords
    topic_ids: list[int]
    effective_random_state: int
    train_selection: SelectedCorpus | None = None
    test_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class BertopicKMeansInferResult:
    test_doc_topic: np.ndarray
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None


def _load_preprocessed_documents_preserve_rows(
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
    ja_dicdir: str | None,
    ja_require_unidic: bool,
) -> list[PreprocessedDocument]:
    documents: list[PreprocessedDocument] = []
    allowed = set(targets) if targets is not None else None
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        if text_column not in frame.columns:
            raise ValueError(f"text_column '{text_column}' not found in CSV {csv_path}")
        if allowed is not None:
            if target_column is None:
                raise ValueError("target filtering requires target_column.")
            if target_column not in frame.columns:
                raise ValueError(
                    f"target_column '{target_column}' not found in CSV {csv_path}"
                )
            frame = frame.loc[frame[target_column].isin(allowed)]

        for row_index, value in frame[text_column].items():
            text = "" if pd.isna(value) else str(value)
            if not text.strip():
                raise ValueError(
                    "bertopic_kmeans requires non-empty document text; "
                    f"csv={csv_path}, row={row_index}, text_column={text_column}"
                )
            documents.append(
                preprocess_document(
                    text,
                    language=language,
                    delimiter=delimiter,
                    segmenter=segmenter,
                    tokenizer=tokenizer,
                    ja_replace_num=ja_replace_num,
                    ja_stopwords=None,
                    ja_dicdir=ja_dicdir,
                    ja_require_unidic=ja_require_unidic,
                )
            )
    return documents


def _document_text(document: PreprocessedDocument) -> str:
    candidates = [
        document.contextual_text,
        " ".join(sentence for sentence in document.sentences_raw if sentence).strip(),
        document.raw_text.strip(),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    raise ValueError("bertopic_kmeans encountered a document with no usable text.")


def _document_texts(documents: Sequence[PreprocessedDocument]) -> list[str]:
    texts = [_document_text(document) for document in documents]
    if not texts:
        raise ValueError("bertopic_kmeans requires at least one document.")
    return texts


def _softmax_negative_distances(
    embeddings: np.ndarray,
    centers: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("soft_temperature must be > 0.")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if embeddings.ndim != 2 or centers.ndim != 2:
        raise ValueError(
            "Expected 2D embeddings and centers, got "
            f"{embeddings.shape} and {centers.shape}."
        )
    if embeddings.shape[1] != centers.shape[1]:
        raise ValueError(
            "Embedding dimension does not match k-means center dimension: "
            f"{embeddings.shape[1]} != {centers.shape[1]}."
        )
    distances = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=2)
    scores = -distances / float(temperature)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    row_sums = exp_scores.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return exp_scores / row_sums


def _validate_doc_topic(
    doc_topic: np.ndarray,
    *,
    num_docs: int,
    num_topics: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(doc_topic, dtype=np.float64)
    if arr.shape != (num_docs, num_topics):
        raise ValueError(
            f"{name} doc-topic shape {arr.shape} != ({num_docs}, {num_topics})."
        )
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"{name} doc-topic rows do not sum to 1.")
    return arr


def _bertopic_topic_id_by_kmeans_label(
    topic_model: object,
    *,
    num_topics: int,
) -> dict[int, int]:
    mapper = getattr(topic_model, "topic_mapper_", None)
    if mapper is not None and hasattr(mapper, "get_mappings"):
        mapping = mapper.get_mappings(original_topics=True)
        label_to_topic = {
            int(label): int(topic_id)
            for label, topic_id in dict(mapping).items()
            if int(label) >= 0 and int(topic_id) >= 0
        }
    else:
        label_to_topic = {topic_id: topic_id for topic_id in range(num_topics)}

    expected_labels = set(range(num_topics))
    if set(label_to_topic) != expected_labels:
        raise ValueError(
            "BERTopic topic mapper did not contain a complete k-means label mapping: "
            f"labels={sorted(label_to_topic)}, expected={sorted(expected_labels)}."
        )
    mapped_topic_ids = [label_to_topic[label] for label in range(num_topics)]
    if len(set(mapped_topic_ids)) != num_topics:
        raise ValueError(
            "BERTopic topic mapper merged or duplicated k-means labels; "
            "cannot align topic words to doc-topic columns."
        )
    return label_to_topic


def _extract_topic_words(topic_model: object, *, num_topics: int) -> TopicWords:
    label_to_topic = _bertopic_topic_id_by_kmeans_label(
        topic_model,
        num_topics=num_topics,
    )
    topic_words: TopicWords = []
    get_topic = getattr(topic_model, "get_topic")
    for kmeans_label in range(num_topics):
        topic_id = label_to_topic[kmeans_label]
        topic = get_topic(topic_id)
        if not topic:
            raise ValueError(
                "BERTopic did not expose c-TF-IDF words for k-means label "
                f"{kmeans_label}. Cannot persist aligned topic words."
            )
        topic_words.append([(str(word), float(score)) for word, score in topic])
    return topic_words


def _fit_bertopic_kmeans(
    *,
    documents: list[str],
    embeddings: np.ndarray,
    num_topics: int,
    params: BertopicKMeansParams,
    effective_random_state: int,
) -> object:
    from bertopic import BERTopic
    from sklearn.cluster import KMeans
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=params.umap_n_neighbors,
        n_components=params.umap_n_components,
        min_dist=params.umap_min_dist,
        metric=params.umap_metric,
        random_state=effective_random_state,
    )
    cluster_model = KMeans(
        n_clusters=num_topics,
        n_init=params.kmeans_n_init,
        random_state=effective_random_state,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=False,
        verbose=params.verbose,
    )
    topic_model.fit_transform(documents, embeddings=embeddings)
    return topic_model


def train_bertopic_kmeans(
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
    effective_random_state: int,
    params: BertopicKMeansParams,
    train_dir: Path,
    use_legacy: bool,
) -> BertopicKMeansTrainResult:
    _ = (train_dir, ja_stopwords_path)
    resolved_delimiter = " / " if use_legacy else delimiter
    resolved_segmenter = "delimiter" if use_legacy else segmenter

    train_preprocessed = _load_preprocessed_documents_preserve_rows(
        csv_paths=train_csvs,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
        delimiter=resolved_delimiter,
        language=language,
        segmenter=resolved_segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )
    test_preprocessed = _load_preprocessed_documents_preserve_rows(
        csv_paths=test_csvs,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
        delimiter=resolved_delimiter,
        language=language,
        segmenter=resolved_segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )
    train_selection = select_modelable_documents(train_preprocessed)
    test_selection = select_modelable_documents(test_preprocessed)
    train_preprocessed = train_selection.documents
    test_preprocessed = test_selection.documents
    train_texts = _document_texts(train_preprocessed)
    test_texts = _document_texts(test_preprocessed)
    if len(train_texts) < num_topics:
        raise ValueError(
            "bertopic_kmeans requires at least num_topics training documents: "
            f"docs={len(train_texts)}, num_topics={num_topics}."
        )

    encoder = SentenceEncoder(
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
    )
    fit_encoder_on_documents(encoder, train_preprocessed)
    if getattr(encoder, "accepts_tokenized", False):
        train_embeddings = encode_documents(
            encoder,
            train_preprocessed,
            show_progress_bar=params.verbose,
        )
        test_embeddings = encode_documents(
            encoder,
            test_preprocessed,
            show_progress_bar=params.verbose,
        )
    else:
        train_embeddings = encoder.encode(train_texts, show_progress_bar=params.verbose)
        test_embeddings = encoder.encode(test_texts, show_progress_bar=params.verbose)

    topic_model = _fit_bertopic_kmeans(
        documents=train_texts,
        embeddings=train_embeddings,
        num_topics=num_topics,
        params=params,
        effective_random_state=effective_random_state,
    )
    umap_model = getattr(topic_model, "umap_model")
    kmeans_model = getattr(topic_model, "hdbscan_model")

    train_umap = getattr(umap_model, "embedding_", None)
    if train_umap is None or np.asarray(train_umap).shape[0] != len(train_texts):
        train_umap = umap_model.transform(train_embeddings)
    test_umap = umap_model.transform(test_embeddings)
    centers = np.asarray(getattr(kmeans_model, "cluster_centers_"), dtype=np.float64)
    if centers.shape[0] != num_topics:
        raise ValueError(
            f"KMeans produced {centers.shape[0]} centers, expected {num_topics}."
        )

    train_doc_topic = _validate_doc_topic(
        _softmax_negative_distances(
            np.asarray(train_umap),
            centers,
            temperature=params.soft_temperature,
        ),
        num_docs=len(train_texts),
        num_topics=num_topics,
        name="train",
    )
    test_doc_topic = _validate_doc_topic(
        _softmax_negative_distances(
            np.asarray(test_umap),
            centers,
            temperature=params.soft_temperature,
        ),
        num_docs=len(test_texts),
        num_topics=num_topics,
        name="test",
    )
    topic_words = _extract_topic_words(topic_model, num_topics=num_topics)

    return BertopicKMeansTrainResult(
        model=topic_model,
        umap_model=umap_model,
        kmeans_model=kmeans_model,
        train_doc_topic=train_doc_topic,
        test_doc_topic=test_doc_topic,
        train_preprocessed=train_preprocessed,
        test_preprocessed=test_preprocessed,
        topic_words=topic_words,
        topic_ids=list(range(num_topics)),
        effective_random_state=int(effective_random_state),
        train_selection=train_selection,
        test_selection=test_selection,
    )


def infer_bertopic_kmeans(
    *, train_result: BertopicKMeansTrainResult
) -> BertopicKMeansInferResult:
    return BertopicKMeansInferResult(
        test_doc_topic=np.asarray(train_result.test_doc_topic, dtype=float),
        test_preprocessed=list(train_result.test_preprocessed),
        test_selection=train_result.test_selection,
    )


def persist_bertopic_kmeans_run(
    *,
    train_result: BertopicKMeansTrainResult,
    infer_result: BertopicKMeansInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    model_path = train_dir / "bertopic_model.pkl"
    umap_path = train_dir / "umap.pkl"
    kmeans_path = train_dir / "kmeans.pkl"
    save_pickle(train_result.model, model_path)
    save_pickle(train_result.umap_model, umap_path)
    save_pickle(train_result.kmeans_model, kmeans_path)

    saved = save_split_pickles(
        [
            PickleArtifactSpec(
                name="train_path",
                filename="bertopic_kmeans.pkl",
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
                name="topic_words",
                filename="topic_words.pkl",
                payload=train_result.topic_words,
                split="train",
            ),
            PickleArtifactSpec(
                name="topic_ids",
                filename="topic_ids.pkl",
                payload=train_result.topic_ids,
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
    return BaselineArtifacts(
        train_path=saved["train_path"],
        infer_path=saved["infer_path"],
        extras={
            "model": model_path,
            "umap_model": umap_path,
            "kmeans_model": kmeans_path,
            "topic_words": saved["topic_words"],
            "topic_ids": saved["topic_ids"],
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

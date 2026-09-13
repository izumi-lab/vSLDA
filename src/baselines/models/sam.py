"""Runner-contract layer for the Spherical Admixture Model baseline.

Wires :mod:`src.baselines.models.sam_features` (documents -> unit-sphere
tf-idf), :mod:`src.baselines.models.sam_inference` (variational EM) and the
repository's artifact contracts together.

SAM is a *document-level* baseline: it produces one topic proportion vector per
document and a signed topic-by-word matrix, and has no sentence-level output.
It therefore sits alongside ``bleilda`` and ``etm`` rather than the
sentence-embedding models.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from src.baselines.artifact_alignment import (
    require_selected_corpus,
    validate_no_additional_document_drop,
    validate_selected_artifact_alignment,
)
from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import (
    load_preprocessed_documents,
    load_preprocessed_documents_with_indices,
)
from src.baselines.models.sam_features import (
    SamVocabulary,
    build_inference_corpus,
    build_training_corpus,
)
from src.baselines.models.sam_inference import (
    SamFitResult,
    document_topic_proportions,
    fit_sam,
    infer_alphatilde,
)
from src.baselines.params import SamParams
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

MODEL_NAME = "sam"
# Persisted orientation of ``topic_word_scores.pkl``.  SAM's natural internal
# layout is (V, T); every evaluation consumer in this repository expects
# (topics, vocabulary), so the transpose happens exactly once, here.
TOPIC_WORD_ORIENTATION = "topic_by_word"
TOP_TERM_COUNT = 20


@dataclass(frozen=True)
class SamTrainResult:
    fit: SamFitResult
    params: SamParams
    vocabulary: SamVocabulary
    train_doc_topic: np.ndarray
    topic_word_scores: np.ndarray  # (T, V), signed, unit rows
    train_preprocessed: list[PreprocessedDocument]
    train_selection: SelectedCorpus
    num_topics: int


@dataclass(frozen=True)
class SamInferResult:
    test_doc_topic: np.ndarray
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus


def _prepare_documents(
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
) -> SelectedCorpus:
    """Load and select documents, preserving source CSV row indices.

    Mirrors ``bleilda._prepare_documents`` including the ``__module__`` guard,
    which lets tests monkeypatch ``load_preprocessed_documents`` while the real
    pipeline keeps the index-preserving loader.
    """

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
    return select_modelable_documents(documents, raw_doc_indices=raw_indices)


def train_sam(
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
    use_legacy: bool,
    params: SamParams,
    train_dir: Path,
    effective_random_state: int,
) -> SamTrainResult:
    _ = train_dir
    selection = _prepare_documents(
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
    vocabulary, corpus = build_training_corpus(selection=selection, params=params)
    # Record the seed that actually drove training so that inference reuses it
    # and params.json reports it instead of the unresolved ``SamParams`` value.
    params = replace(params, random_state=int(effective_random_state))
    result = fit_sam(
        documents=corpus.matrix,
        num_topics=int(num_topics),
        params=params,
        random_state=int(effective_random_state),
    )
    return SamTrainResult(
        fit=result,
        params=params,
        vocabulary=vocabulary,
        train_doc_topic=document_topic_proportions(result.state.alphatilde),
        topic_word_scores=np.ascontiguousarray(result.state.mutilde.T),
        train_preprocessed=list(corpus.documents),
        train_selection=corpus.selection,
        num_topics=int(num_topics),
    )


def infer_sam(
    *,
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
    use_legacy: bool,
    params: SamParams,
    train_result: SamTrainResult,
) -> SamInferResult:
    _ = num_topics
    selection = _prepare_documents(
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
    corpus = build_inference_corpus(
        selection=selection,
        vocabulary=train_result.vocabulary,
        empty_error_message="No SAM-modelable documents remain in the inference split.",
    )
    alphatilde = infer_alphatilde(
        documents=corpus.matrix,
        state=train_result.fit.state,
        hyper=train_result.fit.hyper,
        params=params,
        random_state=train_result.params.random_state,
    )
    return SamInferResult(
        test_doc_topic=document_topic_proportions(alphatilde),
        test_preprocessed=list(corpus.documents),
        test_selection=corpus.selection,
    )


def _top_term_weights(
    topic_word_scores: np.ndarray, words: Sequence[str], *, count: int = TOP_TERM_COUNT
) -> dict[str, list[list[list[object]]]]:
    """Most positive and most negative terms per topic.

    Reproduces the paper's Table 1.  Nothing in the metric pipeline reads this;
    it exists because the negative weights are SAM's headline claim and the
    positive-part coherence protocol cannot see them.
    """

    positive: list[list[list[object]]] = []
    negative: list[list[list[object]]] = []
    limit = min(int(count), len(words))
    for row in np.asarray(topic_word_scores, dtype=np.float64):
        order = np.argsort(-row, kind="stable")
        positive.append([[str(words[i]), float(row[i])] for i in order[:limit]])
        negative.append([[str(words[i]), float(row[i])] for i in order[::-1][:limit]])
    return {"positive": positive, "negative": negative}


def persist_sam_run(
    *,
    train_result: SamTrainResult,
    infer_result: SamInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    train_selection = require_selected_corpus(
        train_result.train_selection, model_name=MODEL_NAME, split="train"
    )
    test_selection = require_selected_corpus(
        infer_result.test_selection, model_name=MODEL_NAME, split="infer"
    )
    validate_selected_artifact_alignment(
        model_name=MODEL_NAME,
        split="train",
        doc_topic=train_result.train_doc_topic,
        preprocessed=train_result.train_preprocessed,
        selection=train_selection,
    )
    validate_selected_artifact_alignment(
        model_name=MODEL_NAME,
        split="infer",
        doc_topic=infer_result.test_doc_topic,
        preprocessed=infer_result.test_preprocessed,
        selection=test_selection,
    )
    validate_no_additional_document_drop(
        model_name=MODEL_NAME,
        split="train",
        selected_count=len(train_selection.documents),
        model_input_count=int(train_result.train_doc_topic.shape[0]),
    )
    validate_no_additional_document_drop(
        model_name=MODEL_NAME,
        split="infer",
        selected_count=len(test_selection.documents),
        model_input_count=int(infer_result.test_doc_topic.shape[0]),
    )

    fit = train_result.fit
    saved = save_split_pickles(
        [
            PickleArtifactSpec(
                name="train_path",
                filename="sam.pkl",
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
                name="test_doc_topic_soft",
                filename=f"{category}_doc_topic_soft.pkl",
                payload=infer_result.test_doc_topic,
                split="infer",
            ),
            PickleArtifactSpec(
                name="topic_word_scores",
                filename="topic_word_scores.pkl",
                payload=train_result.topic_word_scores,
                split="train",
            ),
            PickleArtifactSpec(
                name="alphatilde",
                filename="alphatilde.pkl",
                payload=fit.state.alphatilde,
                split="train",
            ),
            PickleArtifactSpec(
                name="corpus_mean",
                filename="corpus_mean.pkl",
                payload=fit.state.mtilde,
                split="train",
            ),
            PickleArtifactSpec(
                name="idf",
                filename="idf.pkl",
                payload=train_result.vocabulary.idf,
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

    scores = np.asarray(train_result.topic_word_scores, dtype=np.float64)
    positive_mass = np.abs(np.clip(scores, 0.0, None)).sum(axis=1)
    total_mass = np.abs(scores).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        positive_fraction = np.where(total_mass > 0.0, positive_mass / total_mass, 0.0)

    json_saved = save_split_jsons(
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
            "vocabulary": (
                list(train_result.vocabulary.words),
                "vocabulary.json",
                "train",
            ),
            "params_json": (
                {
                    "baseline_params": {
                        key: value
                        for key, value in train_result.params.__dict__.items()
                    },
                    "num_topics": int(train_result.num_topics),
                    "vocab_size": int(train_result.vocabulary.size),
                    "num_train_documents": int(train_result.train_doc_topic.shape[0]),
                    "hyperparameters": {
                        "xi": float(fit.hyper.xi),
                        "kappa": float(fit.hyper.kappa),
                        "kappa0": float(fit.hyper.kappa0),
                        "alpha": [float(value) for value in fit.hyper.alpha],
                    },
                    "elbo_trace": [float(value) for value in fit.elbo_trace],
                    "converged": bool(fit.converged),
                    "iterations": int(fit.iterations),
                    "feature_scheme": str(train_result.vocabulary.feature_scheme),
                    "topic_word_orientation": TOPIC_WORD_ORIENTATION,
                    "positive_mass_fraction_by_topic": [
                        float(value) for value in positive_fraction
                    ],
                    "random_state": train_result.params.random_state,
                    **fit.diagnostics,
                },
                "params.json",
                "train",
            ),
            "topic_term_weights": (
                _top_term_weights(
                    train_result.topic_word_scores, train_result.vocabulary.words
                ),
                "topic_term_weights.json",
                "train",
            ),
        },
        train_dir=train_dir,
        infer_dir=infer_dir,
    )

    return BaselineArtifacts(
        train_path=saved["train_path"],
        infer_path=saved["infer_path"],
        extras={
            "test_doc_topic_soft": saved["test_doc_topic_soft"],
            "topic_word_scores": saved["topic_word_scores"],
            "alphatilde": saved["alphatilde"],
            "corpus_mean": saved["corpus_mean"],
            "idf": saved["idf"],
            "train_preprocessed": saved["train_preprocessed"],
            "infer_preprocessed": saved["infer_preprocessed"],
            "train_preprocessing_selection": json_saved[
                "train_preprocessing_selection"
            ],
            "infer_preprocessing_selection": json_saved[
                "infer_preprocessing_selection"
            ],
            "vocabulary": json_saved["vocabulary"],
            "params_json": json_saved["params_json"],
            "topic_term_weights": json_saved["topic_term_weights"],
        },
    )

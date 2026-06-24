from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gensim
import numpy as np

from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import (
    load_preprocessed_documents,
    load_preprocessed_documents_with_indices,
)
from src.baselines.params import BleiLdaParams
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


@dataclass(frozen=True)
class BleiLdaTrainResult:
    model: gensim.models.ldamodel.LdaModel
    train_doc_topic: np.ndarray
    train_preprocessed: list[PreprocessedDocument]
    train_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class BleiLdaInferResult:
    test_doc_topic: np.ndarray
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None


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
) -> tuple[list[list[str]], list[PreprocessedDocument], SelectedCorpus]:
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
    selection = select_modelable_documents(
        documents,
        raw_doc_indices=raw_indices,
    )
    return (
        [doc.document_tokens for doc in selection.documents],
        selection.documents,
        selection,
    )


def _topic_distribution_from_model(
    model: gensim.models.ldamodel.LdaModel,
    corpus: Sequence[Sequence[tuple[int, int]]],
    *,
    num_topics: int,
) -> np.ndarray:
    output = model.get_document_topics(corpus)
    result = np.zeros((len(corpus), num_topics))
    for row_index, topics in enumerate(output):
        for topic_index, value in topics:
            result[row_index][topic_index] = value
    return result


def train_bleilda(
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
    params: BleiLdaParams,
    train_dir: Path,
    use_legacy: bool,
) -> BleiLdaTrainResult:
    _ = train_dir
    data, train_preprocessed, train_selection = _prepare_documents(
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
    training_data = [tokens for tokens in data if tokens]
    if not training_data:
        raise ValueError("No training documents available for BleiLDA.")

    dictionary = gensim.corpora.Dictionary(training_data)
    if len(dictionary) == 0:
        raise ValueError("Tokenization produced an empty dictionary for BleiLDA.")
    corpus = [dictionary.doc2bow(text) for text in data]
    training_corpus = [bow for bow in corpus if bow]
    model = gensim.models.ldamodel.LdaModel(
        training_corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=params.passes,
        iterations=params.num_iterations,
    )
    return BleiLdaTrainResult(
        model=model,
        train_doc_topic=_topic_distribution_from_model(
            model,
            corpus,
            num_topics=num_topics,
        ),
        train_preprocessed=train_preprocessed,
        train_selection=train_selection,
    )


def infer_bleilda(
    *,
    train_result: BleiLdaTrainResult,
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
) -> BleiLdaInferResult:
    data, test_preprocessed, test_selection = _prepare_documents(
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
    dictionary = train_result.model.id2word
    corpus = [dictionary.doc2bow(text) for text in data]
    return BleiLdaInferResult(
        test_doc_topic=_topic_distribution_from_model(
            train_result.model,
            corpus,
            num_topics=num_topics,
        ),
        test_preprocessed=test_preprocessed,
        test_selection=test_selection,
    )


def persist_bleilda_run(
    *,
    train_result: BleiLdaTrainResult,
    infer_result: BleiLdaInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    model_path = train_dir / "model.gensim"
    train_result.model.save(model_path.as_posix())
    saved = save_split_pickles(
        [
            PickleArtifactSpec(
                name="train_path",
                filename="lda_comp.pkl",
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
            "model_path": model_path,
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

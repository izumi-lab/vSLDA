from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.baselines.dataset_adapters import load_preprocessed_documents_with_indices
from src.baselines.models.ctm import CtmInferResult, CtmTrainResult, persist_ctm_run
from src.baselines.models.gaussian_state import GaussianTrainerState
from src.baselines.models.sentence_gaussianlda import (
    SentenceGaussianLdaInferResult,
    SentenceGaussianLdaTrainResult,
    persist_sentence_gaussianlda_run,
)
from src.core.artifacts import load_json
from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    select_modelable_documents,
)


def _doc(text: str) -> PreprocessedDocument:
    tokens = text.lower().split()
    return PreprocessedDocument(
        raw_text=text,
        sentences_raw=[text],
        sentences_tokenized=[tokens],
        sentences_joined=[" ".join(tokens)],
        document_tokens=tokens,
    )


def _selection(
    docs: list[PreprocessedDocument],
    raw_doc_indices: list[int],
) -> SelectedCorpus:
    return SelectedCorpus(
        documents=docs,
        raw_doc_indices=raw_doc_indices,
        sentence_indices_by_doc=[[0] for _doc in docs],
        dropped_doc_indices=[],
        drop_reasons={},
    )


def test_filtered_preprocessed_loader_preserves_source_row_indices(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "data": [
                "skip this row",
                "alpha beta",
                "other target",
                "gamma delta",
            ],
            "target_str": ["B", "A", "B", "A"],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    documents, raw_indices = load_preprocessed_documents_with_indices(
        csv_paths=[str(csv_path)],
        text_column="data",
        target_column="target_str",
        targets=["A"],
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
    )
    selection = select_modelable_documents(documents, raw_doc_indices=raw_indices)

    assert selection.raw_doc_indices == [1, 3]
    assert selection.raw_doc_indices != list(range(len(selection.raw_doc_indices)))


class _FakeCtmModel:
    def save(self, *, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)


def test_persist_ctm_run_saves_source_raw_doc_indices(tmp_path: Path) -> None:
    train_docs = [_doc("alpha beta"), _doc("gamma delta")]
    test_docs = [_doc("epsilon zeta")]
    train_selection = _selection(train_docs, [4, 9])
    test_selection = _selection(test_docs, [7])

    artifacts = persist_ctm_run(
        train_result=CtmTrainResult(
            model=_FakeCtmModel(),
            topic_preparation=SimpleNamespace(vocab=["alpha", "beta"]),
            encoder=None,
            train_doc_topic=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            model_dir=tmp_path / "train" / "model",
            train_preprocessed=train_docs,
            train_selection=train_selection,
        ),
        infer_result=CtmInferResult(
            test_doc_topic=np.asarray([[0.4, 0.6]]),
            test_preprocessed=test_docs,
            test_selection=test_selection,
        ),
        train_dir=tmp_path / "train",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    train_payload = load_json(artifacts.extras["train_preprocessing_selection"])
    infer_payload = load_json(artifacts.extras["infer_preprocessing_selection"])
    assert train_payload["raw_doc_indices"] == [4, 9]
    assert infer_payload["raw_doc_indices"] == [7]


def _trainer_state(num_topics: int, train_docs: int) -> GaussianTrainerState:
    return GaussianTrainerState(
        average_ll=(0.0,),
        alpha=0.1,
        num_tables=num_topics,
        prior_kappa=0.1,
        table_counts=np.ones(num_topics),
        table_means=np.ones((num_topics, 2)),
        table_inverse_covariances=np.stack(
            [np.eye(2, dtype=float) for _index in range(num_topics)]
        ),
        log_determinants=np.zeros(num_topics),
        sum_table_customers=np.ones((num_topics, 2)),
        sum_squared_table_customers=np.ones((num_topics, 2)),
        table_cholesky_ltriangular_mat=np.stack(
            [np.eye(2, dtype=float) for _index in range(num_topics)]
        ),
        table_counts_per_doc=np.ones((num_topics, train_docs)),
        prior_mu=np.zeros(2),
    )


def test_persist_sentence_gaussianlda_requires_selection(tmp_path: Path) -> None:
    train_docs = [_doc("alpha beta")]
    test_docs = [_doc("gamma delta")]

    with pytest.raises(ValueError, match="preprocessing selection is required"):
        persist_sentence_gaussianlda_run(
            train_result=SentenceGaussianLdaTrainResult(
                trainer_state=_trainer_state(num_topics=2, train_docs=1),
                model=object(),
                train_doc_topic=np.asarray([[1.0, 0.0]]),
                train_sentence_topic_soft=[np.asarray([[1.0, 0.0]])],
                train_preprocessed=train_docs,
                train_selection=None,
            ),
            infer_result=SentenceGaussianLdaInferResult(
                test_doc_topic=np.asarray([[0.0, 1.0]]),
                test_sentence_topic_soft=[np.asarray([[0.0, 1.0]])],
                test_preprocessed=test_docs,
                test_selection=_selection(test_docs, [8]),
            ),
            train_dir=tmp_path / "train",
            infer_dir=tmp_path / "infer",
            category="all",
        )


def test_persist_senclu_run_saves_source_raw_doc_indices(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pysbd")
    from src.baselines.models.senclu import (
        SenCluInferResult,
        SenCluTrainResult,
        persist_senclu_run,
    )

    train_docs = [_doc("alpha beta")]
    test_docs = [_doc("gamma delta")]
    train_selection = _selection(train_docs, [11])
    test_selection = _selection(test_docs, [14])

    artifacts = persist_senclu_run(
        train_result=SenCluTrainResult(
            train_doc_topic=np.asarray([[1.0, 0.0]]),
            test_doc_topic=np.asarray([[0.0, 1.0]]),
            train_sentence_topic_soft=[np.asarray([[1.0, 0.0]])],
            test_sentence_topic_soft=[np.asarray([[0.0, 1.0]])],
            train_preprocessed=train_docs,
            test_preprocessed=test_docs,
            train_selection=train_selection,
            test_selection=test_selection,
        ),
        infer_result=SenCluInferResult(
            test_doc_topic=np.asarray([[0.0, 1.0]]),
            test_sentence_topic_soft=[np.asarray([[0.0, 1.0]])],
            test_preprocessed=test_docs,
            test_selection=test_selection,
        ),
        train_dir=tmp_path / "train",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    train_payload = load_json(artifacts.extras["train_preprocessing_selection"])
    infer_payload = load_json(artifacts.extras["infer_preprocessing_selection"])
    assert train_payload["raw_doc_indices"] == [11]
    assert infer_payload["raw_doc_indices"] == [14]

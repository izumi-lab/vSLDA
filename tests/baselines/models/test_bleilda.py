from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.baselines.models import bleilda
from src.baselines.models.bleilda import (
    infer_bleilda,
    persist_bleilda_run,
    train_bleilda,
)
from src.baselines.params import BleiLdaParams, parse_bleilda_params
from src.core.artifacts import load_pickle
from src.data.preprocessing import PreprocessedDocument


def test_parse_bleilda_params_defaults_inner_iterations() -> None:
    params = parse_bleilda_params({})

    assert params.passes == 20
    assert params.num_iterations == 50


def test_train_bleilda_passes_params_to_gensim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    documents = [
        PreprocessedDocument(
            raw_text="alpha beta",
            sentences_raw=["alpha beta"],
            sentences_tokenized=[["alpha", "beta"]],
            sentences_joined=["alpha beta"],
            document_tokens=["alpha", "beta"],
        ),
        PreprocessedDocument(
            raw_text="beta gamma",
            sentences_raw=["beta gamma"],
            sentences_tokenized=[["beta", "gamma"]],
            sentences_joined=["beta gamma"],
            document_tokens=["beta", "gamma"],
        ),
    ]
    monkeypatch.setattr(
        bleilda,
        "load_preprocessed_documents",
        lambda **_kwargs: documents,
    )

    class _FakeLdaModel:
        def __init__(
            self,
            corpus,
            *,
            num_topics: int,
            id2word,
            passes: int,
            iterations: int,
        ) -> None:
            captured["corpus"] = list(corpus)
            captured["num_topics"] = num_topics
            captured["id2word"] = id2word
            captured["passes"] = passes
            captured["iterations"] = iterations

        def get_document_topics(self, corpus):
            return [[(0, 1.0)] for _ in corpus]

    monkeypatch.setattr(
        bleilda.gensim.models.ldamodel,
        "LdaModel",
        _FakeLdaModel,
    )

    result = train_bleilda(
        train_csvs=["train.csv"],
        targets=None,
        text_column="data",
        target_column=None,
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=3,
        params=BleiLdaParams(passes=19, num_iterations=17),
        train_dir=tmp_path,
        use_legacy=False,
    )

    assert captured["passes"] == 19
    assert captured["iterations"] == 17
    assert result.train_doc_topic.shape == (2, 3)


def test_train_bleilda_drops_empty_token_document_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    documents = [
        PreprocessedDocument(
            raw_text="alpha beta",
            sentences_raw=["alpha beta"],
            sentences_tokenized=[["alpha", "beta"]],
            sentences_joined=["alpha beta"],
            document_tokens=["alpha", "beta"],
        ),
        PreprocessedDocument(
            raw_text="--",
            sentences_raw=["--"],
            sentences_tokenized=[[]],
            sentences_joined=[""],
            document_tokens=[],
        ),
        PreprocessedDocument(
            raw_text="beta gamma",
            sentences_raw=["beta gamma"],
            sentences_tokenized=[["beta", "gamma"]],
            sentences_joined=["beta gamma"],
            document_tokens=["beta", "gamma"],
        ),
    ]
    monkeypatch.setattr(
        bleilda,
        "load_preprocessed_documents",
        lambda **_kwargs: documents,
    )

    class _FakeLdaModel:
        def __init__(
            self,
            corpus,
            *,
            num_topics: int,
            id2word,
            passes: int,
            iterations: int,
        ) -> None:
            captured["training_corpus"] = list(corpus)
            self.num_topics = num_topics

        def get_document_topics(self, corpus):
            captured["doc_topic_corpus"] = list(corpus)
            return [[(0, 1.0)] if row else [] for row in corpus]

    monkeypatch.setattr(
        bleilda.gensim.models.ldamodel,
        "LdaModel",
        _FakeLdaModel,
    )

    result = train_bleilda(
        train_csvs=["train.csv"],
        targets=None,
        text_column="data",
        target_column=None,
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=3,
        params=BleiLdaParams(passes=19, num_iterations=17),
        train_dir=tmp_path,
        use_legacy=False,
    )

    assert len(captured["training_corpus"]) == 2
    assert len(captured["doc_topic_corpus"]) == 2
    assert result.train_doc_topic.shape == (2, 3)
    assert result.train_preprocessed == [documents[0], documents[2]]
    assert result.train_selection is not None
    assert result.train_selection.raw_doc_indices == [0, 2]
    assert result.train_selection.dropped_doc_indices == [1]


@pytest.mark.slow
def test_train_infer_and_persist_bleilda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_docs = [
        PreprocessedDocument(
            raw_text="alpha beta gamma",
            sentences_raw=["alpha beta gamma"],
            sentences_tokenized=[["alpha", "beta", "gamma"]],
            sentences_joined=["alpha beta gamma"],
            document_tokens=["alpha", "beta", "gamma"],
        ),
        PreprocessedDocument(
            raw_text="alpha delta",
            sentences_raw=["alpha delta"],
            sentences_tokenized=[["alpha", "delta"]],
            sentences_joined=["alpha delta"],
            document_tokens=["alpha", "delta"],
        ),
        PreprocessedDocument(
            raw_text="beta gamma epsilon",
            sentences_raw=["beta gamma epsilon"],
            sentences_tokenized=[["beta", "gamma", "epsilon"]],
            sentences_joined=["beta gamma epsilon"],
            document_tokens=["beta", "gamma", "epsilon"],
        ),
    ]
    test_docs = [
        PreprocessedDocument(
            raw_text="alpha beta",
            sentences_raw=["alpha beta"],
            sentences_tokenized=[["alpha", "beta"]],
            sentences_joined=["alpha beta"],
            document_tokens=["alpha", "beta"],
        ),
        PreprocessedDocument(
            raw_text="gamma",
            sentences_raw=["gamma"],
            sentences_tokenized=[["gamma"]],
            sentences_joined=["gamma"],
            document_tokens=["gamma"],
        ),
    ]

    def _fake_loader(*, csv_paths, **_kwargs):
        return train_docs if str(csv_paths[0]) == "train.csv" else test_docs

    monkeypatch.setattr(bleilda, "load_preprocessed_documents", _fake_loader)

    train_result = train_bleilda(
        train_csvs=["train.csv"],
        targets=None,
        text_column="data",
        target_column=None,
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=2,
        params=BleiLdaParams(passes=2, num_iterations=5),
        train_dir=tmp_path / "train",
        use_legacy=False,
    )
    infer_result = infer_bleilda(
        train_result=train_result,
        test_csvs=["test.csv"],
        targets=None,
        text_column="data",
        target_column=None,
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=2,
        use_legacy=False,
    )
    artifacts = persist_bleilda_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "train",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    assert train_result.train_doc_topic.shape == (3, 2)
    assert infer_result.test_doc_topic.shape == (2, 2)
    assert np.allclose(train_result.train_doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(infer_result.test_doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert artifacts.train_path.name == "lda_comp.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert artifacts.extras["model_path"].name == "model.gensim"
    assert load_pickle(tmp_path / "train" / "lda_comp.pkl").shape == (3, 2)

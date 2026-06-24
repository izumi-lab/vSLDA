from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gensim.models import KeyedVectors

from src.baselines.models.gaussianlda import prepare_word_vector_corpus
from src.baselines.models.mvtm import (
    WordVectorEncoder,
    infer_mvtm,
    persist_mvtm_run,
    train_mvtm,
)
from src.baselines.params import MvTMParams, parse_mvtm_params
from src.core.artifacts import load_pickle
from src.data.preprocessing import PreprocessedDocument


def _vectors() -> KeyedVectors:
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        ["alpha", "beta", "gamma", "delta"],
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=np.float32,
        ),
    )
    return vectors


def _docs() -> list[PreprocessedDocument]:
    return [
        PreprocessedDocument(
            raw_text="alpha beta",
            sentences_raw=["alpha beta"],
            sentences_tokenized=[["alpha", "beta"]],
            sentences_joined=["alpha beta"],
            document_tokens=["alpha", "beta"],
        ),
        PreprocessedDocument(
            raw_text="gamma delta",
            sentences_raw=["gamma delta"],
            sentences_tokenized=[["gamma", "delta"]],
            sentences_joined=["gamma delta"],
            document_tokens=["gamma", "delta"],
        ),
    ]


def test_parse_mvtm_params_defaults_to_single_component() -> None:
    params = parse_mvtm_params({})

    assert params.word2vec == "glove-wiki-gigaword-100"
    assert params.num_iterations == 20
    assert params.num_components == 1
    assert params.alpha is None
    assert params.estimate_alpha is False


def test_parse_mvtm_params_validates_positive_values() -> None:
    with pytest.raises(ValueError, match="num_components"):
        parse_mvtm_params({"num_components": 0})


def test_word_vector_encoder_returns_vectors_for_known_tokens() -> None:
    encoder = WordVectorEncoder(_vectors())

    encoded = encoder.encode(["alpha", "missing", "beta"])

    assert encoded.shape == (2, 2)
    assert encoder.get_sentence_embedding_dimension() == 2
    assert np.allclose(encoded[0], np.asarray([1.0, 0.0]))


def test_prepare_word_vector_corpus_drops_oov_with_gaussian_shared_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = _docs()
    docs[0].document_tokens.append("missing")
    monkeypatch.setattr(
        "src.baselines.models.gaussianlda.load_preprocessed_documents",
        lambda **_kwargs: docs,
    )

    prepared = prepare_word_vector_corpus(
        csv_paths=["train.csv"],
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
        use_legacy=False,
        word2vec=_vectors(),
        wikientvec_cache_dir=None,
    )

    assert prepared.index_docs[0] == [0, 1]
    assert prepared.token_docs[0] == ["alpha", "beta"]
    assert prepared.vocab == ["alpha", "beta", "gamma", "delta"]


def test_train_infer_and_persist_mvtm_with_two_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np.random.seed(0)
    monkeypatch.setattr(
        "src.baselines.models.gaussianlda.load_preprocessed_documents",
        lambda **_kwargs: _docs(),
    )
    params = MvTMParams(
        word2vec=_vectors(),
        num_iterations=1,
        num_components=2,
        gibbs_sweeps=1,
        num_samples=1,
        estimate_alpha=False,
    )

    train_result = train_mvtm(
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
        params=params,
        train_dir=tmp_path / "params",
        use_legacy=False,
    )
    infer_result = infer_mvtm(
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
        params=params,
        use_legacy=False,
    )
    artifacts = persist_mvtm_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "params",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    assert train_result.resolved_alpha == pytest.approx(0.5)
    assert train_result.trainer.num_components == 2
    assert train_result.train_doc_topic.shape == (2, 2)
    assert train_result.train_doc_topic_soft.shape == (2, 2)
    assert infer_result.test_doc_topic.shape == (2, 2)
    assert infer_result.test_doc_topic_soft.shape == (2, 2)
    assert artifacts.train_path.name == "table_counts_per_doc.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert load_pickle(tmp_path / "params" / "mixture_weights.pkl").shape == (2, 2)
    assert load_pickle(tmp_path / "params" / "component_means.pkl").shape == (2, 2, 2)
    assert len(load_pickle(tmp_path / "params" / "topic_words.pkl")) == 2
    assert not (tmp_path / "params" / "sentence_topic_train_soft.pkl").exists()
    assert not (tmp_path / "infer" / "all_sentence_topic_soft.pkl").exists()

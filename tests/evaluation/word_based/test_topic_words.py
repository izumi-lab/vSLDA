from __future__ import annotations

import numpy as np
import pytest
from gensim.models import KeyedVectors

from src.core.artifacts import save_json, save_pickle
from src.evaluation.word_based.topic_words import (
    load_bertopic_kmeans_topic_words,
    load_etm_topic_words,
    load_mvtm_topic_words,
)


def test_load_bertopic_kmeans_topic_words_reads_persisted_artifact(
    monkeypatch,
    tmp_path,
) -> None:
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    save_pickle(
        [[("alpha", 0.9), ("beta", 0.5)], [("gamma", 0.8)]],
        param_dir / "topic_words.pkl",
    )

    monkeypatch.setattr(
        "src.evaluation.word_based.topic_words.build_baseline_param_dir",
        lambda **_kwargs: param_dir,
    )

    topic_words = load_bertopic_kmeans_topic_words(
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="all",
        topn=1,
    )

    assert topic_words == [[("alpha", 0.9)], [("gamma", 0.8)]]


def test_load_mvtm_topic_words_scores_saved_vmf_mixture(monkeypatch, tmp_path) -> None:
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        ["alpha", "beta", "gamma"],
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
    )
    vectors.save((param_dir / "local_word2vec.kv").as_posix())
    save_pickle(
        np.asarray([[1.0], [1.0]], dtype=np.float64),
        param_dir / "mixture_weights.pkl",
    )
    save_pickle(
        np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float64),
        param_dir / "component_means.pkl",
    )
    save_pickle(
        np.asarray([2.0, 2.0], dtype=np.float64), param_dir / "kappa_per_topic.pkl"
    )

    monkeypatch.setattr(
        "src.evaluation.word_based.topic_words.build_baseline_param_dir",
        lambda **_kwargs: param_dir,
    )

    topic_words = load_mvtm_topic_words(
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="all",
        topn=1,
        word2vec="glove-wiki-gigaword-100",
    )

    assert topic_words == [
        [("alpha", pytest.approx(2.0))],
        [("beta", pytest.approx(2.0))],
    ]


def test_load_etm_topic_words_reads_beta_and_vocabulary(monkeypatch, tmp_path) -> None:
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    save_pickle(
        np.asarray([[0.2, 0.8, 0.1], [0.7, 0.1, 0.2]], dtype=np.float64),
        param_dir / "topic_word_scores.pkl",
    )
    save_json(["alpha", "beta", "gamma"], param_dir / "vocabulary.json")
    monkeypatch.setattr(
        "src.evaluation.word_based.topic_words.build_baseline_param_dir",
        lambda **_kwargs: param_dir,
    )

    topic_words = load_etm_topic_words(
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="all",
        topn=2,
    )

    assert topic_words == [
        [("beta", pytest.approx(0.8)), ("alpha", pytest.approx(0.2))],
        [("alpha", pytest.approx(0.7)), ("gamma", pytest.approx(0.2))],
    ]

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gensim.models import KeyedVectors

pytestmark = pytest.mark.integration
pytest.importorskip("torch")

from src.baselines.models.etm import (  # noqa: E402
    infer_etm,
    persist_etm_run,
    train_etm,
)
from src.baselines.params import (  # noqa: E402
    EtmParams,
    baseline_params_to_variant,
    parse_etm_params,
)
from src.core.artifacts import load_artifact_json, load_pickle  # noqa: E402
from src.data.preprocessing import PreprocessedDocument  # noqa: E402


def _vectors() -> KeyedVectors:
    vectors = KeyedVectors(vector_size=3)
    vectors.add_vectors(
        ["alpha", "beta", "gamma", "delta"],
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    return vectors


def _doc(tokens: list[str]) -> PreprocessedDocument:
    raw = " ".join(tokens)
    return PreprocessedDocument(
        raw_text=raw,
        sentences_raw=[raw],
        sentences_tokenized=[tokens],
        sentences_joined=[raw],
        document_tokens=list(tokens),
    )


def test_parse_etm_params_defaults_and_validation() -> None:
    params = parse_etm_params({})

    assert params.word2vec == "glove-wiki-gigaword-100"
    assert params.num_epochs == 100
    assert params.batch_size == 128
    assert params.lr == pytest.approx(0.002)
    assert params.random_state is None
    assert params.reference_profile == "repo_default"
    assert "num_epochs=100" in baseline_params_to_variant(params)

    explicit_seed = parse_etm_params({"random_state": "11"})
    assert explicit_seed.random_state == 11

    with pytest.raises(ValueError, match="num_epochs"):
        parse_etm_params({"num_epochs": 0})
    with pytest.raises(ValueError, match="theta_act"):
        parse_etm_params({"theta_act": "invalid"})
    with pytest.raises(ValueError, match="optimizer"):
        parse_etm_params({"optimizer": "invalid"})


def test_train_infer_and_persist_etm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_docs = [
        _doc(["alpha", "beta", "alpha"]),
        _doc(["gamma", "delta"]),
        _doc(["missing"]),
    ]
    test_docs = [
        _doc(["alpha", "gamma"]),
        _doc(["delta", "missing"]),
    ]

    def _fake_loader(*, csv_paths, **_kwargs):
        return train_docs if str(csv_paths[0]) == "train.csv" else test_docs

    monkeypatch.setattr(
        "src.baselines.models.etm.load_preprocessed_documents",
        _fake_loader,
    )
    params = EtmParams(
        word2vec=_vectors(),
        num_epochs=2,
        batch_size=2,
        eval_batch_size=1,
        t_hidden_size=8,
        random_state=11,
    )

    train_result = train_etm(
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
        encoder_device="cpu",
        effective_random_state=7,
    )
    infer_result = infer_etm(
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
    artifacts = persist_etm_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "params",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    assert train_result.train_doc_topic.shape == (2, 2)
    assert infer_result.test_doc_topic.shape == (2, 2)
    assert train_result.topic_word_scores.shape == (2, 4)
    assert np.allclose(train_result.train_doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(infer_result.test_doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert [doc.document_tokens for doc in train_result.train_preprocessed] == [
        ["alpha", "beta", "alpha"],
        ["gamma", "delta"],
    ]
    assert artifacts.train_path.name == "etm.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert artifacts.extras["model_state"].name == "model_state.pt"
    assert artifacts.extras["test_doc_topic_soft"].name == "all_doc_topic_soft.pkl"
    assert load_artifact_json(tmp_path / "params" / "vocabulary.json") == [
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]
    assert load_pickle(tmp_path / "params" / "topic_word_scores.pkl").shape == (2, 4)

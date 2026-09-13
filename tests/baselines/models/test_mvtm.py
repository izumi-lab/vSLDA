from __future__ import annotations

import json
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
from src.baselines.params import (
    MvTMParams,
    baseline_params_to_variant,
    parse_mvtm_params,
)
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

    assert params.word2vec == "word2vec-google-news-300"
    assert params.num_iterations == 20
    assert params.num_components == 1
    assert params.alpha is None
    assert params.estimate_alpha is False
    assert params.max_kappa == 10_000.0


def test_parse_mvtm_params_validates_positive_values() -> None:
    with pytest.raises(ValueError, match="num_components"):
        parse_mvtm_params({"num_components": 0})
    with pytest.raises(ValueError, match="max_kappa"):
        parse_mvtm_params({"max_kappa": 0})
    with pytest.raises(ValueError, match="kappa_default must be <= max_kappa"):
        parse_mvtm_params({"kappa_default": 11, "max_kappa": 10})


def test_parse_mvtm_params_accepts_max_kappa_override() -> None:
    assert parse_mvtm_params({"max_kappa": "2500"}).max_kappa == 2500.0


def test_mvtm_parameter_identity_includes_max_kappa() -> None:
    default_variant = baseline_params_to_variant(parse_mvtm_params({}))
    bounded_variant = baseline_params_to_variant(parse_mvtm_params({"max_kappa": 2500}))

    assert "max_kappa=10000.0" in default_variant
    assert "max_kappa=2500.0" in bounded_variant
    assert default_variant != bounded_variant


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


def _train_and_infer_toy_mvtm(monkeypatch: pytest.MonkeyPatch):
    np.random.seed(0)
    monkeypatch.setattr(
        "src.baselines.models.gaussianlda.load_preprocessed_documents",
        lambda **_kwargs: _docs(),
    )
    params = MvTMParams(
        word2vec=_vectors(),
        num_iterations=1,
        num_components=1,
        gibbs_sweeps=1,
        num_samples=1,
        estimate_alpha=False,
    )
    common = dict(
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
    train_result = train_mvtm(
        train_csvs=["train.csv"], train_dir=Path("unused"), **common
    )
    infer_result = infer_mvtm(
        train_result=train_result, test_csvs=["test.csv"], **common
    )
    return train_result, infer_result


def test_persist_writes_the_fold_in_theta_the_posthoc_command_recognizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Training writes the token-unit fold-in of both splits under ``params/`` and
    ``infer/``; ``vmf-foldin-theta --model mvtm`` rebuilds the same fingerprint (and
    the same values) from the persisted files and therefore skips the run."""

    from src.core.artifacts import load_json
    from src.evaluation.foldin.artifacts import FoldInLayout, foldin_is_current
    from src.evaluation.foldin.mvtm import (
        compute_mvtm_foldin_for_run,
        normalized_word_rows,
    )

    train_result, infer_result = _train_and_infer_toy_mvtm(monkeypatch)
    # The rows the fold-in scores are the trainer's own observations.
    encoded = train_result.trainer.encoded_corpus[0]
    assert np.array_equal(
        normalized_word_rows(train_result.word_vectors, ["alpha", "beta"]),
        np.asarray(encoded, dtype=np.float64),
    )

    artifacts = persist_mvtm_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "params",
        infer_dir=tmp_path / "infer",
        category="all",
        condition_fingerprint="cond-fp",
        dataset="dummy",
        data_run="default",
    )
    expected_keys = {
        "train_doc_topic_foldin": tmp_path / "params" / "all_doc_topic_foldin.pkl",
        "train_doc_topic_foldin_counts": tmp_path
        / "params"
        / "all_doc_topic_foldin_counts.pkl",
        "test_doc_topic_foldin": tmp_path / "infer" / "all_doc_topic_foldin.pkl",
        "test_doc_topic_foldin_counts": tmp_path
        / "infer"
        / "all_doc_topic_foldin_counts.pkl",
    }
    for key, path in expected_keys.items():
        assert artifacts.extras[key] == path
        assert path.exists()
    meta = load_json(tmp_path / "foldin_meta.json")
    assert meta["model"] == "mvtm"
    assert set(meta["splits"]) == {"train", "test"}
    assert meta["splits"]["test"]["written_by"] == "training"
    assert meta["splits"]["test"]["assignment_unit_type"] == "token"
    theta_test = load_pickle(expected_keys["test_doc_topic_foldin"])
    counts_test = load_pickle(expected_keys["test_doc_topic_foldin_counts"])
    assert theta_test.shape == (2, 2)
    assert np.allclose(theta_test.sum(axis=1), 1.0)
    # Every in-vocabulary token is counted: two per document.
    assert np.allclose(counts_test.sum(axis=1), [2.0, 2.0])

    # The post-hoc command reads the persisted parameters and metadata.
    params_json = load_json(tmp_path / "params" / "params.json")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "condition_fingerprint": "cond-fp",
                "baseline_params": params_json["baseline_params"],
            }
        ),
        encoding="utf-8",
    )
    posthoc = compute_mvtm_foldin_for_run(
        tmp_path,
        split="test",
        dataset="dummy",
        data_run="default",
        category="all",
        vectors=train_result.word_vectors,
    )
    assert posthoc.fingerprint == meta["splits"]["test"]["fingerprint"]
    assert foldin_is_current(
        tmp_path,
        split="test",
        fingerprint=posthoc.fingerprint,
        layout=FoldInLayout.for_model("mvtm", category="all"),
    )
    assert np.array_equal(posthoc.theta, theta_test)
    assert np.array_equal(posthoc.expected_counts, counts_test)


def test_persist_without_foldin_writes_no_fold_in_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_result, infer_result = _train_and_infer_toy_mvtm(monkeypatch)
    artifacts = persist_mvtm_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "params",
        infer_dir=tmp_path / "infer",
        category="all",
        foldin=False,
    )
    assert not any(
        key.endswith("foldin") or key.endswith("foldin_counts")
        for key in artifacts.extras
    )
    assert not (tmp_path / "foldin_meta.json").exists()
    assert not (tmp_path / "infer" / "all_doc_topic_foldin.pkl").exists()

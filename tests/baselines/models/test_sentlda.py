from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.baselines.models.sentlda import (
    _build_vocabulary,
    _encode_corpus,
    _fit_sentlda,
    _infer_sentlda,
    infer_sentlda,
    persist_sentlda_run,
    train_sentlda,
)
from src.baselines.models.sentlda_numba import (
    SENTLDA_NUMERICS_BACKEND,
    resolve_sentlda_backend,
)
from src.baselines.params import SentLdaParams, parse_sentlda_params
from src.core.artifacts import load_artifact_json, load_artifact_pickle
from src.data.preprocessing import PreprocessedDocument


def _doc(*sentences: list[str]) -> PreprocessedDocument:
    raw_sentences = [" ".join(tokens) for tokens in sentences]
    return PreprocessedDocument(
        raw_text=" / ".join(raw_sentences),
        sentences_raw=raw_sentences,
        sentences_tokenized=[list(tokens) for tokens in sentences],
        sentences_joined=[" ".join(tokens) for tokens in sentences],
        document_tokens=[token for sentence in sentences for token in sentence],
    )


def _assert_grouped_arrays_equal(
    actual: list[np.ndarray],
    expected: list[np.ndarray],
) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        if np.issubdtype(actual_row.dtype, np.floating):
            assert np.allclose(actual_row, expected_row, atol=1e-6)
        else:
            assert np.array_equal(actual_row, expected_row)


def _softmax_rows(log_scores: np.ndarray) -> np.ndarray:
    shifted = log_scores - log_scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _assert_soft_matches_log_factors(
    soft: list[np.ndarray],
    loglik: list[np.ndarray],
    logprior: list[np.ndarray],
) -> None:
    assert len(soft) == len(loglik) == len(logprior)
    for soft_doc, loglik_doc, logprior_doc in zip(soft, loglik, logprior):
        assert soft_doc.shape == loglik_doc.shape == logprior_doc.shape
        expected = _softmax_rows(loglik_doc + logprior_doc)
        assert np.allclose(soft_doc, expected, atol=1e-5)


def test_encode_corpus_builds_flattened_sentence_views() -> None:
    documents = [
        _doc(["apple", "banana", "apple"], ["banana", "carrot"]),
        _doc(["durian", "apple"], ["carrot", "carrot"]),
    ]

    bundle = _encode_corpus(documents, _build_vocabulary(documents))

    assert bundle.num_docs == 2
    assert bundle.num_sentences == 4
    assert np.array_equal(bundle.doc_offsets, np.asarray([0, 2, 4], dtype=np.int32))
    assert np.array_equal(
        bundle.sentence_offsets,
        np.asarray([0, 3, 5, 7, 9], dtype=np.int32),
    )
    assert np.array_equal(
        bundle.sentence_word_ids_flat,
        np.asarray([0, 1, 0, 1, 2, 3, 0, 2, 2], dtype=np.int32),
    )
    assert np.array_equal(
        bundle.sentence_unique_offsets,
        np.asarray([0, 2, 4, 6, 7], dtype=np.int32),
    )
    assert np.array_equal(
        bundle.sentence_unique_word_ids_flat,
        np.asarray([0, 1, 1, 2, 0, 3, 2], dtype=np.int32),
    )
    assert np.array_equal(
        bundle.sentence_word_counts_flat,
        np.asarray([2, 1, 1, 1, 1, 1, 2], dtype=np.int32),
    )
    assert np.array_equal(
        bundle.sentence_doc_ids,
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    assert np.array_equal(
        bundle.sentence_lengths,
        np.asarray([3, 2, 2, 2], dtype=np.int32),
    )


def test_sentlda_backend_configuration_is_validated() -> None:
    assert SENTLDA_NUMERICS_BACKEND in {"python", "numba"}
    assert parse_sentlda_params({"backend": "NUMBA"}).backend == "numba"
    assert resolve_sentlda_backend("auto") in {"python", "numba"}

    with pytest.raises(ValueError):
        parse_sentlda_params({"backend": "invalid"})

    with pytest.raises(ValueError):
        resolve_sentlda_backend("invalid")


def test_sentlda_fit_and_infer_match_between_python_and_auto_backends() -> None:
    train_docs = [
        _doc(["apple", "banana", "apple"], ["banana", "carrot"]),
        _doc(["durian", "apple"], ["carrot", "carrot"]),
    ]
    test_docs = [
        _doc(["apple", "banana"], ["carrot", "durian"]),
        _doc(["banana", "apple", "banana"]),
    ]
    vocabulary = _build_vocabulary(train_docs)
    train_bundle = _encode_corpus(train_docs, vocabulary)
    test_bundle = _encode_corpus(test_docs, vocabulary)
    auto_backend = resolve_sentlda_backend("auto")

    python_fit = _fit_sentlda(
        bundle=train_bundle,
        num_topics=3,
        alpha=0.2,
        beta=0.05,
        vocab_size=len(vocabulary),
        num_iterations=6,
        random_state=7,
        backend="python",
    )
    auto_fit = _fit_sentlda(
        bundle=train_bundle,
        num_topics=3,
        alpha=0.2,
        beta=0.05,
        vocab_size=len(vocabulary),
        num_iterations=6,
        random_state=7,
        backend=auto_backend,
    )

    assert np.array_equal(python_fit[0], auto_fit[0])
    assert np.array_equal(python_fit[1], auto_fit[1])
    assert np.array_equal(python_fit[2], auto_fit[2])
    _assert_grouped_arrays_equal(python_fit[3], auto_fit[3])
    _assert_grouped_arrays_equal(python_fit[4], auto_fit[4])
    _assert_grouped_arrays_equal(python_fit[5], auto_fit[5])
    _assert_grouped_arrays_equal(python_fit[6], auto_fit[6])
    _assert_soft_matches_log_factors(auto_fit[4], auto_fit[5], auto_fit[6])

    python_infer = _infer_sentlda(
        bundle=test_bundle,
        num_topics=3,
        alpha=0.2,
        beta=0.05,
        vocab_size=len(vocabulary),
        topic_word_counts=python_fit[0],
        topic_total_words=python_fit[1],
        num_iterations=5,
        random_state=13,
        backend="python",
    )
    auto_infer = _infer_sentlda(
        bundle=test_bundle,
        num_topics=3,
        alpha=0.2,
        beta=0.05,
        vocab_size=len(vocabulary),
        topic_word_counts=auto_fit[0],
        topic_total_words=auto_fit[1],
        num_iterations=5,
        random_state=13,
        backend=auto_backend,
    )

    assert np.array_equal(python_infer[0], auto_infer[0])
    _assert_grouped_arrays_equal(python_infer[1], auto_infer[1])
    _assert_grouped_arrays_equal(python_infer[2], auto_infer[2])
    _assert_grouped_arrays_equal(python_infer[3], auto_infer[3])
    _assert_grouped_arrays_equal(python_infer[4], auto_infer[4])
    _assert_soft_matches_log_factors(auto_infer[2], auto_infer[3], auto_infer[4])
    for doc_soft in auto_infer[2]:
        assert np.allclose(doc_soft.sum(axis=1), 1.0, atol=1e-6)


def test_sentlda_public_api_supports_backend_parameter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_docs = [
        _doc(["apple", "banana", "apple"], ["banana", "carrot"]),
        _doc(["durian", "apple"], ["carrot", "carrot"]),
    ]
    test_docs = [
        _doc(["apple", "banana"], ["unknown", "carrot"]),
        _doc(["durian", "apple"]),
    ]
    corpus_by_csv = {
        "train.csv": train_docs,
        "test.csv": test_docs,
    }

    def _fake_loader(*, csv_paths, **_kwargs):
        return corpus_by_csv[str(csv_paths[0])]

    monkeypatch.setattr(
        "src.baselines.models.sentlda.load_preprocessed_documents",
        _fake_loader,
    )

    train_result = train_sentlda(
        train_csvs=["train.csv"],
        targets=None,
        text_column="text",
        target_column=None,
        delimiter=None,
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=3,
        params=SentLdaParams(
            num_iterations=5,
            infer_num_iterations=4,
            random_state=11,
            backend="python",
        ),
        train_dir=tmp_path / "train",
        use_legacy=False,
    )

    infer_result = infer_sentlda(
        train_result=train_result,
        test_csvs=["test.csv"],
        targets=None,
        text_column="text",
        target_column=None,
        delimiter=None,
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=3,
        params=SentLdaParams(
            num_iterations=5,
            infer_num_iterations=4,
            random_state=17,
            backend="python",
        ),
        use_legacy=False,
    )

    assert train_result.train_doc_topic.shape == (2, 3)
    assert infer_result.test_doc_topic.shape == (2, 3)
    assert len(train_result.train_sentence_topic_assignments) == 2
    assert len(infer_result.test_sentence_topic_assignments) == 2
    _assert_soft_matches_log_factors(
        train_result.train_sentence_topic_soft,
        train_result.train_sentence_topic_loglik,
        train_result.train_sentence_topic_logprior,
    )
    _assert_soft_matches_log_factors(
        infer_result.test_sentence_topic_soft,
        infer_result.test_sentence_topic_loglik,
        infer_result.test_sentence_topic_logprior,
    )
    for doc_soft in train_result.train_sentence_topic_soft:
        assert np.allclose(doc_soft.sum(axis=1), 1.0, atol=1e-6)
    for doc_soft in infer_result.test_sentence_topic_soft:
        assert np.allclose(doc_soft.sum(axis=1), 1.0, atol=1e-6)

    artifacts = persist_sentlda_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "artifacts" / "params",
        infer_dir=tmp_path / "artifacts" / "infer",
        category="all",
    )
    assert artifacts.extras["train_sentence_topic_loglik"].name == (
        "all_sentence_topic_loglik.pkl"
    )
    assert artifacts.extras["test_sentence_topic_logprior"].name == (
        "all_sentence_topic_logprior.pkl"
    )
    assert (
        load_artifact_json(artifacts.extras["params_json"])[
            "sentence_topic_soft_definition"
        ]
        == "softmax(logprior + loglik)"
    )
    persisted_loglik = load_artifact_pickle(
        artifacts.extras["train_sentence_topic_loglik"]
    )
    _assert_grouped_arrays_equal(
        persisted_loglik,
        train_result.train_sentence_topic_loglik,
    )

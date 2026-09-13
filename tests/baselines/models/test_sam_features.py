"""Tests for SAM featurization: vocabulary pruning, idf replay, drop bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfTransformer

from src.baselines.models import sam_features as features
from src.baselines.params import parse_sam_params
from src.data.preprocessing import PreprocessedDocument, select_modelable_documents


def _document(tokens: list[str]) -> PreprocessedDocument:
    return PreprocessedDocument(
        raw_text=" ".join(tokens),
        sentences_raw=[" ".join(tokens)],
        sentences_tokenized=[list(tokens)],
        sentences_joined=[" ".join(tokens)],
        document_tokens=list(tokens),
    )


def _selection(token_documents: list[list[str]], *, raw_offset: int = 0):
    documents = [_document(tokens) for tokens in token_documents]
    return select_modelable_documents(
        documents,
        raw_doc_indices=[raw_offset + index for index in range(len(documents))],
    )


_TRAIN = [
    ["alpha", "beta", "gamma", "alpha"],
    ["alpha", "beta", "delta"],
    ["beta", "gamma", "delta", "delta"],
    ["alpha", "gamma", "delta"],
    ["alpha", "beta", "gamma", "delta"],
]


def test_training_corpus_rows_are_unit_norm() -> None:
    params = parse_sam_params({"min_df": 1, "max_df": 1.0})
    vocabulary, corpus = features.build_training_corpus(
        selection=_selection(_TRAIN), params=params
    )
    norms = np.sqrt(
        np.asarray(corpus.matrix.multiply(corpus.matrix).sum(axis=1))
    ).ravel()
    assert np.allclose(norms, 1.0)
    assert corpus.matrix.shape == (len(_TRAIN), vocabulary.size)
    assert vocabulary.size == 4


def test_tfidf_matches_sklearn_transformer() -> None:
    """The hand-rolled weighting must equal ``TfidfTransformer(norm='l2')``.

    We compute it here rather than persisting a scikit-learn estimator, so this
    equivalence is what licenses that choice.
    """

    params = parse_sam_params({"min_df": 1, "max_df": 1.0})
    vocabulary, corpus = features.build_training_corpus(
        selection=_selection(_TRAIN), params=params
    )
    index_map = vocabulary.index_map()
    counts = np.zeros((len(_TRAIN), vocabulary.size))
    for row, tokens in enumerate(_TRAIN):
        for token in tokens:
            counts[row, index_map[token]] += 1.0

    expected = TfidfTransformer(norm="l2", smooth_idf=True).fit_transform(counts)
    assert np.allclose(corpus.matrix.toarray(), expected.toarray())


def test_sublinear_tf_matches_sklearn_transformer() -> None:
    params = parse_sam_params({"min_df": 1, "max_df": 1.0, "sublinear_tf": True})
    vocabulary, corpus = features.build_training_corpus(
        selection=_selection(_TRAIN), params=params
    )
    index_map = vocabulary.index_map()
    counts = np.zeros((len(_TRAIN), vocabulary.size))
    for row, tokens in enumerate(_TRAIN):
        for token in tokens:
            counts[row, index_map[token]] += 1.0
    expected = TfidfTransformer(
        norm="l2", smooth_idf=True, sublinear_tf=True
    ).fit_transform(counts)
    assert np.allclose(corpus.matrix.toarray(), expected.toarray())


def test_vocabulary_pruning_drops_documents_through_the_shared_helper() -> None:
    """A document with no surviving token must be recorded, not silently lost."""

    corpus_tokens = _TRAIN + [["rare_one", "rare_two"]]
    params = parse_sam_params({"min_df": 2, "max_df": 1.0})
    _, corpus = features.build_training_corpus(
        selection=_selection(corpus_tokens), params=params
    )
    assert len(corpus.documents) == len(_TRAIN)
    assert corpus.selection.raw_doc_indices == [0, 1, 2, 3, 4]
    assert corpus.selection.dropped_doc_indices == [5]
    assert corpus.selection.drop_reasons[5] == "no_vocabulary_tokens"


def test_inference_corpus_replays_training_idf() -> None:
    params = parse_sam_params({"min_df": 1, "max_df": 1.0})
    vocabulary, _ = features.build_training_corpus(
        selection=_selection(_TRAIN), params=params
    )
    held_out = [["alpha", "alpha", "beta"], ["gamma", "delta"]]
    corpus = features.build_inference_corpus(
        selection=_selection(held_out, raw_offset=100),
        vocabulary=vocabulary,
        empty_error_message="no held-out documents",
    )
    index_map = vocabulary.index_map()
    counts = np.zeros((len(held_out), vocabulary.size))
    for row, tokens in enumerate(held_out):
        for token in tokens:
            counts[row, index_map[token]] += 1.0
    weighted = counts * vocabulary.idf
    weighted /= np.linalg.norm(weighted, axis=1, keepdims=True)
    assert np.allclose(corpus.matrix.toarray(), weighted)
    assert corpus.selection.raw_doc_indices == [100, 101]


def test_inference_corpus_drops_out_of_vocabulary_documents() -> None:
    params = parse_sam_params({"min_df": 1, "max_df": 1.0})
    vocabulary, _ = features.build_training_corpus(
        selection=_selection(_TRAIN), params=params
    )
    corpus = features.build_inference_corpus(
        selection=_selection([["alpha"], ["unseen_token"]], raw_offset=200),
        vocabulary=vocabulary,
        empty_error_message="no held-out documents",
    )
    assert corpus.selection.raw_doc_indices == [200]
    assert corpus.selection.drop_reasons[201] == "no_vocabulary_tokens"


def test_empty_held_out_split_raises_the_supplied_message() -> None:
    params = parse_sam_params({"min_df": 1, "max_df": 1.0})
    vocabulary, _ = features.build_training_corpus(
        selection=_selection(_TRAIN), params=params
    )
    with pytest.raises(ValueError, match="custom message"):
        features.build_inference_corpus(
            selection=_selection([["unseen_token"]], raw_offset=300),
            vocabulary=vocabulary,
            empty_error_message="custom message",
        )


def test_over_aggressive_pruning_raises_a_named_error() -> None:
    params = parse_sam_params({"min_df": 5, "max_df": 0.5})
    with pytest.raises(ValueError, match="vocabulary collapsed"):
        features.build_training_corpus(selection=_selection(_TRAIN), params=params)


def test_vocabulary_rejects_mismatched_idf_length() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        features.SamVocabulary(
            words=("a", "b"),
            idf=np.ones(3),
            feature_scheme="tfidf",
            sublinear_tf=False,
        )


def test_tf_scheme_is_plain_l2_normalized_term_frequency() -> None:
    """``feature_scheme="tf"`` must drop the idf weighting entirely.

    The paper reports tf and tf-idf as two conditions rather than a default and a
    variant, and the gap between them (~4.7 points on its news-20 tasks) is large
    enough that conflating them would attribute the representation's effect to the
    model.  The corpus below gives the three terms distinct document frequencies,
    without which the idf vector is constant and L2 normalization makes tf-idf and
    tf indistinguishable.
    """

    tokens = [
        ["alpha", "alpha", "beta"],
        ["alpha", "gamma"],
        ["alpha", "gamma", "gamma", "gamma"],
    ]  # df: alpha=3, beta=1, gamma=2
    vocabulary, corpus = features.build_training_corpus(
        selection=_selection(tokens), params=parse_sam_params({"feature_scheme": "tf"})
    )
    assert np.allclose(vocabulary.idf, 1.0)

    index = {word: i for i, word in enumerate(vocabulary.words)}
    counts = np.zeros((len(tokens), len(vocabulary.words)))
    for row, doc in enumerate(tokens):
        for token in doc:
            counts[row, index[token]] += 1.0
    expected = counts / np.linalg.norm(counts, axis=1, keepdims=True)
    assert np.allclose(corpus.matrix.toarray(), expected)

    tfidf_vocabulary, tfidf_corpus = features.build_training_corpus(
        selection=_selection(tokens),
        params=parse_sam_params({"feature_scheme": "tfidf"}),
    )
    assert not np.allclose(tfidf_vocabulary.idf, 1.0)
    assert not np.allclose(tfidf_corpus.matrix.toarray(), expected)


def test_sam_tf_params_default_to_tf_but_honour_an_explicit_scheme() -> None:
    from src.baselines.params import normalize_baseline_params, parse_sam_tf_params

    assert parse_sam_tf_params({}).feature_scheme == "tf"
    assert parse_sam_tf_params({"feature_scheme": "tfidf"}).feature_scheme == "tfidf"
    # Everything else is shared with the tf-idf runner.
    base = parse_sam_params({})
    tf = parse_sam_tf_params({})
    assert (tf.min_df, tf.max_df, tf.kappa_mean_resultant) == (
        base.min_df,
        base.max_df,
        base.kappa_mean_resultant,
    )
    assert normalize_baseline_params("sam_tf", None).feature_scheme == "tf"
    assert normalize_baseline_params("sam", None).feature_scheme == "tfidf"

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from gensim.corpora import Dictionary

from src.evaluation.word_based.topic_word_metrics import (
    PMI_SMOOTHING_EPSILON,
    compute_coherence_score,
    compute_coherence_scores,
    compute_doc_npmi_score,
    compute_palmetto_cv_score,
    compute_streaming_reference_coherence_score,
    compute_streaming_reference_coherence_scores,
    compute_topic_diversity,
    describe_coherence_metric,
    evaluate_topic_words,
)


def test_compute_topic_diversity_uses_unique_word_ratio() -> None:
    topic_words = [
        [("alpha", 1.0), ("beta", 0.8)],
        [("beta", 0.9), ("gamma", 0.7)],
    ]

    diversity = compute_topic_diversity(topic_words)

    assert diversity == 0.75


def test_evaluate_topic_words_returns_both_metrics(monkeypatch) -> None:
    dictionary = Dictionary([["alpha"], ["beta"]])
    monkeypatch.setattr(
        "src.evaluation.word_based.topic_word_metrics.compute_coherence_score",
        lambda **_kwargs: 0.42,
    )

    metrics = evaluate_topic_words(
        topic_words=[[("alpha", 1.0)], [("beta", 0.5)]],
        texts=[["alpha"], ["beta"]],
        dictionary=dictionary,
        corpus_bow=[[(0, 1)], [(1, 1)]],
        coherence="c_v",
    )

    assert metrics == {"coherence": 0.42, "diversity": 1.0}


def test_evaluate_topic_words_uses_distinct_topn_for_coherence_and_diversity(
    monkeypatch,
) -> None:
    dictionary = Dictionary([["alpha"], ["beta"], ["gamma"]])
    captured: dict[str, object] = {}

    def _fake_compute_coherence_score(**kwargs) -> float:
        captured["topic_words"] = kwargs["topic_words"]
        captured["window_size"] = kwargs["window_size"]
        return 0.5

    monkeypatch.setattr(
        "src.evaluation.word_based.topic_word_metrics.compute_coherence_score",
        _fake_compute_coherence_score,
    )

    metrics = evaluate_topic_words(
        topic_words=[
            [("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)],
            [("alpha", 0.9), ("gamma", 0.7), ("beta", 0.5)],
        ],
        texts=[["alpha"], ["beta"]],
        dictionary=dictionary,
        corpus_bow=[[(0, 1)], [(1, 1)]],
        coherence="c_v",
        coherence_topn=1,
        coherence_window_size=110,
        diversity_topn=2,
    )

    assert captured["topic_words"] == [[("alpha", 1.0)], [("alpha", 0.9)]]
    assert captured["window_size"] == 110
    assert metrics == {"coherence": 0.5, "diversity": 0.75}


def test_compute_doc_npmi_score_averages_pairwise_document_npmi() -> None:
    topic_words = [[("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)]]
    texts = [
        ["alpha", "beta"],
        ["alpha", "gamma"],
        ["alpha", "beta", "gamma"],
        ["beta", "gamma"],
    ]

    score = compute_doc_npmi_score(topic_words=topic_words, texts=texts)

    assert score == -0.16992500144231246


def test_compute_doc_npmi_score_uses_minus_one_for_zero_cooccurrence() -> None:
    topic_words = [[("alpha", 1.0), ("delta", 0.8)]]
    texts = [["alpha"], ["beta"], ["gamma"]]

    score = compute_doc_npmi_score(topic_words=topic_words, texts=texts)

    assert score == -1.0


def test_compute_streaming_reference_doc_npmi_reads_jsonl_without_materializing(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    rows = [
        {"tokens": ["alpha", "beta"]},
        {"tokens": ["alpha", "gamma"]},
        {"tokens": ["alpha", "beta", "gamma"]},
        {"tokens": ["beta", "gamma"]},
    ]
    reference_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = compute_streaming_reference_coherence_score(
        topic_words=[[("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)]],
        reference_path=reference_path,
        coherence="doc_npmi",
        min_doc_tokens=1,
    )

    assert result.score == -0.16992500144231246
    assert result.num_docs == 4
    assert result.vocab_size == 3


def test_compute_c_npmi_uses_minus_one_for_reference_oov_terms() -> None:
    dictionary = Dictionary([["alpha"], ["beta"]])
    texts = [["alpha"], ["beta"]]
    corpus_bow = [dictionary.doc2bow(doc) for doc in texts]

    score = compute_coherence_score(
        topic_words=[[("alpha", 1.0), ("missing", 0.8)]],
        texts=texts,
        dictionary=dictionary,
        corpus_bow=corpus_bow,
        coherence="c_npmi",
        window_size=10,
    )

    assert math.isfinite(score)
    assert score == -1.0


def test_compute_c_npmi_penalizes_zero_cooccurrence_for_in_vocab_terms() -> None:
    dictionary = Dictionary([["alpha"], ["beta"]])
    texts = [["alpha"], ["beta"]]
    corpus_bow = [dictionary.doc2bow(doc) for doc in texts]

    score = compute_coherence_score(
        topic_words=[[("alpha", 1.0), ("beta", 0.8)]],
        texts=texts,
        dictionary=dictionary,
        corpus_bow=corpus_bow,
        coherence="c_npmi",
        window_size=10,
    )

    assert math.isfinite(score)
    assert score < 0.0


def test_compute_streaming_reference_c_npmi_uses_minus_one_for_oov_terms(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    rows = [
        {"tokens": ["alpha"]},
        {"tokens": ["beta"]},
    ]
    reference_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = compute_streaming_reference_coherence_score(
        topic_words=[[("alpha", 1.0), ("missing", 0.8)]],
        reference_path=reference_path,
        coherence="c_npmi",
        min_doc_tokens=1,
    )

    assert math.isfinite(result.score)
    assert result.score == -1.0
    assert result.num_docs == 2
    assert result.vocab_size == 2


def test_compute_palmetto_cv_returns_finite_score() -> None:
    score = compute_palmetto_cv_score(
        topic_words=[[("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)]],
        texts=[
            ["alpha", "beta", "gamma"],
            ["alpha", "beta"],
            ["beta", "gamma"],
        ],
        window_size=2,
    )

    assert math.isfinite(score)


def test_compute_palmetto_cv_returns_zero_for_missing_reference_words() -> None:
    score = compute_palmetto_cv_score(
        topic_words=[[("missing_a", 1.0), ("missing_b", 0.8)]],
        texts=[["alpha"], ["beta"]],
    )

    assert score == 0.0


def test_compute_palmetto_cv_zero_vector_similarity_is_finite() -> None:
    score = compute_palmetto_cv_score(
        topic_words=[[("alpha", 1.0), ("missing", 0.8)]],
        texts=[["alpha"], ["alpha"]],
    )

    assert math.isfinite(score)
    assert score == 0.5


def test_compute_coherence_score_uses_palmetto_cv_without_gensim(
    monkeypatch,
) -> None:
    dictionary = Dictionary([["alpha"], ["beta"]])
    texts = [["alpha", "beta"], ["alpha"]]
    corpus_bow = [dictionary.doc2bow(doc) for doc in texts]

    monkeypatch.setattr(
        "src.evaluation.word_based.topic_word_metrics.CoherenceModel",
        lambda **_kwargs: pytest.fail("c_v should not call gensim CoherenceModel"),
    )

    score = compute_coherence_score(
        topic_words=[[("alpha", 1.0), ("beta", 0.8)]],
        texts=texts,
        dictionary=dictionary,
        corpus_bow=corpus_bow,
        coherence="c_v",
    )

    assert math.isfinite(score)


def test_streaming_reference_cv_matches_non_streaming(tmp_path: Path) -> None:
    rows = [
        {"tokens": ["alpha", "beta"]},
        {"tokens": ["alpha", "gamma"]},
        {"tokens": ["beta", "gamma"]},
    ]
    reference_path = tmp_path / "reference.jsonl"
    reference_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    topic_words = [[("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)]]

    expected = compute_palmetto_cv_score(
        topic_words=topic_words,
        texts=[row["tokens"] for row in rows],
        window_size=2,
    )
    result = compute_streaming_reference_coherence_score(
        topic_words=topic_words,
        reference_path=reference_path,
        coherence="c_v",
        window_size=2,
        min_doc_tokens=1,
    )

    assert result.score == expected
    assert result.num_docs == 3
    assert result.vocab_size == 3


def test_evaluate_topic_words_supports_doc_npmi() -> None:
    dictionary = Dictionary([["alpha"], ["beta"], ["gamma"]])

    metrics = evaluate_topic_words(
        topic_words=[[("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)]],
        texts=[
            ["alpha", "beta"],
            ["alpha", "gamma"],
            ["alpha", "beta", "gamma"],
            ["beta", "gamma"],
        ],
        dictionary=dictionary,
        corpus_bow=[
            [(0, 1), (1, 1)],
            [(0, 1), (2, 1)],
            [(0, 1), (1, 1), (2, 1)],
            [(1, 1), (2, 1)],
        ],
        coherence="doc_npmi",
        coherence_topn=3,
        diversity_topn=3,
    )

    assert metrics == {"coherence": -0.16992500144231246, "diversity": 1.0}


def test_compute_coherence_scores_returns_multiple_project_coherences() -> None:
    topic_words = [[("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)]]
    texts = [
        ["alpha", "beta", "gamma"],
        ["alpha", "beta"],
        ["beta", "gamma"],
    ]
    dictionary = Dictionary(texts)
    corpus_bow = [dictionary.doc2bow(doc) for doc in texts]

    scores = compute_coherence_scores(
        topic_words=topic_words,
        texts=texts,
        dictionary=dictionary,
        corpus_bow=corpus_bow,
        coherences=["c_v", "c_npmi", "c_uci"],
        window_size=2,
    )

    assert set(scores) == {"c_v", "c_npmi", "c_uci"}
    for coherence, score in scores.items():
        expected = compute_coherence_score(
            topic_words=topic_words,
            texts=texts,
            dictionary=dictionary,
            corpus_bow=corpus_bow,
            coherence=coherence,
            window_size=2,
        )
        assert score == expected


def test_evaluate_topic_words_returns_named_keys_for_multiple_coherences() -> None:
    topic_words = [[("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)]]
    texts = [
        ["alpha", "beta", "gamma"],
        ["alpha", "beta"],
        ["beta", "gamma"],
    ]
    dictionary = Dictionary(texts)
    corpus_bow = [dictionary.doc2bow(doc) for doc in texts]

    metrics = evaluate_topic_words(
        topic_words=topic_words,
        metric_names=[
            "coherence_c_v",
            "coherence_c_npmi",
            "coherence_c_uci",
            "diversity",
        ],
        texts=texts,
        dictionary=dictionary,
        corpus_bow=corpus_bow,
        coherence=["c_v", "c_npmi", "c_uci"],
        coherence_topn=3,
        diversity_topn=3,
        coherence_window_size=2,
    )

    assert set(metrics) == {
        "coherence_c_v",
        "coherence_c_npmi",
        "coherence_c_uci",
        "diversity",
    }
    assert metrics["diversity"] == 1.0


def test_streaming_reference_multiple_coherences_builds_counts_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    rows = [
        {"tokens": ["alpha", "beta", "gamma"]},
        {"tokens": ["alpha", "beta"]},
        {"tokens": ["beta", "gamma"]},
    ]
    reference_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    call_count = 0
    from src.evaluation.word_based import topic_word_metrics as metrics_module

    original_builder = metrics_module._build_sliding_window_counts

    def _counting_builder(**kwargs):
        nonlocal call_count
        call_count += 1
        return original_builder(**kwargs)

    monkeypatch.setattr(
        "src.evaluation.word_based.topic_word_metrics._build_sliding_window_counts",
        _counting_builder,
    )

    result = compute_streaming_reference_coherence_scores(
        topic_words=[[("alpha", 1.0), ("beta", 0.8), ("gamma", 0.6)]],
        reference_path=reference_path,
        coherences=["c_v", "c_npmi", "c_uci"],
        window_size=2,
        min_doc_tokens=1,
    )

    assert call_count == 1
    assert set(result.scores) == {"c_v", "c_npmi", "c_uci"}
    assert result.num_docs == 3
    assert result.vocab_size == 3


def test_describe_coherence_metric_documents_doc_npmi() -> None:
    assert describe_coherence_metric("doc_npmi") == {
        "definition": (
            "Average pairwise NPMI over top-N topic words using document-level "
            "boolean co-occurrence."
        ),
        "cooccurrence_unit": "document",
        "zero_cooccurrence_policy": "minus_one",
    }


def test_describe_coherence_metric_documents_epsilon_smoothing() -> None:
    details = describe_coherence_metric("c_npmi")

    assert details["zero_cooccurrence_policy"] == "epsilon_smoothing"
    assert details["pmi_smoothing_epsilon"] == PMI_SMOOTHING_EPSILON


def test_describe_coherence_metric_documents_palmetto_cv() -> None:
    details = describe_coherence_metric("c_v")

    assert details["cooccurrence_unit"] == "sliding_window"
    assert details["zero_cooccurrence_policy"] == (
        "undefined_npmi_and_zero_vector_as_zero"
    )
    assert details["pmi_smoothing_epsilon"] == PMI_SMOOTHING_EPSILON

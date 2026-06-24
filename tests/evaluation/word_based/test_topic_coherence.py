from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from gensim.corpora import Dictionary

from src.core.artifacts import load_json, save_pickle
from src.data.preprocessing import PreprocessedDocument
from src.evaluation.reporting import read_evaluation_json
from src.evaluation.word_based.metrics import run_topic_coherence_analysis
from src.evaluation.word_based.topic_words import TopicWordsResult


def _set_fixed_now(
    monkeypatch,
    iso_timestamp: str,
) -> None:
    fixed_dt = datetime.fromisoformat(iso_timestamp).astimezone(UTC)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_dt
            return fixed_dt.astimezone(tz)

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.datetime",
        _FixedDateTime,
    )


def _patch_vmf_proxy_topic_words(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_learned_model",
        lambda **_kwargs: pytest.fail("vmf should use proxy topic words"),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_sentence_topic_npmi",
        lambda **_kwargs: TopicWordsResult(
            topic_words=[[("alpha", 1.0)], [("beta", 0.5)]],
            topic_word_source="sentence_topic_proxy_npmi",
            score_mode="word_npmi",
            score_definition="PMI normalized by -log p(w)",
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_sentence_topics",
        lambda **_kwargs: [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_documents",
        lambda **_kwargs: ["alpha", "beta"],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.tokenize_sentence_documents",
        lambda **_kwargs: [[["alpha"]], [["beta"]]],
    )


def test_run_topic_coherence_analysis_persists_model_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T00:00:00+00:00")
    dictionary = Dictionary([["alpha"], ["beta"]])

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    _patch_vmf_proxy_topic_words(monkeypatch)
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        lambda **_kwargs: {"coherence": 0.42, "diversity": 1.0},
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {
            "model_key": kwargs["model"],
            "metadata_path": str(tmp_path / kwargs["model"] / "metadata.json"),
            "runner_family": kwargs["model"],
            "parameter_variant": "passes=40",
        },
    )

    output_root = run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
    )

    assert output_root == tmp_path

    archive_root = tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all"
    metrics_path = next(archive_root.glob("it0__k2__vmf__*/exec_*/metrics_agg.json"))
    metrics_meta, metrics_results = read_evaluation_json(metrics_path)
    assert metrics_meta["data_run"] == "default"
    assert metrics_meta["condition_id"] == "it0__k2__vmf__palmetto-cv__mpnet"
    assert "__palmetto-cv__" in metrics_meta["condition_id"]
    assert metrics_meta["display_key"] == metrics_meta["condition_id"]
    assert metrics_meta["execution_id"] == "exec_20260413T000000Z"
    assert metrics_meta["archive_dir"].endswith("/exec_20260413T000000Z")
    assert metrics_meta["latest_dir"].endswith(f"/{metrics_meta['display_key']}")
    assert metrics_meta["metric_names"] == ["coherence", "diversity"]
    assert metrics_meta["topic_words"]["topn"] == 25
    assert metrics_meta["topic_words"]["coherence_topn"] == 10
    assert metrics_meta["topic_words"]["diversity_topn"] == 25
    assert metrics_meta["diversity"]["topn"] == 25
    assert metrics_meta["coherence"]["dict_no_below"] == 3
    assert metrics_meta["coherence"]["dict_no_above"] == 0.7
    assert metrics_meta["coherence"]["metric"] == "c_v"
    assert metrics_meta["coherence"]["implementation"] == "palmetto_compatible"
    assert metrics_meta["coherence"]["zero_cooccurrence_policy"] == (
        "undefined_npmi_and_zero_vector_as_zero"
    )
    assert metrics_meta["coherence"]["pmi_smoothing_epsilon"] == 1e-12
    assert (
        metrics_meta["coherence"]["probability_estimation"] == "boolean_sliding_window"
    )
    assert metrics_meta["coherence"]["confirmation_measure"] == (
        "normalized_log_ratio_npmi"
    )
    assert metrics_meta["coherence"]["vector_space"] == "top_word_npmi"
    assert metrics_meta["coherence"]["segmentation"] == "one_set"
    assert metrics_meta["coherence"]["similarity"] == "cosine"
    assert metrics_meta["coherence"]["aggregation"] == "arithmetic_mean"
    assert metrics_meta["coherence"]["undefined_npmi_policy"] == "zero"
    assert metrics_meta["coherence"]["zero_vector_similarity_policy"] == "zero"
    assert metrics_meta["coherence"]["coherence_window_size"] == 110
    assert metrics_meta["coherence"]["coherence_window_size_source"] == (
        "palmetto_compatible_default_c_v"
    )
    assert metrics_meta["coherence"]["coherence_min_window_count"] == 1
    assert metrics_meta["model_provenance"] == {
        "model_key": "vmf",
        "metadata_path": str(tmp_path / "vmf" / "metadata.json"),
        "runner_family": "vmf",
        "parameter_variant": "passes=40",
    }
    assert metrics_results["aggregate"]["coherence"]["mean"] == 0.42
    assert metrics_results["aggregate"]["diversity"]["mean"] == 1.0

    metadata_path = metrics_path.parent / "metadata.json"
    assert metadata_path.exists()

    topic_words_path = metrics_path.parent / "topic_words_topk.json"
    topic_words_meta, _topic_words_results = read_evaluation_json(topic_words_path)
    assert topic_words_meta["data_run"] == "default"
    assert topic_words_meta["model_provenance"]["parameter_variant"] == "passes=40"
    assert topic_words_meta["topn"] == 25
    assert topic_words_meta["coherence_topn"] == 10
    assert topic_words_meta["diversity_topn"] == 25

    latest_pointer = load_json(
        tmp_path
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / metrics_meta["display_key"]
        / "CURRENT.json"
    )
    assert latest_pointer["display_key"] == metrics_meta["display_key"]
    assert latest_pointer["execution_id"] == "exec_20260413T000000Z"
    assert latest_pointer["artifacts"]["metrics"] == "metrics_agg.json"
    assert latest_pointer["artifacts"]["topic_words"] == "topic_words_topk.json"
    assert latest_pointer["artifacts"]["metadata"] == "metadata.json"

    assert not (tmp_path / "summary_metrics.csv").exists()
    assert not (tmp_path / "summary_metrics.json").exists()


def test_run_topic_coherence_analysis_skip_existing_ignores_fingerprint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    dictionary = Dictionary([["alpha"], ["beta"]])

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    _patch_vmf_proxy_topic_words(monkeypatch)
    calls = {"evaluate": 0}

    def _evaluate_topic_words(**_kwargs):
        calls["evaluate"] += 1
        return {"coherence": 0.42, "diversity": 1.0}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        _evaluate_topic_words,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    _set_fixed_now(monkeypatch, "2026-04-13T00:00:00+00:00")
    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        coherence_topn=10,
        skip_existing=True,
    )

    _set_fixed_now(monkeypatch, "2026-04-13T01:00:00+00:00")
    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        coherence_topn=11,
        skip_existing=True,
    )

    assert calls["evaluate"] == 1


def test_run_topic_coherence_analysis_writes_multiple_coherence_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T00:30:00+00:00")
    dictionary = Dictionary([["alpha"], ["beta"]])
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    _patch_vmf_proxy_topic_words(monkeypatch)

    def _evaluate_topic_words(**kwargs):
        captured["coherence"] = kwargs["coherence"]
        captured["metric_names"] = kwargs["metric_names"]
        return {
            "coherence_c_v": 0.42,
            "coherence_c_npmi": -0.12,
            "coherence_c_uci": -3.4,
            "diversity": 1.0,
        }

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        _evaluate_topic_words,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        coherence=["c_v", "c_npmi", "c_uci"],
    )

    assert captured["coherence"] == ["c_v", "c_npmi", "c_uci"]
    assert captured["metric_names"] == [
        "coherence_c_v",
        "coherence_c_npmi",
        "coherence_c_uci",
        "diversity",
    ]

    metrics_path = next(
        (tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all").glob(
            "it0__k2__vmf__*/exec_*/metrics_agg.json"
        )
    )
    metrics_meta, metrics_results = read_evaluation_json(metrics_path)
    assert metrics_meta["metric_names"] == [
        "coherence_c_v",
        "coherence_c_npmi",
        "coherence_c_uci",
        "diversity",
    ]
    assert metrics_meta["coherence"]["metrics"] == ["c_v", "c_npmi", "c_uci"]
    assert metrics_meta["coherence"]["primary_metric"] == "c_v"
    assert metrics_meta["coherence"]["by_metric"]["c_v"]["coherence_window_size"] == 110
    assert (
        metrics_meta["coherence"]["by_metric"]["c_npmi"]["coherence_window_size"] == 10
    )
    assert (
        metrics_meta["coherence"]["by_metric"]["c_uci"]["coherence_window_size"] == 10
    )
    assert metrics_results["aggregate"]["coherence_c_v"]["mean"] == 0.42
    assert metrics_results["aggregate"]["coherence_c_npmi"]["mean"] == -0.12
    assert metrics_results["aggregate"]["coherence_c_uci"]["mean"] == -3.4


def test_run_topic_coherence_analysis_supports_sentlda_proxy_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T01:00:00+00:00")
    dictionary = Dictionary([["alpha"], ["beta"]])

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_sentence_topic_npmi",
        lambda **_kwargs: TopicWordsResult(
            topic_words=[[("alpha", 1.0)], [("beta", 0.5)]],
            topic_word_source="sentence_topic_proxy_npmi",
            score_mode="word_npmi",
            score_definition="PMI normalized by -log p(w)",
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_sentence_topics",
        lambda **_kwargs: [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._load_sentlda_effective_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
            [[["alpha"]], [["beta"]]],
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        lambda **_kwargs: {"coherence": 0.31, "diversity": 0.9},
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {
            "model_key": kwargs["model"],
            "metadata_path": str(tmp_path / kwargs["model"] / "metadata.json"),
            "runner_family": kwargs["model"],
            "parameter_variant": "alpha=0.2",
        },
    )

    output_root = run_topic_coherence_analysis(
        models=["sentlda"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        proxy_word_score_mode="word_npmi",
    )

    assert output_root == tmp_path

    metrics_path = next(
        (tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all").glob(
            "it0__k2__sentlda__*/exec_*/metrics_agg.json"
        )
    )
    metrics_meta, metrics_results = read_evaluation_json(metrics_path)
    assert metrics_meta["model_provenance"]["model_key"] == "sentlda"
    assert metrics_meta["topic_words"]["score_mode"] == "word_npmi"
    assert (
        metrics_meta["topic_words"]["score_definition"] == "PMI normalized by -log p(w)"
    )
    assert metrics_meta["coherence"]["proxy_word_score_mode"] == "word_npmi"
    assert metrics_results["aggregate"]["coherence"]["mean"] == 0.31

    topic_words_path = metrics_path.parent / "topic_words_topk.json"
    topic_words_meta, _topic_words_results = read_evaluation_json(topic_words_path)
    assert topic_words_meta["model_provenance"]["runner_family"] == "sentlda"
    assert topic_words_meta["score_mode"] == "word_npmi"
    assert topic_words_meta["score_definition"] == "PMI normalized by -log p(w)"

    assert not (tmp_path / "summary_metrics.csv").exists()
    assert not (tmp_path / "summary_metrics.json").exists()


def test_run_topic_coherence_analysis_uses_proxy_for_vmf(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T01:30:00+00:00")
    dictionary = Dictionary([["alpha"], ["beta"]])

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_learned_model",
        lambda **_kwargs: pytest.fail("vmf should use proxy topic words"),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_sentence_topic_npmi",
        lambda **_kwargs: TopicWordsResult(
            topic_words=[[("alpha", 1.0)], [("beta", 0.5)]],
            topic_word_source="sentence_topic_proxy_npmi",
            score_mode="word_npmi",
            score_definition="PMI normalized by -log p(w)",
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_sentence_topics",
        lambda **_kwargs: [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_documents",
        lambda **_kwargs: ["alpha", "beta"],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.tokenize_sentence_documents",
        lambda **_kwargs: [[["alpha"]], [["beta"]]],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        lambda **_kwargs: {"coherence": 0.29, "diversity": 0.8},
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {
            "model_key": kwargs["model"],
            "runner_family": kwargs["model"],
        },
    )

    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        proxy_word_score_mode="word_npmi",
    )

    metrics_path = next(
        (tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all").glob(
            "it0__k2__vmf__*/exec_*/metrics_agg.json"
        )
    )
    metrics_meta, metrics_results = read_evaluation_json(metrics_path)
    assert metrics_meta["topic_words"]["source"] == "sentence_topic_proxy_npmi"
    assert metrics_meta["topic_words"]["score_mode"] == "word_npmi"
    assert metrics_meta["coherence"]["proxy_npmi_mode"] == "sentence"
    assert metrics_meta["coherence"]["proxy_word_score_mode"] == "word_npmi"
    assert metrics_results["aggregate"]["coherence"]["mean"] == 0.29


def test_run_topic_coherence_analysis_uses_wikipedia_reference_for_coherence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T01:45:00+00:00")
    target_dictionary = Dictionary([["target_alpha"], ["target_beta"]])
    reference_path = tmp_path / "enwiki-tokenized.jsonl"
    reference_path.write_text(
        "\n".join(
            [
                json.dumps({"tokens": ["wiki_alpha", "wiki_beta"]}),
                json.dumps({"tokens": ["wiki_beta", "wiki_gamma"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["target_alpha"], ["target_beta"]],
            target_dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )

    def _extract_topic_words(**kwargs):
        captured["proxy_dictionary"] = kwargs["dictionary"]
        return TopicWordsResult(
            topic_words=[[("wiki_alpha", 1.0)], [("wiki_beta", 0.5)]],
            topic_word_source="sentence_topic_proxy_npmi",
            score_mode="word_npmi",
            score_definition="PMI normalized by -log p(w)",
        )

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_sentence_topic_npmi",
        _extract_topic_words,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_sentence_topics",
        lambda **_kwargs: [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_documents",
        lambda **_kwargs: ["target alpha", "target beta"],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.tokenize_sentence_documents",
        lambda **_kwargs: [[["target_alpha"]], [["target_beta"]]],
    )

    def _build_shared_reference_counts(**kwargs):
        captured["reference_path"] = kwargs["reference_path"]
        captured["target_words"] = kwargs["target_words"]
        captured["window_sizes"] = kwargs["window_sizes"]
        captured["max_docs"] = kwargs["max_docs"]
        captured["backend"] = kwargs["backend"]
        return SimpleNamespace(num_docs=2, vocab_size=2)

    def _compute_shared_reference_coherence_scores(**kwargs):
        captured["coherence_topic_words"] = kwargs["topic_words"]
        captured["coherence_window_size"] = kwargs["window_size"]
        captured["coherence_min_window_count"] = kwargs["min_window_count"]
        return {"coherence": 0.37, "diversity": 0.8}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_shared_reference_counts",
        _build_shared_reference_counts,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.compute_shared_reference_coherence_scores",
        _compute_shared_reference_coherence_scores,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        dict_no_below=1,
        dict_no_above=1.0,
        coherence_reference="wikipedia",
        coherence_reference_path=reference_path,
        coherence_reference_max_docs=2,
        coherence_window_size=110,
        coherence_min_window_count=7,
    )

    assert captured["proxy_dictionary"] is target_dictionary
    assert captured["reference_path"] == reference_path
    assert captured["target_words"] == {"wiki_alpha", "wiki_beta"}
    assert captured["window_sizes"] == {110}
    assert captured["max_docs"] == 2
    assert captured["backend"] == "numba"
    assert captured["coherence_topic_words"] == [
        [("wiki_alpha", 1.0)],
        [("wiki_beta", 0.5)],
    ]
    assert captured["coherence_window_size"] == 110
    assert captured["coherence_min_window_count"] == 7

    metrics_path = next(
        (tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all").glob(
            "it0__k2__vmf__*/exec_*/metrics_agg.json"
        )
    )
    metrics_meta, metrics_results = read_evaluation_json(metrics_path)
    assert metrics_results["aggregate"]["coherence"]["mean"] == 0.37
    assert metrics_meta["coherence"]["coherence_reference"] == "wikipedia"
    assert metrics_meta["coherence"]["coherence_reference_path"] == str(reference_path)
    assert metrics_meta["coherence"]["coherence_reference_num_docs"] == 2
    assert metrics_meta["coherence"]["coherence_reference_vocab_size"] == 2
    assert metrics_meta["coherence"]["coherence_reference_streaming"] is True
    assert metrics_meta["coherence"]["coherence_window_size"] == 110
    assert metrics_meta["coherence"]["coherence_window_size_source"] == "user"
    assert metrics_meta["coherence"]["coherence_min_window_count"] == 7

    topic_words_meta, _ = read_evaluation_json(
        metrics_path.parent / "topic_words_topk.json"
    )
    assert topic_words_meta["coherence_reference"]["coherence_reference"] == "wikipedia"

    assert not (tmp_path / "summary_metrics.csv").exists()
    assert not (tmp_path / "summary_metrics.json").exists()


def test_run_topic_coherence_analysis_streams_uncapped_wikipedia_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T01:45:00+00:00")
    target_dictionary = Dictionary([["target_alpha"], ["target_beta"]])
    reference_path = tmp_path / "enwiki-tokenized.jsonl"
    reference_path.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["target_alpha"], ["target_beta"]],
            target_dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_reference_corpus_bundle",
        lambda **_kwargs: pytest.fail("full reference corpus should not be loaded"),
    )

    def _extract_topic_words(**_kwargs):
        return TopicWordsResult(
            topic_words=[[("wiki_alpha", 1.0)], [("wiki_beta", 0.5)]],
            topic_word_source="sentence_topic_proxy_npmi",
            score_mode="word_npmi",
            score_definition="PMI normalized by -log p(w)",
        )

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_sentence_topic_npmi",
        _extract_topic_words,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_sentence_topics",
        lambda **_kwargs: [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_documents",
        lambda **_kwargs: ["target alpha", "target beta"],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.tokenize_sentence_documents",
        lambda **_kwargs: [[["target_alpha"]], [["target_beta"]]],
    )

    def _build_shared_reference_counts(**kwargs):
        captured["reference_path"] = kwargs["reference_path"]
        captured["max_docs"] = kwargs["max_docs"]
        captured["target_words"] = kwargs["target_words"]
        return SimpleNamespace(num_docs=123, vocab_size=2)

    def _compute_shared_reference_coherence_scores(**kwargs):
        captured["topic_words"] = kwargs["topic_words"]
        captured["min_window_count"] = kwargs["min_window_count"]
        return {"coherence": 0.44, "diversity": 1.0}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_shared_reference_counts",
        _build_shared_reference_counts,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.compute_shared_reference_coherence_scores",
        _compute_shared_reference_coherence_scores,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        dict_no_below=1,
        dict_no_above=1.0,
        coherence_reference="wikipedia",
        coherence_reference_path=reference_path,
        coherence_window_size=110,
        coherence_min_window_count=5,
    )

    assert captured["reference_path"] == reference_path
    assert captured["max_docs"] is None
    assert captured["target_words"] == {"wiki_alpha", "wiki_beta"}
    assert captured["topic_words"] == [[("wiki_alpha", 1.0)], [("wiki_beta", 0.5)]]
    assert captured["min_window_count"] == 5

    metrics_path = next(
        (tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all").glob(
            "it0__k2__vmf__*/exec_*/metrics_agg.json"
        )
    )
    metrics_meta, metrics_results = read_evaluation_json(metrics_path)
    assert metrics_results["aggregate"]["coherence"]["mean"] == 0.44
    assert metrics_meta["coherence"]["coherence_reference_num_docs"] == 123
    assert metrics_meta["coherence"]["coherence_reference_vocab_size"] == 2
    assert metrics_meta["coherence"]["coherence_reference_streaming"] is True
    assert metrics_meta["coherence"]["coherence_min_window_count"] == 5


def test_run_topic_coherence_analysis_scans_wikipedia_reference_once_for_categories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T01:45:00+00:00")
    target_dictionary = Dictionary([["target_alpha"], ["target_beta"]])
    reference_path = tmp_path / "enwiki-tokenized.jsonl"
    reference_path.write_text("", encoding="utf-8")
    captured: dict[str, object] = {"build_calls": [], "score_calls": []}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["target_alpha"], ["target_beta"]],
            target_dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )

    extracted_categories = iter(["all", "sports"])

    def _extract_topic_words(**_kwargs):
        category = next(extracted_categories)
        return TopicWordsResult(
            topic_words=[
                [(f"wiki_{category}", 1.0)],
                [("wiki_shared", 0.5)],
            ],
            topic_word_source="sentence_topic_proxy_npmi",
            score_mode="word_npmi",
            score_definition="PMI normalized by -log p(w)",
        )

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_sentence_topic_npmi",
        _extract_topic_words,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_sentence_topics",
        lambda **_kwargs: [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_documents",
        lambda **_kwargs: ["target alpha", "target beta"],
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.tokenize_sentence_documents",
        lambda **_kwargs: [[["target_alpha"]], [["target_beta"]]],
    )

    def _build_shared_reference_counts(**kwargs):
        captured["build_calls"].append(kwargs)
        return SimpleNamespace(num_docs=10, vocab_size=len(kwargs["target_words"]))

    def _compute_shared_reference_coherence_scores(**kwargs):
        captured["score_calls"].append(kwargs["topic_words"])
        return {"coherence": 0.5, "diversity": 1.0}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_shared_reference_counts",
        _build_shared_reference_counts,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.compute_shared_reference_coherence_scores",
        _compute_shared_reference_coherence_scores,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all", "sports"],
        out_root=tmp_path,
        dict_no_below=1,
        dict_no_above=1.0,
        coherence_reference="wikipedia",
        coherence_reference_path=reference_path,
    )

    assert len(captured["build_calls"]) == 1
    assert captured["build_calls"][0]["target_words"] == {
        "wiki_all",
        "wiki_sports",
        "wiki_shared",
    }
    assert len(captured["score_calls"]) == 2


def test_run_topic_coherence_analysis_parallelizes_topic_words_and_scoring(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from src.evaluation.word_based import metrics as metrics_module

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T01:45:00+00:00")
    reference_path = tmp_path / "enwiki-tokenized.jsonl"
    reference_path.write_text("", encoding="utf-8")
    captured: dict[str, object] = {
        "collect_categories": [],
        "score_categories": [],
        "build_calls": [],
    }

    def _collect_pending_word_based_group(**kwargs):
        task = kwargs["task"]
        captured["collect_categories"].append(task.category)
        return metrics_module.PendingWordBasedGroup(
            data_run=task.data_run,
            model=task.model,
            num_topics=task.num_topics,
            category=task.category,
            iterations=[
                metrics_module.PendingWordBasedIteration(
                    iteration=0,
                    topic_words=[
                        [(f"wiki_{task.category}", 1.0)],
                        [("wiki_shared", 0.5)],
                    ],
                )
            ],
            topic_word_source="sentence_topic_proxy_npmi",
            proxy_word_score_mode="word_npmi",
            proxy_word_score_definition="PMI normalized by -log p(w)",
        )

    def _score_pending_word_based_group(**kwargs):
        group = kwargs["group"]
        captured["score_categories"].append(group.category)
        return metrics_module.ScoredWordBasedGroup(
            group=group,
            per_iter_metrics=[{"coherence": 0.5, "diversity": 1.0, "num_topics": 2.0}],
            per_iter_topic_words=[
                {
                    "iteration": 0,
                    "topics": metrics_module.serialize_topic_words(
                        group.iterations[0].topic_words
                    ),
                }
            ],
            used_iterations=[0],
        )

    def _build_shared_reference_counts(**kwargs):
        captured["build_calls"].append(kwargs)
        return SimpleNamespace(num_docs=10, vocab_size=len(kwargs["target_words"]))

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._collect_pending_word_based_group",
        _collect_pending_word_based_group,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._score_pending_word_based_group",
        _score_pending_word_based_group,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_shared_reference_counts",
        _build_shared_reference_counts,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all", "sports"],
        out_root=tmp_path,
        dict_no_below=1,
        dict_no_above=1.0,
        coherence_reference="wikipedia",
        coherence_reference_path=reference_path,
        coherence_topic_word_workers=2,
        coherence_score_workers=2,
    )

    assert set(captured["collect_categories"]) == {"all", "sports"}
    assert set(captured["score_categories"]) == {"all", "sports"}
    assert len(captured["build_calls"]) == 1
    assert captured["build_calls"][0]["target_words"] == {
        "wiki_all",
        "wiki_sports",
        "wiki_shared",
    }


def test_run_topic_coherence_analysis_requires_wikipedia_reference_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="coherence_reference_path is required"):
        run_topic_coherence_analysis(
            models=["vmf"],
            dataset="dummy",
            data_runs=["default"],
            iterations=[0],
            num_topics=2,
            categories=["all"],
            out_root=tmp_path,
            coherence_reference="wikipedia",
        )


def test_run_topic_coherence_analysis_records_doc_npmi_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T02:00:00+00:00")
    dictionary = Dictionary([["alpha"], ["beta"]])

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    _patch_vmf_proxy_topic_words(monkeypatch)
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        lambda **_kwargs: {"coherence": -0.1, "diversity": 1.0},
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        coherence="doc_npmi",
    )

    metrics_path = next(
        (tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all").glob(
            "it0__k2__vmf__*/exec_*/metrics_agg.json"
        )
    )
    metrics_meta, _metrics_results = read_evaluation_json(metrics_path)
    assert metrics_meta["coherence"]["metric"] == "doc_npmi"
    assert (
        metrics_meta["coherence"]["definition"]
        == "Average pairwise NPMI over top-N topic words using document-level boolean co-occurrence."
    )
    assert metrics_meta["coherence"]["cooccurrence_unit"] == "document"
    assert metrics_meta["coherence"]["zero_cooccurrence_policy"] == "minus_one"

    assert not (tmp_path / "summary_metrics.csv").exists()
    assert not (tmp_path / "summary_metrics.json").exists()


def test_run_topic_coherence_analysis_aligns_sentlda_to_preprocessed_docs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    generic_dictionary = Dictionary([["alpha"], ["beta"]])
    sentlda_preprocessed_path = tmp_path / "infer" / "preprocessed_corpus.pkl"
    save_pickle(
        [
            PreprocessedDocument(
                raw_text="alpha",
                sentences_raw=["alpha"],
                sentences_tokenized=[["alpha"]],
                sentences_joined=["alpha"],
                document_tokens=["alpha"],
            ),
            PreprocessedDocument(
                raw_text="beta",
                sentences_raw=["beta"],
                sentences_tokenized=[["beta"]],
                sentences_joined=["beta"],
                document_tokens=["beta"],
            ),
        ],
        sentlda_preprocessed_path,
    )

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], [], ["beta"]],
            generic_dictionary,
            [[(0, 1)], [], [(1, 1)]],
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_preprocessed_corpus_path",
        lambda **_kwargs: sentlda_preprocessed_path,
    )
    captured: dict[str, int] = {}

    def _extract_topic_words(**kwargs):
        captured["sentence_docs"] = len(kwargs["sentence_bow_by_doc"])
        return TopicWordsResult(
            topic_words=[[("alpha", 1.0)], [("beta", 0.5)]],
            topic_word_source="sentence_topic_proxy_npmi",
            score_mode="word_npmi",
            score_definition="PMI normalized by -log p(w)",
        )

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.extract_topic_words_from_sentence_topic_npmi",
        _extract_topic_words,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.load_sentence_topics",
        lambda **_kwargs: [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
    )

    def _evaluate_topic_words(**kwargs):
        captured["coherence_docs"] = len(kwargs["texts"])
        return {"coherence": 0.31, "diversity": 0.9}

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        _evaluate_topic_words,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {
            "model_key": kwargs["model"],
            "metadata_path": str(tmp_path / kwargs["model"] / "metadata.json"),
            "runner_family": kwargs["model"],
        },
    )

    run_topic_coherence_analysis(
        models=["sentlda"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        proxy_word_score_mode="word_npmi",
    )

    assert captured == {"sentence_docs": 2, "coherence_docs": 2}


def test_run_topic_coherence_analysis_does_not_update_latest_pointer_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    dictionary = Dictionary([["alpha"], ["beta"]])

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    _patch_vmf_proxy_topic_words(monkeypatch)
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        lambda **_kwargs: {"coherence": 0.42, "diversity": 1.0},
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    _set_fixed_now(monkeypatch, "2026-04-13T00:00:00+00:00")
    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
    )

    display_key = next(
        (tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all").glob(
            "it0__k2__vmf__*"
        )
    ).name
    latest_pointer_path = (
        tmp_path / "latest" / "dummy" / "default" / "all" / display_key / "CURRENT.json"
    )
    before_payload = load_json(latest_pointer_path)

    _set_fixed_now(monkeypatch, "2026-04-13T01:00:00+00:00")
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.save_json",
        lambda _payload, path: (
            (_ for _ in ()).throw(RuntimeError("metadata write failed"))
            if Path(path).name == "metadata.json"
            else None
        ),
    )

    with pytest.raises(RuntimeError, match="metadata write failed"):
        run_topic_coherence_analysis(
            models=["vmf"],
            dataset="dummy",
            data_runs=["default"],
            iterations=[0],
            num_topics=2,
            categories=["all"],
            out_root=tmp_path,
        )

    after_payload = load_json(latest_pointer_path)
    assert after_payload == before_payload
    assert after_payload["execution_id"] == "exec_20260413T000000Z"


def test_run_topic_coherence_analysis_custom_out_root_keeps_direct_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dictionary = Dictionary([["alpha"], ["beta"]])

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_corpus_bundle",
        lambda **_kwargs: (
            [["alpha"], ["beta"]],
            dictionary,
            [[(0, 1)], [(1, 1)]],
        ),
    )
    _patch_vmf_proxy_topic_words(monkeypatch)
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.evaluate_topic_words",
        lambda **_kwargs: {"coherence": 0.42, "diversity": 1.0},
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    run_topic_coherence_analysis(
        models=["vmf"],
        dataset="dummy",
        data_runs=["default"],
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
    )

    metrics_path = next(
        (tmp_path / "dummy" / "default" / "all").glob(
            "it0__k2__vmf__*/metrics_agg.json"
        )
    )
    assert metrics_path.exists()
    assert (metrics_path.parent / "topic_words_topk.json").exists()
    assert (metrics_path.parent / "metadata.json").exists()
    assert not (tmp_path / "archive").exists()
    assert not (tmp_path / "latest").exists()

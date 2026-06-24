from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.core.artifacts import load_json, save_pickle
from src.core.paths import resolve_topic_count_analysis_dir
from src.evaluation.diagnostics.topic_count_diagnostics import (
    run_topic_count_perplexity_analysis,
)
from src.evaluation.reporting import read_evaluation_json


class _FakeEncoder:
    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(
        self,
        sentences,
        *,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        _ = (batch_size, show_progress_bar)
        vectors = []
        for sentence in sentences:
            vectors.append([1.0, 0.0] if str(sentence) == "a" else [0.0, 1.0])
        return np.asarray(vectors, dtype=float)


def test_run_topic_count_perplexity_analysis_writes_summary_reports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metrics_dir = (
        tmp_path
        / "experiments"
        / "dummy"
        / "default"
        / "vmf_sentence_lda"
        / "all"
        / "it0__k10__abcd1234"
    )
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        '{"avg_log_likelihood": -1.25, "perplexity": 3.49, "elapsed_sec": 12.0}',
        encoding="utf-8",
    )
    (metrics_dir / "metadata.json").write_text(
        (
            '{"schema":"vmf_artifact_metadata","axes":{"model_family":"vmf_sentence_lda",'
            '"algorithm_variant":"mixture-2","encoder_model":"intfloat/e5-base",'
            '"embedding_preprocess_variant":"none","iteration":0,"num_topics":10,'
            '"category":"all","data_run":"default"}}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.topic_count_diagnostics._start_execution",
        lambda: ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
    )

    out_path = run_topic_count_perplexity_analysis(
        dataset="dummy",
        iterations=[0],
        topics=[10],
        categories=["all"],
        eval_mode="metrics",
        results_root=tmp_path / "experiments",
        out_root=tmp_path / "analysis",
    )

    assert out_path == (
        tmp_path
        / "analysis"
        / "archive"
        / "2026-04-19"
        / "dummy"
        / "default"
        / "all"
        / out_path.parent.parent.name
        / "exec_20260419T100000Z"
        / "perplexity_summary.csv"
    )
    assert out_path.exists()
    pointer_path = (
        tmp_path
        / "analysis"
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / out_path.parent.parent.name
        / "CURRENT.json"
    )
    assert pointer_path.exists()
    pointer = load_json(pointer_path)
    assert pointer["archive_dir"] == str(out_path.parent)
    assert pointer["artifacts"] == {
        "csv": "perplexity_summary.csv",
        "json": "perplexity_summary.json",
    }
    assert (
        resolve_topic_count_analysis_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id=out_path.parent.parent.name,
            base_root=tmp_path / "analysis",
        )
        == out_path.parent
    )

    meta, results = read_evaluation_json(out_path.with_suffix(".json"))
    assert meta["task"] == "topic_count_diagnostics"
    assert meta["output_kind"] == "tabular"
    assert meta["condition_id"].startswith("it0__k10__perplexity__")
    assert meta["display_key"] == meta["condition_id"]
    assert meta["execution_id"] == "exec_20260419T100000Z"
    assert meta["model_provenance"] == [
        {
            "model": "vmf_sentence_lda",
            "data_run": "default",
            "category": "all",
            "iteration": 0,
            "num_topics": 10,
            "source_condition_id": None,
            "num_components": None,
            "embedding_variant": None,
            "eval_split": "test",
            "eval_mode": "metrics",
            "model_provenance": {
                "model_key": "vmf_sentence_lda",
                "metadata_path": str(metrics_dir / "metadata.json"),
                "artifact_metadata_schema": "vmf_artifact_metadata",
                "model_family": "vmf_sentence_lda",
                "condition_id": None,
                "condition_fingerprint": None,
                "algorithm_variant": "mixture-2",
                "encoder_model": "intfloat/e5-base",
                "embedding_preprocess_variant": "none",
            },
        }
    ]
    assert results["columns"][0] == "dataset"
    assert results["rows"][0]["perplexity"] == 3.49
    assert results["rows"][0]["train_metrics_perplexity"] == 3.49


def test_run_topic_count_perplexity_analysis_rerun_keeps_archives_and_updates_latest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metrics_dir = (
        tmp_path
        / "experiments"
        / "dummy"
        / "default"
        / "vmf_sentence_lda"
        / "all"
        / "it0__k10__abcd1234"
    )
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        '{"avg_log_likelihood": -1.25, "perplexity": 3.49, "elapsed_sec": 12.0}',
        encoding="utf-8",
    )
    (metrics_dir / "metadata.json").write_text(
        '{"schema":"vmf_artifact_metadata","axes":{"iteration":0,"num_topics":10,"category":"all","data_run":"default"}}',
        encoding="utf-8",
    )
    executions = iter(
        [
            ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
            ("2026-04-19T10:00:01+00:00", "exec_20260419T100001Z"),
        ]
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.topic_count_diagnostics._start_execution",
        lambda: next(executions),
    )

    first = run_topic_count_perplexity_analysis(
        dataset="dummy",
        iterations=[0],
        topics=[10],
        categories=["all"],
        eval_mode="metrics",
        results_root=tmp_path / "experiments",
        out_root=tmp_path / "analysis",
    )
    second = run_topic_count_perplexity_analysis(
        dataset="dummy",
        iterations=[0],
        topics=[10],
        categories=["all"],
        eval_mode="metrics",
        results_root=tmp_path / "experiments",
        out_root=tmp_path / "analysis",
    )

    assert first.parent != second.parent
    assert first.exists()
    assert second.exists()
    pointer_path = (
        tmp_path
        / "analysis"
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / first.parent.parent.name
        / "CURRENT.json"
    )
    pointer = load_json(pointer_path)
    assert pointer["execution_id"] == "exec_20260419T100001Z"
    assert (
        resolve_topic_count_analysis_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id=first.parent.parent.name,
            base_root=tmp_path / "analysis",
        )
        == second.parent
    )


def test_run_topic_count_perplexity_analysis_computes_test_predictive_perplexity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    exp_dir = (
        tmp_path
        / "experiments"
        / "dummy"
        / "default"
        / "vmf_sentence_lda"
        / "all"
        / "it0__k2__abcd1234"
    )
    exp_dir.mkdir(parents=True)
    (exp_dir / "metrics.json").write_text(
        '{"avg_log_likelihood": -9.0, "perplexity": 8103.0, "elapsed_sec": 1.0}',
        encoding="utf-8",
    )
    (exp_dir / "params.json").write_text(
        '{"pre_normalize_transform": "none"}',
        encoding="utf-8",
    )
    (exp_dir / "metadata.json").write_text(
        (
            '{"schema":"vmf_artifact_metadata","axes":{"model_family":"vmf_sentence_lda",'
            '"encoder_model":"fake-encoder","iteration":0,"num_topics":2,'
            '"category":"all","data_run":"default"}}'
        ),
        encoding="utf-8",
    )
    save_pickle(
        SimpleNamespace(
            documents=[
                SimpleNamespace(sentences_raw=["a"]),
                SimpleNamespace(sentences_raw=["b"]),
            ]
        ),
        exp_dir / "test_preprocessed.pkl",
    )
    save_pickle(np.eye(2, dtype=float), exp_dir / "topic_means.pkl")
    save_pickle(np.ones(2, dtype=float), exp_dir / "kappa_per_topic.pkl")
    save_pickle(np.ones((2, 1), dtype=float), exp_dir / "mixture_weights.pkl")
    save_pickle(
        np.eye(2, dtype=float).reshape(2, 1, 2), exp_dir / "component_means.pkl"
    )
    save_pickle(np.eye(2, dtype=float), exp_dir / "doc_topic_test_soft.pkl")
    monkeypatch.setattr(
        "src.evaluation.diagnostics.topic_count_diagnostics.build_sentence_encoder",
        lambda **_: _FakeEncoder(),
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.topic_count_diagnostics._start_execution",
        lambda: ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
    )

    out_path = run_topic_count_perplexity_analysis(
        dataset="dummy",
        iterations=[0],
        topics=[2],
        categories=["all"],
        split="test",
        eval_mode="predictive_soft_theta",
        strict=True,
        results_root=tmp_path / "experiments",
        out_root=tmp_path / "analysis",
    )

    _, results = read_evaluation_json(out_path.with_suffix(".json"))
    row = results["rows"][0]
    assert row["eval_split"] == "test"
    assert row["eval_mode"] == "predictive_soft_theta"
    assert row["num_documents"] == 2
    assert row["num_sentences"] == 2
    assert row["train_metrics_avg_log_likelihood"] == -9.0
    assert row["avg_log_likelihood"] != -9.0
    assert row["log_perplexity"] == -row["avg_log_likelihood"]
    assert row["perplexity"] == math.exp(row["log_perplexity"])
    assert row["preprocessed_path"] == str(exp_dir / "test_preprocessed.pkl")

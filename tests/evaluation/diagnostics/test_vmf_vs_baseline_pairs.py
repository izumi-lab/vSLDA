from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.artifacts import load_json, save_pickle
from src.core.paths import resolve_cross_model_pair_diagnostics_dir
from src.data.preprocessing import PreprocessedDocument
from src.evaluation.diagnostics.cross_model_pair_diagnostics import (
    run_vmf_vs_baseline_pair_analysis,
)
from src.evaluation.reporting import read_evaluation_json


def test_run_vmf_vs_baseline_pair_analysis_writes_provenance_aware_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    vmf_path = tmp_path / "results" / "experiments" / "dummy" / "vmf_doc_topic.pkl"
    baseline_path = tmp_path / "results" / "baselines" / "bleilda" / "train.pkl"
    save_pickle([[1.0, 0.0], [0.0, 1.0]], vmf_path)
    save_pickle([[0.9, 0.1], [0.85, 0.15]], baseline_path)
    (vmf_path.parent / "metadata.json").write_text(
        (
            '{"schema":"vmf_artifact_metadata","axes":{"model_family":"vmf_sentence_lda",'
            '"algorithm_variant":"mixture-2","encoder_model":"e5-base",'
            '"embedding_preprocess_variant":"none"}}'
        ),
        encoding="utf-8",
    )
    (baseline_path.parent / "metadata.json").write_text(
        (
            '{"schema":"baseline_artifact_metadata","runner_key":"bleilda",'
            '"runner_family":"bleilda","parameter_variant":"passes=40",'
            '"preprocessing_variant":"language=english","baseline_params":{"passes":40}}'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._resolve_vmf_doc_topic_path",
        lambda **_kwargs: vmf_path,
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._resolve_baseline_doc_topic_path",
        lambda **_kwargs: baseline_path,
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._load_docs",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "target_str": ["science", "science"],
                "data": ["alpha beta", "gamma delta"],
            }
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._start_execution",
        lambda: ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
    )

    out_path = run_vmf_vs_baseline_pair_analysis(
        dataset="dummy",
        category="all",
        iteration=0,
        num_topics=2,
        split="train",
        baseline="bleilda",
        k_neighbors=1,
        baseline_max=0.1,
        vmf_min=0.9,
        topn=5,
        results_root=tmp_path / "results",
        out_root=tmp_path / "analysis",
    )

    assert (
        out_path
        == tmp_path
        / "analysis"
        / "archive"
        / "2026-04-19"
        / "dummy"
        / "default"
        / "all"
        / out_path.parent.parent.name
        / "exec_20260419T100000Z"
        / "pairs_bleilda_train.json"
    )
    assert out_path.exists()
    assert out_path.with_suffix(".csv").exists()
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
    pointer = load_json(pointer_path)
    assert pointer["archive_dir"] == str(out_path.parent)
    assert pointer["artifacts"] == {
        "csv": "pairs_bleilda_train.csv",
        "json": "pairs_bleilda_train.json",
    }
    assert (
        resolve_cross_model_pair_diagnostics_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id=out_path.parent.parent.name,
            base_root=tmp_path / "analysis",
        )
        == out_path.parent
    )

    meta, results = read_evaluation_json(out_path)
    assert meta["task"] == "cross_model_pair_diagnostics"
    assert meta["data_run"] == "default"
    assert meta["condition_id"].startswith("it0__k2__bleilda__train__")
    assert meta["display_key"] == meta["condition_id"]
    assert meta["execution_id"] == "exec_20260419T100000Z"
    assert meta["pair_count"] == 1
    assert meta["seed"] == 42
    assert (
        meta["model_provenance"]["vmf_sentence_lda"]["model_family"]
        == "vmf_sentence_lda"
    )
    assert meta["model_provenance"]["bleilda"]["parameter_variant"] == "passes=40"
    assert results["pairs"][0]["baseline"] == "bleilda"
    assert results["pairs"][0]["i"] == 0
    assert results["pairs"][0]["j"] == 1


def test_run_vmf_vs_baseline_pair_analysis_rerun_keeps_archives_and_updates_latest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    vmf_path = tmp_path / "results" / "experiments" / "dummy" / "vmf_doc_topic.pkl"
    baseline_path = tmp_path / "results" / "baselines" / "bleilda" / "train.pkl"
    save_pickle([[1.0, 0.0], [0.0, 1.0]], vmf_path)
    save_pickle([[0.9, 0.1], [0.85, 0.15]], baseline_path)
    (vmf_path.parent / "metadata.json").write_text(
        '{"schema":"vmf_artifact_metadata","axes":{"model_family":"vmf_sentence_lda"}}',
        encoding="utf-8",
    )
    (baseline_path.parent / "metadata.json").write_text(
        '{"schema":"baseline_artifact_metadata","runner_key":"bleilda","runner_family":"bleilda","parameter_variant":"passes=40","preprocessing_variant":"language=english","baseline_params":{"passes":40}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._resolve_vmf_doc_topic_path",
        lambda **_kwargs: vmf_path,
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._resolve_baseline_doc_topic_path",
        lambda **_kwargs: baseline_path,
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._load_docs",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"target_str": ["science", "science"], "data": ["alpha", "beta"]}
        ),
    )
    executions = iter(
        [
            ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
            ("2026-04-19T10:00:01+00:00", "exec_20260419T100001Z"),
        ]
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._start_execution",
        lambda: next(executions),
    )

    first = run_vmf_vs_baseline_pair_analysis(
        dataset="dummy",
        category="all",
        iteration=0,
        num_topics=2,
        split="train",
        baseline="bleilda",
        k_neighbors=1,
        baseline_max=0.1,
        vmf_min=0.9,
        topn=5,
        results_root=tmp_path / "results",
        out_root=tmp_path / "analysis",
    )
    second = run_vmf_vs_baseline_pair_analysis(
        dataset="dummy",
        category="all",
        iteration=0,
        num_topics=2,
        split="train",
        baseline="bleilda",
        k_neighbors=1,
        baseline_max=0.1,
        vmf_min=0.9,
        topn=5,
        results_root=tmp_path / "results",
        out_root=tmp_path / "analysis",
    )

    assert first.parent != second.parent
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
        resolve_cross_model_pair_diagnostics_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id=first.parent.parent.name,
            base_root=tmp_path / "analysis",
        )
        == second.parent
    )


def test_run_vmf_vs_baseline_pair_analysis_aligns_legacy_bleilda_empty_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    vmf_path = tmp_path / "results" / "experiments" / "dummy" / "vmf_doc_topic.pkl"
    baseline_path = (
        tmp_path / "results" / "baselines" / "bleilda" / "params" / "lda_comp.pkl"
    )
    save_pickle(
        [[1.0, 0.0], [0.5, 0.5], [0.6, 0.4], [0.0, 1.0]],
        vmf_path,
    )
    save_pickle([[0.9, 0.1], [0.1, 0.9], [0.85, 0.15]], baseline_path)
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
                raw_text="--",
                sentences_raw=["--"],
                sentences_tokenized=[[]],
                sentences_joined=[""],
                document_tokens=[],
            ),
            PreprocessedDocument(
                raw_text="beta",
                sentences_raw=["beta"],
                sentences_tokenized=[["beta"]],
                sentences_joined=["beta"],
                document_tokens=["beta"],
            ),
            PreprocessedDocument(
                raw_text="gamma",
                sentences_raw=["gamma"],
                sentences_tokenized=[["gamma"]],
                sentences_joined=["gamma"],
                document_tokens=["gamma"],
            ),
        ],
        baseline_path.with_name("preprocessed_corpus.pkl"),
    )
    (vmf_path.parent / "metadata.json").write_text(
        '{"schema":"vmf_artifact_metadata","axes":{"model_family":"vmf_sentence_lda"}}',
        encoding="utf-8",
    )
    (baseline_path.parent / "metadata.json").write_text(
        '{"schema":"baseline_artifact_metadata","runner_key":"bleilda","runner_family":"bleilda"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._resolve_vmf_doc_topic_path",
        lambda **_kwargs: vmf_path,
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._resolve_baseline_doc_topic_path",
        lambda **_kwargs: baseline_path,
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._load_docs",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "target_str": ["x", "x", "x", "x"],
                "data": ["alpha", "--", "beta", "gamma"],
            }
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.cross_model_pair_diagnostics._start_execution",
        lambda: ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
    )

    out_path = run_vmf_vs_baseline_pair_analysis(
        dataset="dummy",
        category="all",
        iteration=0,
        num_topics=2,
        split="train",
        baseline="bleilda",
        k_neighbors=2,
        baseline_max=0.1,
        vmf_min=0.9,
        topn=5,
        results_root=tmp_path / "results",
        out_root=tmp_path / "analysis",
    )

    meta, results = read_evaluation_json(out_path)

    assert meta["legacy_bleilda_row_alignment"] is True
    assert meta["pair_count"] == 1
    assert results["pairs"][0]["i"] == 0
    assert results["pairs"][0]["j"] == 3
    assert results["pairs"][0]["text_i"] == "alpha"
    assert results["pairs"][0]["text_j"] == "gamma"

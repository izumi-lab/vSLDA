from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.artifacts import load_json, save_pickle
from src.core.paths import (
    build_baseline_dir,
    build_vmf_experiment_dir,
    resolve_sentence_topic_inspection_dir,
)
from src.data.preprocessing import PreprocessedDocument
from src.evaluation.diagnostics.sentence_topic_inspection import (
    _build_output_condition_id,
    _resolve_encoder_model_for_source,
    run_sentence_topic_inspection,
)
from src.evaluation.reporting import read_evaluation_json


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences, **kwargs) -> np.ndarray:
        batch = list(sentences)
        self.calls.append((batch, dict(kwargs)))
        mapping = {
            "alpha": np.array([1.0, 0.0], dtype=np.float64),
            "beta": np.array([0.0, 1.0], dtype=np.float64),
            "gamma": np.array([0.8, 0.2], dtype=np.float64),
        }
        return np.vstack([mapping[text] for text in batch])

    def get_sentence_embedding_dimension(self) -> int:
        return 2


def _prepare_sentence_topic_artifacts(tmp_path: Path) -> None:
    exp_dir = build_vmf_experiment_dir(
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="all",
        dataset_root=tmp_path / "results" / "experiments" / "dummy",
    )
    save_pickle(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float), exp_dir / "topic_means.pkl"
    )
    save_pickle(
        np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=float),
        exp_dir / "doc_topic_train.pkl",
    )
    save_pickle(np.asarray([8.0, 9.0], dtype=float), exp_dir / "kappa_per_topic.pkl")
    save_pickle(
        np.asarray([[1.0], [1.0]], dtype=float), exp_dir / "mixture_weights.pkl"
    )
    save_pickle(
        np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=float),
        exp_dir / "component_means.pkl",
    )
    save_pickle(np.asarray([2.0, 1.0], dtype=float), exp_dir / "topic_counts.pkl")
    (exp_dir / "params.json").write_text(
        json.dumps({"average_ll": [0.1, 0.2], "alpha": 1.0}),
        encoding="utf-8",
    )
    (exp_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema": "vmf_artifact_metadata",
                "condition_id": exp_dir.name,
                "axes": {
                    "dataset": "dummy",
                    "model_family": "vmf_sentence_lda",
                    "algorithm_variant": "mixture-1",
                    "encoder_model": "fake-model",
                    "embedding_preprocess_variant": "none",
                    "num_topics": 2,
                    "iteration": 0,
                    "category": "all",
                    "data_run": "default",
                },
            }
        ),
        encoding="utf-8",
    )

    gaussian_dir = build_baseline_dir(
        model="sentence_gaussianlda",
        split_root="params",
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="all",
        baseline_root=tmp_path / "results" / "baselines",
    )
    save_pickle(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        gaussian_dir / "table_means.pkl",
    )
    save_pickle(
        np.asarray([np.eye(2), np.eye(2)], dtype=float),
        gaussian_dir / "table_cholesky_ltriangular_mat.pkl",
    )
    save_pickle(
        np.asarray([0.0, 0.0], dtype=float), gaussian_dir / "log_determinants.pkl"
    )
    (gaussian_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema": "baseline_artifact_metadata",
                "runner_key": "sentence_gaussianlda",
                "runner_family": "sentence_gaussianlda",
                "data_run": "default",
                "condition_id": gaussian_dir.parent.name,
                "parameter_variant": "default",
                "preprocessing_variant": "language=english",
                "dataset": "dummy",
                "category": "all",
                "num_topics": 2,
                "iteration": 0,
                "baseline_params": {"covariance": "full"},
            }
        ),
        encoding="utf-8",
    )


def _prepare_sentlda_inspection_artifacts(tmp_path: Path) -> None:
    params_dir = build_baseline_dir(
        model="sentlda",
        split_root="params",
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="all",
        baseline_root=tmp_path / "results" / "baselines",
    )
    condition_dir = params_dir.parent
    infer_dir = condition_dir / "infer"
    params_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(
        np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float),
        params_dir / "table_counts_per_doc.pkl",
    )
    save_pickle(
        [
            np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=float),
            np.asarray([[0.4, 0.6]], dtype=float),
        ],
        params_dir / "all_sentence_topic_soft.pkl",
    )
    save_pickle(
        [
            np.asarray([[-1.0, -0.3], [-2.0, -8.0]], dtype=float),
            np.asarray([[-3.0, -0.7]], dtype=float),
        ],
        params_dir / "all_sentence_topic_loglik.pkl",
    )
    save_pickle(
        [
            PreprocessedDocument(
                raw_text="alpha / dropped / beta",
                sentences_raw=["alpha", "beta"],
                sentences_tokenized=[
                    ["alpha"],
                    ["beta"] * 10,
                ],
                sentences_joined=["alpha", "beta"],
                document_tokens=["alpha", *["beta"] * 10],
            ),
            PreprocessedDocument(
                raw_text="gamma",
                sentences_raw=["gamma"],
                sentences_tokenized=[["gamma"]],
                sentences_joined=["gamma"],
                document_tokens=["gamma"],
            ),
        ],
        params_dir / "preprocessed_corpus.pkl",
    )
    (params_dir / "params.json").write_text(
        json.dumps({"average_ll": [-4.0, -3.5]}),
        encoding="utf-8",
    )
    (condition_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema": "baseline_artifact_metadata",
                "runner_key": "sentlda",
                "runner_family": "sentlda",
                "data_run": "default",
                "condition_id": condition_dir.name,
                "parameter_variant": "default",
                "preprocessing_variant": "language=english",
                "dataset": "dummy",
                "category": "all",
                "num_topics": 2,
                "iteration": 0,
                "baseline_params": {"alpha": 0.1},
            }
        ),
        encoding="utf-8",
    )


def _prepare_senclu_inspection_artifacts(tmp_path: Path) -> None:
    params_dir = build_baseline_dir(
        model="senclu",
        split_root="params",
        dataset="dummy",
        iteration=0,
        num_topics=2,
        category="all",
        baseline_root=tmp_path / "results" / "baselines",
    )
    condition_dir = params_dir.parent
    infer_dir = condition_dir / "infer"
    params_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(
        np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float),
        params_dir / "all.pkl",
    )
    save_pickle(
        [
            np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=float),
            np.asarray([[0.4, 0.6]], dtype=float),
        ],
        params_dir / "all_sentence_topic_soft.pkl",
    )
    save_pickle(
        [
            PreprocessedDocument(
                raw_text="alpha / beta",
                sentences_raw=["alpha", "beta"],
                sentences_tokenized=[["alpha"], ["beta"]],
                sentences_joined=["alpha", "beta"],
                document_tokens=["alpha", "beta"],
            ),
            PreprocessedDocument(
                raw_text="gamma",
                sentences_raw=["gamma"],
                sentences_tokenized=[["gamma"]],
                sentences_joined=["gamma"],
                document_tokens=["gamma"],
            ),
        ],
        params_dir / "preprocessed_corpus.pkl",
    )
    (params_dir / "params.json").write_text(
        json.dumps({"average_ll": [-2.0, -1.5]}),
        encoding="utf-8",
    )
    (condition_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema": "baseline_artifact_metadata",
                "runner_key": "senclu",
                "runner_family": "senclu",
                "data_run": "default",
                "condition_id": condition_dir.name,
                "parameter_variant": "default",
                "preprocessing_variant": "language=english",
                "dataset": "dummy",
                "category": "all",
                "num_topics": 2,
                "iteration": 0,
                "baseline_params": {"soft_temperature": 1.0},
            }
        ),
        encoding="utf-8",
    )


def _patch_sentence_topic_runtime(monkeypatch, fake_encoder: _FakeEncoder) -> None:
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection.load_filtered_split_sentences",
        lambda *args, **kwargs: ["alpha", "beta", "gamma"],
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection.build_sentence_encoder",
        lambda **kwargs: fake_encoder,
    )

    def _fake_line_plot(values, out_path):
        _ = values
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        return True

    def _fake_doc_plot(doc_topics, out_path, *, seed=None, title=None):
        _ = (doc_topics, seed, title)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        return True

    def _fake_sphere_plot(**kwargs):
        out_path = kwargs["out_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        return {"num_embeddings_plotted": 3, "num_topics": 2}

    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection.plot_average_ll",
        _fake_line_plot,
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection.plot_doc_topics",
        _fake_doc_plot,
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection.plot_embeddings_on_sphere_3d",
        _fake_sphere_plot,
    )


def test_resolve_encoder_model_uses_source_metadata(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "metadata.json").write_text(
        json.dumps(
            {
                "encoder_config": {
                    "model_name": "sentence-transformers/all-mpnet-base-v2",
                    "embedding_variant": "mpnet",
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        _resolve_encoder_model_for_source(
            requested_encoder_model=None,
            requested_embedding_variant="mpnet",
            exp_dir=exp_dir,
        )
        == "sentence-transformers/all-mpnet-base-v2"
    )


def test_resolve_encoder_model_falls_back_to_variant_default(tmp_path: Path) -> None:
    assert (
        _resolve_encoder_model_for_source(
            requested_encoder_model=None,
            requested_embedding_variant="bge",
            exp_dir=tmp_path,
        )
        == "baai/bge-base-en-v1.5"
    )


def test_resolve_encoder_model_rejects_variant_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Encoder and embedding_variant mismatch"):
        _resolve_encoder_model_for_source(
            requested_encoder_model="sentence-transformers/all-mpnet-base-v2",
            requested_embedding_variant="bge",
            exp_dir=tmp_path,
        )


def test_output_condition_id_includes_embedding_variant_label() -> None:
    condition_id, _ = _build_output_condition_id(
        model="vmf_sentence_lda",
        dataset="dummy",
        data_run="default",
        category="all",
        iteration=0,
        num_topics=2,
        split="train",
        encoder_model="sentence-transformers/all-mpnet-base-v2",
        gaussian_topk=True,
        max_points=2000,
        source_condition_id=None,
        embedding_variant="mpnet",
        num_components=None,
        gaussian_condition_id=None,
        gaussian_embedding_variant="bge",
        gaussian_num_components=None,
    )

    assert "__mpnet__" in condition_id
    assert "__gaussian-bge__" in condition_id


def test_run_sentence_topic_inspection_writes_payloads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _prepare_sentence_topic_artifacts(tmp_path)
    fake_encoder = _FakeEncoder()
    _patch_sentence_topic_runtime(monkeypatch, fake_encoder)
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection._start_execution",
        lambda: ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
    )

    output_dir = run_sentence_topic_inspection(
        dataset="dummy",
        categories=["all"],
        iterations=[0],
        num_topics_list=[2],
        top_k=2,
        encoder_model="fake-model",
        gaussian_topk=True,
        device="cpu",
        results_root=tmp_path / "results",
        out_root=tmp_path / "viz",
        show_progress=False,
    )

    assert output_dir == (
        tmp_path
        / "viz"
        / "archive"
        / "2026-04-19"
        / "dummy"
        / "default"
        / "all"
        / output_dir.parent.name
        / "exec_20260419T100000Z"
    )
    assert not (
        tmp_path
        / "viz"
        / "dummy"
        / "default"
        / "sentence_topic_inspection_summary.json"
    ).exists()
    assert (output_dir / "top_sentences_loglik.json").exists()
    assert (output_dir / "top_sentences_gaussian_loglik.json").exists()
    assert (output_dir / "avg_ll.json").exists()
    assert (output_dir / "doc_topic_tsne.json").exists()
    assert (output_dir / "embeddings_on_sphere_3d.json").exists()
    assert (output_dir / "kappa_per_topic.json").exists()
    pointer_path = (
        tmp_path
        / "viz"
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / output_dir.parent.name
        / "CURRENT.json"
    )
    pointer = load_json(pointer_path)
    assert pointer["archive_dir"] == str(output_dir)
    assert pointer["execution_id"] == "exec_20260419T100000Z"
    assert (
        resolve_sentence_topic_inspection_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id=output_dir.parent.name,
            base_root=tmp_path / "viz",
        )
        == output_dir
    )

    avg_meta, avg_results = read_evaluation_json(output_dir / "avg_ll.json")
    assert avg_meta["artifact"] == "avg_ll_plot"
    assert avg_meta["execution_id"] == "exec_20260419T100000Z"
    assert avg_results["point_count"] == 2

    kappa_meta, kappa_results = read_evaluation_json(
        output_dir / "kappa_per_topic.json"
    )
    assert kappa_meta["artifact"] == "kappa_per_topic"
    assert kappa_results["kappa_per_topic"]["0"] == 8.0

    top_meta, top_results = read_evaluation_json(
        output_dir / "top_sentences_loglik.json"
    )
    assert top_meta["artifact"] == "top_sentences_loglik"
    assert top_results["topics"]["0"][0]["sentence"] == "alpha"

    gaussian_meta, gaussian_results = read_evaluation_json(
        output_dir / "top_sentences_gaussian_loglik.json"
    )
    assert gaussian_meta["artifact"] == "top_sentences_gaussian_loglik"
    assert gaussian_meta["model_provenance"]["runner_key"] == "sentence_gaussianlda"
    assert gaussian_results["topics"]["1"][0]["sentence"] == "beta"

    assert fake_encoder.calls == [
        (
            ["alpha", "beta", "gamma"],
            {"batch_size": 64, "show_progress_bar": False},
        )
    ]


def test_run_sentence_topic_inspection_rerun_keeps_archives_and_updates_latest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _prepare_sentence_topic_artifacts(tmp_path)
    fake_encoder = _FakeEncoder()
    _patch_sentence_topic_runtime(monkeypatch, fake_encoder)
    executions = iter(
        [
            ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
            ("2026-04-19T10:00:01+00:00", "exec_20260419T100001Z"),
        ]
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection._start_execution",
        lambda: next(executions),
    )

    first_dir = run_sentence_topic_inspection(
        dataset="dummy",
        categories=["all"],
        iterations=[0],
        num_topics_list=[2],
        top_k=2,
        encoder_model="fake-model",
        gaussian_topk=True,
        device="cpu",
        results_root=tmp_path / "results",
        out_root=tmp_path / "viz",
        show_progress=False,
    )
    second_dir = run_sentence_topic_inspection(
        dataset="dummy",
        categories=["all"],
        iterations=[0],
        num_topics_list=[2],
        top_k=2,
        encoder_model="fake-model",
        gaussian_topk=True,
        device="cpu",
        results_root=tmp_path / "results",
        out_root=tmp_path / "viz",
        show_progress=False,
    )
    assert first_dir != second_dir
    pointer_path = (
        tmp_path
        / "viz"
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / first_dir.parent.name
        / "CURRENT.json"
    )
    pointer = load_json(pointer_path)
    assert pointer["execution_id"] == "exec_20260419T100001Z"
    assert (
        resolve_sentence_topic_inspection_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id=first_dir.parent.name,
            base_root=tmp_path / "viz",
        )
        == second_dir
    )


def test_run_sentence_topic_inspection_supports_sentlda_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _prepare_sentlda_inspection_artifacts(tmp_path)
    fake_encoder = _FakeEncoder()
    _patch_sentence_topic_runtime(monkeypatch, fake_encoder)
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection.load_filtered_split_sentences",
        lambda *args, **kwargs: pytest.fail(
            "sentLDA inspection should use the persisted preprocessed corpus"
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection._start_execution",
        lambda: ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
    )

    output_dir = run_sentence_topic_inspection(
        model="sentlda",
        dataset="dummy",
        categories=["all"],
        iterations=[0],
        num_topics_list=[2],
        top_k=2,
        split="train",
        results_root=tmp_path / "results",
        out_root=tmp_path / "viz",
        show_progress=False,
    )

    assert (output_dir / "avg_ll.json").exists()
    assert (output_dir / "doc_topic_tsne.json").exists()
    assert (output_dir / "top_sentences_loglik.json").exists()
    assert not (output_dir / "embeddings_on_sphere_3d.json").exists()
    assert fake_encoder.calls == []

    top_meta, top_results = read_evaluation_json(
        output_dir / "top_sentences_loglik.json"
    )
    assert top_meta["top_sentence_method"] == "sentlda_token_mean_loglik"
    assert top_meta["score_definition"] == "mean_token log p(token|topic, rest)"
    assert top_meta["score_normalization"] == "per_token"
    assert top_meta["source_score_definition"] == "log p(sentence|topic, rest)"
    assert top_meta["sentence_source_artifact_path"].endswith(
        "params/preprocessed_corpus.pkl"
    )
    assert top_results["topics"]["0"][0]["sentence"] == "beta"
    assert top_results["topics"]["0"][0]["score"] == -0.2


def test_run_sentence_topic_inspection_supports_senclu_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _prepare_senclu_inspection_artifacts(tmp_path)
    fake_encoder = _FakeEncoder()
    _patch_sentence_topic_runtime(monkeypatch, fake_encoder)
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection.load_filtered_split_sentences",
        lambda *args, **kwargs: pytest.fail(
            "SenClu inspection should use the persisted preprocessed corpus"
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.diagnostics.sentence_topic_inspection._start_execution",
        lambda: ("2026-04-19T10:00:00+00:00", "exec_20260419T100000Z"),
    )

    output_dir = run_sentence_topic_inspection(
        model="senclu",
        dataset="dummy",
        categories=["all"],
        iterations=[0],
        num_topics_list=[2],
        top_k=2,
        split="train",
        results_root=tmp_path / "results",
        out_root=tmp_path / "viz",
        show_progress=False,
    )

    assert (output_dir / "top_sentences_loglik.json").exists()
    assert fake_encoder.calls == []

    top_meta, top_results = read_evaluation_json(
        output_dir / "top_sentences_loglik.json"
    )
    assert top_meta["top_sentence_method"] == "senclu_sentence_given_topic"
    assert top_meta["score_definition"] == (
        "p(sentence|topic) from column-normalized p(topic|sentence, document)"
    )
    assert top_meta["score_normalization"] == "column_over_sentences"
    assert top_meta["source_score_definition"] == "p(topic|sentence, document)"
    assert top_meta["sentence_source_artifact_path"].endswith(
        "params/preprocessed_corpus.pkl"
    )
    assert top_meta["source_artifact_path"].endswith(
        "params/all_sentence_topic_soft.pkl"
    )
    assert top_results["topics"]["0"][0]["sentence"] == "beta"
    assert top_results["topics"]["0"][0]["score"] == pytest.approx(0.8 / 1.3)
    assert top_results["topics"]["1"][0]["sentence"] == "alpha"
    assert top_results["topics"]["1"][0]["score"] == pytest.approx(0.9 / 1.7)
    assert top_results["topics"]["1"][0]["sentence"] == "alpha"

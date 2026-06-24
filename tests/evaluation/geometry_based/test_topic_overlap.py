from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.core.artifacts import load_json
from src.evaluation.geometry_based.metrics import run_topic_overlap_analysis
from src.evaluation.reporting import read_evaluation_json


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
        "src.evaluation.geometry_based.metrics.datetime",
        _FixedDateTime,
    )


def test_run_topic_overlap_analysis_writes_summary_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _fake_load_topic_vectors(
        model: str,
        dataset: str,
        iteration: int,
        num_topics: int,
        category: str,
        data_run: str = "default",
        embedding_variant: str | None = None,
    ) -> np.ndarray:
        _ = (
            model,
            dataset,
            iteration,
            num_topics,
            category,
            data_run,
            embedding_variant,
        )
        return np.asarray([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.load_topic_vectors",
        _fake_load_topic_vectors,
    )

    summary_path = run_topic_overlap_analysis(
        models=["vmf"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
    )

    assert summary_path == tmp_path / "summary.csv"
    assert summary_path.exists()
    meta, results = read_evaluation_json(tmp_path / "summary.json")
    assert meta["task"] == "geometry_based_metrics_summary"
    assert meta["data_runs"] == ["default"]
    assert meta["embedding_variant"] is None
    assert meta["model_provenance"][0]["model"] == "vmf"
    assert meta["model_provenance"][0]["data_run"] == "default"
    assert meta["model_provenance"][0]["category"] == "all"
    assert meta["model_provenance"][0]["model_provenance"]["model_key"] == (
        "vmf_sentence_lda"
    )
    assert meta["model_provenance"][0]["model_provenance"]["metadata_path"].endswith(
        "metadata.json"
    )
    assert results["columns"] == [
        "dataset",
        "data_run",
        "num_topics",
        "category",
        "model",
        "diversity_mean",
        "diversity_std",
        "max_cosine_mean",
        "max_cosine_std",
    ]
    assert results["rows"][0]["dataset"] == "dummy"
    assert results["rows"][0]["data_run"] == "default"
    assert results["rows"][0]["model"] == "vmf"


def test_run_topic_overlap_analysis_persists_model_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.load_topic_vectors",
        lambda **_kwargs: np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.build_result_dir",
        lambda **_kwargs: tmp_path / "baseline" / "params",
    )
    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.load_model_provenance",
        lambda metadata_dir, model_key: {
            "model_key": model_key,
            "metadata_path": str(metadata_dir / "metadata.json"),
            "runner_family": "sentence_gaussianlda",
            "parameter_variant": "soft_temperature=0.8",
        },
    )

    run_topic_overlap_analysis(
        models=["gaussian"],
        dataset="dummy",
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
    )

    out_path = next(
        (tmp_path / "dummy" / "default" / "all").glob(
            "it0__k2__*/overlap_metrics_agg.json"
        )
    )
    meta, _results = read_evaluation_json(out_path)
    assert meta["data_run"] == "default"
    assert meta["condition_id"].startswith("it0__k2__")
    assert meta["model_provenance"] == {
        "model_key": "sentence_gaussianlda",
        "metadata_path": str(tmp_path / "baseline" / "params" / "metadata.json"),
        "runner_family": "sentence_gaussianlda",
        "parameter_variant": "soft_temperature=0.8",
    }


def test_run_topic_overlap_analysis_uses_embedding_variant_for_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.load_topic_vectors",
        lambda **_kwargs: np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )

    def _fake_build_result_dir(**kwargs):
        captured.append((kwargs["model"], kwargs.get("embedding_variant")))
        return tmp_path / str(kwargs["model"]) / "params"

    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.build_result_dir",
        _fake_build_result_dir,
    )
    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.load_model_provenance",
        lambda metadata_dir, model_key: {
            "model_key": model_key,
            "metadata_path": str(metadata_dir / "metadata.json"),
        },
    )

    run_topic_overlap_analysis(
        models=["vmf", "gaussian"],
        dataset="dummy",
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        encoder_model="sentence-transformers/all-mpnet-base-v2",
    )

    assert ("vmf", "mpnet") in captured
    assert ("gaussian", "mpnet_raw") in captured


def test_run_topic_overlap_analysis_default_out_root_uses_archive_latest_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics._uses_default_output_layout",
        lambda _out_root: True,
    )
    _set_fixed_now(monkeypatch, "2026-04-13T00:00:00+00:00")
    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.load_topic_vectors",
        lambda **_kwargs: np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.build_result_dir",
        lambda **_kwargs: tmp_path / "vmf",
    )
    monkeypatch.setattr(
        "src.evaluation.geometry_based.metrics.load_model_provenance",
        lambda metadata_dir, model_key: {
            "model_key": model_key,
            "metadata_path": str(metadata_dir / "metadata.json"),
            "runner_family": "vmf_sentence_lda",
            "parameter_variant": "encoder=mpnet",
        },
    )

    run_topic_overlap_analysis(
        models=["vmf"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        save_per_iter_artifacts=True,
    )

    archive_root = tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all"
    out_path = next(
        archive_root.glob(
            "it0__k2__vmf__*/exec_20260413T000000Z/overlap_metrics_agg.json"
        )
    )
    meta, _results = read_evaluation_json(out_path)
    assert meta["data_run"] == "default"
    assert meta["display_key"] == meta["condition_id"]
    assert meta["execution_id"] == "exec_20260413T000000Z"
    assert meta["archive_dir"].endswith("/exec_20260413T000000Z")
    assert meta["latest_dir"].endswith(f"/{meta['display_key']}")
    assert meta["model_provenance"] == {
        "model_key": "vmf_sentence_lda",
        "metadata_path": str(tmp_path / "vmf" / "metadata.json"),
        "runner_family": "vmf_sentence_lda",
        "parameter_variant": "encoder=mpnet",
    }

    metadata_path = out_path.parent / "metadata.json"
    assert metadata_path.exists()
    assert (out_path.parent / "iter0" / "topic_cosine_similarity.csv").exists()
    assert (out_path.parent / "iter0" / "topic_cosine_similarity.png").exists()
    assert (out_path.parent / "iter1" / "topic_cosine_similarity.csv").exists()
    assert (out_path.parent / "iter1" / "topic_cosine_similarity.png").exists()

    latest_pointer = load_json(
        tmp_path
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / meta["display_key"]
        / "CURRENT.json"
    )
    assert latest_pointer["display_key"] == meta["display_key"]
    assert latest_pointer["execution_id"] == "exec_20260413T000000Z"
    assert latest_pointer["artifacts"]["metrics"] == "overlap_metrics_agg.json"
    assert latest_pointer["artifacts"]["metadata"] == "metadata.json"
    assert (
        latest_pointer["artifacts"]["topic_cosine_similarity_csv_iter0"]
        == "iter0/topic_cosine_similarity.csv"
    )
    assert (
        latest_pointer["artifacts"]["topic_cosine_similarity_png_iter1"]
        == "iter1/topic_cosine_similarity.png"
    )

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.core.artifacts import load_json
from src.evaluation.word_based.label_profile import run_label_topic_profile


def _stub_label_profile_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    doc_topic_path = tmp_path / "bleilda" / "table_counts_per_doc.pkl"
    doc_topic_path.parent.mkdir(parents=True, exist_ok=True)
    doc_topic_path.write_bytes(b"x")
    (doc_topic_path.parent / "metadata.json").write_text(
        (
            '{"schema":"baseline_artifact_metadata","runner_key":"bleilda",'
            '"runner_family":"bleilda","parameter_variant":"passes=40",'
            '"preprocessing_variant":"language=english","baseline_params":{"passes":40}}'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.evaluation.word_based.label_profile._resolve_doc_topic_path",
        lambda **_kwargs: doc_topic_path,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.label_profile._load_doc_topics",
        lambda _path: np.asarray([[0.8, 0.2], [0.1, 0.9]]),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.label_profile._load_labels",
        lambda *_args, **_kwargs: ["a", "b"],
    )
    return doc_topic_path


def _set_fixed_now(
    monkeypatch: pytest.MonkeyPatch,
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
        "src.evaluation.word_based.label_profile.datetime",
        _FixedDateTime,
    )


def test_run_label_topic_profile_persists_model_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    doc_topic_path = _stub_label_profile_inputs(monkeypatch, tmp_path)

    report = run_label_topic_profile(
        model="bleilda",
        dataset="dummy",
        category="all",
        split="train",
        iteration=0,
        num_topics=2,
        results_root=tmp_path,
    )

    assert report["_meta"]["model_provenance"] == {
        "model_key": "bleilda",
        "metadata_path": str(doc_topic_path.parent / "metadata.json"),
        "artifact_metadata_schema": "baseline_artifact_metadata",
        "runner_key": "bleilda",
        "runner_family": "bleilda",
        "data_run": None,
        "condition_id": None,
        "condition_fingerprint": None,
        "parameter_variant": "passes=40",
        "preprocessing_variant": "language=english",
        "baseline_params": {"passes": 40},
    }
    assert report["_meta"]["data_run"] == "default"
    assert report["_meta"]["condition_id"] == "bleilda_train_k2_it0"
    archive_date = str(report["_meta"]["started_at"]).split("T", 1)[0]
    archive_root = (
        tmp_path
        / "topic_analysis"
        / "label_profile"
        / "archive"
        / archive_date
        / "dummy"
        / "default"
        / "all"
        / "bleilda_train_k2_it0"
    )
    json_paths = list(archive_root.glob("exec_*/label_topic_profile.json"))
    csv_paths = list(archive_root.glob("exec_*/label_topic_profile.csv"))
    metadata_paths = list(archive_root.glob("exec_*/metadata.json"))
    assert len(json_paths) == 1
    assert len(csv_paths) == 1
    assert len(metadata_paths) == 1

    latest_pointer = (
        tmp_path
        / "topic_analysis"
        / "label_profile"
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / "bleilda_train_k2_it0"
        / "CURRENT.json"
    )
    assert latest_pointer.exists()
    pointer_payload = load_json(latest_pointer)
    assert pointer_payload["display_key"] == "bleilda_train_k2_it0"
    assert pointer_payload["artifacts"]["json"] == "label_topic_profile.json"


def test_run_label_topic_profile_does_not_update_latest_pointer_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_label_profile_inputs(monkeypatch, tmp_path)
    _set_fixed_now(monkeypatch, "2026-04-10T00:00:00+00:00")

    run_label_topic_profile(
        model="bleilda",
        dataset="dummy",
        category="all",
        split="train",
        iteration=0,
        num_topics=2,
        results_root=tmp_path,
    )

    latest_pointer = (
        tmp_path
        / "topic_analysis"
        / "label_profile"
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / "bleilda_train_k2_it0"
        / "CURRENT.json"
    )
    before_payload = load_json(latest_pointer)

    _set_fixed_now(monkeypatch, "2026-04-10T01:00:00+00:00")
    monkeypatch.setattr(
        "src.evaluation.word_based.label_profile.write_csv_rows",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("csv write failed")),
    )

    with pytest.raises(RuntimeError, match="csv write failed"):
        run_label_topic_profile(
            model="bleilda",
            dataset="dummy",
            category="all",
            split="train",
            iteration=0,
            num_topics=2,
            results_root=tmp_path,
        )

    after_payload = load_json(latest_pointer)
    assert after_payload == before_payload
    assert after_payload["execution_id"] == "exec_20260410T000000Z"
    assert after_payload["archive_dir"].endswith("/exec_20260410T000000Z")


def test_run_label_topic_profile_repeated_runs_keep_archive_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_label_profile_inputs(monkeypatch, tmp_path)
    _set_fixed_now(monkeypatch, "2026-04-10T00:00:00+00:00")
    run_label_topic_profile(
        model="bleilda",
        dataset="dummy",
        category="all",
        split="train",
        iteration=0,
        num_topics=2,
        results_root=tmp_path,
    )

    _set_fixed_now(monkeypatch, "2026-04-10T01:00:00+00:00")
    run_label_topic_profile(
        model="bleilda",
        dataset="dummy",
        category="all",
        split="train",
        iteration=0,
        num_topics=2,
        results_root=tmp_path,
    )

    archive_root = (
        tmp_path
        / "topic_analysis"
        / "label_profile"
        / "archive"
        / "2026-04-10"
        / "dummy"
        / "default"
        / "all"
        / "bleilda_train_k2_it0"
    )
    execution_dirs = sorted(
        path.name for path in archive_root.iterdir() if path.is_dir()
    )
    assert execution_dirs == [
        "exec_20260410T000000Z",
        "exec_20260410T010000Z",
    ]

    latest_pointer = load_json(
        tmp_path
        / "topic_analysis"
        / "label_profile"
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / "bleilda_train_k2_it0"
        / "CURRENT.json"
    )
    assert latest_pointer["execution_id"] == "exec_20260410T010000Z"
    assert latest_pointer["archive_dir"].endswith("/exec_20260410T010000Z")

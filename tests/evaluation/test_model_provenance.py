from __future__ import annotations

from pathlib import Path

from src.evaluation.model_provenance import (
    load_model_provenance,
    load_model_provenance_for_artifact,
)


def test_load_model_provenance_reads_baseline_metadata(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        (
            '{"schema":"baseline_artifact_metadata","runner_key":"ctm",'
            '"runner_family":"ctm","method_kind":"topic_model",'
            '"parameter_variant":"num_epochs=12",'
            '"preprocessing_variant":"language=english",'
            '"baseline_params":{"num_epochs":12}}'
        ),
        encoding="utf-8",
    )

    provenance = load_model_provenance(tmp_path, model_key="ctm")

    assert provenance == {
        "model_key": "ctm",
        "metadata_path": str(tmp_path / "metadata.json"),
        "artifact_metadata_schema": "baseline_artifact_metadata",
        "runner_key": "ctm",
        "runner_family": "ctm",
        "method_kind": "topic_model",
        "data_run": None,
        "condition_id": None,
        "condition_fingerprint": None,
        "parameter_variant": "num_epochs=12",
        "preprocessing_variant": "language=english",
        "baseline_params": {"num_epochs": 12},
    }


def test_load_model_provenance_reads_vmf_metadata(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        (
            '{"schema":"vmf_artifact_metadata","axes":{"model_family":"vmf_sentence_lda",'
            '"algorithm_variant":"num_components=2","encoder_model":"sentence-transformers/all-mpnet-base-v2",'
            '"embedding_preprocess_variant":"none"}}'
        ),
        encoding="utf-8",
    )

    provenance = load_model_provenance(tmp_path, model_key="vmf_sentence_lda")

    assert provenance == {
        "model_key": "vmf_sentence_lda",
        "metadata_path": str(tmp_path / "metadata.json"),
        "artifact_metadata_schema": "vmf_artifact_metadata",
        "model_family": "vmf_sentence_lda",
        "condition_id": None,
        "condition_fingerprint": None,
        "algorithm_variant": "num_components=2",
        "encoder_model": "sentence-transformers/all-mpnet-base-v2",
        "embedding_preprocess_variant": "none",
    }


def test_load_model_provenance_for_artifact_uses_parent_directory(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "params" / "table_counts_per_doc.pkl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"x")
    (artifact_path.parent / "metadata.json").write_text(
        '{"schema":"baseline_artifact_metadata","runner_key":"bleilda"}',
        encoding="utf-8",
    )

    provenance = load_model_provenance_for_artifact(
        artifact_path,
        model_key="bleilda",
    )

    assert provenance["runner_key"] == "bleilda"

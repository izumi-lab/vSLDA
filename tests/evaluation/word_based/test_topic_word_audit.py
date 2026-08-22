from __future__ import annotations

from pathlib import Path

import pytest

from src.core.artifacts import save_json, save_pickle
from src.evaluation.word_based.topic_word_audit import (
    audit_model_artifacts,
    evaluation_vocabulary_fingerprint,
    validate_document_alignment,
)


def test_artifact_audit_reports_missing_without_fallback(tmp_path: Path) -> None:
    result = audit_model_artifacts(model="bleilda", condition_dir=tmp_path)
    assert not result.ok
    assert result.missing_groups == (("params/model.gensim", "model.gensim"),)
    assert result.protocol == "collapsed_posthoc"


def _write_vmf_condition(tmp_path: Path, transform: str) -> Path:
    save_json({"pre_normalize_transform": transform}, tmp_path / "params.json")
    for name in (
        "kappa_per_topic.pkl",
        "mixture_weights.pkl",
        "component_means.pkl",
    ):
        save_pickle([0.0], tmp_path / name)
    return tmp_path


def test_vmf_mean_center_audit_requires_only_mean(tmp_path: Path) -> None:
    condition_dir = _write_vmf_condition(tmp_path, "mean_center")
    save_pickle([0.0], condition_dir / "embedding_transform_mean.pkl")
    result = audit_model_artifacts(model="vmf", condition_dir=condition_dir)
    assert result.ok, (result.missing_groups, result.inconsistencies)


def test_vmf_whitening_audit_fails_on_missing_matrix(tmp_path: Path) -> None:
    condition_dir = _write_vmf_condition(tmp_path, "whitening")
    save_pickle([0.0], condition_dir / "embedding_transform_mean.pkl")
    result = audit_model_artifacts(model="vmf", condition_dir=condition_dir)
    assert not result.ok
    assert ("embedding_transform_whitening_matrix.pkl",) in result.missing_groups


def test_vmf_none_audit_flags_unexpected_transform_artifacts(tmp_path: Path) -> None:
    condition_dir = _write_vmf_condition(tmp_path, "none")
    save_pickle([0.0], condition_dir / "embedding_transform_mean.pkl")
    result = audit_model_artifacts(model="vmf", condition_dir=condition_dir)
    assert not result.ok
    assert any("does not use it" in item for item in result.inconsistencies)


def test_etm_audit_requires_params_json(tmp_path: Path) -> None:
    result = audit_model_artifacts(model="etm", condition_dir=tmp_path)
    assert ("params/params.json", "params.json") in result.missing_groups


def test_word_vector_audit_checks_local_path_existence(tmp_path: Path) -> None:
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    save_json({}, params_dir / "params.json")
    for name in (
        "kappa_per_topic.pkl",
        "mixture_weights.pkl",
        "component_means.pkl",
    ):
        save_pickle([0.0], params_dir / name)
    save_json(
        {"baseline_params": {"word2vec": "vectors/missing_local.kv"}},
        tmp_path / "metadata.json",
    )
    result = audit_model_artifacts(model="mvtm", condition_dir=tmp_path)
    assert not result.ok
    assert ("word2vec source: vectors/missing_local.kv",) in result.missing_groups


def test_gaussianlda_audit_matches_loader_artifacts_without_prior_mu(
    tmp_path: Path,
) -> None:
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    save_json({}, params_dir / "params.json")
    for name in (
        "table_counts.pkl",
        "table_means.pkl",
        "log_determinants.pkl",
        "table_cholesky_ltriangular_mat.pkl",
    ):
        save_pickle([0.0], params_dir / name)
    save_json(
        {"baseline_params": {"word2vec": "glove-wiki-gigaword-100"}},
        tmp_path / "metadata.json",
    )

    result = audit_model_artifacts(model="gaussianlda", condition_dir=tmp_path)

    assert result.ok, (result.missing_groups, result.inconsistencies)
    assert all("prior_mu.pkl" not in group for group in result.missing_groups)


def test_gaussianlda_audit_requires_loader_cholesky_artifact(tmp_path: Path) -> None:
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    save_json({}, params_dir / "params.json")
    for name in ("table_counts.pkl", "table_means.pkl", "log_determinants.pkl"):
        save_pickle([0.0], params_dir / name)
    save_json(
        {"baseline_params": {"word2vec": "glove-wiki-gigaword-100"}},
        tmp_path / "metadata.json",
    )

    result = audit_model_artifacts(model="gaussianlda", condition_dir=tmp_path)

    assert not result.ok
    assert (
        "params/table_cholesky_ltriangular_mat.pkl",
        "table_cholesky_ltriangular_mat.pkl",
    ) in result.missing_groups


def test_vmf_audit_accepts_combined_selection_structure(tmp_path: Path) -> None:
    condition_dir = _write_vmf_condition(tmp_path, "none")
    save_pickle([0.0], condition_dir / "train_preprocessed.pkl")
    save_json(
        {
            "train": {"raw_doc_indices": [0, 1]},
            "test": {"raw_doc_indices": [2]},
        },
        condition_dir / "preprocessing_selection.json",
    )
    result = audit_model_artifacts(model="vmf", condition_dir=condition_dir)
    assert result.ok, (result.missing_groups, result.inconsistencies)


def test_vmf_audit_flags_missing_selection_file(tmp_path: Path) -> None:
    condition_dir = _write_vmf_condition(tmp_path, "none")
    save_pickle([0.0], condition_dir / "train_preprocessed.pkl")
    result = audit_model_artifacts(model="vmf", condition_dir=condition_dir)
    assert not result.ok
    assert any(
        "missing preprocessing selection" in item for item in result.inconsistencies
    )


def test_vmf_audit_flags_selection_without_requested_split(tmp_path: Path) -> None:
    condition_dir = _write_vmf_condition(tmp_path, "none")
    save_pickle([0.0], condition_dir / "train_preprocessed.pkl")
    save_pickle([0.0], condition_dir / "test_preprocessed.pkl")
    save_json(
        {"train": {"raw_doc_indices": [0]}},
        condition_dir / "preprocessing_selection.json",
    )
    result = audit_model_artifacts(model="vmf", condition_dir=condition_dir)
    assert not result.ok
    assert any("no 'test' split" in item for item in result.inconsistencies)


def test_baseline_audit_flags_duplicate_selection_ids(tmp_path: Path) -> None:
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    save_pickle([0.0], params_dir / "model.gensim")
    save_pickle([0.0], params_dir / "preprocessed_corpus.pkl")
    save_json({"raw_doc_indices": [1, 1]}, params_dir / "preprocessing_selection.json")
    result = audit_model_artifacts(model="bleilda", condition_dir=tmp_path)
    assert not result.ok
    assert any("Duplicate raw document ID" in item for item in result.inconsistencies)


def test_document_alignment_checks_order_not_only_set() -> None:
    assert validate_document_alignment({"a": [1, 2], "b": [1, 2]}) == (1, 2)
    with pytest.raises(ValueError, match="alignment differs"):
        validate_document_alignment({"a": [1, 2], "b": [2, 1]})


def test_vocabulary_fingerprint_depends_on_fixed_order() -> None:
    assert evaluation_vocabulary_fingerprint(["a", "b"]) != (
        evaluation_vocabulary_fingerprint(["b", "a"])
    )

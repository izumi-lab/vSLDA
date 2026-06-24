from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.artifacts import CURRENT_POINTER_FILENAME, METADATA_FILENAME, save_json
from src.core.errors import MissingArtifactError
from src.evaluation.classification.feature_registry import (
    FEATURE_REGISTRY,
    FeatureSpec,
    build_feature_specs,
    get_feature_specs,
    iter_available_features,
    load_pickle_array,
    register_feature_spec,
    resolve_feature_catalog_entry,
    resolve_feature_display_name,
    topic_distribution,
)


def _write_latest_pointer_case(
    *,
    archive_dir: Path,
    latest_dir: Path,
    metadata: dict[str, object],
    artifacts: dict[str, str],
    display_key: str,
    embedding_variant: str | None = None,
) -> None:
    archive_dir.mkdir(parents=True)
    save_json(metadata, archive_dir / METADATA_FILENAME)
    for relative_path in artifacts.values():
        path = archive_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    save_json(
        {
            "schema": "latest_result_pointer",
            "schema_version": 1,
            "task": "test",
            "display_key": display_key,
            "dataset": "dummy",
            "data_run": "default",
            "category": "science",
            "archive_dir": str(archive_dir),
            "started_at": "2026-01-01T00:00:00Z",
            "execution_id": "exec",
            "condition_fingerprint": metadata.get("condition_fingerprint"),
            "embedding_variant": embedding_variant,
            "artifacts": artifacts,
        },
        latest_dir / CURRENT_POINTER_FILENAME,
    )


def test_topic_distribution_normalizes_rows() -> None:
    arr = topic_distribution(np.asarray([[2.0, 2.0], [0.0, 0.0]]))
    assert np.allclose(arr[0], np.asarray([0.5, 0.5]))
    assert np.allclose(arr[1], np.asarray([0.0, 0.0]))


def test_build_feature_specs_includes_expected_models() -> None:
    specs = build_feature_specs("hard")
    assert [spec.display_name for spec in specs] == [
        "Contextual TM",
        "Blei LDA",
        "BERTopic (UMAP + k-means)",
        "Gaussian LDA",
        "ETM",
        "MvTM",
        "Spherical k-means",
        "Gaussian k-means",
        "movMF",
        "Gaussian mixture",
        "SenClu",
        "Sentence LDA",
        "sentLDA",
        "vMF Sentence LDA",
    ]


def test_feature_registry_exposes_builtin_model_keys() -> None:
    assert list(FEATURE_REGISTRY.keys()) == [
        "ctm",
        "bleilda",
        "bertopic_kmeans",
        "gaussianlda",
        "etm",
        "mvtm",
        "spherical_kmeans",
        "gaussian_kmeans",
        "movmf",
        "gaussian_mixture",
        "senclu",
        "sentence_gaussianlda",
        "sentlda",
        "vmf_sentence_lda",
    ]


def test_register_feature_spec_adds_builder() -> None:
    original_registry = dict(FEATURE_REGISTRY)
    register_feature_spec(
        "dummy_model",
        lambda vmf_assignment: FeatureSpec(
            model_key="dummy_model",
            display_name=f"Dummy ({vmf_assignment})",
            train_path_resolver=lambda dataset, iteration, num_topics, category: Path(
                "train.pkl"
            ),
            test_path_resolver=lambda dataset, iteration, num_topics, category: Path(
                "test.pkl"
            ),
            train_loader=lambda path: np.asarray([]),
            test_loader=lambda path: np.asarray([]),
        ),
    )

    try:
        specs = get_feature_specs("hard")
        assert specs[-1].display_name == "Dummy (hard)"
    finally:
        FEATURE_REGISTRY.clear()
        FEATURE_REGISTRY.update(original_registry)


def test_iter_available_features_skips_unresolved_paths_in_all_mode(
    tmp_path: Path,
) -> None:
    original_registry = dict(FEATURE_REGISTRY)

    def _raise_missing(*_args: object) -> Path:
        raise MissingArtifactError(tmp_path / "missing")

    def _builder(_: str) -> FeatureSpec:
        return FeatureSpec(
            model_key="raising",
            display_name="Raising",
            train_path_resolver=_raise_missing,
            test_path_resolver=_raise_missing,
            train_loader=lambda path: np.asarray([]),
            test_loader=lambda path: np.asarray([]),
        )

    try:
        FEATURE_REGISTRY.clear()
        FEATURE_REGISTRY["raising"] = _builder

        specs = iter_available_features(
            dataset="dummy",
            data_run="default",
            iteration=0,
            num_topics=2,
            category="science",
            vmf_assignment="hard",
            feature_resolve_mode="all",
            selected_models=["raising"],
        )

        assert specs == []
    finally:
        FEATURE_REGISTRY.clear()
        FEATURE_REGISTRY.update(original_registry)


def test_iter_available_features_raises_unresolved_paths_in_strict_mode(
    tmp_path: Path,
) -> None:
    original_registry = dict(FEATURE_REGISTRY)

    def _raise_missing(*_args: object) -> Path:
        raise MissingArtifactError(tmp_path / "missing")

    def _builder(_: str) -> FeatureSpec:
        return FeatureSpec(
            model_key="raising",
            display_name="Raising",
            train_path_resolver=_raise_missing,
            test_path_resolver=_raise_missing,
            train_loader=lambda path: np.asarray([]),
            test_loader=lambda path: np.asarray([]),
        )

    try:
        FEATURE_REGISTRY.clear()
        FEATURE_REGISTRY["raising"] = _builder

        with pytest.raises(MissingArtifactError):
            iter_available_features(
                dataset="dummy",
                data_run="default",
                iteration=0,
                num_topics=2,
                category="science",
                vmf_assignment="hard",
                feature_resolve_mode="strict",
                selected_models=["raising"],
            )
    finally:
        FEATURE_REGISTRY.clear()
        FEATURE_REGISTRY.update(original_registry)


def test_iter_available_features_resolves_vmf_soft_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.core.paths.EXPERIMENT_RESULTS_ROOT",
        tmp_path / "experiments",
    )
    monkeypatch.setattr(
        "src.core.paths.BASELINE_RESULTS_ROOT",
        tmp_path / "baselines",
    )
    specs = iter_available_features(
        dataset="20newsgroup",
        iteration=2,
        num_topics=30,
        category="science",
        vmf_assignment="soft",
    )

    vmf_spec, vmf_train_path, vmf_test_path = next(
        item for item in specs if item[0].model_key == "vmf_sentence_lda"
    )
    assert vmf_spec.display_name == "vMF Sentence LDA (soft)"
    assert vmf_train_path.name == "doc_topic_train_soft.pkl"
    assert vmf_test_path.name == "doc_topic_test_soft.pkl"

    ctm_spec, ctm_train_path, ctm_test_path = specs[0]
    assert ctm_spec.model_key == "ctm"
    assert ctm_train_path.name == "ctm.pkl"
    assert ctm_test_path.name == "science.pkl"


def test_iter_available_features_reads_all_matching_latest_baseline_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "results"
    baseline_root = root / "baselines"
    monkeypatch.setattr("src.core.paths.BASELINE_RESULTS_ROOT", baseline_root)
    monkeypatch.setattr("src.core.paths.EXPERIMENT_RESULTS_ROOT", root / "experiments")

    for variant in ("mpnet", "e5"):
        display_key = f"k2_it0_{variant}"
        archive_dir = (
            baseline_root
            / "dummy"
            / "default"
            / "ctm"
            / "archive"
            / "2026-01-01"
            / "science"
            / display_key
            / f"baseline_{variant}"
        )
        latest_dir = (
            baseline_root
            / "dummy"
            / "default"
            / "ctm"
            / "latest"
            / "science"
            / display_key
        )
        _write_latest_pointer_case(
            archive_dir=archive_dir,
            latest_dir=latest_dir,
            metadata={
                "runner_key": "ctm",
                "runner_family": "ctm",
                "dataset": "dummy",
                "data_run": "default",
                "category": "science",
                "num_topics": 2,
                "iteration": 0,
                "embedding_variant": variant,
                "condition_fingerprint": f"fingerprint-{variant}",
            },
            artifacts={
                "train_path": "params/ctm.pkl",
                "infer_path": "infer/science.pkl",
            },
            display_key=display_key,
            embedding_variant=variant,
        )

    specs = iter_available_features(
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="science",
        vmf_assignment="hard",
    )
    ctm_specs = [item for item in specs if item[0].model_key == "ctm"]

    assert [spec.display_name for spec, _train, _test in ctm_specs] == [
        "Contextual TM [e5]",
        "Contextual TM [mpnet]",
    ]
    for spec, train_path, test_path in ctm_specs:
        entry = resolve_feature_catalog_entry(spec, train_path)
        assert train_path.exists()
        assert test_path.exists()
        assert entry["display_key"] in {"k2_it0_e5", "k2_it0_mpnet"}
        assert entry["embedding_variant"] in {"e5", "mpnet"}
        assert entry["condition_fingerprint"] in {
            "fingerprint-e5",
            "fingerprint-mpnet",
        }


def test_iter_available_features_filters_latest_embedding_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "results"
    baseline_root = root / "baselines"
    monkeypatch.setattr("src.core.paths.BASELINE_RESULTS_ROOT", baseline_root)
    monkeypatch.setattr("src.core.paths.EXPERIMENT_RESULTS_ROOT", root / "experiments")

    for variant in ("mpnet", "e5"):
        display_key = f"k2_it0_{variant}"
        _write_latest_pointer_case(
            archive_dir=(
                baseline_root
                / "dummy"
                / "default"
                / "ctm"
                / "archive"
                / "2026-01-01"
                / "science"
                / display_key
                / f"baseline_{variant}"
            ),
            latest_dir=(
                baseline_root
                / "dummy"
                / "default"
                / "ctm"
                / "latest"
                / "science"
                / display_key
            ),
            metadata={
                "runner_key": "ctm",
                "runner_family": "ctm",
                "dataset": "dummy",
                "data_run": "default",
                "category": "science",
                "num_topics": 2,
                "iteration": 0,
                "embedding_variant": variant,
            },
            artifacts={
                "train_path": "params/ctm.pkl",
                "infer_path": "infer/science.pkl",
            },
            display_key=display_key,
            embedding_variant=variant,
        )

    specs = iter_available_features(
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="science",
        vmf_assignment="hard",
        embedding_variants=["mpnet"],
    )
    ctm_specs = [item for item in specs if item[0].model_key == "ctm"]

    assert [spec.display_name for spec, _train, _test in ctm_specs] == [
        "Contextual TM [mpnet]"
    ]


def test_iter_available_features_ignores_sentence_embedding_filter_for_word_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "results"
    baseline_root = root / "baselines"
    monkeypatch.setattr("src.core.paths.BASELINE_RESULTS_ROOT", baseline_root)
    monkeypatch.setattr("src.core.paths.EXPERIMENT_RESULTS_ROOT", root / "experiments")

    display_key = "k2_it0_glove100"
    _write_latest_pointer_case(
        archive_dir=(
            baseline_root
            / "dummy"
            / "default"
            / "gaussianlda"
            / "archive"
            / "2026-01-01"
            / "science"
            / display_key
            / "baseline_glove100"
        ),
        latest_dir=(
            baseline_root
            / "dummy"
            / "default"
            / "gaussianlda"
            / "latest"
            / "science"
            / display_key
        ),
        metadata={
            "runner_key": "gaussianlda",
            "runner_family": "gaussianlda",
            "dataset": "dummy",
            "data_run": "default",
            "category": "science",
            "num_topics": 2,
            "iteration": 0,
            "embedding_variant": "glove100",
        },
        artifacts={
            "train_path": "params/table_counts_per_doc.pkl",
            "infer_path": "infer/science.pkl",
        },
        display_key=display_key,
        embedding_variant="glove100",
    )

    specs = iter_available_features(
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="science",
        vmf_assignment="hard",
        embedding_variants=["mpnet"],
    )
    gaussian_specs = [item for item in specs if item[0].model_key == "gaussianlda"]

    assert [spec.display_name for spec, _train, _test in gaussian_specs] == [
        "Gaussian LDA [glove100]"
    ]
    assert gaussian_specs[0][1].name == "table_counts_per_doc.pkl"
    assert gaussian_specs[0][2].name == "science.pkl"


def test_iter_available_features_matches_raw_sentence_embedding_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "results"
    baseline_root = root / "baselines"
    monkeypatch.setattr("src.core.paths.BASELINE_RESULTS_ROOT", baseline_root)
    monkeypatch.setattr("src.core.paths.EXPERIMENT_RESULTS_ROOT", root / "experiments")

    display_key = "k2_it0_mpnet_raw"
    _write_latest_pointer_case(
        archive_dir=(
            baseline_root
            / "dummy"
            / "default"
            / "sentence_gaussianlda"
            / "archive"
            / "2026-01-01"
            / "science"
            / display_key
            / "baseline_mpnet_raw"
        ),
        latest_dir=(
            baseline_root
            / "dummy"
            / "default"
            / "sentence_gaussianlda"
            / "latest"
            / "science"
            / display_key
        ),
        metadata={
            "runner_key": "sentence_gaussianlda",
            "runner_family": "sentence_gaussianlda",
            "dataset": "dummy",
            "data_run": "default",
            "category": "science",
            "num_topics": 2,
            "iteration": 0,
            "embedding_variant": "mpnet_raw",
        },
        artifacts={
            "train_path": "params/table_counts_per_doc.pkl",
            "infer_path": "infer/science.pkl",
        },
        display_key=display_key,
        embedding_variant="mpnet_raw",
    )

    specs = iter_available_features(
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="science",
        vmf_assignment="hard",
        embedding_variants=["mpnet"],
    )
    sentence_specs = [
        item for item in specs if item[0].model_key == "sentence_gaussianlda"
    ]

    assert [spec.display_name for spec, _train, _test in sentence_specs] == [
        "Sentence LDA [mpnet_raw]"
    ]


def test_iter_available_features_reads_latest_vmf_soft_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "results"
    experiment_root = root / "experiments"
    monkeypatch.setattr("src.core.paths.EXPERIMENT_RESULTS_ROOT", experiment_root)
    monkeypatch.setattr("src.core.paths.BASELINE_RESULTS_ROOT", root / "baselines")

    display_key = "k2_it0_c1_mpnet"
    _write_latest_pointer_case(
        archive_dir=(
            experiment_root
            / "dummy"
            / "default"
            / "vmf_sentence_lda"
            / "archive"
            / "2026-01-01"
            / "science"
            / display_key
            / "vmf_mpnet"
        ),
        latest_dir=(
            experiment_root
            / "dummy"
            / "default"
            / "vmf_sentence_lda"
            / "latest"
            / "science"
            / display_key
        ),
        metadata={
            "axes": {
                "dataset": "dummy",
                "data_run": "default",
                "category": "science",
                "num_topics": 2,
                "iteration": 0,
                "model_family": "vmf_sentence_lda",
                "embedding_variant": "mpnet",
            },
            "condition_fingerprint": "fingerprint-vmf",
            "encoder_config": {"embedding_variant": "mpnet"},
        },
        artifacts={
            "train_path": "doc_topic_train.pkl",
            "infer_path": "doc_topic_test.pkl",
            "train_doc_topic_soft": "doc_topic_train_soft.pkl",
            "test_doc_topic_soft": "doc_topic_test_soft.pkl",
        },
        display_key=display_key,
        embedding_variant="mpnet",
    )

    specs = iter_available_features(
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="science",
        vmf_assignment="soft",
    )
    vmf_spec, train_path, test_path = next(
        item for item in specs if item[0].model_key == "vmf_sentence_lda"
    )

    assert vmf_spec.display_name == "vMF Sentence LDA (soft) [c1_mpnet]"
    assert train_path.name == "doc_topic_train_soft.pkl"
    assert test_path.name == "doc_topic_test_soft.pkl"


def test_load_pickle_array_raises_missing_artifact_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pkl"

    with pytest.raises(MissingArtifactError) as exc_info:
        load_pickle_array(missing_path)

    assert str(missing_path) in str(exc_info.value)


def test_resolve_feature_display_name_appends_non_default_variant(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "params" / "train.pkl"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(b"x")
    (train_path.parent / "metadata.json").write_text(
        '{"parameter_variant":"num_epochs=12","runner_key":"ctm","runner_family":"ctm"}',
        encoding="utf-8",
    )
    spec = FeatureSpec(
        model_key="ctm",
        display_name="Contextual TM",
        train_path_resolver=lambda *_args: train_path,
        test_path_resolver=lambda *_args: tmp_path / "test.pkl",
        train_loader=lambda path: np.asarray([]),
        test_loader=lambda path: np.asarray([]),
    )

    assert (
        resolve_feature_display_name(spec, train_path)
        == "Contextual TM [num_epochs=12]"
    )


def test_resolve_feature_catalog_entry_reads_metadata_axes(tmp_path: Path) -> None:
    train_path = tmp_path / "params" / "ctm.pkl"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(b"x")
    (train_path.parent / "metadata.json").write_text(
        '{"runner_key":"ctm","runner_family":"ctm","parameter_variant":"num_epochs=12","preprocessing_variant":"language=english","baseline_params":{"num_epochs":12}}',
        encoding="utf-8",
    )
    spec = FeatureSpec(
        model_key="ctm",
        display_name="Contextual TM",
        train_path_resolver=lambda *_args: train_path,
        test_path_resolver=lambda *_args: tmp_path / "test.pkl",
        train_loader=lambda path: np.asarray([]),
        test_loader=lambda path: np.asarray([]),
    )

    entry = resolve_feature_catalog_entry(spec, train_path)

    assert entry["feature_name"] == "Contextual TM [num_epochs=12]"
    assert entry["runner_family"] == "ctm"
    assert entry["baseline_params"] == {"num_epochs": 12}


def test_sentlda_available_index_resolver_uses_training_vocabulary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_path = tmp_path / "params" / "table_counts_per_doc.pkl"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(b"x")
    (train_path.parent / "vocabulary.json").write_text(
        '{"known": 0}',
        encoding="utf-8",
    )
    (train_path.parent.parent / "metadata.json").write_text(
        '{"language":"english"}',
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_build_preprocessed_available_indices(
        dataset,
        category,
        metadata,
        *,
        availability_predicate,
        target_column,
        label_schema,
    ):
        captured["dataset"] = dataset
        captured["category"] = category
        captured["metadata"] = metadata
        captured["target_column"] = target_column
        captured["label_schema"] = label_schema
        doc_with_vocab = type(
            "Doc",
            (),
            {"sentences_tokenized": [["known"], ["other"]]},
        )()
        doc_without_vocab = type(
            "Doc",
            (),
            {"sentences_tokenized": [["other"]]},
        )()
        captured["with_vocab"] = availability_predicate(doc_with_vocab)
        captured["without_vocab"] = availability_predicate(doc_without_vocab)
        return ("train", "test")

    monkeypatch.setattr(
        "src.evaluation.classification.feature_registry.build_preprocessed_available_indices",
        _fake_build_preprocessed_available_indices,
    )

    spec = next(
        spec for spec in get_feature_specs("hard") if spec.model_key == "sentlda"
    )
    resolved = spec.available_index_resolver(
        "20newsgroup",
        "computer",
        train_path,
        tmp_path / "infer" / "computer.pkl",
        "target_str",
        "identity",
    )

    assert resolved == ("train", "test")
    assert captured["dataset"] == "20newsgroup"
    assert captured["category"] == "computer"
    assert captured["metadata"] == {"language": "english"}
    assert captured["target_column"] == "target_str"
    assert captured["label_schema"] == "identity"
    assert captured["with_vocab"] is True
    assert captured["without_vocab"] is False


def test_etm_available_index_resolver_uses_training_vocabulary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_path = tmp_path / "params" / "etm.pkl"
    train_path.parent.mkdir(parents=True)
    train_path.write_bytes(b"x")
    (train_path.parent / "vocabulary.json").write_text(
        '["known"]',
        encoding="utf-8",
    )
    (train_path.parent.parent / "metadata.json").write_text(
        '{"language":"english"}',
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_build_preprocessed_available_indices(
        dataset,
        category,
        metadata,
        *,
        availability_predicate,
        target_column,
        label_schema,
    ):
        captured["dataset"] = dataset
        captured["category"] = category
        captured["metadata"] = metadata
        captured["target_column"] = target_column
        captured["label_schema"] = label_schema
        doc_with_vocab = type(
            "Doc",
            (),
            {"document_tokens": ["known", "other"]},
        )()
        doc_without_vocab = type(
            "Doc",
            (),
            {"document_tokens": ["other"]},
        )()
        captured["with_vocab"] = availability_predicate(doc_with_vocab)
        captured["without_vocab"] = availability_predicate(doc_without_vocab)
        return ("train", "test")

    monkeypatch.setattr(
        "src.evaluation.classification.feature_registry.build_preprocessed_available_indices",
        _fake_build_preprocessed_available_indices,
    )

    spec = next(spec for spec in get_feature_specs("hard") if spec.model_key == "etm")
    resolved = spec.available_index_resolver(
        "20newsgroup",
        "computer",
        train_path,
        tmp_path / "infer" / "computer.pkl",
        "target_str",
        "identity",
    )

    assert resolved == ("train", "test")
    assert captured["dataset"] == "20newsgroup"
    assert captured["category"] == "computer"
    assert captured["metadata"] == {"language": "english"}
    assert captured["target_column"] == "target_str"
    assert captured["label_schema"] == "identity"
    assert captured["with_vocab"] is True
    assert captured["without_vocab"] is False

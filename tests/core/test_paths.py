from __future__ import annotations

from pathlib import Path

import pytest

from src.core.artifacts import (
    CURRENT_POINTER_FILENAME,
    METADATA_FILENAME,
    load_json,
    save_json,
)
from src.core.contracts import RunSpec
from src.core.errors import MissingArtifactError
from src.core.paths import (
    REPO_ROOT,
    ResultPathBuilder,
    build_archive_result_dir,
    build_baseline_archive_dir,
    build_baseline_condition_id,
    build_baseline_dir,
    build_baseline_display_key,
    build_baseline_doc_topic_path,
    build_baseline_latest_dir,
    build_latest_result_dir,
    build_result_display_key,
    build_vmf_archive_dir,
    build_vmf_condition_id,
    build_vmf_display_key,
    build_vmf_doc_topic_path,
    build_vmf_experiment_dir,
    build_vmf_latest_dir,
    resolve_baseline_condition_dir,
    resolve_cross_model_pair_diagnostics_dir,
    resolve_latest_result_dir,
    resolve_project_path,
    resolve_sentence_topic_inspection_dir,
    resolve_topic_count_analysis_dir,
    resolve_vmf_experiment_dir,
    write_baseline_latest_pointer,
    write_latest_result_pointer,
    write_vmf_latest_pointer,
)


def test_resolve_project_path_uses_repo_root_for_relative_paths() -> None:
    assert resolve_project_path("configs/experiments/20newsgroup.example.yaml") == (
        REPO_ROOT / "configs/experiments/20newsgroup.example.yaml"
    )


def test_result_path_builder_run_dir() -> None:
    builder = ResultPathBuilder(Path("results"))
    spec = RunSpec(
        dataset_name="20newsgroups",
        model_name="vmf_sentence_lda",
        num_topics=20,
        seed=0,
    )

    assert builder.run_dir(spec) == Path(
        "results/20newsgroups/vmf_sentence_lda/20topic/seed0"
    )


def test_result_path_builder_metric_path() -> None:
    builder = ResultPathBuilder(Path("results"))
    spec = RunSpec(
        dataset_name="20newsgroups",
        model_name="vmf_sentence_lda",
        num_topics=20,
        seed=0,
    )

    assert builder.metrics_path(spec, "classification") == Path(
        "results/20newsgroups/vmf_sentence_lda/20topic/seed0/metrics/classification.json"
    )


def test_build_vmf_experiment_dir_uses_run_name_when_present() -> None:
    path = build_vmf_experiment_dir(
        dataset="demo_dataset",
        iteration=2,
        num_topics=30,
        category="all",
        run_name="fy2024",
    )
    assert path == (
        REPO_ROOT
        / "results/experiments/demo_dataset/fy2024/vmf_sentence_lda/all"
        / path.name
    )
    assert path.name.startswith("it2__k30__")


def test_resolve_project_path_keeps_absolute_path() -> None:
    path = Path("/tmp/example")
    assert resolve_project_path(path) == path


def test_build_result_display_key_uses_short_human_readable_order() -> None:
    assert (
        build_result_display_key(
            num_topics=50,
            iteration=0,
            extra_labels=["bleilda", "train"],
        )
        == "bleilda_train_k50_it0"
    )


def test_build_component_display_keys_use_k_iteration_component_order() -> None:
    assert (
        build_vmf_display_key(
            iteration=0,
            num_topics=10,
            num_components=1,
        )
        == "k10_it0_c1"
    )
    assert build_vmf_display_key(iteration=0, num_topics=10) == "k10_it0"
    assert (
        build_baseline_display_key(
            iteration=1,
            num_topics=5,
            num_components=2,
        )
        == "k5_it1_c2"
    )


def test_build_archive_result_dir_uses_date_and_execution_id() -> None:
    archive_dir = build_archive_result_dir(
        base_root=Path("results/topic_analysis/label_profile"),
        dataset="dummy",
        data_run="default",
        category="all",
        display_key="bleilda_train_k50_it0",
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="exec_20260410T021530Z",
    )

    assert archive_dir == Path(
        "results/topic_analysis/label_profile/archive/2026-04-10/dummy/default/all/bleilda_train_k50_it0/exec_20260410T021530Z"
    )


def test_write_and_resolve_latest_result_pointer(tmp_path: Path) -> None:
    base_root = tmp_path / "topic_analysis" / "label_profile"
    archive_dir = build_archive_result_dir(
        base_root=base_root,
        dataset="dummy",
        data_run="default",
        category="all",
        display_key="bleilda_train_k50_it0",
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="exec_20260410T021530Z",
    )
    archive_dir.mkdir(parents=True)

    pointer_path = write_latest_result_pointer(
        base_root=base_root,
        task="word_based_label_profile",
        dataset="dummy",
        data_run="default",
        category="all",
        display_key="bleilda_train_k50_it0",
        archive_dir=archive_dir,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="exec_20260410T021530Z",
        condition_fingerprint="abcd1234ef",
        artifacts={"json": "label_topic_profile.json", "metadata": "metadata.json"},
    )

    assert pointer_path == (
        build_latest_result_dir(
            base_root=base_root,
            dataset="dummy",
            data_run="default",
            category="all",
            display_key="bleilda_train_k50_it0",
        )
        / CURRENT_POINTER_FILENAME
    )
    payload = load_json(pointer_path)
    assert payload["display_key"] == "bleilda_train_k50_it0"
    assert payload["execution_id"] == "exec_20260410T021530Z"
    assert (
        resolve_latest_result_dir(
            base_root=base_root,
            dataset="dummy",
            data_run="default",
            category="all",
            display_key="bleilda_train_k50_it0",
        )
        == archive_dir
    )


def test_resolve_topic_count_analysis_dir_prefers_latest_pointer(
    tmp_path: Path,
) -> None:
    base_root = tmp_path / "topic_count_analysis"
    archive_dir = build_archive_result_dir(
        base_root=base_root,
        dataset="dummy",
        data_run="default",
        category="all",
        display_key="it0__k10__perplexity__abcd1234",
        started_at="2026-04-19T10:00:00+00:00",
        execution_id="exec_20260419T100000Z",
    )
    archive_dir.mkdir(parents=True)
    write_latest_result_pointer(
        base_root=base_root,
        task="topic_count_diagnostics",
        dataset="dummy",
        data_run="default",
        category="all",
        display_key="it0__k10__perplexity__abcd1234",
        archive_dir=archive_dir,
        started_at="2026-04-19T10:00:00+00:00",
        execution_id="exec_20260419T100000Z",
        condition_fingerprint="abcd1234ef",
        artifacts={"json": "perplexity_summary.json"},
    )

    assert (
        resolve_topic_count_analysis_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id="it0__k10__perplexity__abcd1234",
            base_root=base_root,
        )
        == archive_dir
    )


def test_resolve_analysis_wrappers_fall_back_to_legacy_dirs(tmp_path: Path) -> None:
    topic_count_legacy = (
        tmp_path / "topic_count_analysis" / "dummy" / "default" / "all" / "it0"
    )
    pair_legacy = (
        tmp_path / "analysis" / "vmf_vs_baseline" / "dummy" / "default" / "all" / "it1"
    )
    inspect_legacy = tmp_path / "visualization" / "dummy" / "default" / "all" / "it2"
    topic_count_legacy.mkdir(parents=True)
    pair_legacy.mkdir(parents=True)
    inspect_legacy.mkdir(parents=True)

    assert (
        resolve_topic_count_analysis_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id="it0",
            base_root=tmp_path / "topic_count_analysis",
        )
        == topic_count_legacy
    )
    assert (
        resolve_cross_model_pair_diagnostics_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id="it1",
            base_root=tmp_path / "analysis" / "vmf_vs_baseline",
        )
        == pair_legacy
    )
    assert (
        resolve_sentence_topic_inspection_dir(
            dataset="dummy",
            data_run="default",
            category="all",
            condition_id="it2",
            base_root=tmp_path / "visualization",
        )
        == inspect_legacy
    )


def test_build_vmf_doc_topic_path_uses_assignment_suffix() -> None:
    path = build_vmf_doc_topic_path(
        dataset="missing_dataset_for_assignment_suffix_test",
        iteration=0,
        num_topics=20,
        category="science",
        split="test",
        assignment="soft",
    )
    assert path.name == "doc_topic_test_soft.pkl"
    assert "vmf_sentence_lda" in path.parts
    assert "science" in path.parts


def test_embedding_variant_is_appended_after_component_display_key() -> None:
    assert (
        build_vmf_display_key(
            iteration=3,
            num_topics=10,
            num_components=1,
            embedding_variant="bge",
        )
        == "k10_it3_c1_bge"
    )
    latest_dir = build_vmf_latest_dir(
        category="politics",
        iteration=3,
        num_topics=10,
        num_components=1,
        embedding_variant="bge",
        run_name="default",
        dataset_root=REPO_ROOT / "results/experiments/20newsgroup",
    )
    assert latest_dir.name == "k10_it3_c1_bge"


def test_build_vmf_doc_topic_path_resolves_embedding_variant_latest_pointer(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dummy"
    for variant in ["mpnet", "bge"]:
        archive_dir = build_vmf_archive_dir(
            category="science",
            iteration=0,
            num_topics=20,
            num_components=1,
            embedding_variant=variant,
            started_at=f"2026-04-10T02:15:3{len(variant)}+00:00",
            execution_id=f"vmf_{variant}",
            run_name="default",
            dataset_root=dataset_root,
        )
        archive_dir.mkdir(parents=True)
        save_json(
            {
                "condition_id": f"it0__k20__{variant}",
                "num_components": 1,
                "axes": {
                    "iteration": 0,
                    "num_topics": 20,
                    "category": "science",
                    "data_run": "default",
                    "embedding_variant": variant,
                },
            },
            archive_dir / METADATA_FILENAME,
        )
        write_vmf_latest_pointer(
            dataset="dummy",
            data_run="default",
            category="science",
            iteration=0,
            num_topics=20,
            num_components=1,
            embedding_variant=variant,
            archive_dir=archive_dir,
            started_at=f"2026-04-10T02:15:3{len(variant)}+00:00",
            execution_id=f"vmf_{variant}",
            condition_fingerprint=f"abcd1234{len(variant)}f",
            artifacts={"metadata": "metadata.json"},
            dataset_root=dataset_root,
        )

    path = build_vmf_doc_topic_path(
        dataset="dummy",
        iteration=0,
        num_topics=20,
        category="science",
        split="train",
        num_components=1,
        embedding_variant="mpnet",
        dataset_root=dataset_root,
    )

    assert path == (
        dataset_root
        / "default"
        / "vmf_sentence_lda"
        / "archive"
        / "2026-04-10"
        / "science"
        / "k20_it0_c1_mpnet"
        / "vmf_mpnet"
        / "doc_topic_train.pkl"
    )
    with pytest.raises(MissingArtifactError, match="Multiple latest pointers"):
        build_vmf_doc_topic_path(
            dataset="dummy",
            iteration=0,
            num_topics=20,
            category="science",
            split="train",
            num_components=1,
            dataset_root=dataset_root,
        )


def test_build_baseline_doc_topic_path_resolves_embedding_variant_latest_pointer(
    tmp_path: Path,
) -> None:
    for variant in ["glove50", "glove100"]:
        archive_dir = build_baseline_archive_dir(
            model="gaussianlda",
            dataset="dummy",
            data_run="default",
            category="all",
            iteration=1,
            num_topics=5,
            embedding_variant=variant,
            started_at=f"2026-04-10T02:15:3{len(variant)}+00:00",
            execution_id=f"baseline_{variant}",
            baseline_root=tmp_path,
        )
        (archive_dir / "params").mkdir(parents=True)
        save_json(
            {
                "runner_key": "gaussianlda",
                "condition_id": f"it1__k5__{variant}",
                "iteration": 1,
                "num_topics": 5,
                "category": "all",
                "data_run": "default",
                "embedding_variant": variant,
            },
            archive_dir / "metadata.json",
        )
        write_baseline_latest_pointer(
            model="gaussianlda",
            dataset="dummy",
            data_run="default",
            category="all",
            iteration=1,
            num_topics=5,
            archive_dir=archive_dir,
            started_at=f"2026-04-10T02:15:3{len(variant)}+00:00",
            execution_id=f"baseline_{variant}",
            condition_fingerprint=f"abcd1234{len(variant)}f",
            artifacts={"metadata": "metadata.json"},
            embedding_variant=variant,
            baseline_root=tmp_path,
        )

    path = build_baseline_doc_topic_path(
        model="gaussianlda",
        dataset="dummy",
        iteration=1,
        num_topics=5,
        category="all",
        split="train",
        embedding_variant="glove100",
        baseline_root=tmp_path,
    )

    assert path == (
        tmp_path
        / "dummy"
        / "default"
        / "gaussianlda"
        / "archive"
        / "2026-04-10"
        / "all"
        / "k5_it1_glove100"
        / "baseline_glove100"
        / "params"
        / "table_counts_per_doc.pkl"
    )
    with pytest.raises(MissingArtifactError, match="Multiple latest pointers"):
        build_baseline_doc_topic_path(
            model="gaussianlda",
            dataset="dummy",
            iteration=1,
            num_topics=5,
            category="all",
            split="train",
            baseline_root=tmp_path,
        )


def test_build_baseline_dir_supports_optional_category() -> None:
    params_dir = build_baseline_dir(
        model="bleilda",
        split_root="params",
        dataset="20newsgroup",
        iteration=1,
        num_topics=10,
        category="computer",
    )
    assert params_dir == (
        REPO_ROOT
        / "results/baselines/20newsgroup/default/bleilda"
        / "computer"
        / params_dir.parent.name
        / "params"
    )
    assert params_dir.parent.name.startswith("it1__k10__")
    assert "__computer__" not in params_dir.parent.name
    infer_dir = build_baseline_dir(
        model="bleilda",
        split_root="infer",
        dataset="20newsgroup",
        iteration=1,
        num_topics=10,
    )
    assert infer_dir == (
        REPO_ROOT
        / "results/baselines/20newsgroup/default/bleilda"
        / "all"
        / infer_dir.parent.name
        / "infer"
    )


def test_build_baseline_doc_topic_path_handles_model_specific_layouts() -> None:
    dataset = "missing_dataset_for_baseline_layout_test"
    blei_train = build_baseline_doc_topic_path(
        model="bleilda",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )
    senclu_train = build_baseline_doc_topic_path(
        model="senclu",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )
    ctm_train = build_baseline_doc_topic_path(
        model="ctm",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )
    gaussian_train = build_baseline_doc_topic_path(
        model="gaussianlda",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )
    etm_train = build_baseline_doc_topic_path(
        model="etm",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )
    etm_test_soft = build_baseline_doc_topic_path(
        model="etm",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="test",
        prefer_soft=True,
    )
    mvtm_train = build_baseline_doc_topic_path(
        model="mvtm",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )
    sentlda_train = build_baseline_doc_topic_path(
        model="sentlda",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )
    bertopic_kmeans_train = build_baseline_doc_topic_path(
        model="bertopic_kmeans",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )
    spherical_kmeans_train = build_baseline_doc_topic_path(
        model="spherical_kmeans",
        dataset=dataset,
        iteration=0,
        num_topics=20,
        category="computer",
        split="train",
    )

    assert blei_train.name == "lda_comp.pkl"
    assert senclu_train.name == "computer.pkl"
    assert ctm_train.name == "ctm.pkl"
    assert gaussian_train.name == "table_counts_per_doc.pkl"
    assert etm_train.name == "etm.pkl"
    assert etm_test_soft.name == "computer_doc_topic_soft.pkl"
    assert mvtm_train.name == "table_counts_per_doc.pkl"
    assert sentlda_train.name == "table_counts_per_doc.pkl"
    assert bertopic_kmeans_train.name == "bertopic_kmeans.pkl"
    assert spherical_kmeans_train.name == "computer.pkl"
    assert "bleilda" in blei_train.parts
    assert "senclu" in senclu_train.parts
    assert "ctm" in ctm_train.parts
    assert "gaussianlda" in gaussian_train.parts
    assert "etm" in etm_train.parts
    assert "mvtm" in mvtm_train.parts
    assert "sentlda" in sentlda_train.parts
    assert "bertopic_kmeans" in bertopic_kmeans_train.parts
    assert "spherical_kmeans" in spherical_kmeans_train.parts


def test_vmf_condition_id_changes_when_condition_payload_changes() -> None:
    first, _ = build_vmf_condition_id(
        iteration=0,
        num_topics=20,
        category="science",
        fingerprint_payload={"encoder_model": "mpnet", "assignment": "hard"},
    )
    second, _ = build_vmf_condition_id(
        iteration=0,
        num_topics=20,
        category="science",
        fingerprint_payload={"encoder_model": "mpnet", "assignment": "soft"},
    )

    assert first != second


def test_baseline_condition_id_changes_when_data_run_changes() -> None:
    first, _ = build_baseline_condition_id(
        model="ctm",
        iteration=0,
        num_topics=20,
        category="science",
        fingerprint_payload={
            "data_run": "default",
            "parameter_variant": "num_epochs=10",
        },
    )
    second, _ = build_baseline_condition_id(
        model="ctm",
        iteration=0,
        num_topics=20,
        category="science",
        fingerprint_payload={
            "data_run": "fy2024",
            "parameter_variant": "num_epochs=10",
        },
    )

    assert first != second


def test_vmf_condition_id_omits_category_from_readable_prefix() -> None:
    condition_id, _ = build_vmf_condition_id(
        iteration=0,
        num_topics=20,
        category="science",
        fingerprint_payload={"category": "science", "encoder_model": "mpnet"},
    )

    assert condition_id.startswith("it0__k20__")
    assert "__science__" not in condition_id


def test_baseline_condition_id_omits_category_from_readable_prefix() -> None:
    condition_id, _ = build_baseline_condition_id(
        model="ctm",
        iteration=0,
        num_topics=20,
        category="science",
        fingerprint_payload={"category": "science", "parameter_variant": "default"},
    )

    assert condition_id.startswith("it0__k20__")
    assert "__science__" not in condition_id


def test_resolve_vmf_experiment_dir_supports_legacy_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dummy"
    legacy_dir = (
        dataset_root / "default" / "vmf_sentence_lda" / "it0__k20__science__abcd1234"
    )
    legacy_dir.mkdir(parents=True)
    save_json(
        {
            "axes": {
                "iteration": 0,
                "num_topics": 20,
                "category": "science",
                "data_run": "default",
            }
        },
        legacy_dir / METADATA_FILENAME,
    )

    resolved_dir = resolve_vmf_experiment_dir(
        dataset="dummy",
        iteration=0,
        num_topics=20,
        category="science",
        dataset_root=dataset_root,
    )

    assert resolved_dir == legacy_dir
    assert (
        build_vmf_doc_topic_path(
            dataset="dummy",
            iteration=0,
            num_topics=20,
            category="science",
            split="test",
            condition_id=legacy_dir.name,
            dataset_root=dataset_root,
        )
        == legacy_dir / "doc_topic_test.pkl"
    )


def test_resolve_vmf_experiment_dir_prefers_latest_pointer(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dummy"
    archive_dir = build_vmf_archive_dir(
        category="science",
        iteration=0,
        num_topics=20,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="vmf_20260410T021530Z",
        run_name="default",
        dataset_root=dataset_root,
    )
    archive_dir.mkdir(parents=True)
    save_json(
        {
            "axes": {
                "iteration": 0,
                "num_topics": 20,
                "category": "science",
                "data_run": "default",
            }
        },
        archive_dir / METADATA_FILENAME,
    )
    write_vmf_latest_pointer(
        dataset="dummy",
        data_run="default",
        category="science",
        iteration=0,
        num_topics=20,
        archive_dir=archive_dir,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="vmf_20260410T021530Z",
        condition_fingerprint="abcd1234ef",
        artifacts={"metadata": "metadata.json"},
        dataset_root=dataset_root,
    )

    resolved_dir = resolve_vmf_experiment_dir(
        dataset="dummy",
        iteration=0,
        num_topics=20,
        category="science",
        dataset_root=dataset_root,
    )

    assert resolved_dir == archive_dir
    assert (
        build_vmf_latest_dir(
            category="science",
            iteration=0,
            num_topics=20,
            run_name="default",
            dataset_root=dataset_root,
        )
        / CURRENT_POINTER_FILENAME
    ).exists()


def test_resolve_vmf_experiment_dir_prefers_component_latest_pointer(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dummy"
    archive_dir = build_vmf_archive_dir(
        category="science",
        iteration=0,
        num_topics=20,
        num_components=1,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="vmf_20260410T021530Z",
        run_name="default",
        dataset_root=dataset_root,
    )
    archive_dir.mkdir(parents=True)
    save_json(
        {
            "num_components": 1,
            "axes": {
                "iteration": 0,
                "num_topics": 20,
                "category": "science",
                "data_run": "default",
                "algorithm_variant": "components_1__estimate_alpha_every_1",
            },
        },
        archive_dir / METADATA_FILENAME,
    )
    write_vmf_latest_pointer(
        dataset="dummy",
        data_run="default",
        category="science",
        iteration=0,
        num_topics=20,
        num_components=1,
        archive_dir=archive_dir,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="vmf_20260410T021530Z",
        condition_fingerprint="abcd1234ef",
        artifacts={"metadata": "metadata.json"},
        dataset_root=dataset_root,
    )

    resolved_dir = resolve_vmf_experiment_dir(
        dataset="dummy",
        iteration=0,
        num_topics=20,
        category="science",
        num_components=1,
        dataset_root=dataset_root,
    )

    assert resolved_dir == archive_dir
    pointer_path = (
        build_vmf_latest_dir(
            category="science",
            iteration=0,
            num_topics=20,
            num_components=1,
            run_name="default",
            dataset_root=dataset_root,
        )
        / CURRENT_POINTER_FILENAME
    )
    assert pointer_path.exists()
    assert load_json(pointer_path)["display_key"] == "k20_it0_c1"


def test_resolve_vmf_experiment_dir_supports_nested_dataset_root_layout(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dummy"
    nested_dataset_root = dataset_root / "dummy"
    archive_dir = build_vmf_archive_dir(
        category="science",
        iteration=0,
        num_topics=20,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="vmf_20260410T021530Z",
        run_name="default",
        dataset_root=nested_dataset_root,
    )
    archive_dir.mkdir(parents=True)
    save_json(
        {
            "axes": {
                "iteration": 0,
                "num_topics": 20,
                "category": "science",
                "data_run": "default",
            }
        },
        archive_dir / METADATA_FILENAME,
    )
    write_vmf_latest_pointer(
        dataset="dummy",
        data_run="default",
        category="science",
        iteration=0,
        num_topics=20,
        archive_dir=archive_dir,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="vmf_20260410T021530Z",
        condition_fingerprint="abcd1234ef",
        artifacts={"metadata": "metadata.json"},
        dataset_root=nested_dataset_root,
    )

    resolved_dir = resolve_vmf_experiment_dir(
        dataset="dummy",
        iteration=0,
        num_topics=20,
        category="science",
        dataset_root=dataset_root,
    )

    assert resolved_dir == archive_dir
    assert (
        build_vmf_doc_topic_path(
            dataset="dummy",
            iteration=0,
            num_topics=20,
            category="science",
            split="test",
            dataset_root=dataset_root,
        )
        == archive_dir / "doc_topic_test.pkl"
    )


def test_resolve_baseline_condition_dir_supports_legacy_layout(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "dummy" / "default" / "ctm" / "it0__k20__science__abcd1234"
    (legacy_dir / "params").mkdir(parents=True)
    save_json(
        {
            "iteration": 0,
            "num_topics": 20,
            "category": "science",
            "data_run": "default",
        },
        legacy_dir / "params" / METADATA_FILENAME,
    )

    resolved_dir = resolve_baseline_condition_dir(
        model="ctm",
        dataset="dummy",
        iteration=0,
        num_topics=20,
        category="science",
        baseline_root=tmp_path,
    )

    assert resolved_dir == legacy_dir
    assert (
        build_baseline_doc_topic_path(
            model="ctm",
            dataset="dummy",
            iteration=0,
            num_topics=20,
            category="science",
            split="train",
            condition_id=legacy_dir.name,
            baseline_root=tmp_path,
        )
        == legacy_dir / "params" / "ctm.pkl"
    )


def test_resolve_baseline_condition_dir_prefers_latest_pointer(tmp_path: Path) -> None:
    archive_dir = build_baseline_archive_dir(
        model="ctm",
        dataset="dummy",
        data_run="default",
        category="science",
        iteration=0,
        num_topics=20,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="baseline_20260410T021530Z",
        baseline_root=tmp_path,
    )
    (archive_dir / "params").mkdir(parents=True)
    save_json(
        {
            "iteration": 0,
            "num_topics": 20,
            "category": "science",
            "data_run": "default",
        },
        archive_dir / "metadata.json",
    )
    write_baseline_latest_pointer(
        model="ctm",
        dataset="dummy",
        data_run="default",
        category="science",
        iteration=0,
        num_topics=20,
        archive_dir=archive_dir,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="baseline_20260410T021530Z",
        condition_fingerprint="abcd1234ef",
        artifacts={"metadata": "metadata.json", "train_path": "params/ctm.pkl"},
        baseline_root=tmp_path,
    )

    resolved_dir = resolve_baseline_condition_dir(
        model="ctm",
        dataset="dummy",
        iteration=0,
        num_topics=20,
        category="science",
        baseline_root=tmp_path,
    )

    assert resolved_dir == archive_dir
    assert (
        build_baseline_latest_dir(
            model="ctm",
            dataset="dummy",
            data_run="default",
            category="science",
            iteration=0,
            num_topics=20,
            baseline_root=tmp_path,
        )
        / CURRENT_POINTER_FILENAME
    ).exists()


def test_resolve_mvtm_condition_dir_prefers_component_latest_pointer(
    tmp_path: Path,
) -> None:
    archive_dir = build_baseline_archive_dir(
        model="mvtm",
        dataset="dummy",
        data_run="default",
        category="all",
        iteration=1,
        num_topics=5,
        num_components=1,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="baseline_20260410T021530Z",
        baseline_root=tmp_path,
    )
    (archive_dir / "params").mkdir(parents=True)
    save_json(
        {
            "runner_key": "mvtm",
            "iteration": 1,
            "num_topics": 5,
            "category": "all",
            "data_run": "default",
            "baseline_params": {"num_components": 1},
        },
        archive_dir / "metadata.json",
    )
    write_baseline_latest_pointer(
        model="mvtm",
        dataset="dummy",
        data_run="default",
        category="all",
        iteration=1,
        num_topics=5,
        num_components=1,
        archive_dir=archive_dir,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="baseline_20260410T021530Z",
        condition_fingerprint="abcd1234ef",
        artifacts={"metadata": "metadata.json"},
        baseline_root=tmp_path,
    )

    resolved_dir = resolve_baseline_condition_dir(
        model="mvtm",
        dataset="dummy",
        iteration=1,
        num_topics=5,
        category="all",
        num_components=1,
        baseline_root=tmp_path,
    )

    assert resolved_dir == archive_dir
    pointer_path = (
        build_baseline_latest_dir(
            model="mvtm",
            dataset="dummy",
            data_run="default",
            category="all",
            iteration=1,
            num_topics=5,
            num_components=1,
            baseline_root=tmp_path,
        )
        / CURRENT_POINTER_FILENAME
    )
    assert pointer_path.exists()
    assert load_json(pointer_path)["display_key"] == "k5_it1_c1"


def test_resolve_baseline_condition_dir_finds_single_embedding_variant_pointer(
    tmp_path: Path,
) -> None:
    archive_dir = build_baseline_archive_dir(
        model="gaussianlda",
        dataset="dummy",
        data_run="default",
        category="all",
        iteration=1,
        num_topics=5,
        embedding_variant="glove100",
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="baseline_20260410T021530Z",
        baseline_root=tmp_path,
    )
    (archive_dir / "params").mkdir(parents=True)
    save_json(
        {
            "runner_key": "gaussianlda",
            "iteration": 1,
            "num_topics": 5,
            "category": "all",
            "data_run": "default",
            "embedding_variant": "glove100",
        },
        archive_dir / "metadata.json",
    )
    write_baseline_latest_pointer(
        model="gaussianlda",
        dataset="dummy",
        data_run="default",
        category="all",
        iteration=1,
        num_topics=5,
        archive_dir=archive_dir,
        started_at="2026-04-10T02:15:30+00:00",
        execution_id="baseline_20260410T021530Z",
        condition_fingerprint="abcd1234ef",
        artifacts={"metadata": "metadata.json"},
        embedding_variant="glove100",
        baseline_root=tmp_path,
    )

    resolved_dir = resolve_baseline_condition_dir(
        model="gaussianlda",
        dataset="dummy",
        iteration=1,
        num_topics=5,
        category="all",
        baseline_root=tmp_path,
    )

    assert resolved_dir == archive_dir


def test_resolve_baseline_condition_dir_requires_variant_when_multiple_match(
    tmp_path: Path,
) -> None:
    for variant in ["glove50", "glove100"]:
        archive_dir = build_baseline_archive_dir(
            model="gaussianlda",
            dataset="dummy",
            data_run="default",
            category="all",
            iteration=1,
            num_topics=5,
            embedding_variant=variant,
            started_at=f"2026-04-10T02:15:3{len(variant)}+00:00",
            execution_id=f"baseline_{variant}",
            baseline_root=tmp_path,
        )
        (archive_dir / "params").mkdir(parents=True)
        save_json(
            {
                "runner_key": "gaussianlda",
                "iteration": 1,
                "num_topics": 5,
                "category": "all",
                "data_run": "default",
                "embedding_variant": variant,
            },
            archive_dir / "metadata.json",
        )
        write_baseline_latest_pointer(
            model="gaussianlda",
            dataset="dummy",
            data_run="default",
            category="all",
            iteration=1,
            num_topics=5,
            archive_dir=archive_dir,
            started_at=f"2026-04-10T02:15:3{len(variant)}+00:00",
            execution_id=f"baseline_{variant}",
            condition_fingerprint=f"abcd1234{len(variant)}f",
            artifacts={"metadata": "metadata.json"},
            embedding_variant=variant,
            baseline_root=tmp_path,
        )

    with pytest.raises(MissingArtifactError, match="Multiple latest pointers"):
        resolve_baseline_condition_dir(
            model="gaussianlda",
            dataset="dummy",
            iteration=1,
            num_topics=5,
            category="all",
            baseline_root=tmp_path,
        )

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.core.result_identity import (
    build_condition_id,
    build_display_key,
    build_execution_date,
    build_execution_id,
)


def _current_experiment_results_root() -> Path:
    from . import paths as public_paths

    return public_paths.EXPERIMENT_RESULTS_ROOT


def _current_baseline_results_root() -> Path:
    from . import paths as public_paths

    return public_paths.BASELINE_RESULTS_ROOT


def build_result_display_key(
    *,
    num_topics: int,
    iteration: int,
    extra_labels: tuple[Any, ...] | list[Any] = (),
) -> str:
    return build_display_key(
        num_topics=num_topics,
        iteration=iteration,
        extra_labels=tuple(extra_labels),
    )


def _build_component_display_key(
    *,
    iteration: int,
    num_topics: int,
    num_components: int | None,
    embedding_variant: str | None = None,
) -> str:
    base_key = build_result_display_key(num_topics=num_topics, iteration=iteration)
    if num_components is None:
        display_key = base_key
    else:
        display_key = f"{base_key}_c{int(num_components)}"
    if embedding_variant:
        return f"{display_key}_{embedding_variant}"
    return display_key


def build_vmf_display_key(
    *,
    iteration: int,
    num_topics: int,
    num_components: int | None = None,
    embedding_variant: str | None = None,
) -> str:
    return _build_component_display_key(
        iteration=iteration,
        num_topics=num_topics,
        num_components=num_components,
        embedding_variant=embedding_variant,
    )


def build_baseline_display_key(
    *,
    iteration: int,
    num_topics: int,
    num_components: int | None = None,
    embedding_variant: str | None = None,
) -> str:
    return _build_component_display_key(
        iteration=iteration,
        num_topics=num_topics,
        num_components=num_components,
        embedding_variant=embedding_variant,
    )


def build_vmf_legacy_display_key(*, iteration: int, num_topics: int) -> str:
    return build_result_display_key(num_topics=num_topics, iteration=iteration)


def build_baseline_legacy_display_key(*, iteration: int, num_topics: int) -> str:
    return build_result_display_key(num_topics=num_topics, iteration=iteration)


def build_vmf_model_root(*, run_name: str, dataset_root: Path | None = None) -> Path:
    resolved_dataset_root = dataset_root or _current_experiment_results_root()
    return resolved_dataset_root / str(run_name or "default") / "vmf_sentence_lda"


def build_vmf_latest_dir(
    *,
    category: str,
    iteration: int,
    num_topics: int,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    run_name: str = "default",
    dataset_root: Path | None = None,
) -> Path:
    return (
        build_vmf_model_root(run_name=run_name, dataset_root=dataset_root)
        / "latest"
        / str(category)
        / build_vmf_display_key(
            iteration=iteration,
            num_topics=num_topics,
            num_components=num_components,
            embedding_variant=embedding_variant,
        )
    )


def build_vmf_archive_dir(
    *,
    category: str,
    iteration: int,
    num_topics: int,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    started_at: str | None = None,
    execution_id: str | None = None,
    run_name: str = "default",
    dataset_root: Path | None = None,
) -> Path:
    resolved_execution_id = execution_id or build_execution_id(started_at=started_at)
    return (
        build_vmf_model_root(run_name=run_name, dataset_root=dataset_root)
        / "archive"
        / build_execution_date(started_at=started_at)
        / str(category)
        / build_vmf_display_key(
            iteration=iteration,
            num_topics=num_topics,
            num_components=num_components,
            embedding_variant=embedding_variant,
        )
        / resolved_execution_id
    )


def build_baseline_model_root(
    *,
    model: str,
    dataset: str,
    data_run: str,
    baseline_root: Path | None = None,
) -> Path:
    resolved_baseline_root = baseline_root or _current_baseline_results_root()
    return resolved_baseline_root / dataset / data_run / model


def build_baseline_latest_dir(
    *,
    model: str,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    baseline_root: Path | None = None,
) -> Path:
    return (
        build_baseline_model_root(
            model=model,
            dataset=dataset,
            data_run=data_run,
            baseline_root=baseline_root,
        )
        / "latest"
        / str(category)
        / build_baseline_display_key(
            iteration=iteration,
            num_topics=num_topics,
            num_components=num_components,
            embedding_variant=embedding_variant,
        )
    )


def build_baseline_archive_dir(
    *,
    model: str,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    started_at: str | None = None,
    execution_id: str | None = None,
    baseline_root: Path | None = None,
) -> Path:
    resolved_execution_id = execution_id or build_execution_id(started_at=started_at)
    return (
        build_baseline_model_root(
            model=model,
            dataset=dataset,
            data_run=data_run,
            baseline_root=baseline_root,
        )
        / "archive"
        / build_execution_date(started_at=started_at)
        / str(category)
        / build_baseline_display_key(
            iteration=iteration,
            num_topics=num_topics,
            num_components=num_components,
            embedding_variant=embedding_variant,
        )
        / resolved_execution_id
    )


def build_latest_result_dir(
    *,
    base_root: Path,
    dataset: str,
    data_run: str,
    category: str,
    display_key: str,
) -> Path:
    return (
        Path(base_root)
        / "latest"
        / str(dataset)
        / str(data_run)
        / str(category)
        / str(display_key)
    )


def build_archive_result_dir(
    *,
    base_root: Path,
    dataset: str,
    data_run: str,
    category: str,
    display_key: str,
    started_at: str | None = None,
    execution_id: str | None = None,
) -> Path:
    resolved_execution_id = execution_id or build_execution_id(started_at=started_at)
    return (
        Path(base_root)
        / "archive"
        / build_execution_date(started_at=started_at)
        / str(dataset)
        / str(data_run)
        / str(category)
        / str(display_key)
        / resolved_execution_id
    )


def build_vmf_condition_id(
    *,
    iteration: int,
    num_topics: int,
    category: str,
    fingerprint_payload: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    payload = (
        dict(fingerprint_payload)
        if fingerprint_payload is not None
        else {
            "iteration": int(iteration),
            "num_topics": int(num_topics),
            "category": str(category),
        }
    )
    return build_condition_id(
        iteration=iteration,
        num_topics=num_topics,
        category=None,
        fingerprint_payload=payload,
    )


def build_baseline_condition_id(
    *,
    model: str,
    iteration: int,
    num_topics: int,
    category: str,
    fingerprint_payload: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    payload = (
        dict(fingerprint_payload)
        if fingerprint_payload is not None
        else {
            "model": str(model),
            "iteration": int(iteration),
            "num_topics": int(num_topics),
            "category": str(category),
        }
    )
    return build_condition_id(
        iteration=iteration,
        num_topics=num_topics,
        category=None,
        fingerprint_payload=payload,
    )


def legacy_vmf_experiment_dir(
    *,
    dataset_root: Path,
    run_name: str,
    condition_id: str,
) -> Path:
    return dataset_root / str(run_name or "default") / "vmf_sentence_lda" / condition_id


def legacy_baseline_condition_dir(
    *,
    baseline_root: Path,
    dataset: str,
    data_run: str,
    model: str,
    condition_id: str,
) -> Path:
    return baseline_root / dataset / data_run / model / condition_id


def build_vmf_experiment_dir(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    run_name: str = "default",
    condition_id: str | None = None,
    condition_payload: Mapping[str, Any] | None = None,
    dataset_root: Path | None = None,
) -> Path:
    resolved_dataset_root = dataset_root or (
        _current_experiment_results_root() / dataset
    )
    data_run = str(run_name or "default")
    resolved_condition_id = condition_id
    if resolved_condition_id is None:
        resolved_condition_id, _ = build_vmf_condition_id(
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            fingerprint_payload=condition_payload,
        )
    category_first_dir = (
        resolved_dataset_root
        / data_run
        / "vmf_sentence_lda"
        / str(category)
        / resolved_condition_id
    )
    if condition_id is not None:
        legacy_dir = legacy_vmf_experiment_dir(
            dataset_root=resolved_dataset_root,
            run_name=data_run,
            condition_id=resolved_condition_id,
        )
        if legacy_dir.exists():
            return legacy_dir
    return category_first_dir


def build_baseline_dir(
    *,
    model: str,
    split_root: str,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str | None = None,
    data_run: str = "default",
    condition_id: str | None = None,
    condition_payload: Mapping[str, Any] | None = None,
    baseline_root: Path | None = None,
) -> Path:
    resolved_condition_id = condition_id
    if resolved_condition_id is None:
        resolved_condition_id, _ = build_baseline_condition_id(
            model=model,
            iteration=iteration,
            num_topics=num_topics,
            category=category or "all",
            fingerprint_payload=condition_payload,
        )
    resolved_baseline_root = baseline_root or _current_baseline_results_root()
    resolved_category = str(category or "all")
    category_first_dir = (
        resolved_baseline_root
        / dataset
        / data_run
        / model
        / resolved_category
        / resolved_condition_id
        / split_root
    )
    if condition_id is not None:
        legacy_dir = (
            legacy_baseline_condition_dir(
                baseline_root=resolved_baseline_root,
                dataset=dataset,
                data_run=data_run,
                model=model,
                condition_id=resolved_condition_id,
            )
            / split_root
        )
        if legacy_dir.exists():
            return legacy_dir
    return category_first_dir

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.core.artifacts import (
    CURRENT_POINTER_FILENAME,
    build_latest_result_pointer,
    load_artifact_json,
    save_latest_result_pointer,
)
from src.core.errors import MissingArtifactError

from .path_builders import (
    build_baseline_display_key,
    build_baseline_latest_dir,
    build_latest_result_dir,
    build_vmf_display_key,
    build_vmf_latest_dir,
)
from .paths_roots import resolve_project_path, stringify_project_path


def write_pointer_at(
    *,
    pointer_path: Path,
    task: str,
    display_key: str,
    dataset: str,
    data_run: str,
    category: str,
    archive_dir: Path,
    started_at: str,
    execution_id: str,
    condition_fingerprint: str | None,
    artifacts: Mapping[str, str],
    embedding_variant: str | None = None,
    encoder_config: Mapping[str, Any] | None = None,
) -> Path:
    payload = build_latest_result_pointer(
        task=task,
        display_key=display_key,
        dataset=dataset,
        data_run=data_run,
        category=category,
        archive_dir=stringify_project_path(Path(archive_dir)),
        started_at=str(started_at),
        execution_id=str(execution_id),
        condition_fingerprint=condition_fingerprint,
        artifacts=artifacts,
        embedding_variant=embedding_variant,
        encoder_config=encoder_config,
    )
    save_latest_result_pointer(payload, pointer_path)
    return pointer_path


def write_vmf_latest_pointer(
    *,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    archive_dir: Path,
    started_at: str,
    execution_id: str,
    condition_fingerprint: str | None,
    artifacts: Mapping[str, str],
    num_components: int | None = None,
    embedding_variant: str | None = None,
    encoder_config: Mapping[str, Any] | None = None,
    dataset_root: Path | None = None,
) -> Path:
    latest_dir = build_vmf_latest_dir(
        category=category,
        iteration=iteration,
        num_topics=num_topics,
        num_components=num_components,
        embedding_variant=embedding_variant,
        run_name=data_run,
        dataset_root=dataset_root,
    )
    return write_pointer_at(
        pointer_path=latest_dir / CURRENT_POINTER_FILENAME,
        task="vmf_experiment",
        display_key=build_vmf_display_key(
            iteration=iteration,
            num_topics=num_topics,
            num_components=num_components,
            embedding_variant=embedding_variant,
        ),
        dataset=dataset,
        data_run=data_run,
        category=category,
        archive_dir=archive_dir,
        started_at=started_at,
        execution_id=execution_id,
        condition_fingerprint=condition_fingerprint,
        artifacts=artifacts,
        embedding_variant=embedding_variant,
        encoder_config=encoder_config,
    )


def write_baseline_latest_pointer(
    *,
    model: str,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    archive_dir: Path,
    started_at: str,
    execution_id: str,
    condition_fingerprint: str | None,
    artifacts: Mapping[str, str],
    num_components: int | None = None,
    embedding_variant: str | None = None,
    encoder_config: Mapping[str, Any] | None = None,
    baseline_root: Path | None = None,
) -> Path:
    latest_dir = build_baseline_latest_dir(
        model=model,
        dataset=dataset,
        data_run=data_run,
        category=category,
        iteration=iteration,
        num_topics=num_topics,
        num_components=num_components,
        embedding_variant=embedding_variant,
        baseline_root=baseline_root,
    )
    return write_pointer_at(
        pointer_path=latest_dir / CURRENT_POINTER_FILENAME,
        task=f"baseline_{model}",
        display_key=build_baseline_display_key(
            iteration=iteration,
            num_topics=num_topics,
            num_components=num_components,
            embedding_variant=embedding_variant,
        ),
        dataset=dataset,
        data_run=data_run,
        category=category,
        archive_dir=archive_dir,
        started_at=started_at,
        execution_id=execution_id,
        condition_fingerprint=condition_fingerprint,
        artifacts=artifacts,
        embedding_variant=embedding_variant,
        encoder_config=encoder_config,
    )


def write_latest_result_pointer(
    *,
    base_root: Path,
    task: str,
    dataset: str,
    data_run: str,
    category: str,
    display_key: str,
    archive_dir: Path,
    started_at: str,
    execution_id: str,
    condition_fingerprint: str | None,
    artifacts: Mapping[str, str],
) -> Path:
    latest_dir = build_latest_result_dir(
        base_root=base_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        display_key=display_key,
    )
    return write_pointer_at(
        pointer_path=latest_dir / CURRENT_POINTER_FILENAME,
        task=task,
        display_key=display_key,
        dataset=dataset,
        data_run=data_run,
        category=category,
        archive_dir=archive_dir,
        started_at=started_at,
        execution_id=execution_id,
        condition_fingerprint=condition_fingerprint,
        artifacts=artifacts,
    )


def resolve_latest_result_dir(
    *,
    base_root: Path,
    dataset: str,
    data_run: str,
    category: str,
    display_key: str,
    fallback_dir: Path | None = None,
) -> Path:
    latest_dir = build_latest_result_dir(
        base_root=base_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        display_key=display_key,
    )
    pointer_path = latest_dir / CURRENT_POINTER_FILENAME
    if pointer_path.exists():
        payload = load_artifact_json(pointer_path)
        if isinstance(payload, dict) and payload.get("archive_dir"):
            archive_dir = resolve_project_path(str(payload["archive_dir"]))
            if archive_dir.exists():
                return archive_dir
    if fallback_dir is not None and fallback_dir.exists():
        return fallback_dir
    raise MissingArtifactError(
        pointer_path,
        detail="Expected latest result pointer or fallback result directory.",
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.artifacts import METADATA_FILENAME, load_artifact_json


@dataclass(frozen=True)
class ArtifactSplitConfig:
    split_csvs: tuple[str, ...] | None
    text_column: str
    target_column: str
    data_run: str | None


def find_metadata_path_near_artifact(artifact_path: Path) -> Path:
    for candidate in [artifact_path.parent, *artifact_path.parents[:3]]:
        metadata_path = candidate / METADATA_FILENAME
        if metadata_path.exists():
            return metadata_path
    return artifact_path.parent / METADATA_FILENAME


def load_artifact_metadata(artifact_path: Path) -> dict[str, Any]:
    metadata_path = find_metadata_path_near_artifact(artifact_path)
    if not metadata_path.exists():
        return {}
    payload = load_artifact_json(metadata_path)
    return payload if isinstance(payload, dict) else {}


def resolve_artifact_split_config(
    artifact_path: Path,
    *,
    split: str,
    default_text_column: str = "data",
    default_target_column: str = "target_str",
) -> ArtifactSplitConfig:
    payload = load_artifact_metadata(artifact_path)
    key = "train_csvs" if split == "train" else "test_csvs"
    raw_paths = payload.get(key)
    split_csvs: tuple[str, ...] | None = None
    if isinstance(raw_paths, (list, tuple)):
        normalized = tuple(str(path) for path in raw_paths if str(path).strip())
        split_csvs = normalized or None

    axes = payload.get("axes")
    data_run = payload.get("data_run")
    if data_run is None and isinstance(axes, dict):
        axes_data_run = axes.get("data_run")
        if isinstance(axes_data_run, str) and axes_data_run.strip():
            data_run = axes_data_run

    return ArtifactSplitConfig(
        split_csvs=split_csvs,
        text_column=str(payload.get("text_column") or default_text_column),
        target_column=str(payload.get("target_column") or default_target_column),
        data_run=str(data_run) if data_run is not None else None,
    )

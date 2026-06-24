from __future__ import annotations

import bz2
import csv
import json
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any, Mapping, Sequence, TypeVar

import yaml

from src.core.errors import MissingArtifactError, require_artifact_path

METADATA_FILENAME = "metadata.json"
PREPROCESSING_SELECTION_FILENAME = "preprocessing_selection.json"
VMF_PARAMS_FILENAME = "params.json"
VMF_METRICS_FILENAME = "metrics.json"
CURRENT_POINTER_FILENAME = "CURRENT.json"

ARTIFACT_METADATA_SCHEMA_VERSION = 1
VMF_METADATA_SCHEMA = "vmf_artifact_metadata"
BASELINE_METADATA_SCHEMA = "baseline_artifact_metadata"
LATEST_POINTER_SCHEMA = "latest_result_pointer"
LATEST_POINTER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentAxes:
    dataset: str
    model_family: str
    algorithm_variant: str
    encoder_model: str
    embedding_preprocess_variant: str
    num_topics: int
    iteration: int
    category: str
    data_run: str
    embedding_variant: str = "default"


@dataclass(frozen=True)
class VmfArtifactMetadata:
    axes: ExperimentAxes
    condition_id: str
    condition_fingerprint: str
    started_at: str | None
    execution_id: str | None
    language: str
    delimiter: str | None
    segmenter: str
    tokenizer: str
    text_column: str
    target_column: str | None
    has_labels: bool
    ja_replace_num: bool
    ja_stopwords_path: str | None
    ja_dicdir: str | None
    ja_require_unidic: bool
    train_csvs: tuple[str, ...]
    test_csvs: tuple[str, ...]
    fiscal_years: tuple[int, ...] | None
    num_components: int = 1
    encoder_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class BaselineArtifactMetadata:
    runner_key: str
    runner_family: str
    method_kind: str
    data_run: str
    condition_id: str
    condition_fingerprint: str
    started_at: str | None
    execution_id: str | None
    parameter_variant: str
    preprocessing_variant: str
    dataset: str
    category: str
    num_topics: int
    iteration: int
    baseline_params: dict[str, Any]
    targets: tuple[str, ...] | None
    language: str
    delimiter: str | None
    segmenter: str
    tokenizer: str
    legacy_preprocessing: bool | None
    text_column: str
    target_column: str | None
    ja_replace_num: bool
    ja_stopwords_path: str | None
    ja_dicdir: str | None
    ja_require_unidic: bool
    encoder_device: str | None
    runtime_num_workers: int
    train_csvs: tuple[str, ...]
    test_csvs: tuple[str, ...]
    train_dir: str
    infer_dir: str
    effective_random_state: int | None = None
    doc_topic_source: str | None = None
    doc_topic_space: str | None = None
    embedding_variant: str | None = None
    encoder_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: Path


@dataclass(frozen=True)
class PickleArtifactSpec:
    name: str
    filename: str
    payload: Any
    split: str


def build_artifact_refs(artifacts: Mapping[str, Path]) -> tuple[ArtifactRef, ...]:
    return tuple(
        ArtifactRef(name=str(name), path=Path(path))
        for name, path in sorted(artifacts.items())
    )


def artifact_refs_to_path_map(artifacts: Sequence[ArtifactRef]) -> dict[str, Path]:
    return {artifact.name: artifact.path for artifact in artifacts}


def artifact_refs_to_string_map(artifacts: Sequence[ArtifactRef]) -> dict[str, str]:
    return {artifact.name: str(artifact.path) for artifact in artifacts}


def ensure_artifact_paths_exist(artifacts: Mapping[str, Path]) -> None:
    missing = [str(path) for path in artifacts.values() if not Path(path).exists()]
    if missing:
        raise MissingArtifactError(
            missing[0],
            detail=f"Expected artifact paths were not found: {missing}",
        )


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_metadata_payload(metadata: Any, *, schema: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "artifact_kind": "metadata",
        **asdict(metadata),
    }


def save_vmf_metadata(metadata: VmfArtifactMetadata, path: Path) -> None:
    save_json(_build_metadata_payload(metadata, schema=VMF_METADATA_SCHEMA), path)


def save_baseline_metadata(metadata: BaselineArtifactMetadata, path: Path) -> None:
    save_json(
        _build_metadata_payload(metadata, schema=BASELINE_METADATA_SCHEMA),
        path,
    )


def build_latest_result_pointer(
    *,
    task: str,
    display_key: str,
    dataset: str,
    data_run: str,
    category: str,
    archive_dir: str,
    started_at: str,
    execution_id: str,
    condition_fingerprint: str | None,
    artifacts: Mapping[str, str],
    embedding_variant: str | None = None,
    encoder_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": LATEST_POINTER_SCHEMA,
        "schema_version": LATEST_POINTER_SCHEMA_VERSION,
        "task": task,
        "display_key": display_key,
        "dataset": dataset,
        "data_run": data_run,
        "category": category,
        "archive_dir": archive_dir,
        "started_at": started_at,
        "execution_id": execution_id,
        "condition_fingerprint": condition_fingerprint,
        "embedding_variant": embedding_variant,
        "encoder_config": None if encoder_config is None else dict(encoder_config),
        "artifacts": {str(name): str(path) for name, path in sorted(artifacts.items())},
    }


def save_latest_result_pointer(payload: Mapping[str, Any], path: Path) -> None:
    save_json(dict(payload), path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_artifact_json(path: Path) -> Any:
    return load_json(
        require_artifact_path(path, detail="Expected a JSON artifact for evaluation.")
    )


def save_yaml(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text_lines(path: Path, *, encoding: str = "utf-8") -> list[str]:
    with path.open("r", encoding=encoding) as f:
        return f.read().splitlines()


def save_csv_rows(
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    encoding: str = "utf-8",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding=encoding) as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def copy_binary_stream_to_path(
    src: IO[bytes],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def extract_bz2_file(src: Path, dst: Path) -> None:
    with bz2.open(src, "rb") as fin:
        copy_binary_stream_to_path(fin, dst)


def save_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def save_pickles(artifacts: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for name, data in artifacts.items():
        path = output_dir / f"{name}.pkl"
        save_pickle(data, path)
        saved[str(name)] = path
    return saved


def save_split_pickles(
    artifacts: Sequence[PickleArtifactSpec],
    *,
    train_dir: Path,
    infer_dir: Path,
) -> dict[str, Path]:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for artifact in artifacts:
        if artifact.split == "train":
            output_dir = train_dir
        elif artifact.split == "infer":
            output_dir = infer_dir
        else:
            raise ValueError(f"Unsupported artifact split: {artifact.split}")
        path = output_dir / artifact.filename
        save_pickle(artifact.payload, path)
        saved[artifact.name] = path
    return saved


def save_split_jsons(
    artifacts: Mapping[str, tuple[Any, str, str]],
    *,
    train_dir: Path,
    infer_dir: Path,
) -> dict[str, Path]:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for name, (payload, filename, split) in artifacts.items():
        if split == "train":
            output_dir = train_dir
        elif split == "infer":
            output_dir = infer_dir
        else:
            raise ValueError(f"Unsupported artifact split: {split}")
        path = output_dir / filename
        save_json(payload, path)
        saved[str(name)] = path
    return saved


T = TypeVar("T")


def load_pickle(path: Path) -> T:
    with path.open("rb") as f:
        return pickle.load(f)


def load_artifact_pickle(path: Path) -> T:
    return load_pickle(
        require_artifact_path(path, detail="Expected a pickle artifact for evaluation.")
    )

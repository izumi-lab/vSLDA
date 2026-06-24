from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.artifacts import (
    BASELINE_METADATA_SCHEMA,
    METADATA_FILENAME,
    VMF_METADATA_SCHEMA,
    load_artifact_json,
)


def _find_metadata_path(metadata_dir: Path) -> Path:
    for candidate in [metadata_dir, *metadata_dir.parents[:3]]:
        metadata_path = candidate / METADATA_FILENAME
        if metadata_path.exists():
            return metadata_path
    return metadata_dir / METADATA_FILENAME


def load_model_provenance(
    metadata_dir: Path,
    *,
    model_key: str,
) -> dict[str, Any]:
    metadata_path = _find_metadata_path(metadata_dir)
    provenance: dict[str, Any] = {
        "model_key": model_key,
        "metadata_path": str(metadata_path),
    }
    if not metadata_path.exists():
        return provenance

    payload = load_artifact_json(metadata_path)
    if not isinstance(payload, dict):
        return provenance

    schema = payload.get("schema")
    provenance["artifact_metadata_schema"] = schema

    if schema == BASELINE_METADATA_SCHEMA or "runner_key" in payload:
        baseline_payload = {
            "runner_key": payload.get("runner_key", model_key),
            "runner_family": payload.get("runner_family", model_key),
            "data_run": payload.get("data_run"),
            "condition_id": payload.get("condition_id"),
            "condition_fingerprint": payload.get("condition_fingerprint"),
            "parameter_variant": payload.get("parameter_variant"),
            "preprocessing_variant": payload.get("preprocessing_variant"),
            "baseline_params": (
                dict(payload["baseline_params"])
                if isinstance(payload.get("baseline_params"), dict)
                else None
            ),
        }
        if payload.get("method_kind") is not None:
            baseline_payload["method_kind"] = payload.get("method_kind")
        provenance.update(baseline_payload)
        return provenance

    axes = payload.get("axes")
    if schema == VMF_METADATA_SCHEMA or isinstance(axes, dict):
        axes_payload = axes if isinstance(axes, dict) else {}
        provenance.update(
            {
                "model_family": axes_payload.get("model_family", model_key),
                "condition_id": payload.get("condition_id"),
                "condition_fingerprint": payload.get("condition_fingerprint"),
                "algorithm_variant": axes_payload.get("algorithm_variant"),
                "encoder_model": axes_payload.get("encoder_model"),
                "embedding_preprocess_variant": axes_payload.get(
                    "embedding_preprocess_variant"
                ),
            }
        )
    return provenance


def load_model_provenance_for_artifact(
    artifact_path: Path,
    *,
    model_key: str,
) -> dict[str, Any]:
    return load_model_provenance(artifact_path.parent, model_key=model_key)

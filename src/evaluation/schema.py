from __future__ import annotations

from typing import Any

EVALUATION_SCHEMA_NAME = "evaluation_result"
EVALUATION_SCHEMA_VERSION = 1


def build_evaluation_meta(
    *,
    task: str,
    output_kind: str = "payload",
    **extra: Any,
) -> dict[str, Any]:
    meta = {
        "task": task,
        "schema": EVALUATION_SCHEMA_NAME,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "output_kind": output_kind,
    }
    meta.update(extra)
    return meta


def normalize_evaluation_meta(
    meta: dict[str, Any],
    *,
    default_output_kind: str = "payload",
) -> dict[str, Any]:
    normalized = dict(meta)
    normalized.setdefault("schema", EVALUATION_SCHEMA_NAME)
    normalized.setdefault("schema_version", EVALUATION_SCHEMA_VERSION)
    normalized.setdefault("output_kind", default_output_kind)
    return normalized


def build_evaluation_payload(
    *,
    meta: dict[str, Any],
    results: Any,
) -> dict[str, Any]:
    return {
        "_meta": normalize_evaluation_meta(meta),
        "results": results,
    }


def split_evaluation_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    meta = payload.get("_meta")
    results = payload.get("results")
    if isinstance(meta, dict) and "results" in payload:
        return normalize_evaluation_meta(meta), results
    return {}, payload

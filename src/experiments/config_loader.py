from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.core.artifacts import load_yaml
from src.core.paths import resolve_project_path


def load_config_yaml(path: str | Path) -> Dict[str, Any]:
    payload = load_yaml(resolve_project_path(path))
    return payload or {}


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and bool(value.get("__replace__", False)):
            replaced = dict(value)
            replaced.pop("__replace__", None)
            merged[key] = replaced
            continue
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(base_value, value)
        else:
            merged[key] = value
    return merged


def load_config_yaml_resolved(
    path: str | Path,
    *,
    visited: set[Path] | None = None,
) -> Dict[str, Any]:
    resolved_path = resolve_project_path(path)
    visited_paths = set() if visited is None else set(visited)
    if resolved_path in visited_paths:
        raise ValueError(f"Cyclic config extends detected at {resolved_path}")
    visited_paths.add(resolved_path)

    cfg = load_config_yaml(resolved_path)
    extends = cfg.pop("extends", None)
    if extends is None:
        return cfg

    extends_path = Path(str(extends))
    if not extends_path.is_absolute():
        extends_path = resolved_path.parent / extends_path

    parent_cfg = load_config_yaml_resolved(extends_path, visited=visited_paths)
    return deep_merge_dict(parent_cfg, cfg)

from __future__ import annotations

from typing import get_args

from src.evaluation.registry import (
    OutputKind,
    get_task,
    list_tasks,
    register_builtin_tasks,
)
from src.evaluation.schema import build_evaluation_meta

REQUIRED_META_KEYS = {"task", "output_kind", "schema", "schema_version"}


def test_all_registered_tasks_have_known_output_kind() -> None:
    register_builtin_tasks()
    allowed_output_kinds = set(get_args(OutputKind))

    for task in list_tasks():
        registered = get_task(task.name)
        assert registered.output_kind in allowed_output_kinds


def test_build_evaluation_meta_includes_required_keys() -> None:
    meta = build_evaluation_meta(
        task="classification",
        output_kind="payload",
        dataset="dummy",
    )

    for key in REQUIRED_META_KEYS:
        assert key in meta

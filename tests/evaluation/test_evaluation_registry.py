from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.evaluation import registry as registry_module
from src.evaluation.registry import (
    get_task,
    list_run_from_config_tasks,
    list_tasks,
    register_builtin_tasks,
    register_task,
    run_task,
)


def test_register_builtin_tasks_exposes_classification_tasks() -> None:
    register_builtin_tasks()
    names = [task.name for task in list_tasks()]
    assert "classification" in names
    assert "classification_limited" in names
    assert "classification_summary" in names
    assert "geometry_based_metrics" in names
    assert "sentence_topic_inspection" in names
    assert "word_based_metrics" in names
    assert "topic_count_diagnostics" in names
    assert "word_based_label_profile" in names
    assert "word_based_topic_word_table" in names
    assert "cross_model_pair_diagnostics" in names


def test_get_task_raises_for_unknown_name() -> None:
    register_builtin_tasks()
    with pytest.raises(ValueError):
        get_task("missing_task")


def test_builtin_task_exposes_output_kind() -> None:
    register_builtin_tasks()
    task = get_task("geometry_based_metrics")
    assert task.output_kind == "path"
    assert task.run_from_config_supported is True


def test_legacy_task_alias_is_rejected_with_canonical_hint() -> None:
    register_builtin_tasks()
    with pytest.raises(ValueError, match="Use 'geometry_based_metrics' instead"):
        get_task("topic_overlap")
    with pytest.raises(ValueError, match="Use 'word_based_metrics' instead"):
        get_task("topic_coherence")
    with pytest.raises(ValueError, match="Use 'sentence_topic_inspection' instead"):
        get_task("geometry_doc_topic_tsne")


def test_list_run_from_config_tasks_returns_supported_subset() -> None:
    register_builtin_tasks()
    names = [task.name for task in list_run_from_config_tasks()]
    assert "classification" in names
    assert "classification_summary" in names
    assert "geometry_based_metrics" in names
    assert "word_based_metrics" in names
    assert "topic_count_diagnostics" in names
    assert "word_based_label_profile" not in names


def test_run_task_returns_runner_result() -> None:
    register_task(
        name="dummy_return",
        description="dummy",
        runner=lambda **_kwargs: {"ok": True},
        output_kind="payload",
        run_from_config_supported=False,
    )

    result = run_task("dummy_return")

    assert result == {"ok": True}


def test_resolve_models_for_word_based_metrics_supports_sentlda() -> None:
    cfg = SimpleNamespace(
        selection=SimpleNamespace(models=None),
        baselines=[
            SimpleNamespace(runner="ctm"),
            SimpleNamespace(runner="sentlda"),
        ],
    )

    resolved = registry_module._resolve_models_for_task(
        cfg,
        task_name="word_based_metrics",
    )

    assert resolved == ["vmf", "ctm", "sentlda"]

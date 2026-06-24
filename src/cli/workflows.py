from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.evaluation.registry import (
    RunFromConfigContext,
    get_task,
    list_run_from_config_tasks,
    register_builtin_tasks,
    run_task_from_config,
)
from src.experiments.comparison_runner import run_comparison
from src.experiments.config import (
    ComparisonConfig,
    load_config,
    normalize_evaluation_task_name,
    resolve_run_selection,
)
from src.experiments.job_planning import resolve_data_runs

DEFAULT_ALL_EXPERIMENT_CONFIGS: tuple[str, ...] = (
    "configs/experiments/20newsgroup.example.yaml",
)


def run_experiments_workflow(
    *,
    config: Path,
    models: str | None,
    seed: int | None,
    seed_base: int | None,
    num_workers: int | None,
    vmf_soft_temp: float | None,
    categories: Sequence[str] | None,
    topics: Sequence[int] | None,
    iterations: Sequence[int] | None,
    encoder_model: str | None = None,
    strip_terminal_normalize: bool | None = None,
) -> Path:
    return run_comparison(
        config_path=config,
        models=models,
        seed=seed,
        seed_base=seed_base,
        num_workers=num_workers,
        vmf_soft_temp=vmf_soft_temp,
        encoder_model=encoder_model,
        strip_terminal_normalize=strip_terminal_normalize,
        categories=None if categories is None else list(categories),
        num_topics=None if topics is None else list(topics),
        iterations=None if iterations is None else list(iterations),
    )


def resolve_limited_classification_setting(
    *,
    ratio: float | None,
    count: int | None,
) -> tuple[str, float | int]:
    if ratio is None and count is None:
        raise ValueError("either --ratio or --count is required")
    if ratio is not None and count is not None:
        raise ValueError("use only one of --ratio or --count")
    if ratio is not None:
        return "ratio", ratio
    return "count", int(count)


def run_all_experiments_workflow(
    *,
    configs: Sequence[Path],
    models: str,
    seed_base: int | None,
    num_workers: int | None,
    vmf_soft_temp: float | None,
    include_all_category_runs: bool,
    all_category_topics: Sequence[int],
    all_category_iterations: Sequence[int],
) -> None:
    all_category_datasets = {"20newsgroup", "nyt"}

    for config_path in configs:
        if not config_path.exists():
            print(f"[skip] config not found: {config_path}")
            continue
        print(f"running {config_path} ...")
        run_comparison(
            config_path=config_path,
            models=models,
            seed_base=seed_base,
            num_workers=num_workers,
            vmf_soft_temp=vmf_soft_temp,
        )
        cfg = load_config(config_path)
        dataset_name = cfg.dataset.name

        if include_all_category_runs and dataset_name in all_category_datasets:
            topics = sorted({int(k) for k in all_category_topics})
            iterations = sorted({int(i) for i in all_category_iterations})
            print(
                f"running {config_path} with overrides "
                f"(category=all, topics={topics}, iterations={iterations}) ..."
            )
            run_comparison(
                config_path=config_path,
                models=models,
                seed_base=seed_base,
                num_workers=num_workers,
                vmf_soft_temp=vmf_soft_temp,
                categories=["all"],
                num_topics=topics,
                iterations=iterations,
            )
    print(
        "experiments run-all finished. Evaluation is not run automatically; "
        "use evaluation run-from-config or an explicit evaluation command."
    )


def run_smoke_workflow(
    *,
    config: Path,
    models: str | None,
    seed: int,
    num_workers: int | None,
    category: Sequence[str],
    topic: Sequence[int],
    iteration: Sequence[int],
) -> Path:
    return run_comparison(
        config_path=config,
        models=models,
        seed=seed,
        num_workers=num_workers,
        categories=list(category) or None,
        num_topics=list(topic) or None,
        iterations=list(iteration) or None,
    )


def _resolve_run_from_config_tasks(
    *,
    cfg: ComparisonConfig,
    task: str | None,
) -> list[str]:
    requested = [task] if task is not None else list(cfg.evaluation.tasks or [])
    if not requested:
        raise ValueError(
            "No evaluation task specified. Use --task or set evaluation.tasks in the config."
        )

    register_builtin_tasks()
    supported_task_names = {item.name for item in list_run_from_config_tasks()}
    resolved: list[str] = []
    for raw_task in requested:
        task_name = normalize_evaluation_task_name(raw_task)
        task_info = get_task(task_name)
        if task_info.name not in supported_task_names:
            raise ValueError(
                f"Evaluation task '{task_name}' is not supported by run-from-config. "
                f"Supported: {sorted(supported_task_names)}"
            )
        if task_info.name not in resolved:
            resolved.append(task_info.name)
    return resolved


def _resolve_run_from_config_axes(
    cfg: ComparisonConfig,
) -> tuple[list[str], list[int], list[int]]:
    categories, topics, iterations = resolve_run_selection(cfg)
    return (
        list(categories.keys()),
        [int(topic) for topic in topics],
        [int(iteration) for iteration in iterations],
    )


def run_evaluation_from_config_workflow(
    *,
    config: Path,
    task: str | None,
    classifiers: Sequence[str],
    vmf_assignment: str,
    result_root: Path,
    target_column: str | None,
    label_schema: str,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str | None = None,
) -> None:
    cfg = load_config(config)
    task_names = _resolve_run_from_config_tasks(cfg=cfg, task=task)
    categories, topics, iterations = _resolve_run_from_config_axes(cfg)
    data_run_names = [run.name for run in resolve_data_runs(cfg)]
    resolved_target_column = (
        target_column or cfg.preprocess.target_column or "target_str"
    )
    classifiers_to_use = list(classifiers) or list(
        cfg.evaluation.classifiers or ["svm"]
    )
    resolved_embedding_variants = (
        list(embedding_variants)
        if embedding_variants is not None
        else cfg.evaluation.embedding_variants
    )
    resolved_feature_resolve_mode = (
        feature_resolve_mode or cfg.evaluation.feature_resolve_mode
    )
    context = RunFromConfigContext(
        cfg=cfg,
        result_root=result_root,
        data_run_names=data_run_names,
        categories=categories,
        topics=topics,
        iterations=iterations,
        classifiers=classifiers_to_use,
        vmf_assignment=vmf_assignment,
        target_column=resolved_target_column,
        label_schema=label_schema,
        alignment_mode=cfg.evaluation.alignment_mode,
        embedding_variants=resolved_embedding_variants,
        feature_resolve_mode=resolved_feature_resolve_mode,
    )

    for task_name in task_names:
        print(f"running evaluation task '{task_name}' from {config} ...")
        run_task_from_config(task_name, context)
    print(f"evaluation run-from-config finished for {config}")

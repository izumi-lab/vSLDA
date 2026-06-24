from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal

TaskRunner = Callable[..., Any]
RunFromConfigRunner = Callable[["RunFromConfigContext"], Any]
OutputKind = Literal["none", "path", "payload", "text"]


@dataclass(frozen=True)
class EvaluationTask:
    name: str
    description: str
    runner: TaskRunner
    output_kind: OutputKind = "none"
    run_from_config_supported: bool = False
    run_from_config_runner: RunFromConfigRunner | None = None


@dataclass(frozen=True)
class RunFromConfigContext:
    cfg: Any
    result_root: Path
    data_run_names: list[str]
    categories: list[str]
    topics: list[int]
    iterations: list[int]
    classifiers: list[str]
    vmf_assignment: str
    target_column: str
    label_schema: str
    alignment_mode: str
    embedding_variants: list[str] | None
    feature_resolve_mode: str


_TASKS: Dict[str, EvaluationTask] = {}
_TASK_ALIASES: Dict[str, str] = {}
_LEGACY_BUILTIN_TASK_ALIASES: Dict[str, str] = {
    "baseline_doc_topic_tsne": "sentence_topic_inspection",
    "geometry_doc_topic_tsne": "sentence_topic_inspection",
    "topic_overlap": "geometry_based_metrics",
    "topic_coherence": "word_based_metrics",
    "topic_count_perplexity": "topic_count_diagnostics",
    "inspect_slda": "sentence_topic_inspection",
    "label_topic_profile": "word_based_label_profile",
    "topic_table_tex": "word_based_topic_word_table",
    "vmf_vs_baseline_pairs": "cross_model_pair_diagnostics",
}


def register_task(
    *,
    name: str,
    description: str,
    runner: TaskRunner,
    output_kind: OutputKind = "none",
    run_from_config_supported: bool = False,
    run_from_config_runner: RunFromConfigRunner | None = None,
) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Task name must be non-empty.")
    if key in _TASKS:
        raise ValueError(f"Evaluation task '{key}' is already registered.")
    _TASKS[key] = EvaluationTask(
        name=key,
        description=description,
        runner=runner,
        output_kind=output_kind,
        run_from_config_supported=run_from_config_supported,
        run_from_config_runner=run_from_config_runner,
    )


def register_task_alias(*, alias: str, target: str) -> None:
    normalized_alias = alias.strip().lower().replace("-", "_")
    normalized_target = target.strip().lower().replace("-", "_")
    if not normalized_alias:
        raise ValueError("Task alias must be non-empty.")
    if normalized_target not in _TASKS:
        raise ValueError(f"Cannot alias unknown task '{target}'.")
    _TASK_ALIASES[normalized_alias] = normalized_target


def get_task(name: str) -> EvaluationTask:
    key = name.strip().lower().replace("-", "_")
    key = _TASK_ALIASES.get(key, key)
    if key in _LEGACY_BUILTIN_TASK_ALIASES:
        canonical = _LEGACY_BUILTIN_TASK_ALIASES[key]
        raise ValueError(
            f"Evaluation task '{name}' is no longer accepted. "
            f"Use '{canonical}' instead."
        )
    if key not in _TASKS:
        available = sorted(_TASKS.keys())
        raise ValueError(f"Unknown evaluation task '{name}'. Available: {available}")
    return _TASKS[key]


def list_tasks() -> list[EvaluationTask]:
    return [_TASKS[name] for name in sorted(_TASKS.keys())]


def list_run_from_config_tasks() -> list[EvaluationTask]:
    return [task for task in list_tasks() if task.run_from_config_supported]


def run_task(name: str, **kwargs: Any) -> Any:
    task = get_task(name)
    return task.runner(**kwargs)


def _configured_experiment_models(cfg: Any) -> list[str]:
    resolved = ["vmf_sentence_lda"]
    for baseline in cfg.baselines:
        runner = str(baseline.runner).strip().lower()
        if runner and runner not in resolved:
            resolved.append(runner)
    return resolved


def _resolve_models_for_task(
    cfg: Any,
    *,
    task_name: str,
) -> list[str]:
    explicit_selection = cfg.selection.models is not None
    configured_models = cfg.selection.models or _configured_experiment_models(cfg)

    if task_name == "geometry_based_metrics":
        aliases = {
            "vmf_sentence_lda": "vmf",
            "vmf": "vmf",
            "sentence_gaussianlda": "gaussian",
            "gaussian": "gaussian",
        }
        supported = {"vmf", "gaussian"}
    elif task_name == "word_based_metrics":
        aliases = {
            "vmf_sentence_lda": "vmf",
            "vmf": "vmf",
        }
        supported = {
            "vmf",
            "gaussian",
            "sentence_gaussianlda",
            "bleilda",
            "bertopic_kmeans",
            "gaussianlda",
            "mvtm",
            "ctm",
            "senclu",
            "sentlda",
            "spherical_kmeans",
            "gaussian_kmeans",
            "movmf",
            "gaussian_mixture",
        }
    else:
        raise ValueError(f"Task '{task_name}' does not use model selection.")

    resolved: list[str] = []
    unsupported: list[str] = []
    for model_name in configured_models:
        normalized = str(model_name).strip().lower()
        mapped = aliases.get(normalized, normalized)
        if mapped not in supported:
            unsupported.append(normalized)
            continue
        if mapped not in resolved:
            resolved.append(mapped)

    if explicit_selection and unsupported:
        raise ValueError(
            f"Task '{task_name}' does not support selection.models={unsupported}."
        )
    if not resolved:
        raise ValueError(
            f"Task '{task_name}' could not resolve any supported model from selection.models."
        )
    return resolved


def _run_classification_from_config(context: RunFromConfigContext) -> Any:
    return run_task(
        "classification",
        iterations=context.iterations,
        datasets=[context.cfg.dataset.name],
        data_runs=context.data_run_names,
        categories=context.categories,
        topics=context.topics,
        classifiers=context.classifiers,
        vmf_assignment=context.vmf_assignment,
        result_root=context.result_root,
        target_column=context.target_column,
        label_schema=context.label_schema,
        alignment_mode=context.alignment_mode,
        embedding_variants=context.embedding_variants,
        feature_resolve_mode=context.feature_resolve_mode,
    )


def _run_classification_summary_from_config(context: RunFromConfigContext) -> None:
    for topic in context.topics:
        for data_run in context.data_run_names:
            run_task(
                "classification_summary",
                metric="acc",
                dataset=context.cfg.dataset.name,
                data_run=data_run,
                topics=int(topic),
                iterations=context.iterations,
                classifiers=context.classifiers,
                vmf_assignment=context.vmf_assignment,
                alignment_mode=context.alignment_mode,
                result_root=context.result_root,
                target_column=context.target_column,
                label_schema=context.label_schema,
                embedding_variants=context.embedding_variants,
                feature_resolve_mode=context.feature_resolve_mode,
                selected_models=None,
            )


def _run_topic_count_diagnostics_from_config(context: RunFromConfigContext) -> Any:
    return run_task(
        "topic_count_diagnostics",
        dataset=context.cfg.dataset.name,
        iterations=context.iterations,
        topics=context.topics,
        categories=context.categories,
        data_runs=context.data_run_names,
    )


def _run_geometry_based_metrics_from_config(context: RunFromConfigContext) -> None:
    models = _resolve_models_for_task(context.cfg, task_name="geometry_based_metrics")
    embedding_variants = context.embedding_variants or [
        context.cfg.encoder.embedding_variant
    ]
    for topic in context.topics:
        for embedding_variant in embedding_variants:
            run_task(
                "geometry_based_metrics",
                models=models,
                dataset=context.cfg.dataset.name,
                data_runs=context.data_run_names,
                iterations=context.iterations,
                num_topics=int(topic),
                categories=context.categories,
                embedding_variant=embedding_variant,
                encoder_model=context.cfg.encoder.model_name,
            )


def _run_word_based_metrics_from_config(context: RunFromConfigContext) -> None:
    models = _resolve_models_for_task(context.cfg, task_name="word_based_metrics")
    run_task(
        "word_based_metrics",
        models=models,
        dataset=context.cfg.dataset.name,
        data_runs=context.data_run_names,
        iterations=context.iterations,
        num_topics=[int(topic) for topic in context.topics],
        categories=context.categories,
        language=context.cfg.preprocess.language,
        delimiter=context.cfg.preprocess.delimiter or " / ",
        ja_replace_num=context.cfg.preprocess.ja_replace_num,
        ja_dicdir=context.cfg.preprocess.ja_dicdir,
        ja_require_unidic=context.cfg.preprocess.ja_require_unidic,
    )


def run_task_from_config(name: str, context: RunFromConfigContext) -> Any:
    task = get_task(name)
    if not task.run_from_config_supported or task.run_from_config_runner is None:
        raise ValueError(
            f"Evaluation task '{name}' is not supported by run-from-config."
        )
    return task.run_from_config_runner(context)


def register_builtin_tasks() -> None:
    if _TASKS:
        return

    from src.evaluation.classification import (
        run_classification_suite,
        run_limited_classification_suite,
        write_classification_summary,
    )
    from src.evaluation.diagnostics.cross_model_pair_diagnostics import (
        run_cross_model_pair_diagnostics,
    )
    from src.evaluation.diagnostics.sentence_topic_inspection import (
        run_sentence_topic_inspection,
    )
    from src.evaluation.diagnostics.topic_count_diagnostics import (
        run_topic_count_diagnostics,
    )
    from src.evaluation.geometry_based.metrics import run_geometry_based_metrics
    from src.evaluation.word_based.label_profile import (
        run_word_based_label_profile,
    )
    from src.evaluation.word_based.metrics import run_word_based_metrics
    from src.evaluation.word_based.topic_word_table import (
        run_word_based_topic_word_table,
    )

    register_task(
        name="classification",
        description="Run full classification evaluation.",
        runner=run_classification_suite,
        output_kind="none",
        run_from_config_supported=True,
        run_from_config_runner=_run_classification_from_config,
    )
    register_task(
        name="classification_limited",
        description="Run limited-data classification evaluation.",
        runner=run_limited_classification_suite,
        output_kind="none",
    )
    register_task(
        name="classification_summary",
        description="Aggregate classification metrics for reporting.",
        runner=write_classification_summary,
        output_kind="text",
        run_from_config_supported=True,
        run_from_config_runner=_run_classification_summary_from_config,
    )
    register_task(
        name="geometry_based_metrics",
        description="Analyze geometry-based metrics across iterations.",
        runner=run_geometry_based_metrics,
        output_kind="path",
        run_from_config_supported=True,
        run_from_config_runner=_run_geometry_based_metrics_from_config,
    )
    register_task(
        name="word_based_metrics",
        description="Analyze topic-word metrics such as coherence and diversity across iterations.",
        runner=run_word_based_metrics,
        output_kind="path",
        run_from_config_supported=True,
        run_from_config_runner=_run_word_based_metrics_from_config,
    )
    register_task(
        name="sentence_topic_inspection",
        description="Inspect sentence-topic alignment and generate provenance-aware visual diagnostics for vMF Sentence LDA runs.",
        runner=run_sentence_topic_inspection,
        output_kind="path",
    )
    register_task(
        name="topic_count_diagnostics",
        description="Summarize topic-count diagnostics such as perplexity across runs.",
        runner=run_topic_count_diagnostics,
        output_kind="path",
        run_from_config_supported=True,
        run_from_config_runner=_run_topic_count_diagnostics_from_config,
    )
    register_task(
        name="word_based_label_profile",
        description="Analyze label-wise profiles in the representative-word-based pipeline.",
        runner=run_word_based_label_profile,
        output_kind="payload",
    )
    register_task(
        name="word_based_topic_word_table",
        description="Render representative-word-based outputs as LaTeX tables with provenance-aware sidecars.",
        runner=run_word_based_topic_word_table,
        output_kind="path",
    )
    register_task(
        name="cross_model_pair_diagnostics",
        description="Find document pairs that are far in one model space but close in another.",
        runner=run_cross_model_pair_diagnostics,
        output_kind="path",
    )

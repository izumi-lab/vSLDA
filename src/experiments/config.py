from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from src.data.catalog import resolve_category_targets

from .config_loader import load_config_yaml_resolved
from .config_parsers import (
    ensure_list,
    normalize_evaluation_task_name,
    parse_baselines,
    parse_dataset_config,
    parse_encoder_config,
    parse_evaluation_config,
    parse_experiment_config,
    parse_models_arg,
    parse_output_root,
    parse_preprocess_config,
    parse_runtime_config,
    parse_selection_config,
    parse_train_config,
    parse_vmf_config,
    validate_preset_config,
)
from .config_schema import (
    BaselineConfig,
    ComparisonConfig,
    DatasetConfig,
    EncoderConfig,
    EvaluationConfig,
    ExperimentConfig,
    PreprocessConfig,
    PresetConfig,
    RuntimeConfig,
    SelectionConfig,
    TrainConfig,
    VmfConfig,
    VmfInferenceConfig,
)


def apply_encoder_overrides(
    raw_cfg: dict,
    *,
    encoder_model: str | None = None,
    strip_terminal_normalize: bool | None = None,
) -> dict:
    if encoder_model is None and strip_terminal_normalize is None:
        return raw_cfg
    cfg = dict(raw_cfg)
    encoder_cfg = dict(cfg.get("encoder", {}))
    if encoder_model is not None:
        encoder_cfg["model_name"] = encoder_model
    if strip_terminal_normalize is not None:
        encoder_cfg["strip_terminal_normalize"] = strip_terminal_normalize
    cfg["encoder"] = encoder_cfg
    return cfg


def load_config(
    path: str | Path,
    *,
    encoder_model: str | None = None,
    strip_terminal_normalize: bool | None = None,
) -> ComparisonConfig:
    cfg = load_config_yaml_resolved(path)
    cfg = apply_encoder_overrides(
        cfg,
        encoder_model=encoder_model,
        strip_terminal_normalize=strip_terminal_normalize,
    )
    preset = validate_preset_config(cfg)
    dataset = parse_dataset_config(cfg)
    train = parse_train_config(cfg)
    encoder = parse_encoder_config(cfg)
    preprocess = parse_preprocess_config(cfg)
    experiments = parse_experiment_config(cfg)
    selection = parse_selection_config(cfg)
    evaluation = parse_evaluation_config(cfg)
    runtime = parse_runtime_config(cfg)
    vmf = parse_vmf_config(cfg)
    baselines = parse_baselines(cfg, encoder=encoder)
    output_root = parse_output_root(cfg, dataset_name=dataset.name)

    return ComparisonConfig(
        dataset=dataset,
        train=train,
        encoder=encoder,
        experiments=experiments,
        baselines=baselines,
        output_root=output_root,
        preprocess=preprocess,
        selection=selection,
        preset=preset,
        evaluation=evaluation,
        runtime=runtime,
        vmf=vmf,
    )


def resolve_model_selection(
    cfg: ComparisonConfig,
    *,
    models: str | None = None,
) -> set[str] | None:
    """Resolve CLI and config model filters into normalized runner keys."""
    cli_models = parse_models_arg(models)
    if cli_models is not None:
        return cli_models
    if cfg.selection.models is None:
        return None
    return {str(model).strip().lower() for model in cfg.selection.models}


def resolve_run_selection(
    cfg: ComparisonConfig,
    *,
    categories: Sequence[str] | None = None,
    num_topics: Sequence[int] | None = None,
    iterations: Sequence[int] | None = None,
) -> tuple[Mapping[str, Sequence[str] | None], list[int], list[int]]:
    """Resolve preset defaults and CLI overrides into concrete run axes."""
    selected_categories = (
        categories if categories is not None else cfg.selection.categories
    )
    resolved_categories: Mapping[str, Sequence[str] | None]
    if selected_categories is None:
        resolved_categories = cfg.dataset.categories
    else:
        selected: dict[str, Sequence[str] | None] = {}
        for raw_category in selected_categories:
            category = str(raw_category).strip()
            if not category:
                continue
            if category == "all":
                selected["all"] = None
                continue
            if category not in cfg.dataset.categories:
                available = sorted(cfg.dataset.categories.keys())
                raise ValueError(
                    f"Unknown category '{category}'. "
                    f"Available categories: {available} and 'all'."
                )
            selected[category] = cfg.dataset.categories[category]
        if not selected:
            raise ValueError("At least one category must be selected.")
        resolved_categories = selected

    resolved_topics = (
        [int(value) for value in num_topics]
        if num_topics is not None
        else [int(value) for value in (cfg.selection.topics or cfg.train.num_topics)]
    )
    resolved_iterations = (
        [int(value) for value in iterations]
        if iterations is not None
        else [
            int(value)
            for value in (cfg.selection.iterations or cfg.experiments.iterations)
        ]
    )
    return resolved_categories, resolved_topics, resolved_iterations


def resolve_targets(
    dataset_cfg: DatasetConfig,
    preprocess_cfg: PreprocessConfig,
    category: str,
    targets: Sequence[str] | None,
) -> Sequence[str] | None:
    return resolve_category_targets(
        dataset_cfg.name,
        category,
        targets,
        target_column=preprocess_cfg.target_column or "target_str",
        train_csv=dataset_cfg.train_csv,
        has_labels=preprocess_cfg.has_labels,
        allow_all_unfiltered=False,
    )

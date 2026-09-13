from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from src.baselines.params import normalize_covariance_type
from src.core.vmf_variant import (
    VMF_HYPERPARAMETER_CONFIG_FIELDS,
    VMF_HYPERPARAMETER_KEYS,
)
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
    DEFAULT_SAEM_BURN_IN,
    DEFAULT_SAEM_DECAY,
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

# Baseline runners whose topic parameters live in a pretrained word-vector space.
WORD_EMBEDDING_RUNNERS = {
    "gaussianlda",
    "etm",
    "mvtm",
}


def apply_encoder_overrides(
    raw_cfg: dict,
    *,
    encoder_model: str | None = None,
    strip_terminal_normalize: bool | None = None,
    encoder_device: str | None = None,
) -> dict:
    """Override the encoder block from the command line.

    ``encoder_device`` selects the torch device the sentence encoder runs on ("auto" resolves
    to cuda when available, else cpu). It is not part of the condition fingerprint, so the same
    condition trains identically on a GPU host and on a CPU-only one; only the run's recorded
    ``encoder_config`` differs.
    """
    if (
        encoder_model is None
        and strip_terminal_normalize is None
        and encoder_device is None
    ):
        return raw_cfg
    cfg = dict(raw_cfg)
    encoder_cfg = dict(cfg.get("encoder", {}))
    if encoder_model is not None:
        encoder_cfg["model_name"] = encoder_model
    if strip_terminal_normalize is not None:
        encoder_cfg["strip_terminal_normalize"] = strip_terminal_normalize
    if encoder_device is not None:
        encoder_cfg["device"] = encoder_device
    cfg["encoder"] = encoder_cfg
    return cfg


def apply_gaussian_prior_scale_override(
    raw_cfg: dict,
    *,
    prior_scale: float | None = None,
) -> dict:
    if prior_scale is None:
        return raw_cfg
    cfg = dict(raw_cfg)
    baselines = []
    for raw_baseline in cfg.get("baselines", []):
        baseline = dict(raw_baseline)
        if str(baseline.get("runner", "")).strip().lower() in {
            "gaussianlda",
            "sentence_gaussianlda",
        }:
            params = dict(baseline.get("params") or {})
            params["prior_scale"] = float(prior_scale)
            baseline["params"] = params
        baselines.append(baseline)
    cfg["baselines"] = baselines
    return cfg


def apply_gaussian_covariance_type_override(
    raw_cfg: dict,
    *,
    covariance_type: str | None = None,
) -> dict:
    """Set ``params.covariance_type`` of every ``sentence_gaussianlda`` baseline."""
    if covariance_type is None:
        return raw_cfg
    normalized = normalize_covariance_type(covariance_type)
    cfg = dict(raw_cfg)
    baselines = []
    for raw_baseline in cfg.get("baselines", []):
        baseline = dict(raw_baseline)
        if str(baseline.get("runner", "")).strip().lower() == "sentence_gaussianlda":
            params = dict(baseline.get("params") or {})
            params["covariance_type"] = normalized
            baseline["params"] = params
        baselines.append(baseline)
    cfg["baselines"] = baselines
    return cfg


def apply_vmf_hyperparameter_override(
    raw_cfg: dict,
    *,
    kappa0: float | None = None,
    alpha0: float | None = None,
    gibbs_sweeps: int | None = None,
    num_samples: int | None = None,
    num_iterations: int | None = None,
    saem_burn_in: int | None = None,
    saem_decay: float | None = None,
) -> dict:
    """Override the vMF training hyperparameters of ``train`` from the command line.

    A key is recorded in ``train.hyperparameter_overrides`` only when the new value differs
    from the YAML value, so passing the configured value leaves the result path unchanged
    (mirrors the unsuffixed default of ``--prior-scale 0.1``).
    """
    requested = {
        "kappa0": kappa0,
        "alpha0": alpha0,
        "gibbs_sweeps": gibbs_sweeps,
        "num_samples": num_samples,
        "num_iterations": num_iterations,
        "saem_burn_in": saem_burn_in,
        "saem_decay": saem_decay,
    }
    if all(value is None for value in requested.values()):
        return raw_cfg
    cfg = dict(raw_cfg)
    train_cfg = dict(cfg.get("train", {}))
    overrides = [str(key) for key in (train_cfg.get("hyperparameter_overrides") or [])]
    for key in VMF_HYPERPARAMETER_KEYS:
        value = requested[key]
        if value is None:
            continue
        field_name = VMF_HYPERPARAMETER_CONFIG_FIELDS[key]
        if key == "gibbs_sweeps":
            current = train_cfg.get("gibbs_sweeps", train_cfg.get("zeta", 1))
            train_cfg.pop("zeta", None)
        elif key == "num_samples":
            current = train_cfg.get("num_samples", train_cfg.get("B", 1))
            train_cfg.pop("B", None)
        elif key == "kappa0":
            current = train_cfg.get("kappa_default", 10.0)
        elif key == "saem_burn_in":
            current = train_cfg.get("saem_burn_in", DEFAULT_SAEM_BURN_IN)
        elif key == "saem_decay":
            current = train_cfg.get("saem_decay", DEFAULT_SAEM_DECAY)
        else:
            current = train_cfg.get(field_name)
        if key in {"gibbs_sweeps", "num_samples", "num_iterations"}:
            new_value: float | int = int(value)
            if new_value < 1:
                raise ValueError(f"--{key.replace('_', '-')} must be >= 1.")
        elif key == "saem_burn_in":
            # The burn-in T_0 of the SAEM (config_schema.TrainConfig): 0 <= T_0 <= T.
            new_value = int(value)
            if new_value < 0:
                raise ValueError("--saem-burn-in must be >= 0.")
        elif key == "saem_decay":
            new_value = float(value)
            if not (0.5 < new_value <= 1.0):
                raise ValueError("--saem-decay must lie in (1/2, 1].")
        else:
            new_value = float(value)
            if new_value <= 0.0:
                raise ValueError(f"--{key} must be > 0.")
        train_cfg[field_name] = new_value
        differs = (
            current is None
            or isinstance(current, (list, tuple))
            or float(current) != float(new_value)
        )
        if differs and key not in overrides:
            overrides.append(key)
    burn_in = train_cfg.get("saem_burn_in")
    iterations = train_cfg.get("num_iterations")
    if (
        burn_in is not None
        and iterations is not None
        and int(burn_in) > int(iterations)
    ):
        raise ValueError(
            f"--saem-burn-in ({burn_in}) must not exceed train.num_iterations ({iterations})."
        )
    train_cfg["hyperparameter_overrides"] = overrides
    cfg["train"] = train_cfg
    return cfg


def apply_word2vec_override(
    raw_cfg: dict,
    *,
    word2vec: str | None = None,
) -> dict:
    if word2vec is None:
        return raw_cfg
    cfg = dict(raw_cfg)
    baselines = []
    for raw_baseline in cfg.get("baselines", []):
        baseline = dict(raw_baseline)
        if str(baseline.get("runner", "")).strip().lower() in WORD_EMBEDDING_RUNNERS:
            params = dict(baseline.get("params") or {})
            params["word2vec"] = str(word2vec)
            baseline["params"] = params
        baselines.append(baseline)
    cfg["baselines"] = baselines
    return cfg


def load_config(
    path: str | Path,
    *,
    encoder_model: str | None = None,
    strip_terminal_normalize: bool | None = None,
    encoder_device: str | None = None,
    prior_scale: float | None = None,
    word2vec: str | None = None,
    covariance_type: str | None = None,
    kappa0: float | None = None,
    alpha0: float | None = None,
    gibbs_sweeps: int | None = None,
    num_samples: int | None = None,
    num_iterations: int | None = None,
    saem_burn_in: int | None = None,
    saem_decay: float | None = None,
) -> ComparisonConfig:
    cfg = load_config_yaml_resolved(path)
    cfg = apply_encoder_overrides(
        cfg,
        encoder_model=encoder_model,
        strip_terminal_normalize=strip_terminal_normalize,
        encoder_device=encoder_device,
    )
    cfg = apply_gaussian_prior_scale_override(cfg, prior_scale=prior_scale)
    cfg = apply_gaussian_covariance_type_override(cfg, covariance_type=covariance_type)
    cfg = apply_word2vec_override(cfg, word2vec=word2vec)
    cfg = apply_vmf_hyperparameter_override(
        cfg,
        kappa0=kappa0,
        alpha0=alpha0,
        gibbs_sweeps=gibbs_sweeps,
        num_samples=num_samples,
        num_iterations=num_iterations,
        saem_burn_in=saem_burn_in,
        saem_decay=saem_decay,
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

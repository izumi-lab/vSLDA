from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.baselines.params import normalize_baseline_params
from src.core.paths import resolve_project_path
from src.data.text_processing import normalize_segmenter_name, normalize_tokenizer_name
from src.utils.encoder_profiles import resolve_encoder_settings
from src.utils.random import DEFAULT_RANDOM_SEED

from .config_schema import (
    BaselineConfig,
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


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def parse_models_arg(models_arg: str | None) -> set[str] | None:
    if not models_arg:
        return None
    return {m.strip().lower() for m in models_arg.split(",") if m.strip()}


def normalize_evaluation_task_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def normalize_optional_str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    normalized = [str(item).strip() for item in ensure_list(value) if str(item).strip()]
    return normalized or None


def normalize_optional_int_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    normalized = [int(item) for item in ensure_list(value)]
    return normalized or None


def normalize_optional_model_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    normalized = [
        str(item).strip().lower() for item in ensure_list(value) if str(item).strip()
    ]
    return normalized or None


def normalize_optional_task_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    normalized = [
        normalize_evaluation_task_name(item)
        for item in ensure_list(value)
        if str(item).strip()
    ]
    return normalized or None


def normalize_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def validate_preset_config(raw_cfg: Dict[str, Any]) -> PresetConfig:
    preset_cfg = raw_cfg.get("preset", {})
    kind = str(preset_cfg.get("kind", "standard")).strip().lower()
    purpose = str(preset_cfg.get("purpose", "quantitative")).strip().lower()

    allowed_kinds = {"standard", "smoke", "qualitative_allfit"}
    if kind not in allowed_kinds:
        raise ValueError(f"preset.kind must be one of {sorted(allowed_kinds)}.")

    allowed_purposes = {"quantitative", "qualitative"}
    if purpose not in allowed_purposes:
        raise ValueError(f"preset.purpose must be one of {sorted(allowed_purposes)}.")

    if kind == "qualitative_allfit" and purpose != "qualitative":
        raise ValueError(
            "preset.kind='qualitative_allfit' requires preset.purpose='qualitative'."
        )

    return PresetConfig(kind=kind, purpose=purpose)


def parse_dataset_config(raw_cfg: Dict[str, Any]) -> DatasetConfig:
    dataset_cfg = raw_cfg["dataset"]
    fiscal_years_raw = dataset_cfg.get("fiscal_years")
    fiscal_years: Optional[List[int]] = None
    if fiscal_years_raw is not None:
        fiscal_years = [int(y) for y in ensure_list(fiscal_years_raw)]
    by_fy_root_value = dataset_cfg.get("by_fy_root")
    by_fy_root = (
        resolve_project_path(by_fy_root_value) if by_fy_root_value is not None else None
    )

    train_csv_value = dataset_cfg.get("train_csv")
    test_csv_value = dataset_cfg.get("test_csv")
    if train_csv_value is None:
        if fiscal_years and by_fy_root is not None:
            train_csv_value = by_fy_root / f"fy{fiscal_years[0]}" / "train.csv"
        else:
            raise ValueError("dataset.train_csv is required.")
    if test_csv_value is None:
        if fiscal_years and by_fy_root is not None:
            test_csv_value = by_fy_root / f"fy{fiscal_years[0]}" / "test.csv"
        else:
            test_csv_value = train_csv_value

    dataset = DatasetConfig(
        name=dataset_cfg["name"],
        train_csv=resolve_project_path(train_csv_value),
        test_csv=resolve_project_path(test_csv_value),
        categories=dataset_cfg["categories"],
        by_fy_root=by_fy_root,
        fiscal_years=fiscal_years,
        fiscal_year_mode=dataset_cfg.get("fiscal_year_mode", "concat_years"),
    )
    if dataset.fiscal_year_mode not in {"per_year", "concat_years"}:
        raise ValueError(
            "dataset.fiscal_year_mode must be either 'per_year' or 'concat_years'."
        )
    if dataset.fiscal_years and dataset.by_fy_root is None:
        raise ValueError(
            "dataset.by_fy_root is required when dataset.fiscal_years is set."
        )
    return dataset


def parse_train_config(raw_cfg: Dict[str, Any]) -> TrainConfig:
    train_cfg = raw_cfg["train"]
    gibbs_sweeps = int(train_cfg.get("gibbs_sweeps", train_cfg.get("zeta", 1)))
    num_samples = int(train_cfg.get("num_samples", train_cfg.get("B", 1)))
    alpha_raw = train_cfg.get("alpha", None)
    if isinstance(alpha_raw, (list, tuple)):
        alpha = [float(a) for a in alpha_raw]
    elif alpha_raw is None:
        alpha = None
    else:
        alpha = float(alpha_raw)

    train = TrainConfig(
        num_topics=ensure_list(train_cfg["num_topics"]),
        num_iterations=int(train_cfg["num_iterations"]),
        alpha=alpha,
        kappa_default=float(train_cfg.get("kappa_default", 10.0)),
        num_components=int(train_cfg.get("num_components", 1)),
        gibbs_sweeps=gibbs_sweeps,
        num_samples=num_samples,
        estimate_alpha=bool(train_cfg.get("estimate_alpha", True)),
        alpha_update_every=int(train_cfg.get("alpha_update_every", 1)),
        alpha_max_iter=int(train_cfg.get("alpha_max_iter", 100)),
        alpha_tol=float(train_cfg.get("alpha_tol", 1e-5)),
        alpha_min_value=float(train_cfg.get("alpha_min_value", 1e-3)),
        repair_empty_topics=bool(train_cfg.get("repair_empty_topics", True)),
        min_topic_count_for_repair=int(train_cfg.get("min_topic_count_for_repair", 1)),
        avg_log_likelihood_every=int(train_cfg.get("avg_log_likelihood_every", 1)),
        invariant_check_every=int(train_cfg.get("invariant_check_every", 1)),
    )
    if train.num_components < 1:
        raise ValueError("train.num_components must be >= 1.")
    if train.alpha_min_value <= 0.0:
        raise ValueError("train.alpha_min_value must be > 0.")
    if train.min_topic_count_for_repair < 1:
        raise ValueError("train.min_topic_count_for_repair must be >= 1.")
    if train.avg_log_likelihood_every < 1:
        raise ValueError("train.avg_log_likelihood_every must be >= 1.")
    if train.invariant_check_every < 1:
        raise ValueError("train.invariant_check_every must be >= 1.")
    return train


def parse_encoder_config(raw_cfg: Dict[str, Any]) -> EncoderConfig:
    enc_cfg = raw_cfg.get("encoder", {})
    encode_prefix_raw = enc_cfg.get("encode_prefix")
    encode_prefix = None if encode_prefix_raw is None else str(encode_prefix_raw)
    encode_prompt_raw = enc_cfg.get("encode_prompt")
    encode_prompt = None if encode_prompt_raw is None else str(encode_prompt_raw)
    encode_prompt_name_raw = enc_cfg.get("encode_prompt_name")
    encode_prompt_name = (
        None if encode_prompt_name_raw is None else str(encode_prompt_name_raw)
    )
    pooling_raw = enc_cfg.get("pooling")
    pooling = None if pooling_raw is None else str(pooling_raw)

    raw_transform = str(enc_cfg.get("pre_normalize_transform", "none")).strip().lower()
    transform_aliases = {
        "none": "none",
        "mean_center": "mean_center",
        "mean-center": "mean_center",
        "meancenter": "mean_center",
        "whitening": "whitening",
        "whiten": "whitening",
    }
    if raw_transform not in transform_aliases:
        raise ValueError(
            "encoder.pre_normalize_transform must be one of {none, mean_center, whitening}."
        )

    whitening_eps = float(enc_cfg.get("whitening_eps", 1e-5))
    if whitening_eps <= 0.0:
        raise ValueError("encoder.whitening_eps must be > 0.")

    encode_batch_size_raw = enc_cfg.get("encode_batch_size")
    encode_batch_size = (
        None if encode_batch_size_raw is None else int(encode_batch_size_raw)
    )
    if encode_batch_size is not None and encode_batch_size <= 0:
        raise ValueError("encoder.encode_batch_size must be > 0.")

    model_kwargs = dict(enc_cfg.get("model_kwargs") or {})
    tokenizer_kwargs = dict(enc_cfg.get("tokenizer_kwargs") or {})
    normalize_embeddings_raw = enc_cfg.get("normalize_embeddings")
    normalize_embeddings = (
        None if normalize_embeddings_raw is None else bool(normalize_embeddings_raw)
    )
    truncate_dim_raw = enc_cfg.get("truncate_dim")
    truncate_dim = None if truncate_dim_raw is None else int(truncate_dim_raw)
    resolved = resolve_encoder_settings(
        model_name=str(
            enc_cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2")
        ),
        backend=str(enc_cfg.get("backend", "auto")),
        pooling=pooling,
        encode_prefix=encode_prefix,
        encode_prompt=encode_prompt,
        encode_prompt_name=encode_prompt_name,
        encode_batch_size=encode_batch_size,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        normalize_embeddings=normalize_embeddings,
        truncate_dim=truncate_dim,
    )

    return EncoderConfig(
        model_name=resolved.model_name,
        device=str(enc_cfg.get("device", "cuda")),
        encode_prefix=resolved.encode_prefix,
        backend=resolved.backend,
        pooling=resolved.pooling,
        encode_prompt=resolved.encode_prompt,
        encode_prompt_name=resolved.encode_prompt_name,
        encode_batch_size=resolved.encode_batch_size,
        model_kwargs=resolved.model_kwargs,
        tokenizer_kwargs=resolved.tokenizer_kwargs,
        normalize_embeddings=resolved.normalize_embeddings,
        truncate_dim=resolved.truncate_dim,
        strip_terminal_normalize=normalize_bool(
            enc_cfg.get("strip_terminal_normalize", True),
            name="encoder.strip_terminal_normalize",
        ),
        embedding_variant=resolved.embedding_variant,
        pre_normalize_transform=transform_aliases[raw_transform],
        whitening_eps=whitening_eps,
    )


def parse_preprocess_config(raw_cfg: Dict[str, Any]) -> PreprocessConfig:
    dataset_cfg = raw_cfg["dataset"]
    preprocess_cfg = raw_cfg.get("preprocess", {})
    raw_segmenter = str(
        preprocess_cfg.get("segmenter", dataset_cfg.get("segmenter", "delimiter"))
    ).strip()
    raw_tokenizer = str(
        preprocess_cfg.get("tokenizer", dataset_cfg.get("tokenizer", "default"))
    ).strip()
    if not raw_segmenter:
        raise ValueError("preprocess.segmenter must not be empty.")
    if not raw_tokenizer:
        raise ValueError("preprocess.tokenizer must not be empty.")

    dataset_language = str(dataset_cfg.get("language", "english"))
    dataset_delimiter = dataset_cfg.get("delimiter", " / ")
    dataset_text_column = str(dataset_cfg.get("text_column", "data"))
    dataset_target_column = dataset_cfg.get("target_column", "target_str")
    dataset_has_labels = bool(dataset_cfg.get("has_labels", True))

    language = str(preprocess_cfg.get("language", dataset_language))
    segmenter = normalize_segmenter_name(raw_segmenter)
    tokenizer = normalize_tokenizer_name(language, raw_tokenizer)

    return PreprocessConfig(
        language=language,
        delimiter=preprocess_cfg.get("delimiter", dataset_delimiter),
        text_column=str(preprocess_cfg.get("text_column", dataset_text_column)),
        target_column=preprocess_cfg.get("target_column", dataset_target_column),
        has_labels=bool(preprocess_cfg.get("has_labels", dataset_has_labels)),
        segmenter=segmenter,
        tokenizer=tokenizer,
        legacy_preprocessing=(
            None
            if preprocess_cfg.get("legacy_preprocessing") is None
            else bool(preprocess_cfg.get("legacy_preprocessing"))
        ),
        ja_replace_num=bool(
            preprocess_cfg.get(
                "ja_replace_num", dataset_cfg.get("ja_replace_num", True)
            )
        ),
        ja_stopwords_path=preprocess_cfg.get(
            "ja_stopwords_path", dataset_cfg.get("ja_stopwords_path")
        ),
        ja_dicdir=preprocess_cfg.get("ja_dicdir", dataset_cfg.get("ja_dicdir")),
        ja_require_unidic=bool(
            preprocess_cfg.get(
                "ja_require_unidic", dataset_cfg.get("ja_require_unidic", True)
            )
        ),
    )


def parse_experiment_config(raw_cfg: Dict[str, Any]) -> ExperimentConfig:
    exp_cfg = raw_cfg.get("experiments", {})
    return ExperimentConfig(iterations=ensure_list(exp_cfg.get("iterations", [0])))


def parse_selection_config(raw_cfg: Dict[str, Any]) -> SelectionConfig:
    selection_cfg = raw_cfg.get("selection", {})
    return SelectionConfig(
        models=normalize_optional_model_list(selection_cfg.get("models")),
        categories=normalize_optional_str_list(selection_cfg.get("categories")),
        topics=normalize_optional_int_list(selection_cfg.get("topics")),
        iterations=normalize_optional_int_list(selection_cfg.get("iterations")),
    )


def parse_evaluation_config(raw_cfg: Dict[str, Any]) -> EvaluationConfig:
    evaluation_cfg = raw_cfg.get("evaluation", {})
    embedding_variants_raw = evaluation_cfg.get(
        "embedding_variants",
        evaluation_cfg.get("embedding_variant"),
    )
    feature_resolve_mode = (
        str(evaluation_cfg.get("feature_resolve_mode", "all")).strip() or "all"
    )
    if feature_resolve_mode not in {"all", "strict"}:
        raise ValueError(
            "evaluation.feature_resolve_mode must be one of {all, strict}."
        )
    return EvaluationConfig(
        tasks=normalize_optional_task_list(evaluation_cfg.get("tasks")),
        classifiers=normalize_optional_model_list(evaluation_cfg.get("classifiers")),
        alignment_mode=str(evaluation_cfg.get("alignment_mode", "intersection")).strip()
        or "intersection",
        embedding_variants=normalize_optional_str_list(embedding_variants_raw),
        feature_resolve_mode=feature_resolve_mode,
    )


def parse_runtime_config(raw_cfg: Dict[str, Any]) -> RuntimeConfig:
    runtime_cfg = raw_cfg.get("runtime", {})
    seed_base_raw = runtime_cfg.get("seed_base", DEFAULT_RANDOM_SEED)
    return RuntimeConfig(
        seed_base=None if seed_base_raw is None else int(seed_base_raw),
        num_workers=int(runtime_cfg.get("num_workers", 1)),
    )


def parse_vmf_config(raw_cfg: Dict[str, Any]) -> VmfConfig:
    vmf_cfg = raw_cfg.get("vmf", {})
    vmf_inference_cfg = vmf_cfg.get("inference", {})
    return VmfConfig(
        inference=VmfInferenceConfig(
            soft_temperature=float(vmf_inference_cfg.get("soft_temperature", 1.0))
        )
    )


def _baseline_params_with_encoder_defaults(
    *,
    runner: str,
    params: Any,
    encoder: EncoderConfig,
) -> dict[str, Any]:
    options = {} if params is None else dict(params)
    runner_key = str(runner).strip().lower()

    def _set_common_encoder_defaults(*, model_key: str, prefix_key: str) -> None:
        options.setdefault(model_key, encoder.model_name)
        options.setdefault(prefix_key, encoder.encode_prefix)
        options.setdefault("encoder_backend", encoder.backend)
        options.setdefault("pooling", encoder.pooling)
        options.setdefault("encode_prompt", encoder.encode_prompt)
        options.setdefault("encode_prompt_name", encoder.encode_prompt_name)
        options.setdefault("model_kwargs", encoder.model_kwargs)
        options.setdefault("tokenizer_kwargs", encoder.tokenizer_kwargs)
        options.setdefault("normalize_embeddings", encoder.normalize_embeddings)
        options.setdefault("truncate_dim", encoder.truncate_dim)
        if encoder.encode_batch_size is not None:
            options.setdefault("encode_batch_size", encoder.encode_batch_size)

    if runner_key == "ctm":
        _set_common_encoder_defaults(
            model_key="contextual_model_name",
            prefix_key="contextual_encode_prefix",
        )
    elif runner_key == "senclu":
        _set_common_encoder_defaults(
            model_key="encoder_model_name",
            prefix_key="encode_prefix",
        )
    elif runner_key == "sentence_gaussianlda":
        _set_common_encoder_defaults(
            model_key="encoder_model_name",
            prefix_key="encode_prefix",
        )
        options.setdefault("strip_terminal_normalize", encoder.strip_terminal_normalize)
    elif runner_key == "bertopic_kmeans":
        _set_common_encoder_defaults(
            model_key="encoder_model_name",
            prefix_key="encode_prefix",
        )
    elif runner_key in {
        "spherical_kmeans",
        "gaussian_kmeans",
        "movmf",
        "gaussian_mixture",
    }:
        _set_common_encoder_defaults(
            model_key="encoder_model_name",
            prefix_key="encode_prefix",
        )
        if runner_key in {"gaussian_kmeans", "gaussian_mixture"}:
            options.setdefault(
                "strip_terminal_normalize", encoder.strip_terminal_normalize
            )

    return options


def parse_baselines(
    raw_cfg: Dict[str, Any],
    *,
    encoder: EncoderConfig | None = None,
) -> list[BaselineConfig]:
    encoder_config = encoder or parse_encoder_config(raw_cfg)
    return [
        BaselineConfig(
            name=baseline.get("name", baseline.get("runner", "baseline")),
            runner=baseline["runner"],
            params=normalize_baseline_params(
                baseline["runner"],
                _baseline_params_with_encoder_defaults(
                    runner=baseline["runner"],
                    params=baseline.get("params"),
                    encoder=encoder_config,
                ),
            ),
        )
        for baseline in raw_cfg.get("baselines", [])
    ]


def parse_output_root(raw_cfg: Dict[str, Any], *, dataset_name: str) -> Path:
    output_root = resolve_project_path(
        raw_cfg.get("output_root", "results/experiments")
    )
    return output_root / dataset_name

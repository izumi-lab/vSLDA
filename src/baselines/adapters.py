from __future__ import annotations

from importlib import import_module
from typing import Any

from src.baselines.contracts import BaselineArtifacts, BaselineRunRequest
from src.baselines.params import (
    parse_bertopic_kmeans_params,
    parse_bleilda_params,
    parse_ctm_params,
    parse_etm_params,
    parse_gaussian_kmeans_params,
    parse_gaussian_mixture_params,
    parse_gaussianlda_params,
    parse_movmf_params,
    parse_mvtm_params,
    parse_senclu_params,
    parse_sentence_gaussianlda_params,
    parse_sentlda_params,
    parse_spherical_kmeans_params,
)

from . import adapter_runtime as adapter_runtime_module
from .adapter_runtime import (
    _baseline_archive_root,
    _baseline_dir,
    _build_artifacts,
    _build_baseline_identity,
    _build_persisted_artifacts,
    _build_preprocessing_variant,
    _build_split_artifacts,
    _resolve_legacy_preprocessing,
    _save_runner_metadata,
    _with_split_dirs,
    _write_baseline_pointer,
    execute_adapter,
    use_legacy_category_behavior,
)
from .adapter_specs import BaselineAdapterSpec


def _call_model_function(
    module_name: str,
    function_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    module = import_module(f"src.baselines.models.{module_name}")
    return getattr(module, function_name)(*args, **kwargs)


def train_ctm(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("ctm", "train_ctm", *args, **kwargs)


def infer_ctm(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("ctm", "infer_ctm", *args, **kwargs)


def persist_ctm_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("ctm", "persist_ctm_run", *args, **kwargs)


def train_bleilda(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("bleilda", "train_bleilda", *args, **kwargs)


def infer_bleilda(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("bleilda", "infer_bleilda", *args, **kwargs)


def persist_bleilda_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("bleilda", "persist_bleilda_run", *args, **kwargs)


def train_bertopic_kmeans(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "bertopic_kmeans",
        "train_bertopic_kmeans",
        *args,
        **kwargs,
    )


def infer_bertopic_kmeans(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "bertopic_kmeans",
        "infer_bertopic_kmeans",
        *args,
        **kwargs,
    )


def persist_bertopic_kmeans_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "bertopic_kmeans",
        "persist_bertopic_kmeans_run",
        *args,
        **kwargs,
    )


def train_gaussianlda(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("gaussianlda", "train_gaussianlda", *args, **kwargs)


def infer_gaussianlda(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("gaussianlda", "infer_gaussianlda", *args, **kwargs)


def persist_gaussianlda_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "gaussianlda",
        "persist_gaussianlda_run",
        *args,
        **kwargs,
    )


def train_etm(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("etm", "train_etm", *args, **kwargs)


def infer_etm(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("etm", "infer_etm", *args, **kwargs)


def persist_etm_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("etm", "persist_etm_run", *args, **kwargs)


def train_mvtm(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("mvtm", "train_mvtm", *args, **kwargs)


def infer_mvtm(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("mvtm", "infer_mvtm", *args, **kwargs)


def persist_mvtm_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("mvtm", "persist_mvtm_run", *args, **kwargs)


def train_spherical_kmeans(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "train_spherical_kmeans",
        *args,
        **kwargs,
    )


def infer_spherical_kmeans(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "infer_spherical_kmeans",
        *args,
        **kwargs,
    )


def persist_spherical_kmeans_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "persist_spherical_kmeans_run",
        *args,
        **kwargs,
    )


def train_gaussian_kmeans(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "train_gaussian_kmeans",
        *args,
        **kwargs,
    )


def infer_gaussian_kmeans(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "infer_gaussian_kmeans",
        *args,
        **kwargs,
    )


def persist_gaussian_kmeans_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "persist_gaussian_kmeans_run",
        *args,
        **kwargs,
    )


def train_movmf(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "train_movmf",
        *args,
        **kwargs,
    )


def infer_movmf(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "infer_movmf",
        *args,
        **kwargs,
    )


def persist_movmf_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "persist_movmf_run",
        *args,
        **kwargs,
    )


def train_gaussian_mixture(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "train_gaussian_mixture",
        *args,
        **kwargs,
    )


def infer_gaussian_mixture(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "infer_gaussian_mixture",
        *args,
        **kwargs,
    )


def persist_gaussian_mixture_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_embedding_clustering",
        "persist_gaussian_mixture_run",
        *args,
        **kwargs,
    )


def train_senclu(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("senclu", "train_senclu", *args, **kwargs)


def infer_senclu(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("senclu", "infer_senclu", *args, **kwargs)


def persist_senclu_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("senclu", "persist_senclu_run", *args, **kwargs)


def train_sentlda(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("sentlda", "train_sentlda", *args, **kwargs)


def infer_sentlda(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("sentlda", "infer_sentlda", *args, **kwargs)


def persist_sentlda_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function("sentlda", "persist_sentlda_run", *args, **kwargs)


def train_sentence_gaussianlda(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_gaussianlda",
        "train_sentence_gaussianlda",
        *args,
        **kwargs,
    )


def infer_sentence_gaussianlda(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_gaussianlda",
        "infer_sentence_gaussianlda",
        *args,
        **kwargs,
    )


def persist_sentence_gaussianlda_run(*args: Any, **kwargs: Any) -> Any:
    return _call_model_function(
        "sentence_gaussianlda",
        "persist_sentence_gaussianlda_run",
        *args,
        **kwargs,
    )


def run_ctm(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="ctm",
            runner_family="ctm",
            train_passes_encoder_device=True,
        ),
        request=request,
        parse_params=parse_ctm_params,
        train_fn=train_ctm,
        infer_fn=infer_ctm,
        persist_fn=persist_ctm_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_bleilda(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="bleilda",
            runner_family="bleilda",
            infer_mode="no_params",
        ),
        request=request,
        parse_params=parse_bleilda_params,
        train_fn=train_bleilda,
        infer_fn=infer_bleilda,
        persist_fn=persist_bleilda_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_bertopic_kmeans(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="bertopic_kmeans",
            runner_family="bertopic_kmeans",
            infer_mode="train_only",
            train_passes_test_csvs=True,
            train_passes_encoder_device=True,
            train_passes_effective_random_state=True,
        ),
        request=request,
        parse_params=parse_bertopic_kmeans_params,
        train_fn=train_bertopic_kmeans,
        infer_fn=infer_bertopic_kmeans,
        persist_fn=persist_bertopic_kmeans_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_gaussianlda(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(model="gaussianlda", runner_family="gaussianlda"),
        request=request,
        parse_params=parse_gaussianlda_params,
        train_fn=train_gaussianlda,
        infer_fn=infer_gaussianlda,
        persist_fn=persist_gaussianlda_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_etm(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="etm",
            runner_family="etm",
            train_passes_encoder_device=True,
            train_passes_effective_random_state=True,
        ),
        request=request,
        parse_params=parse_etm_params,
        train_fn=train_etm,
        infer_fn=infer_etm,
        persist_fn=persist_etm_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_mvtm(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(model="mvtm", runner_family="mvtm"),
        request=request,
        parse_params=parse_mvtm_params,
        train_fn=train_mvtm,
        infer_fn=infer_mvtm,
        persist_fn=persist_mvtm_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_spherical_kmeans(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="spherical_kmeans",
            runner_family="spherical_kmeans",
            train_passes_encoder_device=True,
        ),
        request=request,
        parse_params=parse_spherical_kmeans_params,
        train_fn=train_spherical_kmeans,
        infer_fn=infer_spherical_kmeans,
        persist_fn=persist_spherical_kmeans_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_gaussian_kmeans(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="gaussian_kmeans",
            runner_family="gaussian_kmeans",
            train_passes_encoder_device=True,
        ),
        request=request,
        parse_params=parse_gaussian_kmeans_params,
        train_fn=train_gaussian_kmeans,
        infer_fn=infer_gaussian_kmeans,
        persist_fn=persist_gaussian_kmeans_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_movmf(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="movmf",
            runner_family="movmf",
            train_passes_encoder_device=True,
        ),
        request=request,
        parse_params=parse_movmf_params,
        train_fn=train_movmf,
        infer_fn=infer_movmf,
        persist_fn=persist_movmf_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_gaussian_mixture(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="gaussian_mixture",
            runner_family="gaussian_mixture",
            train_passes_encoder_device=True,
        ),
        request=request,
        parse_params=parse_gaussian_mixture_params,
        train_fn=train_gaussian_mixture,
        infer_fn=infer_gaussian_mixture,
        persist_fn=persist_gaussian_mixture_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_senclu(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="senclu",
            runner_family="senclu",
            infer_mode="train_only",
            train_passes_test_csvs=True,
            train_passes_encoder_device=True,
        ),
        request=request,
        parse_params=parse_senclu_params,
        train_fn=train_senclu,
        infer_fn=infer_senclu,
        persist_fn=persist_senclu_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_sentlda(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(model="sentlda", runner_family="sentlda"),
        request=request,
        parse_params=parse_sentlda_params,
        train_fn=train_sentlda,
        infer_fn=infer_sentlda,
        persist_fn=persist_sentlda_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )


def run_sentence_gaussianlda(request: BaselineRunRequest) -> BaselineArtifacts:
    adapter_runtime_module.use_legacy_category_behavior = use_legacy_category_behavior
    return execute_adapter(
        spec=BaselineAdapterSpec(
            model="sentence_gaussianlda",
            runner_family="sentence_gaussianlda",
            train_passes_encoder_device=True,
        ),
        request=request,
        parse_params=parse_sentence_gaussianlda_params,
        train_fn=train_sentence_gaussianlda,
        infer_fn=infer_sentence_gaussianlda,
        persist_fn=persist_sentence_gaussianlda_run,
        save_metadata=_save_runner_metadata,
        build_persisted_artifacts=_build_persisted_artifacts,
    )

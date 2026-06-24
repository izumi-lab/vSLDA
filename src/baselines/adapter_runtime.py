from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping

from src.baselines.contracts import BaselineArtifacts, BaselineRunRequest
from src.baselines.dataset_adapters import use_legacy_category_behavior
from src.baselines.model_kinds import baseline_method_kind
from src.baselines.params import (
    baseline_params_to_options,
    baseline_params_to_variant,
    normalize_baseline_params,
)
from src.core.artifacts import (
    METADATA_FILENAME,
    BaselineArtifactMetadata,
    ensure_artifact_paths_exist,
    save_baseline_metadata,
    save_json,
)
from src.core.paths import (
    build_baseline_archive_dir,
    build_baseline_condition_id,
    build_baseline_dir,
    write_baseline_latest_pointer,
)
from src.core.result_identity import slugify_label
from src.utils.encoder_profiles import encoder_model_alias

from .adapter_specs import BaselineAdapterSpec

EMBEDDING_AWARE_RUNNERS = {
    "ctm",
    "senclu",
    "sentence_gaussianlda",
    "bertopic_kmeans",
    "spherical_kmeans",
    "gaussian_kmeans",
    "movmf",
    "gaussian_mixture",
}

WORD_EMBEDDING_AWARE_RUNNERS = {
    "gaussianlda",
    "etm",
    "mvtm",
}

GAUSSIAN_TERMINAL_NORMALIZE_RUNNERS = {
    "sentence_gaussianlda",
    "gaussian_kmeans",
    "gaussian_mixture",
}

_GLOVE_RE = re.compile(r"^glove-wiki-gigaword-(?P<dim>\d+)$")
_VECTOR_DIM_RE = re.compile(r"(?P<dim>\d+)d(?:\D|$)")


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _word_embedding_alias(word2vec: object) -> str:
    text = str(word2vec).strip()
    normalized = text.lower()
    glove_match = _GLOVE_RE.match(normalized)
    if glove_match is not None:
        return f"glove{glove_match.group('dim')}"
    if normalized.startswith("wikientvec"):
        dim_match = _VECTOR_DIM_RE.search(normalized)
        if dim_match is not None:
            return f"wikient{dim_match.group('dim')}"
        return "wikient"
    if normalized in {"local", "local-word2vec"}:
        return "local"
    path_name = Path(text).name
    slug_source = path_name if path_name else text
    return f"wordvec_{slugify_label(slug_source, max_length=32)}"


def _baseline_encoder_config(
    *,
    runner: str,
    baseline_params: object,
) -> dict[str, Any] | None:
    runner_key = str(runner).strip().lower()
    if runner_key in WORD_EMBEDDING_AWARE_RUNNERS:
        normalized = baseline_params_to_options(baseline_params)
        word2vec = normalized.get("word2vec", "glove-wiki-gigaword-100")
        return {
            "embedding_type": "word_vectors",
            "word2vec": str(word2vec),
            "wikientvec_cache_dir": normalized.get("wikientvec_cache_dir"),
            "embedding_variant": _word_embedding_alias(word2vec),
        }
    if runner_key not in EMBEDDING_AWARE_RUNNERS:
        return None
    normalized = baseline_params_to_options(baseline_params)
    if runner_key == "ctm":
        model_name = normalized.get("contextual_model_name")
        encode_prefix = normalized.get("contextual_encode_prefix")
    else:
        model_name = normalized.get("encoder_model_name")
        encode_prefix = normalized.get("encode_prefix")
    if model_name is None:
        return None
    return {
        "model_name": str(model_name),
        "backend": normalized.get("encoder_backend"),
        "pooling": normalized.get("pooling"),
        "encode_prefix": encode_prefix,
        "encode_prompt": normalized.get("encode_prompt"),
        "encode_prompt_name": normalized.get("encode_prompt_name"),
        "encode_batch_size": normalized.get("encode_batch_size"),
        "model_kwargs": normalized.get("model_kwargs") or {},
        "tokenizer_kwargs": normalized.get("tokenizer_kwargs") or {},
        "normalize_embeddings": normalized.get("normalize_embeddings"),
        "truncate_dim": normalized.get("truncate_dim"),
        "strip_terminal_normalize": normalized.get("strip_terminal_normalize"),
        "embedding_variant": encoder_model_alias(str(model_name)),
    }


def _baseline_embedding_variant(
    *,
    runner: str,
    baseline_params: object,
) -> str | None:
    runner_key = str(runner).strip().lower()
    config = _baseline_encoder_config(
        runner=runner_key, baseline_params=baseline_params
    )
    if config is None:
        return None
    variant = str(config["embedding_variant"])
    normalized = baseline_params_to_options(baseline_params)
    default_strip_terminal_normalize = True
    strip_terminal_normalize = _coerce_bool(
        normalized.get("strip_terminal_normalize"),
        default=default_strip_terminal_normalize,
    )
    if runner_key in GAUSSIAN_TERMINAL_NORMALIZE_RUNNERS:
        return f"{variant}_{'raw' if strip_terminal_normalize else 'norm'}"
    return variant


def _baseline_dir(
    model: str,
    *,
    split_root: str,
    request: BaselineRunRequest,
) -> Path:
    condition_id, _ = _build_baseline_identity(model=model, request=request)
    return build_baseline_dir(
        model=model,
        split_root=split_root,
        dataset=request.dataset,
        iteration=request.iteration,
        num_topics=request.num_topics,
        category=request.category,
        data_run=str(request.options.get("data_run", "default")),
        condition_id=condition_id,
    )


def _baseline_archive_root(
    model: str,
    *,
    request: BaselineRunRequest,
    num_components: int | None = None,
    embedding_variant: str | None = None,
) -> Path:
    options = dict(request.options)
    return build_baseline_archive_dir(
        model=model,
        dataset=request.dataset,
        data_run=str(options.get("data_run", "default")),
        category=request.category,
        iteration=request.iteration,
        num_topics=request.num_topics,
        num_components=num_components,
        embedding_variant=embedding_variant,
        started_at=(
            None
            if options.get("started_at") is None
            else str(options.get("started_at"))
        ),
        execution_id=(
            None
            if options.get("execution_id") is None
            else str(options.get("execution_id"))
        ),
    )


def _build_artifacts(
    *,
    train_path: Path,
    infer_path: Path,
    extras: dict[str, Path] | None = None,
) -> BaselineArtifacts:
    all_artifacts = {
        "train_path": train_path,
        "infer_path": infer_path,
        **({} if extras is None else dict(extras)),
    }
    ensure_artifact_paths_exist(all_artifacts)
    return BaselineArtifacts(
        train_path=train_path,
        infer_path=infer_path,
        extras={} if extras is None else dict(extras),
    )


def _with_split_dirs(
    *,
    train_dir: Path,
    infer_dir: Path,
    extras: dict[str, Path] | None = None,
) -> dict[str, Path]:
    payload = {"train_dir": train_dir, "infer_dir": infer_dir}
    if extras is not None:
        payload.update(extras)
    return payload


def _build_split_artifacts(
    *,
    train_dir: Path,
    infer_dir: Path,
    train_filename: str,
    infer_filename: str,
    extras: dict[str, Path] | None = None,
) -> BaselineArtifacts:
    return _build_artifacts(
        train_path=train_dir / train_filename,
        infer_path=infer_dir / infer_filename,
        extras=_with_split_dirs(
            train_dir=train_dir, infer_dir=infer_dir, extras=extras
        ),
    )


def _build_persisted_artifacts(
    *,
    artifacts: BaselineArtifacts,
    train_dir: Path,
    infer_dir: Path,
    extras: dict[str, Path] | None = None,
) -> BaselineArtifacts:
    persisted_extras = dict(artifacts.extras)
    if extras is not None:
        persisted_extras.update(extras)
    return _build_artifacts(
        train_path=artifacts.train_path,
        infer_path=artifacts.infer_path,
        extras=_with_split_dirs(
            train_dir=train_dir,
            infer_dir=infer_dir,
            extras=persisted_extras,
        ),
    )


def _save_runner_metadata(
    *,
    request: BaselineRunRequest,
    runner_family: str,
    condition_id: str | None = None,
    condition_fingerprint: str | None = None,
    metadata_dir: Path | None = None,
    train_dir: Path,
    infer_dir: Path,
) -> Path:
    options = dict(request.options)
    if condition_id is None or condition_fingerprint is None:
        resolved_condition_id, resolved_condition_fingerprint = (
            _build_baseline_identity(
                model=request.name,
                request=request,
            )
        )
        condition_id = condition_id or resolved_condition_id
        condition_fingerprint = condition_fingerprint or resolved_condition_fingerprint
    if metadata_dir is None:
        metadata_dir = train_dir
    baseline_params = normalize_baseline_params(request.name, options)
    normalized_params = baseline_params_to_options(baseline_params)
    encoder_config = _baseline_encoder_config(
        runner=request.name,
        baseline_params=baseline_params,
    )
    metadata = BaselineArtifactMetadata(
        runner_key=request.name,
        runner_family=runner_family,
        method_kind=baseline_method_kind(request.name),
        data_run=str(options.get("data_run", "default")),
        condition_id=condition_id,
        condition_fingerprint=condition_fingerprint,
        started_at=(
            None
            if options.get("started_at") is None
            else str(options.get("started_at"))
        ),
        execution_id=(
            None
            if options.get("execution_id") is None
            else str(options.get("execution_id"))
        ),
        parameter_variant=baseline_params_to_variant(baseline_params),
        preprocessing_variant=_build_preprocessing_variant(options),
        dataset=request.dataset,
        category=request.category,
        num_topics=int(request.num_topics),
        iteration=int(request.iteration),
        baseline_params=normalized_params,
        targets=(
            None
            if options.get("targets") is None
            else tuple(str(target) for target in options["targets"])
        ),
        language=str(options.get("language", "english")),
        delimiter=options.get("delimiter"),
        segmenter=str(options.get("segmenter", "delimiter")),
        tokenizer=str(options.get("tokenizer", "default")),
        legacy_preprocessing=(
            None
            if options.get("legacy_preprocessing") is None
            else bool(options.get("legacy_preprocessing"))
        ),
        text_column=str(options.get("text_column", "data")),
        target_column=options.get("target_column"),
        ja_replace_num=bool(options.get("ja_replace_num", True)),
        ja_stopwords_path=options.get("ja_stopwords_path"),
        ja_dicdir=options.get("ja_dicdir"),
        ja_require_unidic=bool(options.get("ja_require_unidic", True)),
        encoder_device=options.get("encoder_device"),
        runtime_num_workers=int(options.get("runtime_num_workers", 1)),
        train_csvs=tuple(str(path) for path in options.get("train_csvs", []) or []),
        test_csvs=tuple(str(path) for path in options.get("test_csvs", []) or []),
        train_dir=str(train_dir),
        infer_dir=str(infer_dir),
        effective_random_state=(
            None
            if options.get("effective_random_state") is None
            else int(options["effective_random_state"])
        ),
        doc_topic_source=(
            None
            if options.get("doc_topic_source") is None
            else str(options["doc_topic_source"])
        ),
        doc_topic_space=(
            None
            if options.get("doc_topic_space") is None
            else str(options["doc_topic_space"])
        ),
        embedding_variant=(
            None if encoder_config is None else str(encoder_config["embedding_variant"])
        ),
        encoder_config=encoder_config,
    )
    metadata_path = metadata_dir / METADATA_FILENAME
    save_json(
        {
            "dataset_name": request.dataset,
            "model_name": request.name,
            "num_topics": int(request.num_topics),
            "seed": options.get("seed", options.get("effective_random_state")),
            "category": request.category,
            "iteration": int(request.iteration),
            "data_run": str(options.get("data_run", "default")),
            "condition_id": condition_id,
            "condition_fingerprint": condition_fingerprint,
            "baseline_params": normalized_params,
            "targets": metadata.targets,
        },
        metadata_dir / "config.json",
    )
    save_baseline_metadata(metadata, metadata_path)
    return metadata_path


def _write_baseline_pointer(
    *,
    model: str,
    request: BaselineRunRequest,
    archive_dir: Path,
    condition_fingerprint: str,
    artifacts: Mapping[str, Path],
    num_components: int | None = None,
    embedding_variant: str | None = None,
    encoder_config: Mapping[str, Any] | None = None,
) -> Path:
    options = dict(request.options)
    serialized_artifacts: dict[str, str] = {}
    for name, path in sorted(artifacts.items()):
        try:
            serialized_artifacts[str(name)] = path.relative_to(archive_dir).as_posix()
        except ValueError:
            serialized_artifacts[str(name)] = str(path)
    return write_baseline_latest_pointer(
        model=model,
        dataset=request.dataset,
        data_run=str(options.get("data_run", "default")),
        category=request.category,
        iteration=request.iteration,
        num_topics=request.num_topics,
        num_components=num_components,
        archive_dir=archive_dir,
        started_at=(
            "" if options.get("started_at") is None else str(options.get("started_at"))
        ),
        execution_id=(
            ""
            if options.get("execution_id") is None
            else str(options.get("execution_id"))
        ),
        condition_fingerprint=condition_fingerprint,
        artifacts=serialized_artifacts,
        embedding_variant=embedding_variant,
        encoder_config=encoder_config,
    )


def _display_num_components_for_request(
    *,
    model: str,
    request: BaselineRunRequest,
) -> int | None:
    if model != "mvtm" and request.name != "mvtm":
        return None
    params = normalize_baseline_params(request.name, dict(request.options))
    num_components = getattr(params, "num_components", None)
    if num_components is None:
        return None
    return int(num_components)


def _build_baseline_identity(
    *,
    model: str,
    request: BaselineRunRequest,
) -> tuple[str, str]:
    options = dict(request.options)
    baseline_params = normalize_baseline_params(request.name, options)
    encoder_config = _baseline_encoder_config(
        runner=request.name,
        baseline_params=baseline_params,
    )
    payload: dict[str, Any] = {
        "model": model,
        "runner_key": request.name,
        "dataset": request.dataset,
        "data_run": str(options.get("data_run", "default")),
        "iteration": int(request.iteration),
        "num_topics": int(request.num_topics),
        "category": request.category,
        "parameter_variant": baseline_params_to_variant(baseline_params),
        "baseline_params": baseline_params_to_options(baseline_params),
        "preprocessing_variant": _build_preprocessing_variant(options),
        "train_csvs": [str(path) for path in options.get("train_csvs", []) or []],
        "test_csvs": [str(path) for path in options.get("test_csvs", []) or []],
        "targets": (
            None
            if options.get("targets") is None
            else [str(target) for target in options.get("targets", []) or []]
        ),
        "encoder_config": encoder_config,
    }
    if options.get("effective_random_state") is not None:
        payload["effective_random_state"] = int(options["effective_random_state"])
    return build_baseline_condition_id(
        model=model,
        iteration=request.iteration,
        num_topics=request.num_topics,
        category=request.category,
        fingerprint_payload=payload,
    )


def _build_preprocessing_variant(options: dict[str, object]) -> str:
    legacy_preprocessing = options.get("legacy_preprocessing")
    if legacy_preprocessing is None:
        legacy_preprocessing_part = "auto"
    else:
        legacy_preprocessing_part = str(bool(legacy_preprocessing)).lower()
    parts = [
        f"language={options.get('language', 'english')}",
        f"delimiter={options.get('delimiter') if options.get('delimiter') is not None else 'none'}",
        f"segmenter={options.get('segmenter', 'delimiter')}",
        f"tokenizer={options.get('tokenizer', 'default')}",
        f"legacy_preprocessing={legacy_preprocessing_part}",
        f"text_column={options.get('text_column', 'data')}",
        f"target_column={options.get('target_column') if options.get('target_column') is not None else 'none'}",
        f"ja_replace_num={str(bool(options.get('ja_replace_num', True))).lower()}",
        f"ja_require_unidic={str(bool(options.get('ja_require_unidic', True))).lower()}",
    ]
    if options.get("ja_stopwords_path") is not None:
        parts.append(f"ja_stopwords_path={options['ja_stopwords_path']}")
    if options.get("ja_dicdir") is not None:
        parts.append(f"ja_dicdir={options['ja_dicdir']}")
    return "__".join(parts)


def _resolve_legacy_preprocessing(
    *,
    dataset: str,
    options: dict[str, object],
) -> bool:
    override = options.get("legacy_preprocessing")
    if override is None:
        return use_legacy_category_behavior(
            dataset,
            str(options.get("language", "english")),
        )
    return bool(override)


def execute_adapter(
    *,
    spec: BaselineAdapterSpec,
    request: BaselineRunRequest,
    parse_params: Callable[[dict[str, object]], Any],
    train_fn: Callable[..., Any],
    infer_fn: Callable[..., Any],
    persist_fn: Callable[..., BaselineArtifacts],
    save_metadata: Callable[..., Path] = _save_runner_metadata,
    build_persisted_artifacts: Callable[
        ..., BaselineArtifacts
    ] = _build_persisted_artifacts,
) -> BaselineArtifacts:
    options = dict(request.options)
    params = parse_params(options)
    encoder_config = _baseline_encoder_config(
        runner=request.name, baseline_params=params
    )
    embedding_variant = _baseline_embedding_variant(
        runner=request.name,
        baseline_params=params,
    )
    use_legacy = _resolve_legacy_preprocessing(dataset=request.dataset, options=options)
    display_num_components = _display_num_components_for_request(
        model=spec.model,
        request=request,
    )
    condition_id, condition_fingerprint = _build_baseline_identity(
        model=spec.model,
        request=request,
    )
    condition_root = _baseline_archive_root(
        spec.model,
        request=request,
        num_components=display_num_components,
        embedding_variant=embedding_variant,
    )
    train_dir = condition_root / "params"
    infer_dir = condition_root / "infer"

    common_kwargs = {
        "targets": options.get("targets"),
        "text_column": str(options.get("text_column", "data")),
        "target_column": options.get("target_column"),
        "delimiter": options.get("delimiter"),
        "language": str(options.get("language", "english")),
        "segmenter": str(options.get("segmenter", "delimiter")),
        "tokenizer": str(options.get("tokenizer", "default")),
        "ja_replace_num": bool(options.get("ja_replace_num", True)),
        "ja_stopwords_path": options.get("ja_stopwords_path"),
        "ja_dicdir": options.get("ja_dicdir"),
        "ja_require_unidic": bool(options.get("ja_require_unidic", True)),
        "num_topics": request.num_topics,
        "use_legacy": use_legacy,
    }

    train_kwargs = dict(common_kwargs)
    if spec.train_passes_encoder_device:
        train_kwargs["encoder_device"] = str(options.get("encoder_device", "auto"))
    if spec.train_passes_effective_random_state:
        train_kwargs["effective_random_state"] = int(
            options.get("effective_random_state", 0)
        )
    train_kwargs["train_csvs"] = list(options.get("train_csvs", []) or [])
    if spec.train_passes_test_csvs:
        train_kwargs["test_csvs"] = list(options.get("test_csvs", []) or [])
    train_kwargs["params"] = params
    train_kwargs["train_dir"] = train_dir
    train_result = train_fn(**train_kwargs)

    if spec.infer_mode == "train_only":
        infer_result = infer_fn(train_result=train_result)
    else:
        infer_kwargs = dict(common_kwargs)
        infer_kwargs["train_result"] = train_result
        infer_kwargs["test_csvs"] = list(options.get("test_csvs", []) or [])
        if spec.infer_mode == "standard":
            infer_kwargs["params"] = params
        infer_result = infer_fn(**infer_kwargs)

    persisted = persist_fn(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=train_dir,
        infer_dir=infer_dir,
        category=request.category,
    )
    metadata_path = save_metadata(
        request=request,
        runner_family=spec.runner_family,
        condition_id=condition_id,
        condition_fingerprint=condition_fingerprint,
        metadata_dir=condition_root,
        train_dir=train_dir,
        infer_dir=infer_dir,
    )
    artifacts = build_persisted_artifacts(
        artifacts=persisted,
        train_dir=train_dir,
        infer_dir=infer_dir,
        extras={"metadata": metadata_path},
    )
    _write_baseline_pointer(
        model=spec.model,
        request=request,
        archive_dir=condition_root,
        condition_fingerprint=condition_fingerprint,
        artifacts=artifacts.as_dict(),
        num_components=display_num_components,
        embedding_variant=embedding_variant,
        encoder_config=encoder_config,
    )
    return artifacts

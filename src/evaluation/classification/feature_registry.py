from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.core.artifacts import (
    CURRENT_POINTER_FILENAME,
    METADATA_FILENAME,
    PREPROCESSING_SELECTION_FILENAME,
    load_artifact_json,
    load_artifact_pickle,
)
from src.core.errors import MissingArtifactError
from src.core.paths import (
    build_baseline_doc_topic_path,
    build_vmf_doc_topic_path,
    resolve_baseline_condition_dir,
    resolve_project_path,
    resolve_vmf_experiment_dir,
)
from src.data.preprocessing import PreprocessedDocument

from .alignment import (
    SplitAlignment,
    build_baseline_available_indices,
    build_label_space_indices,
    build_preprocessed_available_indices,
)
from .config import model_matches_selector

FeatureLoader = Callable[[Path], np.ndarray]
AvailableIndexResolver = Callable[
    [str, str, Path, Path, str, str], tuple[SplitAlignment, SplitAlignment]
]

LOGGER = logging.getLogger(__name__)


def topic_distribution(x_data: np.ndarray | list[np.ndarray]) -> np.ndarray:
    rows = []
    for row in x_data:
        total = row.sum()
        rows.append(row if total == 0 else row / total)
    return np.asarray(rows)


def load_pickle_array(path: Path) -> np.ndarray:
    return np.asarray(load_artifact_pickle(path))


def load_topic_distribution(path: Path) -> np.ndarray:
    return topic_distribution(load_pickle_array(path))


def resolve_feature_display_name(spec: "FeatureSpec", train_path: Path) -> str:
    spec_metadata = getattr(spec, "metadata", None)
    if spec_metadata is not None:
        return spec.display_name

    metadata = _load_feature_metadata(train_path)
    if metadata is None:
        return spec.display_name

    parameter_variant = metadata.get("parameter_variant")
    if not isinstance(parameter_variant, str) or parameter_variant == "default":
        return spec.display_name
    return f"{spec.display_name} [{parameter_variant}]"


def resolve_feature_catalog_entry(
    spec: "FeatureSpec",
    train_path: Path,
) -> dict[str, object]:
    feature_name = resolve_feature_display_name(spec, train_path)
    spec_metadata = getattr(spec, "metadata", None)
    metadata = (
        spec_metadata
        if spec_metadata is not None
        else _load_feature_metadata(train_path)
    )
    display_key = getattr(spec, "display_key", None)
    embedding_variant = getattr(spec, "embedding_variant", None)
    condition_fingerprint = getattr(spec, "condition_fingerprint", None)
    encoder_config = getattr(spec, "encoder_config", None)
    if metadata is None:
        return {
            "feature_name": feature_name,
            "display_name": spec.display_name,
            "model_key": spec.model_key,
            "runner_key": spec.model_key,
            "runner_family": spec.model_key,
            "parameter_variant": None,
            "preprocessing_variant": None,
            "baseline_params": None,
            "display_key": display_key,
            "embedding_variant": embedding_variant,
            "condition_fingerprint": condition_fingerprint,
            "encoder_config": encoder_config,
        }

    return {
        "feature_name": feature_name,
        "display_name": spec.display_name,
        "model_key": spec.model_key,
        "runner_key": metadata.get("runner_key", spec.model_key),
        "runner_family": metadata.get("runner_family", spec.model_key),
        "parameter_variant": metadata.get("parameter_variant"),
        "preprocessing_variant": metadata.get("preprocessing_variant"),
        "baseline_params": (
            dict(metadata["baseline_params"])
            if isinstance(metadata.get("baseline_params"), dict)
            else None
        ),
        "display_key": display_key or _metadata_display_key(metadata),
        "embedding_variant": embedding_variant or _metadata_embedding_variant(metadata),
        "condition_fingerprint": condition_fingerprint
        or _metadata_condition_fingerprint(metadata),
        "encoder_config": encoder_config or _metadata_encoder_config(metadata),
    }


def _load_feature_metadata(train_path: Path) -> dict[str, object] | None:
    for candidate in [train_path.parent, *train_path.parents[:3]]:
        metadata_path = candidate / METADATA_FILENAME
        if not metadata_path.exists():
            continue
        payload = load_artifact_json(metadata_path)
        if isinstance(payload, dict):
            return payload
    return None


def _current_experiment_results_root() -> Path:
    from src.core import paths as public_paths

    return public_paths.EXPERIMENT_RESULTS_ROOT


def _current_baseline_results_root() -> Path:
    from src.core import paths as public_paths

    return public_paths.BASELINE_RESULTS_ROOT


def _metadata_axes(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    axes = metadata.get("axes")
    return axes if isinstance(axes, Mapping) else {}


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    axes = _metadata_axes(metadata)
    if key in axes:
        return axes[key]
    return metadata.get(key)


def _metadata_display_key(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("display_key")
    return None if value is None else str(value)


def _metadata_embedding_variant(metadata: Mapping[str, Any]) -> str | None:
    for value in (
        metadata.get("embedding_variant"),
        _metadata_axes(metadata).get("embedding_variant"),
    ):
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _metadata_condition_fingerprint(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("condition_fingerprint")
    return None if value is None else str(value)


def _metadata_encoder_config(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    value = metadata.get("encoder_config")
    return dict(value) if isinstance(value, Mapping) else None


@dataclass(frozen=True)
class FeatureSpec:
    model_key: str
    display_name: str
    train_path_resolver: Callable[[str, str, int, int, str], Path]
    test_path_resolver: Callable[[str, str, int, int, str], Path]
    train_loader: FeatureLoader
    test_loader: FeatureLoader
    available_index_resolver: AvailableIndexResolver | None = None
    metadata: dict[str, Any] | None = None
    display_key: str | None = None
    embedding_variant: str | None = None
    condition_fingerprint: str | None = None
    encoder_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedFeatureArtifact:
    train_path: Path
    test_path: Path
    metadata: dict[str, Any]
    display_key: str | None
    embedding_variant: str | None
    condition_fingerprint: str | None
    encoder_config: dict[str, Any] | None


FeatureSpecBuilder = Callable[[str], FeatureSpec]


_BASELINE_TRAIN_FILENAMES: dict[str, str] = {
    "bleilda": "lda_comp.pkl",
    "ctm": "ctm.pkl",
    "etm": "etm.pkl",
    "gaussianlda": "table_counts_per_doc.pkl",
    "mvtm": "table_counts_per_doc.pkl",
    "sentence_gaussianlda": "table_counts_per_doc.pkl",
    "sentlda": "table_counts_per_doc.pkl",
    "bertopic_kmeans": "bertopic_kmeans.pkl",
}

_SENTENCE_EMBEDDING_FILTERED_MODELS = {
    "ctm",
    "bertopic_kmeans",
    "spherical_kmeans",
    "gaussian_kmeans",
    "movmf",
    "gaussian_mixture",
    "senclu",
    "sentence_gaussianlda",
    "vmf_sentence_lda",
}


def _base_display_key(*, iteration: int, num_topics: int) -> str:
    return f"k{int(num_topics)}_it{int(iteration)}"


def _display_key_matches(
    display_key: str | None,
    *,
    iteration: int,
    num_topics: int,
) -> bool:
    if display_key is None:
        return False
    base_key = _base_display_key(iteration=iteration, num_topics=num_topics)
    return display_key == base_key or display_key.startswith(f"{base_key}_")


def _display_key_suffix(
    display_key: str | None,
    *,
    iteration: int,
    num_topics: int,
) -> str | None:
    if display_key is None:
        return None
    base_key = _base_display_key(iteration=iteration, num_topics=num_topics)
    prefix = f"{base_key}_"
    if display_key.startswith(prefix):
        suffix = display_key[len(prefix) :].strip("_")
        return suffix or None
    return None


def _clean_variant(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "default":
        return None
    return text


def _pointer_variant_values(
    *,
    pointer_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    iteration: int,
    num_topics: int,
) -> set[str]:
    values: set[str] = set()
    for value in (
        pointer_payload.get("embedding_variant"),
        _metadata_embedding_variant(metadata),
        _display_key_suffix(
            (
                None
                if pointer_payload.get("display_key") is None
                else str(pointer_payload.get("display_key"))
            ),
            iteration=iteration,
            num_topics=num_topics,
        ),
    ):
        cleaned = _clean_variant(value)
        if cleaned is not None:
            values.add(cleaned)
    encoder_config = pointer_payload.get("encoder_config")
    if isinstance(encoder_config, Mapping):
        cleaned = _clean_variant(encoder_config.get("embedding_variant"))
        if cleaned is not None:
            values.add(cleaned)
    return values


def _variant_matches(
    *,
    model_key: str,
    pointer_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    iteration: int,
    num_topics: int,
    embedding_variants: Sequence[str] | None,
) -> bool:
    if not embedding_variants:
        return True
    if model_key not in _SENTENCE_EMBEDDING_FILTERED_MODELS:
        return True
    requested = {str(item).strip() for item in embedding_variants if str(item).strip()}
    if not requested:
        return True
    values = _pointer_variant_values(
        pointer_payload=pointer_payload,
        metadata=metadata,
        iteration=iteration,
        num_topics=num_topics,
    )
    if not values:
        return True
    return any(
        _variant_value_matches(value=value, requested=requested_value)
        for value in values
        for requested_value in requested
    )


def _variant_value_matches(*, value: str, requested: str) -> bool:
    return (
        value == requested
        or value.startswith(f"{requested}_")
        or value.endswith(f"_{requested}")
    )


def _pointer_metadata_matches(
    *,
    model_key: str,
    pointer_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
) -> bool:
    display_key = pointer_payload.get("display_key")
    if not _display_key_matches(
        None if display_key is None else str(display_key),
        iteration=iteration,
        num_topics=num_topics,
    ):
        return False

    expected = {
        "dataset": str(dataset),
        "data_run": str(data_run),
        "category": str(category),
    }
    for key, expected_value in expected.items():
        value = _metadata_value(metadata, key)
        if value is None:
            value = pointer_payload.get(key)
        if value is not None and str(value) != expected_value:
            return False

    for key, expected_value in {
        "iteration": int(iteration),
        "num_topics": int(num_topics),
    }.items():
        value = _metadata_value(metadata, key)
        if value is not None and int(value) != expected_value:
            return False

    if model_key == "vmf_sentence_lda":
        model_family = _metadata_value(metadata, "model_family")
        return model_family in {None, "vmf_sentence_lda"}

    runner_key = metadata.get("runner_key")
    runner_family = metadata.get("runner_family")
    return model_key in {
        None if runner_key is None else str(runner_key),
        None if runner_family is None else str(runner_family),
    } or (runner_key is None and runner_family is None)


def _resolve_archive_dir(pointer_payload: Mapping[str, Any]) -> Path | None:
    archive_dir = pointer_payload.get("archive_dir")
    if archive_dir is None or not str(archive_dir).strip():
        return None
    resolved = resolve_project_path(str(archive_dir))
    return resolved if resolved.exists() else None


def _load_pointer_payload(pointer_path: Path) -> dict[str, Any] | None:
    try:
        payload = load_artifact_json(pointer_path)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _load_archive_metadata(archive_dir: Path) -> dict[str, Any]:
    metadata_path = archive_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    try:
        payload = load_artifact_json(metadata_path)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _artifact_path(
    *,
    archive_dir: Path,
    artifacts: Mapping[str, Any],
    keys: Sequence[str],
) -> Path | None:
    for key in keys:
        value = artifacts.get(key)
        if value is None or not str(value).strip():
            continue
        return archive_dir / str(value)
    return None


def _fallback_baseline_artifact_pair(
    *,
    model_key: str,
    archive_dir: Path,
    category: str,
) -> tuple[Path, Path] | None:
    if model_key in {
        "spherical_kmeans",
        "gaussian_kmeans",
        "movmf",
        "gaussian_mixture",
        "senclu",
    }:
        train_name = f"{category}.pkl"
    else:
        train_name = _BASELINE_TRAIN_FILENAMES.get(model_key)
    if train_name is None:
        return None
    return (
        archive_dir / "params" / train_name,
        archive_dir / "infer" / f"{category}.pkl",
    )


def _resolve_pointer_artifact_pair(
    *,
    model_key: str,
    archive_dir: Path,
    pointer_payload: Mapping[str, Any],
    category: str,
    vmf_assignment: str,
) -> tuple[Path, Path] | None:
    raw_artifacts = pointer_payload.get("artifacts")
    artifacts = raw_artifacts if isinstance(raw_artifacts, Mapping) else {}
    if model_key == "vmf_sentence_lda":
        if vmf_assignment == "soft":
            train_path = (
                _artifact_path(
                    archive_dir=archive_dir,
                    artifacts=artifacts,
                    keys=("train_doc_topic_soft", "doc_topic_train_soft"),
                )
                or archive_dir / "doc_topic_train_soft.pkl"
            )
            test_path = (
                _artifact_path(
                    archive_dir=archive_dir,
                    artifacts=artifacts,
                    keys=("test_doc_topic_soft", "doc_topic_test_soft"),
                )
                or archive_dir / "doc_topic_test_soft.pkl"
            )
        else:
            train_path = (
                _artifact_path(
                    archive_dir=archive_dir,
                    artifacts=artifacts,
                    keys=("train_path", "train_doc_topic", "doc_topic_train"),
                )
                or archive_dir / "doc_topic_train.pkl"
            )
            test_path = (
                _artifact_path(
                    archive_dir=archive_dir,
                    artifacts=artifacts,
                    keys=(
                        "infer_path",
                        "test_path",
                        "test_doc_topic",
                        "doc_topic_test",
                    ),
                )
                or archive_dir / "doc_topic_test.pkl"
            )
        return train_path, test_path

    train_path = _artifact_path(
        archive_dir=archive_dir,
        artifacts=artifacts,
        keys=("train_path",),
    )
    test_path = _artifact_path(
        archive_dir=archive_dir,
        artifacts=artifacts,
        keys=("infer_path", "test_path"),
    )
    if train_path is not None and test_path is not None:
        return train_path, test_path
    return _fallback_baseline_artifact_pair(
        model_key=model_key,
        archive_dir=archive_dir,
        category=category,
    )


def _feature_variant_label(
    *,
    pointer_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    iteration: int,
    num_topics: int,
) -> str | None:
    display_key = pointer_payload.get("display_key")
    suffix = _display_key_suffix(
        None if display_key is None else str(display_key),
        iteration=iteration,
        num_topics=num_topics,
    )
    return (
        _clean_variant(suffix)
        or _clean_variant(pointer_payload.get("embedding_variant"))
        or _clean_variant(_metadata_embedding_variant(metadata))
    )


def _resolved_display_name(
    spec: FeatureSpec,
    artifact: ResolvedFeatureArtifact,
    *,
    iteration: int,
    num_topics: int,
) -> str:
    variant_label = _feature_variant_label(
        pointer_payload={
            "display_key": artifact.display_key,
            "embedding_variant": artifact.embedding_variant,
        },
        metadata=artifact.metadata,
        iteration=iteration,
        num_topics=num_topics,
    )
    if variant_label is None:
        return spec.display_name
    return f"{spec.display_name} [{variant_label}]"


def _with_resolved_artifact(
    spec: FeatureSpec,
    artifact: ResolvedFeatureArtifact,
    *,
    iteration: int,
    num_topics: int,
) -> FeatureSpec:
    return FeatureSpec(
        model_key=spec.model_key,
        display_name=_resolved_display_name(
            spec,
            artifact,
            iteration=iteration,
            num_topics=num_topics,
        ),
        train_path_resolver=spec.train_path_resolver,
        test_path_resolver=spec.test_path_resolver,
        train_loader=spec.train_loader,
        test_loader=spec.test_loader,
        available_index_resolver=spec.available_index_resolver,
        metadata=dict(artifact.metadata),
        display_key=artifact.display_key,
        embedding_variant=artifact.embedding_variant,
        condition_fingerprint=artifact.condition_fingerprint,
        encoder_config=artifact.encoder_config,
    )


def _latest_root_for_spec(
    *,
    model_key: str,
    dataset: str,
    data_run: str,
    category: str,
) -> Path:
    if model_key == "vmf_sentence_lda":
        return (
            _current_experiment_results_root()
            / dataset
            / data_run
            / "vmf_sentence_lda"
            / "latest"
            / category
        )
    return (
        _current_baseline_results_root()
        / dataset
        / data_run
        / model_key
        / "latest"
        / category
    )


def _resolve_latest_feature_artifacts(
    *,
    spec: FeatureSpec,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    vmf_assignment: str,
    embedding_variants: Sequence[str] | None,
    feature_resolve_mode: str,
) -> list[ResolvedFeatureArtifact]:
    latest_root = _latest_root_for_spec(
        model_key=spec.model_key,
        dataset=dataset,
        data_run=data_run,
        category=category,
    )
    if not latest_root.exists():
        return []

    resolved: list[ResolvedFeatureArtifact] = []
    invalid_matches: list[str] = []
    for pointer_path in sorted(latest_root.glob(f"*/{CURRENT_POINTER_FILENAME}")):
        pointer_payload = _load_pointer_payload(pointer_path)
        if pointer_payload is None:
            invalid_matches.append(str(pointer_path))
            continue
        archive_dir = _resolve_archive_dir(pointer_payload)
        if archive_dir is None:
            invalid_matches.append(str(pointer_path))
            continue
        metadata = _load_archive_metadata(archive_dir)
        if not _pointer_metadata_matches(
            model_key=spec.model_key,
            pointer_payload=pointer_payload,
            metadata=metadata,
            dataset=dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
        ):
            continue
        if not _variant_matches(
            model_key=spec.model_key,
            pointer_payload=pointer_payload,
            metadata=metadata,
            iteration=iteration,
            num_topics=num_topics,
            embedding_variants=embedding_variants,
        ):
            continue
        artifact_pair = _resolve_pointer_artifact_pair(
            model_key=spec.model_key,
            archive_dir=archive_dir,
            pointer_payload=pointer_payload,
            category=category,
            vmf_assignment=vmf_assignment,
        )
        if artifact_pair is None:
            invalid_matches.append(str(pointer_path))
            continue
        train_path, test_path = artifact_pair
        if not train_path.exists() or not test_path.exists():
            invalid_matches.append(str(pointer_path))
            continue
        encoder_config = pointer_payload.get("encoder_config") or metadata.get(
            "encoder_config"
        )
        resolved.append(
            ResolvedFeatureArtifact(
                train_path=train_path,
                test_path=test_path,
                metadata=metadata,
                display_key=(
                    None
                    if pointer_payload.get("display_key") is None
                    else str(pointer_payload.get("display_key"))
                ),
                embedding_variant=(
                    _clean_variant(pointer_payload.get("embedding_variant"))
                    or _metadata_embedding_variant(metadata)
                ),
                condition_fingerprint=(
                    None
                    if pointer_payload.get("condition_fingerprint") is None
                    else str(pointer_payload.get("condition_fingerprint"))
                )
                or _metadata_condition_fingerprint(metadata),
                encoder_config=(
                    dict(encoder_config)
                    if isinstance(encoder_config, Mapping)
                    else None
                ),
            )
        )

    if feature_resolve_mode == "strict" and invalid_matches:
        raise MissingArtifactError(
            invalid_matches[0],
            detail=(
                "Invalid latest feature pointer(s) found while resolving "
                f"{spec.model_key}: {invalid_matches}"
            ),
        )
    return resolved


def _baseline_feature_spec(
    *,
    model_key: str,
    display_name: str,
    train_loader: FeatureLoader = load_topic_distribution,
    test_loader: FeatureLoader = load_topic_distribution,
    available_index_resolver: AvailableIndexResolver | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        model_key=model_key,
        display_name=display_name,
        train_path_resolver=lambda dataset, data_run, iteration, num_topics, category: _resolve_baseline_doc_topic_path(
            model_key, dataset, data_run, iteration, num_topics, category, "train"
        ),
        test_path_resolver=lambda dataset, data_run, iteration, num_topics, category: _resolve_baseline_doc_topic_path(
            model_key, dataset, data_run, iteration, num_topics, category, "test"
        ),
        train_loader=train_loader,
        test_loader=test_loader,
        available_index_resolver=available_index_resolver,
    )


def _resolve_baseline_doc_topic_path(
    model: str,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
) -> Path:
    try:
        condition_dir = resolve_baseline_condition_dir(
            model=model,
            dataset=dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
        )
    except MissingArtifactError:
        path = build_baseline_doc_topic_path(
            model=model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            data_run=data_run,
        )
    else:
        path = build_baseline_doc_topic_path(
            model=model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            data_run=data_run,
            condition_id=condition_dir.name,
        )
    if path is None:
        raise ValueError(
            f"Baseline model '{model}' does not expose a '{split}' doc-topic path."
        )
    return path


def _build_ctm_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="ctm",
        display_name="Contextual TM",
        available_index_resolver=_baseline_available_resolver(
            require_document_tokens=True,
            require_contextual_text=True,
        ),
    )


def _build_blei_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="bleilda",
        display_name="Blei LDA",
        available_index_resolver=_baseline_available_resolver(
            require_document_tokens=True,
        ),
    )


def _build_bertopic_kmeans_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="bertopic_kmeans",
        display_name="BERTopic (UMAP + k-means)",
        available_index_resolver=_baseline_available_resolver(
            require_document_tokens=False,
            require_contextual_text=True,
        ),
    )


def _build_gaussian_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="gaussianlda",
        display_name="Gaussian LDA",
        available_index_resolver=_baseline_available_resolver(
            require_document_tokens=True,
        ),
    )


def _build_mvtm_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="mvtm",
        display_name="MvTM",
        available_index_resolver=_baseline_available_resolver(
            require_document_tokens=True,
        ),
    )


def _build_etm_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="etm",
        display_name="ETM",
        available_index_resolver=_etm_available_resolver,
    )


def _build_senclu_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="senclu",
        display_name="SenClu",
        train_loader=load_topic_distribution,
        test_loader=load_pickle_array,
        available_index_resolver=_baseline_available_resolver(
            require_sentences=True,
            require_document_tokens=False,
        ),
    )


def _sentence_clustering_feature_spec(
    *,
    model_key: str,
    display_name: str,
) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key=model_key,
        display_name=display_name,
        train_loader=load_topic_distribution,
        test_loader=load_pickle_array,
        available_index_resolver=_baseline_available_resolver(
            require_sentences=True,
            require_document_tokens=False,
        ),
    )


def _build_spherical_kmeans_feature_spec(_: str) -> FeatureSpec:
    return _sentence_clustering_feature_spec(
        model_key="spherical_kmeans",
        display_name="Spherical k-means",
    )


def _build_gaussian_kmeans_feature_spec(_: str) -> FeatureSpec:
    return _sentence_clustering_feature_spec(
        model_key="gaussian_kmeans",
        display_name="Gaussian k-means",
    )


def _build_movmf_feature_spec(_: str) -> FeatureSpec:
    return _sentence_clustering_feature_spec(
        model_key="movmf",
        display_name="movMF",
    )


def _build_gaussian_mixture_feature_spec(_: str) -> FeatureSpec:
    return _sentence_clustering_feature_spec(
        model_key="gaussian_mixture",
        display_name="Gaussian mixture",
    )


def _build_sentence_gaussian_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="sentence_gaussianlda",
        display_name="Sentence LDA",
        available_index_resolver=_baseline_available_resolver(
            require_sentences=True,
            require_document_tokens=False,
        ),
    )


def _build_sentlda_feature_spec(_: str) -> FeatureSpec:
    return _baseline_feature_spec(
        model_key="sentlda",
        display_name="sentLDA",
        available_index_resolver=_sentlda_available_resolver,
    )


def _build_vmf_feature_spec(vmf_assignment: str) -> FeatureSpec:
    if vmf_assignment == "soft":
        vmf_display_name = "vMF Sentence LDA (soft)"
    elif vmf_assignment == "hard":
        vmf_display_name = "vMF Sentence LDA"
    else:
        raise ValueError(f"Unknown vmf_assignment: {vmf_assignment}")
    return FeatureSpec(
        model_key="vmf_sentence_lda",
        display_name=vmf_display_name,
        train_path_resolver=lambda dataset, data_run, iteration, num_topics, category: _vmf_doc_topic_path(
            dataset=dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split="train",
            assignment=vmf_assignment,
        ),
        test_path_resolver=lambda dataset, data_run, iteration, num_topics, category: _vmf_doc_topic_path(
            dataset=dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split="test",
            assignment=vmf_assignment,
        ),
        train_loader=load_topic_distribution,
        test_loader=load_topic_distribution,
        available_index_resolver=_vmf_available_resolver,
    )


def _vmf_doc_topic_path(
    *,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    assignment: str,
) -> Path:
    try:
        condition_dir = resolve_vmf_experiment_dir(
            dataset=dataset,
            run_name=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
        )
    except MissingArtifactError:
        return build_vmf_doc_topic_path(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            assignment=assignment,
            run_name=data_run,
        )
    return build_vmf_doc_topic_path(
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        assignment=assignment,
        run_name=data_run,
        condition_id=condition_dir.name,
    )


def _label_space_available_resolver(
    dataset: str,
    category: str,
    _train_path: Path,
    _test_path: Path,
    target_column: str,
    label_schema: str,
) -> tuple[SplitAlignment, SplitAlignment]:
    return build_label_space_indices(
        dataset,
        category,
        target_column=target_column,
        label_schema=label_schema,
    )


def _baseline_available_resolver(
    *,
    require_document_tokens: bool,
    require_contextual_text: bool = False,
    require_sentences: bool = False,
) -> AvailableIndexResolver:
    def _resolver(
        dataset: str,
        category: str,
        train_path: Path,
        _test_path: Path,
        target_column: str,
        label_schema: str,
    ) -> tuple[SplitAlignment, SplitAlignment]:
        selection_alignment = _selection_alignment_from_artifacts(
            train_path,
            _test_path,
        )
        if selection_alignment is not None:
            return selection_alignment
        metadata = _load_feature_metadata(train_path)
        return build_baseline_available_indices(
            dataset,
            category,
            metadata,
            target_column=target_column,
            label_schema=label_schema,
            require_document_tokens=require_document_tokens,
            require_contextual_text=require_contextual_text,
            require_sentences=require_sentences,
        )

    return _resolver


def _sentlda_available_resolver(
    dataset: str,
    category: str,
    train_path: Path,
    _test_path: Path,
    target_column: str,
    label_schema: str,
) -> tuple[SplitAlignment, SplitAlignment]:
    metadata = _load_feature_metadata(train_path)
    vocabulary_path = train_path.parent / "vocabulary.json"
    vocabulary_payload = load_artifact_json(vocabulary_path)
    if not isinstance(vocabulary_payload, dict):
        raise ValueError(f"Invalid sentLDA vocabulary artifact: {vocabulary_path}")
    vocabulary = {str(token) for token in vocabulary_payload}

    return build_preprocessed_available_indices(
        dataset,
        category,
        metadata,
        availability_predicate=lambda document: _sentlda_document_available(
            document,
            vocabulary,
        ),
        target_column=target_column,
        label_schema=label_schema,
    )


def _etm_available_resolver(
    dataset: str,
    category: str,
    train_path: Path,
    _test_path: Path,
    target_column: str,
    label_schema: str,
) -> tuple[SplitAlignment, SplitAlignment]:
    metadata = _load_feature_metadata(train_path)
    vocabulary_path = train_path.parent / "vocabulary.json"
    vocabulary_payload = load_artifact_json(vocabulary_path)
    if isinstance(vocabulary_payload, dict):
        vocabulary = {str(token) for token in vocabulary_payload}
    elif isinstance(vocabulary_payload, list):
        vocabulary = {str(token) for token in vocabulary_payload}
    else:
        raise ValueError(f"Invalid ETM vocabulary artifact: {vocabulary_path}")

    return build_preprocessed_available_indices(
        dataset,
        category,
        metadata,
        availability_predicate=lambda document: _document_has_vocabulary_token(
            document,
            vocabulary,
        ),
        target_column=target_column,
        label_schema=label_schema,
    )


def _vmf_available_resolver(
    dataset: str,
    category: str,
    train_path: Path,
    _test_path: Path,
    target_column: str,
    label_schema: str,
) -> tuple[SplitAlignment, SplitAlignment]:
    metadata = _load_feature_metadata(train_path)
    selection_alignment = _selection_alignment_from_artifacts(train_path, _test_path)
    if selection_alignment is not None:
        return selection_alignment
    if metadata is None:
        return _label_space_available_resolver(
            dataset,
            category,
            train_path,
            _test_path,
            target_column,
            label_schema,
        )

    return build_preprocessed_available_indices(
        dataset,
        category,
        metadata,
        availability_predicate=_sentence_available,
        target_column=target_column,
        label_schema=label_schema,
    )


def _sentence_available(document: PreprocessedDocument) -> bool:
    tokenized_sentences = getattr(document, "sentences_tokenized", None)
    if tokenized_sentences is None:
        return bool(getattr(document, "sentences_raw", None))
    return any(
        str(raw_sentence).strip() and bool(sentence_tokens)
        for raw_sentence, sentence_tokens in zip(
            document.sentences_raw,
            tokenized_sentences,
        )
    )


def _selection_alignment_from_artifacts(
    train_path: Path,
    test_path: Path,
) -> tuple[SplitAlignment, SplitAlignment] | None:
    train_selection_path = train_path.parent / PREPROCESSING_SELECTION_FILENAME
    test_selection_path = test_path.parent / PREPROCESSING_SELECTION_FILENAME
    if train_selection_path.exists() and test_selection_path.exists():
        return (
            _split_alignment_from_selection_payload(
                load_artifact_json(train_selection_path),
                split_key="train",
            ),
            _split_alignment_from_selection_payload(
                load_artifact_json(test_selection_path),
                split_key="test",
            ),
        )
    if train_selection_path.exists() and train_selection_path == test_selection_path:
        payload = load_artifact_json(train_selection_path)
        if isinstance(payload, Mapping) and "train" in payload and "test" in payload:
            return (
                _split_alignment_from_selection_payload(payload, split_key="train"),
                _split_alignment_from_selection_payload(payload, split_key="test"),
            )
    return None


def _split_alignment_from_selection_payload(
    payload: object,
    *,
    split_key: str,
) -> SplitAlignment:
    if isinstance(payload, Mapping) and split_key in payload:
        payload = payload[split_key]
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid preprocessing selection artifact payload.")
    raw_doc_indices = payload.get("raw_doc_indices")
    if not isinstance(raw_doc_indices, list):
        raise ValueError("preprocessing_selection.json is missing raw_doc_indices.")
    available = np.asarray([int(index) for index in raw_doc_indices], dtype=int)
    return SplitAlignment(raw_indices=available.copy(), available_indices=available)


def _sentlda_document_available(
    document: PreprocessedDocument,
    vocabulary: set[str],
) -> bool:
    return any(
        any(token in vocabulary for token in sentence_tokens)
        for sentence_tokens in document.sentences_tokenized
    )


def _document_has_vocabulary_token(
    document: PreprocessedDocument,
    vocabulary: set[str],
) -> bool:
    return any(token in vocabulary for token in document.document_tokens)


FEATURE_REGISTRY: dict[str, FeatureSpecBuilder] = {
    "ctm": _build_ctm_feature_spec,
    "bleilda": _build_blei_feature_spec,
    "bertopic_kmeans": _build_bertopic_kmeans_feature_spec,
    "gaussianlda": _build_gaussian_feature_spec,
    "etm": _build_etm_feature_spec,
    "mvtm": _build_mvtm_feature_spec,
    "spherical_kmeans": _build_spherical_kmeans_feature_spec,
    "gaussian_kmeans": _build_gaussian_kmeans_feature_spec,
    "movmf": _build_movmf_feature_spec,
    "gaussian_mixture": _build_gaussian_mixture_feature_spec,
    "senclu": _build_senclu_feature_spec,
    "sentence_gaussianlda": _build_sentence_gaussian_feature_spec,
    "sentlda": _build_sentlda_feature_spec,
    "vmf_sentence_lda": _build_vmf_feature_spec,
}


def register_feature_spec(model_key: str, builder: FeatureSpecBuilder) -> None:
    normalized_key = str(model_key).strip()
    if not normalized_key:
        raise ValueError("model_key must not be empty.")
    FEATURE_REGISTRY[normalized_key] = builder


def get_feature_specs(vmf_assignment: str) -> list[FeatureSpec]:
    return [builder(vmf_assignment) for builder in FEATURE_REGISTRY.values()]


def build_feature_specs(vmf_assignment: str) -> list[FeatureSpec]:
    return get_feature_specs(vmf_assignment)


def iter_available_features(
    *,
    dataset: str,
    data_run: str = "default",
    iteration: int,
    num_topics: int,
    category: str,
    vmf_assignment: str,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = "all",
    selected_models: Sequence[str] | None = None,
) -> list[tuple[FeatureSpec, Path, Path]]:
    if feature_resolve_mode not in {"all", "strict"}:
        raise ValueError(f"Unknown feature_resolve_mode: {feature_resolve_mode}")

    selectors = [str(model) for model in (selected_models or []) if str(model).strip()]
    resolved_features: list[tuple[FeatureSpec, Path, Path]] = []
    for spec in get_feature_specs(vmf_assignment):
        if selectors and not any(
            model_matches_selector(
                spec.display_name,
                selector,
                model_key=spec.model_key,
            )
            for selector in selectors
        ):
            continue

        latest_artifacts = _resolve_latest_feature_artifacts(
            spec=spec,
            dataset=dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            vmf_assignment=vmf_assignment,
            embedding_variants=embedding_variants,
            feature_resolve_mode=feature_resolve_mode,
        )
        if latest_artifacts:
            for artifact in latest_artifacts:
                resolved_spec = _with_resolved_artifact(
                    spec,
                    artifact,
                    iteration=iteration,
                    num_topics=num_topics,
                )
                resolved_features.append(
                    (resolved_spec, artifact.train_path, artifact.test_path)
                )
            continue

        try:
            train_path = spec.train_path_resolver(
                dataset, data_run, iteration, num_topics, category
            )
            test_path = spec.test_path_resolver(
                dataset, data_run, iteration, num_topics, category
            )
        except MissingArtifactError as exc:
            if feature_resolve_mode == "strict":
                raise
            LOGGER.warning(
                "[skip] %s: unable to resolve feature paths (%s)",
                spec.display_name,
                exc,
            )
            continue

        resolved_features.append((spec, train_path, test_path))
    return resolved_features

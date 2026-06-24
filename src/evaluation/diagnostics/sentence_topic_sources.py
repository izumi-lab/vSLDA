from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.artifacts import (
    METADATA_FILENAME,
    load_artifact_json,
    load_artifact_pickle,
)
from src.core.paths import (
    RESULTS_ROOT,
    build_baseline_doc_topic_path,
    resolve_baseline_condition_dir,
)

from .sentence_topic_artifacts import resolve_vmf_experiment_dir

VMF_MODEL_ALIASES = {"vmf", "vmf_sentence_lda"}


@dataclass(frozen=True)
class InspectionSource:
    model: str
    model_family: str
    condition_dir: Path
    artifact_dir: Path
    doc_topic_path: Path
    metadata_path: Path | None
    primary_artifact_path: Path
    average_ll: list[float]
    model_provenance_key: str
    encoder_model: str | None
    embedding_variant: str | None
    supports_vmf_sphere: bool
    supports_vmf_top_sentences: bool
    supports_gaussian_top_sentences: bool
    supports_sentlda_top_sentences: bool
    supports_senclu_top_sentences: bool
    top_sentence_method: str | None


def normalize_inspection_model(model: str) -> str:
    normalized = str(model).strip().lower().replace("-", "_")
    if normalized in VMF_MODEL_ALIASES:
        return "vmf_sentence_lda"
    return normalized


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = load_artifact_json(path)
    return payload if isinstance(payload, dict) else {}


def _metadata_encoder_model(metadata: dict[str, Any]) -> str | None:
    encoder_config = metadata.get("encoder_config")
    if isinstance(encoder_config, dict):
        model_name = encoder_config.get("model_name")
        if model_name not in {None, ""}:
            return str(model_name)
    axes = metadata.get("axes")
    if isinstance(axes, dict):
        model_name = axes.get("encoder_model")
        if model_name not in {None, ""}:
            return str(model_name)
    return None


def _metadata_embedding_variant(metadata: dict[str, Any]) -> str | None:
    encoder_config = metadata.get("encoder_config")
    if isinstance(encoder_config, dict):
        variant = encoder_config.get("embedding_variant")
        if variant not in {None, ""}:
            return str(variant)
    variant = metadata.get("embedding_variant")
    if variant not in {None, ""}:
        return str(variant)
    axes = metadata.get("axes")
    if isinstance(axes, dict):
        variant = axes.get("embedding_variant")
        if variant not in {None, ""}:
            return str(variant)
    return None


def _load_average_ll(params_path: Path) -> list[float]:
    candidate_paths = [params_path]
    if params_path.parent.exists():
        candidate_paths.extend(
            path
            for path in sorted(params_path.parent.glob("*.json"))
            if path != params_path
        )
    for candidate_path in candidate_paths:
        payload = _load_json_if_exists(candidate_path)
        values = payload.get("average_ll")
        if isinstance(values, list):
            return [float(value) for value in values]
    return []


def resolve_inspection_source(
    *,
    model: str,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    data_run: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    results_root: Path = RESULTS_ROOT,
) -> InspectionSource:
    normalized_model = normalize_inspection_model(model)
    if normalized_model == "vmf_sentence_lda":
        exp_dir = resolve_vmf_experiment_dir(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            condition_id=condition_id,
            num_components=num_components,
            embedding_variant=embedding_variant,
            results_root=results_root,
        )
        metadata_path = exp_dir / METADATA_FILENAME
        metadata = _load_json_if_exists(metadata_path)
        return InspectionSource(
            model=normalized_model,
            model_family=normalized_model,
            condition_dir=exp_dir,
            artifact_dir=exp_dir,
            doc_topic_path=exp_dir / f"doc_topic_{split}.pkl",
            metadata_path=metadata_path if metadata_path.exists() else None,
            primary_artifact_path=exp_dir / "topic_means.pkl",
            average_ll=_load_average_ll(exp_dir / "params.json"),
            model_provenance_key=normalized_model,
            encoder_model=_metadata_encoder_model(metadata),
            embedding_variant=embedding_variant
            or _metadata_embedding_variant(metadata),
            supports_vmf_sphere=True,
            supports_vmf_top_sentences=True,
            supports_gaussian_top_sentences=False,
            supports_sentlda_top_sentences=False,
            supports_senclu_top_sentences=False,
            top_sentence_method="vmf_loglik",
        )

    condition_dir = resolve_baseline_condition_dir(
        model=normalized_model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        condition_id=condition_id,
        num_components=num_components,
        embedding_variant=embedding_variant,
        baseline_root=results_root / "baselines",
    )
    doc_topic_path = build_baseline_doc_topic_path(
        model=normalized_model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        data_run=data_run,
        condition_id=condition_id,
        num_components=num_components,
        embedding_variant=embedding_variant,
        baseline_root=results_root / "baselines",
    )
    if doc_topic_path is None:
        raise ValueError(
            f"Model '{normalized_model}' does not expose {split} doc-topic artifacts."
        )
    metadata_path = condition_dir / METADATA_FILENAME
    metadata = _load_json_if_exists(metadata_path)
    params_dir = condition_dir / "params"
    primary_artifact_path = doc_topic_path
    if (
        normalized_model == "sentence_gaussianlda"
        and (params_dir / "table_means.pkl").exists()
    ):
        primary_artifact_path = params_dir / "table_means.pkl"
    return InspectionSource(
        model=normalized_model,
        model_family=str(metadata.get("runner_family") or normalized_model),
        condition_dir=condition_dir,
        artifact_dir=doc_topic_path.parent,
        doc_topic_path=doc_topic_path,
        metadata_path=metadata_path if metadata_path.exists() else None,
        primary_artifact_path=primary_artifact_path,
        average_ll=_load_average_ll(params_dir / "params.json"),
        model_provenance_key=normalized_model,
        encoder_model=_metadata_encoder_model(metadata),
        embedding_variant=embedding_variant or _metadata_embedding_variant(metadata),
        supports_vmf_sphere=False,
        supports_vmf_top_sentences=False,
        supports_gaussian_top_sentences=normalized_model == "sentence_gaussianlda",
        supports_sentlda_top_sentences=normalized_model == "sentlda",
        supports_senclu_top_sentences=normalized_model == "senclu",
        top_sentence_method=(
            "gaussian_loglik"
            if normalized_model == "sentence_gaussianlda"
            else (
                "sentlda_token_loglik"
                if normalized_model == "sentlda"
                else (
                    "senclu_sentence_given_topic"
                    if normalized_model == "senclu"
                    else None
                )
            )
        ),
    )


def load_source_doc_topics(source: InspectionSource) -> Any:
    return load_artifact_pickle(source.doc_topic_path)

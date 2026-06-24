from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core.artifacts import METADATA_FILENAME, load_artifact_pickle, load_json
from src.core.paths import (
    RESULTS_ROOT,
    VISUALIZATION_RESULTS_ROOT,
    build_archive_result_dir,
    build_latest_result_dir,
    write_latest_result_pointer,
)
from src.core.result_identity import build_condition_id, build_execution_id
from src.data.splits import load_filtered_split_sentences
from src.evaluation.model_provenance import load_model_provenance_for_artifact
from src.evaluation.reporting import write_evaluation_json
from src.evaluation.schema import build_evaluation_meta
from src.evaluation.source_data import resolve_artifact_split_config
from src.utils.encoder import SentenceEncoder
from src.utils.encoder_profiles import (
    default_encoder_model_for_embedding_variant,
    embedding_variant_base,
    encoder_model_alias,
)
from src.utils.random import DEFAULT_RANDOM_SEED

from .sentence_topic_artifacts import (
    load_gaussian_params,
    load_topic_means,
    load_vmf_params,
    resolve_sentence_gaussian_dir,
)
from .sentence_topic_scoring import (
    top_sentences_by_topic_gaussian_loglik,
    top_sentences_by_topic_log_score_matrix,
    top_sentences_by_topic_vmf_loglik,
)
from .sentence_topic_sources import (
    InspectionSource,
    load_source_doc_topics,
    resolve_inspection_source,
)
from .sentence_topic_visualization import (
    plot_average_ll,
    plot_doc_topics,
    plot_embeddings_on_sphere_3d,
)


def _select_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _iter_unique(values: list[str] | list[int]) -> list[str] | list[int]:
    seen: set[str | int] = set()
    ordered: list[str] | list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_sentence_encoder(
    *,
    model_name: str,
    device: str,
) -> SentenceEncoder:
    return SentenceEncoder(
        model_name,
        device=device,
        strip_terminal_normalize=False,
    )


DEFAULT_SENTENCE_TOPIC_ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"


def _metadata_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = load_json(path)
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


def _resolve_encoder_model_for_source(
    *,
    requested_encoder_model: str | None,
    requested_embedding_variant: str | None,
    exp_dir: Path,
) -> str:
    metadata = _metadata_mapping(exp_dir / METADATA_FILENAME)
    metadata_encoder_model = _metadata_encoder_model(metadata)
    metadata_embedding_variant = _metadata_embedding_variant(metadata)
    effective_variant = requested_embedding_variant or metadata_embedding_variant

    if requested_encoder_model not in {None, ""}:
        resolved_encoder_model = str(requested_encoder_model)
    elif metadata_encoder_model:
        resolved_encoder_model = metadata_encoder_model
    elif effective_variant:
        resolved_encoder_model = (
            default_encoder_model_for_embedding_variant(effective_variant)
            or DEFAULT_SENTENCE_TOPIC_ENCODER_MODEL
        )
    else:
        resolved_encoder_model = DEFAULT_SENTENCE_TOPIC_ENCODER_MODEL

    if effective_variant:
        encoder_variant = embedding_variant_base(
            encoder_model_alias(resolved_encoder_model)
        )
        expected_variant = embedding_variant_base(effective_variant)
        if encoder_variant != expected_variant:
            raise ValueError(
                "Encoder and embedding_variant mismatch: "
                f"embedding_variant='{effective_variant}' expects encoder variant "
                f"'{expected_variant}', but encoder='{resolved_encoder_model}' resolves "
                f"to '{encoder_variant}'. Omit --encoder to use the source artifact's "
                "encoder, or pass a matching --encoder."
            )

    return resolved_encoder_model


def encode_sentences(
    encoder: SentenceEncoder,
    sentences: list[str],
    *,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    return np.asarray(
        encoder.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        ),
        dtype=float,
    )


def _start_execution() -> tuple[str, str]:
    started_at = datetime.now(UTC).isoformat()
    return started_at, build_execution_id(prefix="exec", started_at=started_at)


def _build_artifact_meta_context(
    *,
    display_key: str,
    started_at: str,
    execution_id: str,
    archive_dir: Path,
    latest_dir: Path,
) -> dict[str, Any]:
    return {
        "display_key": display_key,
        "started_at": started_at,
        "execution_id": execution_id,
        "archive_dir": str(archive_dir),
        "latest_dir": str(latest_dir),
    }


def _build_output_condition_id(
    *,
    model: str,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    split: str,
    encoder_model: str,
    gaussian_topk: bool,
    max_points: int,
    source_condition_id: str | None,
    embedding_variant: str | None,
    num_components: int | None,
    gaussian_condition_id: str | None,
    gaussian_embedding_variant: str | None,
    gaussian_num_components: int | None,
) -> tuple[str, str]:
    extra_labels = ["inspect", model]
    if embedding_variant:
        extra_labels.append(embedding_variant)
    if gaussian_topk and gaussian_embedding_variant:
        extra_labels.append(f"gaussian-{gaussian_embedding_variant}")

    return build_condition_id(
        iteration=int(iteration),
        num_topics=int(num_topics),
        fingerprint_payload={
            "task": "sentence_topic_inspection",
            "model": model,
            "dataset": dataset,
            "data_run": data_run,
            "category": category,
            "iteration": int(iteration),
            "num_topics": int(num_topics),
            "split": split,
            "encoder_model": encoder_model,
            "gaussian_topk": bool(gaussian_topk),
            "max_points": int(max_points),
            "source_condition_id": source_condition_id,
            "embedding_variant": embedding_variant,
            "num_components": None if num_components is None else int(num_components),
            "gaussian_condition_id": gaussian_condition_id,
            "gaussian_embedding_variant": gaussian_embedding_variant,
            "gaussian_num_components": (
                None
                if gaussian_num_components is None
                else int(gaussian_num_components)
            ),
        },
        extra_labels=extra_labels,
    )


def _validate_single_condition_selector(
    *,
    selector_name: str,
    selector_value: str | None,
    categories: list[str],
    iterations: list[int],
    num_topics_list: list[int],
) -> None:
    if selector_value is None:
        return
    condition_count = len(categories) * len(iterations) * len(num_topics_list)
    if condition_count > 1:
        raise ValueError(
            f"{selector_name} can only be used when selecting a single category, "
            "iteration, and topic count."
        )


def _source_selection_payload(
    *,
    model: str,
    source_condition_id: str | None,
    embedding_variant: str | None,
    num_components: int | None,
    gaussian_condition_id: str | None,
    gaussian_embedding_variant: str | None,
    gaussian_num_components: int | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "source_condition_id": source_condition_id,
        "embedding_variant": embedding_variant,
        "num_components": None if num_components is None else int(num_components),
        "gaussian_condition_id": gaussian_condition_id,
        "gaussian_embedding_variant": gaussian_embedding_variant,
        "gaussian_num_components": (
            None if gaussian_num_components is None else int(gaussian_num_components)
        ),
    }


def _write_avg_ll_sidecar(
    *,
    out_dir: Path,
    dataset: str,
    category: str,
    iteration: int,
    num_topics: int,
    condition_id: str,
    condition_fingerprint: str,
    average_ll: list[float],
    results_root: Path,
    source_artifact_path: Path,
    model_provenance: dict[str, Any],
    artifact_meta: dict[str, Any],
) -> Path | None:
    png_path = out_dir / "avg_ll.png"
    if not plot_average_ll(average_ll, png_path):
        return None
    sidecar_path = png_path.with_suffix(".json")
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="sentence_topic_inspection",
            output_kind="payload",
            artifact="avg_ll_plot",
            dataset=dataset,
            category=category,
            iteration=int(iteration),
            num_topics=int(num_topics),
            condition_id=condition_id,
            condition_fingerprint=condition_fingerprint,
            png_path=str(png_path),
            results_root=str(results_root),
            source_artifact_path=str(source_artifact_path),
            model_provenance=model_provenance,
            **artifact_meta,
        ),
        results={
            "point_count": int(len(average_ll)),
            "average_ll": [float(value) for value in average_ll],
        },
        path=sidecar_path,
    )
    return sidecar_path


def _write_doc_topic_sidecar(
    *,
    out_dir: Path,
    dataset: str,
    category: str,
    iteration: int,
    num_topics: int,
    condition_id: str,
    condition_fingerprint: str,
    seed: int | None,
    doc_topics: np.ndarray,
    results_root: Path,
    source_artifact_path: Path,
    doc_topic_path: Path,
    model_provenance: dict[str, Any],
    artifact_meta: dict[str, Any],
) -> Path | None:
    png_path = out_dir / "doc_topic_tsne.png"
    ok = plot_doc_topics(
        doc_topics,
        png_path,
        seed=seed,
        title="Doc-topic TSNE (color = argmax topic)",
    )
    if not ok:
        return None
    sidecar_path = png_path.with_suffix(".json")
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="sentence_topic_inspection",
            output_kind="payload",
            artifact="doc_topic_tsne",
            dataset=dataset,
            category=category,
            iteration=int(iteration),
            num_topics=int(num_topics),
            condition_id=condition_id,
            condition_fingerprint=condition_fingerprint,
            seed=seed,
            png_path=str(png_path),
            results_root=str(results_root),
            source_artifact_path=str(source_artifact_path),
            doc_topic_path=str(doc_topic_path),
            model_provenance=model_provenance,
            **artifact_meta,
        ),
        results={
            "num_docs": int(doc_topics.shape[0]),
            "num_topics": int(doc_topics.shape[1]),
            "argmax_topic_histogram": np.bincount(
                np.argmax(doc_topics, axis=1),
                minlength=doc_topics.shape[1],
            )
            .astype(int)
            .tolist(),
        },
        path=sidecar_path,
    )
    return sidecar_path


def _write_sphere_sidecar(
    *,
    out_dir: Path,
    dataset: str,
    category: str,
    iteration: int,
    num_topics: int,
    condition_id: str,
    condition_fingerprint: str,
    seed: int | None,
    embeddings: np.ndarray,
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    max_points: int,
    results_root: Path,
    source_artifact_path: Path,
    model_provenance: dict[str, Any],
    artifact_meta: dict[str, Any],
) -> Path:
    png_path = out_dir / "embeddings_on_sphere_3d.png"
    plot_stats = plot_embeddings_on_sphere_3d(
        embeddings=embeddings,
        topic_means=topic_means,
        kappa_per_topic=kappa_per_topic,
        mixture_weights=mixture_weights,
        component_means=component_means,
        out_path=png_path,
        max_points=max_points,
        seed=seed,
    )
    sidecar_path = png_path.with_suffix(".json")
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="sentence_topic_inspection",
            output_kind="payload",
            artifact="embeddings_on_sphere_3d",
            dataset=dataset,
            category=category,
            iteration=int(iteration),
            num_topics=int(num_topics),
            condition_id=condition_id,
            condition_fingerprint=condition_fingerprint,
            seed=seed,
            max_points=int(max_points),
            png_path=str(png_path),
            results_root=str(results_root),
            source_artifact_path=str(source_artifact_path),
            model_provenance=model_provenance,
            **artifact_meta,
        ),
        results=plot_stats,
        path=sidecar_path,
    )
    return sidecar_path


def _write_sentence_payloads(
    *,
    out_dir: Path,
    dataset: str,
    category: str,
    iteration: int,
    num_topics: int,
    condition_id: str,
    condition_fingerprint: str,
    top_k: int,
    sentences: list[str],
    embeddings: np.ndarray,
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    results_root: Path,
    source_artifact_path: Path,
    model_provenance: dict[str, Any],
    gaussian_topk: bool,
    gaussian_dir: Path | None = None,
    artifact_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    resolved_artifact_meta = {} if artifact_meta is None else dict(artifact_meta)

    kappa_path = out_dir / "kappa_per_topic.json"
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="sentence_topic_inspection",
            output_kind="payload",
            artifact="kappa_per_topic",
            dataset=dataset,
            category=category,
            iteration=int(iteration),
            num_topics=int(num_topics),
            condition_id=condition_id,
            condition_fingerprint=condition_fingerprint,
            results_root=str(results_root),
            source_artifact_path=str(source_artifact_path),
            model_provenance=model_provenance,
            **resolved_artifact_meta,
        ),
        results={
            "kappa_per_topic": {
                int(topic_idx): float(kappa)
                for topic_idx, kappa in enumerate(kappa_per_topic.tolist())
            }
        },
        path=kappa_path,
    )
    paths["kappa_per_topic_path"] = str(kappa_path)

    top_sentences = top_sentences_by_topic_vmf_loglik(
        topic_means=topic_means,
        kappa_per_topic=kappa_per_topic,
        mixture_weights=mixture_weights,
        component_means=component_means,
        sentences=sentences,
        embeddings=embeddings,
        top_k=top_k,
    )
    top_sentences_path = out_dir / "top_sentences_loglik.json"
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="sentence_topic_inspection",
            output_kind="payload",
            artifact="top_sentences_loglik",
            dataset=dataset,
            category=category,
            iteration=int(iteration),
            num_topics=int(num_topics),
            condition_id=condition_id,
            condition_fingerprint=condition_fingerprint,
            top_k=int(top_k),
            sentence_count=int(len(sentences)),
            results_root=str(results_root),
            source_artifact_path=str(source_artifact_path),
            top_sentence_method="vmf_loglik",
            model_provenance=model_provenance,
            **resolved_artifact_meta,
        ),
        results={"topics": top_sentences},
        path=top_sentences_path,
    )
    paths["top_sentences_loglik_path"] = str(top_sentences_path)

    if gaussian_topk:
        if gaussian_dir is None:
            raise ValueError("gaussian_dir is required when gaussian_topk=True")
        gaussian_path = gaussian_dir / "table_means.pkl"
        if not gaussian_path.exists():
            raise FileNotFoundError(
                f"Gaussian Sentence LDA topic means not found: {gaussian_path}"
            )
        gaussian_provenance = load_model_provenance_for_artifact(
            gaussian_path,
            model_key="sentence_gaussianlda",
        )
        gaussian_means, gaussian_cholesky, gaussian_log_dets = load_gaussian_params(
            gaussian_dir
        )
        gaussian_top_sentences = top_sentences_by_topic_gaussian_loglik(
            gaussian_means=gaussian_means,
            gaussian_cholesky=gaussian_cholesky,
            gaussian_log_determinants=gaussian_log_dets,
            sentences=sentences,
            embeddings=embeddings,
            top_k=top_k,
        )
        gaussian_top_sentences_path = out_dir / "top_sentences_gaussian_loglik.json"
        write_evaluation_json(
            meta=build_evaluation_meta(
                task="sentence_topic_inspection",
                output_kind="payload",
                artifact="top_sentences_gaussian_loglik",
                dataset=dataset,
                category=category,
                iteration=int(iteration),
                num_topics=int(num_topics),
                condition_id=condition_id,
                condition_fingerprint=condition_fingerprint,
                top_k=int(top_k),
                sentence_count=int(len(sentences)),
                results_root=str(results_root),
                gaussian_artifact_path=str(gaussian_path),
                model_provenance=gaussian_provenance,
                **resolved_artifact_meta,
            ),
            results={"topics": gaussian_top_sentences},
            path=gaussian_top_sentences_path,
        )
        paths["top_sentences_gaussian_loglik_path"] = str(gaussian_top_sentences_path)

    return paths


def _flatten_grouped_topic_arrays(grouped: Any, *, path: Path) -> np.ndarray:
    if isinstance(grouped, np.ndarray):
        if grouped.ndim != 2:
            raise ValueError(
                f"Expected a 2D array at {path}, got shape {grouped.shape}."
            )
        return np.asarray(grouped, dtype=float)
    if not isinstance(grouped, (list, tuple)):
        raise ValueError(f"Expected a list of arrays at {path}, got {type(grouped)}.")
    arrays = [np.asarray(item, dtype=float) for item in grouped]
    if not arrays:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(arrays)


def _matrix_sentence_score_path(
    source: InspectionSource, *, split: str
) -> tuple[Path, str, str]:
    # File names include the original category, not the display key. Derive them from
    # the soft artifact pattern by scanning the selected split directory.
    split_dir = source.condition_dir / ("params" if split == "train" else "infer")
    if source.supports_senclu_top_sentences:
        soft_matches = sorted(split_dir.glob("*_sentence_topic_soft.pkl"))
        if soft_matches:
            return (
                soft_matches[0],
                "senclu_sentence_given_topic",
                "p(sentence|topic) from column-normalized p(topic|sentence, document)",
            )
        raise FileNotFoundError(
            f"SenClu sentence-topic soft artifact not found under {split_dir}"
        )

    loglik_matches = sorted(split_dir.glob("*_sentence_topic_loglik.pkl"))
    if loglik_matches:
        return (
            loglik_matches[0],
            "sentlda_token_loglik",
            "log p(sentence|topic, rest)",
        )
    soft_matches = sorted(split_dir.glob("*_sentence_topic_soft.pkl"))
    if soft_matches:
        return (
            soft_matches[0],
            "sentlda_collapsed_conditional_logprob",
            "log(max(p(topic|sentence, rest), eps))",
        )
    raise FileNotFoundError(
        f"sentLDA sentence-topic score artifact not found under {split_dir}"
    )


def _baseline_preprocessed_sentences_path(
    source: InspectionSource, *, split: str
) -> Path:
    split_dir = source.condition_dir / ("params" if split == "train" else "infer")
    return split_dir / "preprocessed_corpus.pkl"


def _load_preprocessed_sentences(path: Path) -> list[str]:
    sentences, _token_counts = _load_preprocessed_sentences_with_token_counts(path)
    return sentences


def _load_preprocessed_sentences_with_token_counts(
    path: Path,
) -> tuple[list[str], np.ndarray]:
    payload = load_artifact_pickle(path)
    documents = getattr(payload, "documents", payload)
    if not isinstance(documents, (list, tuple)):
        raise ValueError(
            f"Expected a preprocessed corpus or list of documents at {path}, "
            f"got {type(payload)}."
        )

    sentences: list[str] = []
    token_counts: list[int] = []
    for doc_index, document in enumerate(documents):
        raw_sentences = getattr(document, "sentences_raw", None)
        if raw_sentences is None:
            raise ValueError(
                f"Expected document {doc_index} at {path} to expose sentences_raw."
            )
        tokenized_sentences = getattr(document, "sentences_tokenized", None)
        if tokenized_sentences is None:
            raise ValueError(
                f"Expected document {doc_index} at {path} to expose sentences_tokenized."
            )
        if len(raw_sentences) != len(tokenized_sentences):
            raise ValueError(
                f"Document {doc_index} at {path} has mismatched sentence counts: "
                f"raw={len(raw_sentences)} tokenized={len(tokenized_sentences)}."
            )
        for raw_sentence, tokenized_sentence in zip(
            raw_sentences,
            tokenized_sentences,
            strict=True,
        ):
            if not str(raw_sentence).strip():
                continue
            sentences.append(str(raw_sentence))
            token_counts.append(len(tokenized_sentence))
    return sentences, np.asarray(token_counts, dtype=float)


def _normalize_sentlda_loglik_by_token_count(
    log_scores: np.ndarray,
    token_counts: np.ndarray,
    *,
    path: Path,
) -> np.ndarray:
    if log_scores.shape[0] != token_counts.shape[0]:
        raise ValueError(
            f"Sentence-topic score rows {log_scores.shape[0]} do not match "
            f"token counts {token_counts.shape[0]} at {path}."
        )
    normalized = np.asarray(log_scores, dtype=float).copy()
    valid = token_counts > 0
    normalized[valid] = normalized[valid] / token_counts[valid, None]
    normalized[~valid] = -np.inf
    return normalized


def _column_normalize_sentence_topic_soft(soft_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(soft_scores, dtype=float).copy()
    if scores.ndim != 2:
        raise ValueError(f"Expected a 2D sentence-topic matrix, got {scores.shape}.")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        return scores

    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    column_sums = scores.sum(axis=0, keepdims=True)
    normalized = np.empty_like(scores, dtype=float)
    valid_columns = column_sums.squeeze(0) > 0.0
    if np.any(valid_columns):
        normalized[:, valid_columns] = (
            scores[:, valid_columns] / column_sums[:, valid_columns]
        )
    if np.any(~valid_columns):
        normalized[:, ~valid_columns] = 1.0 / float(scores.shape[0])
    return normalized


def _write_matrix_top_sentences_sidecar(
    *,
    out_dir: Path,
    dataset: str,
    category: str,
    iteration: int,
    num_topics: int,
    condition_id: str,
    condition_fingerprint: str,
    top_k: int,
    sentences: list[str],
    log_scores: np.ndarray,
    score_artifact_path: Path,
    top_sentence_method: str,
    score_definition: str,
    results_root: Path,
    model_provenance: dict[str, Any],
    artifact_meta: dict[str, Any],
) -> Path:
    if log_scores.shape[0] != len(sentences):
        raise ValueError(
            f"Sentence-topic score rows {log_scores.shape[0]} do not match "
            f"loaded sentences {len(sentences)} at {score_artifact_path}."
        )
    top_sentences = top_sentences_by_topic_log_score_matrix(
        log_scores=log_scores,
        sentences=sentences,
        top_k=top_k,
    )
    path = out_dir / "top_sentences_loglik.json"
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="sentence_topic_inspection",
            output_kind="payload",
            artifact="top_sentences_loglik",
            dataset=dataset,
            category=category,
            iteration=int(iteration),
            num_topics=int(num_topics),
            condition_id=condition_id,
            condition_fingerprint=condition_fingerprint,
            top_k=int(top_k),
            sentence_count=int(len(sentences)),
            results_root=str(results_root),
            source_artifact_path=str(score_artifact_path),
            top_sentence_method=top_sentence_method,
            score_definition=score_definition,
            model_provenance=model_provenance,
            **artifact_meta,
        ),
        results={"topics": top_sentences},
        path=path,
    )
    return path


def _write_sentence_gaussian_top_sentences_sidecar(
    *,
    out_dir: Path,
    dataset: str,
    category: str,
    iteration: int,
    num_topics: int,
    condition_id: str,
    condition_fingerprint: str,
    top_k: int,
    sentences: list[str],
    embeddings: np.ndarray,
    source: InspectionSource,
    results_root: Path,
    model_provenance: dict[str, Any],
    artifact_meta: dict[str, Any],
) -> Path:
    params_dir = source.condition_dir / "params"
    means, cholesky, log_dets = load_gaussian_params(params_dir)
    top_sentences = top_sentences_by_topic_gaussian_loglik(
        gaussian_means=means,
        gaussian_cholesky=cholesky,
        gaussian_log_determinants=log_dets,
        sentences=sentences,
        embeddings=embeddings,
        top_k=top_k,
    )
    path = out_dir / "top_sentences_loglik.json"
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="sentence_topic_inspection",
            output_kind="payload",
            artifact="top_sentences_loglik",
            dataset=dataset,
            category=category,
            iteration=int(iteration),
            num_topics=int(num_topics),
            condition_id=condition_id,
            condition_fingerprint=condition_fingerprint,
            top_k=int(top_k),
            sentence_count=int(len(sentences)),
            results_root=str(results_root),
            source_artifact_path=str(params_dir / "table_means.pkl"),
            top_sentence_method="gaussian_loglik",
            score_definition="log p(embedding|topic)",
            model_provenance=model_provenance,
            **artifact_meta,
        ),
        results={"topics": top_sentences},
        path=path,
    )
    return path


def run_sentence_topic_inspection(
    *,
    model: str = "vmf_sentence_lda",
    dataset: str,
    categories: list[str],
    iterations: list[int],
    num_topics_list: list[int],
    source_condition_id: str | None = None,
    embedding_variant: str | None = None,
    num_components: int | None = None,
    gaussian_condition_id: str | None = None,
    gaussian_embedding_variant: str | None = None,
    gaussian_num_components: int | None = None,
    top_k: int = 5,
    encoder_model: str | None = None,
    split: str = "train",
    data_column: str = "data",
    target_column: str = "target_str",
    delimiter: str | None = " / ",
    language: str = "english",
    segmenter: str = "delimiter",
    seed: int | None = DEFAULT_RANDOM_SEED,
    gaussian_topk: bool = False,
    data_run: str = "default",
    device: str | None = None,
    encode_batch_size: int = 64,
    show_progress: bool = True,
    max_points: int = 2000,
    results_root: Path = RESULTS_ROOT,
    out_root: Path = VISUALIZATION_RESULTS_ROOT,
) -> Path:
    resolved_categories = list(_iter_unique(categories))
    resolved_iterations = [int(value) for value in _iter_unique(iterations)]
    resolved_num_topics = [int(value) for value in _iter_unique(num_topics_list)]
    resolved_device = _select_device(device)
    _validate_single_condition_selector(
        selector_name="source_condition_id",
        selector_value=source_condition_id,
        categories=resolved_categories,
        iterations=resolved_iterations,
        num_topics_list=resolved_num_topics,
    )
    _validate_single_condition_selector(
        selector_name="gaussian_condition_id",
        selector_value=gaussian_condition_id,
        categories=resolved_categories,
        iterations=resolved_iterations,
        num_topics_list=resolved_num_topics,
    )
    source_selection = _source_selection_payload(
        model=model,
        source_condition_id=source_condition_id,
        embedding_variant=embedding_variant,
        num_components=num_components,
        gaussian_condition_id=gaussian_condition_id,
        gaussian_embedding_variant=gaussian_embedding_variant,
        gaussian_num_components=gaussian_num_components,
    )

    encoders: dict[str, SentenceEncoder] = {}
    encoder_embedding_dims: dict[str, int] = {}
    sentences_cache: dict[str, list[str]] = {}
    embeddings_cache: dict[tuple[str, str], np.ndarray] = {}
    first_output_dir: Path | None = None

    for category in resolved_categories:
        for iteration in resolved_iterations:
            for num_topics in resolved_num_topics:
                source = resolve_inspection_source(
                    model=model,
                    dataset=dataset,
                    iteration=iteration,
                    num_topics=num_topics,
                    category=category,
                    split=split,
                    data_run=data_run,
                    condition_id=source_condition_id,
                    num_components=num_components,
                    embedding_variant=embedding_variant,
                    results_root=results_root,
                )
                if not source.condition_dir.exists():
                    raise FileNotFoundError(
                        f"Source condition dir not found: {source.condition_dir}"
                    )
                needs_encoder = (
                    source.supports_vmf_sphere
                    or source.supports_vmf_top_sentences
                    or source.supports_gaussian_top_sentences
                )
                resolved_encoder_model = None
                if needs_encoder:
                    resolved_encoder_model = _resolve_encoder_model_for_source(
                        requested_encoder_model=encoder_model,
                        requested_embedding_variant=source.embedding_variant,
                        exp_dir=source.condition_dir,
                    )
                resolved_source_embedding_variant = (
                    source.embedding_variant or embedding_variant
                )
                artifact_split = resolve_artifact_split_config(
                    source.primary_artifact_path,
                    split=split,
                    default_text_column=data_column,
                    default_target_column=target_column,
                )

                sentence_source_artifact_path: Path | None = None
                sentences: list[str] | None = None
                sentence_token_counts: np.ndarray | None = None
                if (
                    source.supports_sentlda_top_sentences
                    or source.supports_senclu_top_sentences
                ):
                    preprocessed_path = _baseline_preprocessed_sentences_path(
                        source,
                        split=split,
                    )
                    if preprocessed_path.exists():
                        sentences, sentence_token_counts = (
                            _load_preprocessed_sentences_with_token_counts(
                                preprocessed_path
                            )
                        )
                        sentence_source_artifact_path = preprocessed_path

                if sentences is None:
                    sentences = sentences_cache.get(category)
                    if sentences is None:
                        sentences = load_filtered_split_sentences(
                            dataset,
                            category,
                            split,
                            data_column=artifact_split.text_column,
                            target_column=artifact_split.target_column,
                            language=language,
                            delimiter=delimiter,
                            segmenter=segmenter,
                            split_csvs=artifact_split.split_csvs,
                        )
                        sentences_cache[category] = sentences
                if category != "all" and not sentences:
                    raise ValueError(
                        f"No sentences found for dataset '{dataset}' and category '{category}'. "
                        "If this dataset is unlabeled, use category 'all'."
                    )

                if (
                    sentences
                    and resolved_encoder_model is not None
                    and resolved_encoder_model not in encoders
                ):
                    encoder = build_sentence_encoder(
                        model_name=resolved_encoder_model,
                        device=resolved_device,
                    )
                    encoders[resolved_encoder_model] = encoder
                    encoder_embedding_dims[resolved_encoder_model] = int(
                        encoder.get_sentence_embedding_dimension()
                    )

                condition_id, condition_fingerprint = _build_output_condition_id(
                    model=source.model,
                    dataset=dataset,
                    data_run=artifact_split.data_run or data_run,
                    category=category,
                    iteration=iteration,
                    num_topics=num_topics,
                    split=split,
                    encoder_model=resolved_encoder_model or "",
                    gaussian_topk=gaussian_topk,
                    max_points=max_points,
                    source_condition_id=source_condition_id,
                    embedding_variant=resolved_source_embedding_variant,
                    num_components=num_components,
                    gaussian_condition_id=gaussian_condition_id,
                    gaussian_embedding_variant=gaussian_embedding_variant,
                    gaussian_num_components=gaussian_num_components,
                )
                resolved_data_run = artifact_split.data_run or data_run
                display_key = condition_id
                started_at, execution_id = _start_execution()
                out_dir = build_archive_result_dir(
                    base_root=out_root,
                    dataset=dataset,
                    data_run=resolved_data_run,
                    category=category,
                    display_key=display_key,
                    started_at=started_at,
                    execution_id=execution_id,
                )
                latest_dir = build_latest_result_dir(
                    base_root=out_root,
                    dataset=dataset,
                    data_run=resolved_data_run,
                    category=category,
                    display_key=display_key,
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                if first_output_dir is None:
                    first_output_dir = out_dir
                artifact_meta = _build_artifact_meta_context(
                    display_key=display_key,
                    started_at=started_at,
                    execution_id=execution_id,
                    archive_dir=out_dir,
                    latest_dir=latest_dir,
                )
                if sentence_source_artifact_path is not None:
                    artifact_meta["sentence_source_artifact_path"] = str(
                        sentence_source_artifact_path
                    )
                artifact_meta["source_selection"] = source_selection

                doc_topics = np.asarray(load_source_doc_topics(source), dtype=float)
                average_ll = source.average_ll
                source_artifact_path = source.primary_artifact_path
                model_provenance = load_model_provenance_for_artifact(
                    source_artifact_path,
                    model_key=source.model_provenance_key,
                )

                row: dict[str, Any] = {
                    "model": source.model,
                    "dataset": dataset,
                    "data_run": resolved_data_run,
                    "category": category,
                    "split": split,
                    "iteration": int(iteration),
                    "num_topics": int(num_topics),
                    "condition_id": condition_id,
                    "display_key": display_key,
                    "condition_fingerprint": condition_fingerprint,
                    "started_at": started_at,
                    "execution_id": execution_id,
                    "encoder_model": resolved_encoder_model,
                    "device": resolved_device,
                    "results_root": str(results_root),
                    "latest_dir": str(latest_dir),
                    "output_dir": str(out_dir),
                    "doc_topic_path": str(source.doc_topic_path),
                    "source_artifact_path": str(source_artifact_path),
                    "sentence_source_artifact_path": (
                        None
                        if sentence_source_artifact_path is None
                        else str(sentence_source_artifact_path)
                    ),
                    "top_sentence_method": source.top_sentence_method,
                    "source_condition_id": source_condition_id,
                    "embedding_variant": source.embedding_variant,
                    "num_components": (
                        None if num_components is None else int(num_components)
                    ),
                    "gaussian_condition_id": gaussian_condition_id,
                    "gaussian_embedding_variant": gaussian_embedding_variant,
                    "gaussian_num_components": (
                        None
                        if gaussian_num_components is None
                        else int(gaussian_num_components)
                    ),
                    "avg_ll_sidecar_path": None,
                    "doc_topic_tsne_sidecar_path": None,
                    "embeddings_on_sphere_3d_sidecar_path": None,
                    "kappa_per_topic_path": None,
                    "top_sentences_loglik_path": None,
                    "top_sentences_gaussian_loglik_path": None,
                    "warnings": [],
                }

                avg_ll_sidecar_path = _write_avg_ll_sidecar(
                    out_dir=out_dir,
                    dataset=dataset,
                    category=category,
                    iteration=iteration,
                    num_topics=num_topics,
                    condition_id=condition_id,
                    condition_fingerprint=condition_fingerprint,
                    average_ll=average_ll,
                    results_root=results_root,
                    source_artifact_path=source_artifact_path,
                    model_provenance=model_provenance,
                    artifact_meta=artifact_meta,
                )
                if avg_ll_sidecar_path is not None:
                    row["avg_ll_sidecar_path"] = str(avg_ll_sidecar_path)
                elif not average_ll:
                    row["warnings"].append(
                        f"avg_ll: not available for model '{source.model}'"
                    )

                try:
                    doc_topic_sidecar_path = _write_doc_topic_sidecar(
                        out_dir=out_dir,
                        dataset=dataset,
                        category=category,
                        iteration=iteration,
                        num_topics=num_topics,
                        condition_id=condition_id,
                        condition_fingerprint=condition_fingerprint,
                        seed=seed,
                        doc_topics=doc_topics,
                        results_root=results_root,
                        source_artifact_path=source_artifact_path,
                        doc_topic_path=source.doc_topic_path,
                        model_provenance=model_provenance,
                        artifact_meta=artifact_meta,
                    )
                    if doc_topic_sidecar_path is not None:
                        row["doc_topic_tsne_sidecar_path"] = str(doc_topic_sidecar_path)
                except Exception as exc:
                    row["warnings"].append(f"doc_topic_tsne: {exc}")

                gaussian_dir = None
                if gaussian_topk:
                    gaussian_dir = resolve_sentence_gaussian_dir(
                        dataset=dataset,
                        iteration=iteration,
                        num_topics=num_topics,
                        category=category,
                        data_run=data_run,
                        condition_id=gaussian_condition_id,
                        num_components=gaussian_num_components,
                        embedding_variant=gaussian_embedding_variant,
                        results_root=results_root,
                    )

                if sentences:
                    embeddings: np.ndarray | None = None
                    if needs_encoder:
                        if resolved_encoder_model is None:
                            raise RuntimeError(
                                "Encoder model was not resolved for sentence scoring."
                            )
                        encoder = encoders.get(resolved_encoder_model)
                        if encoder is None:
                            raise RuntimeError(
                                "Encoder was not initialized as expected."
                            )
                        encoder_embedding_dim = encoder_embedding_dims.get(
                            resolved_encoder_model
                        )
                        if encoder_embedding_dim is None:
                            encoder_embedding_dim = int(
                                encoder.get_sentence_embedding_dimension()
                            )
                            encoder_embedding_dims[resolved_encoder_model] = (
                                encoder_embedding_dim
                            )
                        if source.supports_vmf_top_sentences:
                            topic_dim = int(
                                load_topic_means(source.condition_dir).shape[1]
                            )
                        elif source.supports_gaussian_top_sentences:
                            topic_dim = int(
                                load_gaussian_params(source.condition_dir / "params")[
                                    0
                                ].shape[1]
                            )
                        else:
                            topic_dim = encoder_embedding_dim
                        if encoder_embedding_dim != topic_dim:
                            raise ValueError(
                                "Encoder embedding dimension mismatch: "
                                f"encoder='{resolved_encoder_model}' -> {encoder_embedding_dim}, "
                                f"topic parameters -> {topic_dim} "
                                f"(dataset='{dataset}', category='{category}', "
                                f"iteration={iteration}, num_topics={num_topics})."
                            )

                        embedding_cache_key = (category, resolved_encoder_model)
                        embeddings = embeddings_cache.get(embedding_cache_key)
                        if embeddings is None:
                            embeddings = encode_sentences(
                                encoder,
                                sentences,
                                batch_size=encode_batch_size,
                                show_progress=show_progress,
                            )
                            embeddings_cache[embedding_cache_key] = embeddings
                        elif embeddings.shape[1] != topic_dim:
                            raise ValueError(
                                f"Cached embedding dim {embeddings.shape[1]} does not match topic dim {topic_dim}."
                            )

                    if source.supports_vmf_top_sentences:
                        if embeddings is None:
                            raise RuntimeError(
                                "vMF sentence scoring requires embeddings."
                            )
                        topic_means = load_topic_means(source.condition_dir)
                        (
                            kappa_per_topic,
                            mixture_weights,
                            component_means,
                            _topic_counts,
                            _alpha,
                        ) = load_vmf_params(source.condition_dir)
                        row.update(
                            _write_sentence_payloads(
                                out_dir=out_dir,
                                dataset=dataset,
                                category=category,
                                iteration=iteration,
                                num_topics=num_topics,
                                condition_id=condition_id,
                                condition_fingerprint=condition_fingerprint,
                                top_k=top_k,
                                sentences=sentences,
                                embeddings=embeddings,
                                topic_means=topic_means,
                                kappa_per_topic=kappa_per_topic,
                                mixture_weights=mixture_weights,
                                component_means=component_means,
                                results_root=results_root,
                                source_artifact_path=source_artifact_path,
                                model_provenance=model_provenance,
                                gaussian_topk=gaussian_topk,
                                gaussian_dir=gaussian_dir,
                                artifact_meta=artifact_meta,
                            )
                        )

                        try:
                            sphere_sidecar_path = _write_sphere_sidecar(
                                out_dir=out_dir,
                                dataset=dataset,
                                category=category,
                                iteration=iteration,
                                num_topics=num_topics,
                                condition_id=condition_id,
                                condition_fingerprint=condition_fingerprint,
                                seed=seed,
                                embeddings=embeddings,
                                topic_means=topic_means,
                                kappa_per_topic=kappa_per_topic,
                                mixture_weights=mixture_weights,
                                component_means=component_means,
                                max_points=max_points,
                                results_root=results_root,
                                source_artifact_path=source_artifact_path,
                                model_provenance=model_provenance,
                                artifact_meta=artifact_meta,
                            )
                            row["embeddings_on_sphere_3d_sidecar_path"] = str(
                                sphere_sidecar_path
                            )
                        except Exception as exc:
                            row["warnings"].append(f"embeddings_on_sphere_3d: {exc}")

                    elif source.supports_gaussian_top_sentences:
                        if embeddings is None:
                            raise RuntimeError(
                                "Sentence Gaussian LDA scoring requires embeddings."
                            )
                        top_path = _write_sentence_gaussian_top_sentences_sidecar(
                            out_dir=out_dir,
                            dataset=dataset,
                            category=category,
                            iteration=iteration,
                            num_topics=num_topics,
                            condition_id=condition_id,
                            condition_fingerprint=condition_fingerprint,
                            top_k=top_k,
                            sentences=sentences,
                            embeddings=embeddings,
                            source=source,
                            results_root=results_root,
                            model_provenance=model_provenance,
                            artifact_meta=artifact_meta,
                        )
                        row["top_sentences_loglik_path"] = str(top_path)

                    elif (
                        source.supports_sentlda_top_sentences
                        or source.supports_senclu_top_sentences
                    ):
                        score_path, method, score_definition = (
                            _matrix_sentence_score_path(source, split=split)
                        )
                        raw_scores = load_artifact_pickle(score_path)
                        score_matrix = _flatten_grouped_topic_arrays(
                            raw_scores,
                            path=score_path,
                        )
                        if method in {
                            "sentlda_collapsed_conditional_logprob",
                        }:
                            score_matrix = np.log(np.maximum(score_matrix, 1e-12))
                        elif method == "senclu_sentence_given_topic":
                            score_matrix = _column_normalize_sentence_topic_soft(
                                score_matrix
                            )
                            artifact_meta = {
                                **artifact_meta,
                                "score_normalization": "column_over_sentences",
                                "source_score_definition": (
                                    "p(topic|sentence, document)"
                                ),
                            }
                        elif method == "sentlda_token_loglik":
                            if sentence_token_counts is None:
                                raise ValueError(
                                    "sentLDA token-mean log-likelihood inspection "
                                    "requires preprocessed sentence token counts."
                                )
                            score_matrix = _normalize_sentlda_loglik_by_token_count(
                                score_matrix,
                                sentence_token_counts,
                                path=score_path,
                            )
                            method = "sentlda_token_mean_loglik"
                            score_definition = "mean_token log p(token|topic, rest)"
                            artifact_meta = {
                                **artifact_meta,
                                "score_normalization": "per_token",
                                "source_score_definition": (
                                    "log p(sentence|topic, rest)"
                                ),
                            }
                        top_path = _write_matrix_top_sentences_sidecar(
                            out_dir=out_dir,
                            dataset=dataset,
                            category=category,
                            iteration=iteration,
                            num_topics=num_topics,
                            condition_id=condition_id,
                            condition_fingerprint=condition_fingerprint,
                            top_k=top_k,
                            sentences=sentences,
                            log_scores=score_matrix,
                            score_artifact_path=score_path,
                            top_sentence_method=method,
                            score_definition=score_definition,
                            results_root=results_root,
                            model_provenance=model_provenance,
                            artifact_meta=artifact_meta,
                        )
                        row["top_sentences_loglik_path"] = str(top_path)
                        row["top_sentence_method"] = method

                pointer_artifacts: dict[str, str] = {}
                if row["avg_ll_sidecar_path"] is not None:
                    pointer_artifacts["avg_ll_png"] = "avg_ll.png"
                    pointer_artifacts["avg_ll_json"] = "avg_ll.json"
                if row["doc_topic_tsne_sidecar_path"] is not None:
                    pointer_artifacts["doc_topic_tsne_png"] = "doc_topic_tsne.png"
                    pointer_artifacts["doc_topic_tsne_json"] = "doc_topic_tsne.json"
                if row["embeddings_on_sphere_3d_sidecar_path"] is not None:
                    pointer_artifacts["embeddings_on_sphere_3d_png"] = (
                        "embeddings_on_sphere_3d.png"
                    )
                    pointer_artifacts["embeddings_on_sphere_3d_json"] = (
                        "embeddings_on_sphere_3d.json"
                    )
                if row["kappa_per_topic_path"] is not None:
                    pointer_artifacts["kappa_per_topic_json"] = "kappa_per_topic.json"
                if row["top_sentences_loglik_path"] is not None:
                    pointer_artifacts["top_sentences_loglik_json"] = (
                        "top_sentences_loglik.json"
                    )
                if row["top_sentences_gaussian_loglik_path"] is not None:
                    pointer_artifacts["top_sentences_gaussian_loglik_json"] = (
                        "top_sentences_gaussian_loglik.json"
                    )
                write_latest_result_pointer(
                    base_root=out_root,
                    task="sentence_topic_inspection",
                    dataset=dataset,
                    data_run=resolved_data_run,
                    category=category,
                    display_key=display_key,
                    archive_dir=out_dir,
                    started_at=started_at,
                    execution_id=execution_id,
                    condition_fingerprint=condition_fingerprint,
                    artifacts=pointer_artifacts,
                )
    return first_output_dir or (out_root / dataset / data_run)


run_sentence_topic_inspection_report = run_sentence_topic_inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect sentence-topic structure and visualize trained vMF Sentence LDA runs."
    )
    parser.add_argument("--model", default="vmf_sentence_lda")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--category", nargs="+", required=True)
    parser.add_argument("--iteration", type=int, nargs="+", default=[0])
    parser.add_argument("--num-topics", type=int, nargs="+", required=True)
    parser.add_argument("--condition-id", dest="source_condition_id", default=None)
    parser.add_argument("--embedding-variant", default=None)
    parser.add_argument("--num-components", type=int, default=None)
    parser.add_argument("--gaussian-condition-id", default=None)
    parser.add_argument("--gaussian-embedding-variant", default=None)
    parser.add_argument("--gaussian-num-components", type=int, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--data-column", default="data")
    parser.add_argument("--target-column", default="target_str")
    parser.add_argument("--delimiter", default=" / ")
    parser.add_argument("--language", default="english")
    parser.add_argument("--segmenter", default="delimiter")
    parser.add_argument("--data-run", default="default")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--gaussian_topk", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--out-root", type=Path, default=VISUALIZATION_RESULTS_ROOT)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_sentence_topic_inspection(
        model=args.model,
        dataset=args.dataset,
        categories=args.category,
        iterations=args.iteration,
        num_topics_list=args.num_topics,
        source_condition_id=args.source_condition_id,
        embedding_variant=args.embedding_variant,
        num_components=args.num_components,
        gaussian_condition_id=args.gaussian_condition_id,
        gaussian_embedding_variant=args.gaussian_embedding_variant,
        gaussian_num_components=args.gaussian_num_components,
        top_k=args.topk,
        encoder_model=args.encoder,
        split=args.split,
        data_column=args.data_column,
        target_column=args.target_column,
        delimiter=args.delimiter,
        language=args.language,
        segmenter=args.segmenter,
        data_run=args.data_run,
        seed=args.seed,
        gaussian_topk=args.gaussian_topk,
        device=args.device,
        encode_batch_size=args.encode_batch_size,
        show_progress=not args.no_progress,
        max_points=args.max_points,
        results_root=args.results_root,
        out_root=args.out_root,
    )
    print(f"[ok] {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import ive, logsumexp

from src.core.artifacts import (
    METADATA_FILENAME,
    VMF_PARAMS_FILENAME,
    load_artifact_pickle,
)
from src.core.paths import (
    EXPERIMENT_RESULTS_ROOT,
    RESULTS_ROOT,
    build_archive_result_dir,
    build_latest_result_dir,
    resolve_vmf_experiment_dir,
    write_latest_result_pointer,
)
from src.core.result_identity import build_condition_id, build_execution_id
from src.data.splits import load_filtered_split_texts
from src.data.text_processing import split_sentences
from src.evaluation.model_provenance import load_model_provenance
from src.evaluation.reporting import (
    read_json,
    write_csv_rows,
    write_tabular_report_json,
)
from src.evaluation.source_data import resolve_artifact_split_config
from src.utils.encoder import SentenceEncoder
from src.utils.encoder_profiles import (
    default_encoder_model_for_embedding_variant,
    embedding_variant_base,
    encoder_model_alias,
)
from src.utils.logging import get_logger

DEFAULT_OUT_ROOT = RESULTS_ROOT / "topic_count_analysis"
DEFAULT_ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"
SUPPORTED_EVAL_MODES = {"predictive_soft_theta", "metrics"}
logger = get_logger(__name__)


def _start_execution() -> tuple[str, str]:
    started_at = datetime.now(UTC).isoformat()
    return started_at, build_execution_id(prefix="exec", started_at=started_at)


def _metrics_path(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    results_root: Path = EXPERIMENT_RESULTS_ROOT,
) -> Path:
    return (
        resolve_vmf_experiment_dir(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            run_name=data_run,
            condition_id=condition_id,
            num_components=num_components,
            embedding_variant=embedding_variant,
            dataset_root=results_root / dataset,
        )
        / "metrics.json"
    )


def _experiment_dir(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    results_root: Path = EXPERIMENT_RESULTS_ROOT,
) -> Path:
    return resolve_vmf_experiment_dir(
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        run_name=data_run,
        condition_id=condition_id,
        num_components=num_components,
        embedding_variant=embedding_variant,
        dataset_root=results_root / dataset,
    )


def _build_output_condition_id(
    *,
    dataset: str,
    data_run: str,
    category: str,
    iterations: Sequence[int],
    topics: Sequence[int],
    source_condition_id: str | None,
    num_components: int | None,
    embedding_variant: str | None,
    split: str,
    eval_mode: str,
    strict: bool,
) -> tuple[str, str]:
    return build_condition_id(
        iteration=int(min(iterations)),
        num_topics=int(min(topics)),
        fingerprint_payload={
            "task": "topic_count_diagnostics",
            "dataset": dataset,
            "data_run": data_run,
            "category": category,
            "iterations": [int(value) for value in iterations],
            "topics": [int(value) for value in topics],
            "source_condition_id": source_condition_id,
            "num_components": None if num_components is None else int(num_components),
            "embedding_variant": embedding_variant,
            "split": split,
            "eval_mode": eval_mode,
            "strict": bool(strict),
        },
        extra_labels=["perplexity", split],
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_metrics_provenance(metrics_path: Path) -> dict[str, Any]:
    return load_model_provenance(
        metrics_path.parent,
        model_key="vmf_sentence_lda",
    )


def _metadata_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
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


def _resolve_encoder_model(
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
            or DEFAULT_ENCODER_MODEL
        )
    else:
        resolved_encoder_model = DEFAULT_ENCODER_MODEL

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


def build_sentence_encoder(*, model_name: str, device: str | None) -> SentenceEncoder:
    return SentenceEncoder(
        model_name,
        device=device,
        strip_terminal_normalize=False,
    )


def _load_document_sentences_from_preprocessed(path: Path) -> list[list[str]]:
    payload = load_artifact_pickle(path)
    documents = getattr(payload, "documents", payload)
    if not isinstance(documents, (list, tuple)):
        raise ValueError(
            f"Expected a preprocessed corpus or list of documents at {path}, "
            f"got {type(payload)}."
        )

    corpus: list[list[str]] = []
    for doc_index, document in enumerate(documents):
        raw_sentences = getattr(document, "sentences_raw", None)
        if raw_sentences is None:
            if isinstance(document, (list, tuple)):
                raw_sentences = document
            else:
                raise ValueError(
                    f"Expected document {doc_index} at {path} to expose sentences_raw."
                )
        corpus.append(
            [str(sentence) for sentence in raw_sentences if str(sentence).strip()]
        )
    return corpus


def _load_document_sentences_from_csv(
    *,
    dataset: str,
    category: str,
    split: str,
    data_column: str,
    target_column: str,
    language: str,
    delimiter: str | None,
    segmenter: str,
    split_csvs: tuple[str, ...] | None,
) -> list[list[str]]:
    texts = load_filtered_split_texts(
        dataset,
        category,
        split,
        data_column=data_column,
        target_column=target_column,
        split_csvs=split_csvs,
    )
    return [
        [
            sentence
            for sentence in split_sentences(
                text,
                language=language,
                delimiter=delimiter,
                segmenter=segmenter,
            )
            if sentence.strip()
        ]
        for text in texts
    ]


def _load_evaluation_corpus(
    *,
    exp_dir: Path,
    dataset: str,
    category: str,
    split: str,
    data_column: str,
    target_column: str,
    language: str,
    delimiter: str | None,
    segmenter: str,
    strict: bool,
) -> tuple[list[list[str]], Path | None]:
    preprocessed_path = exp_dir / f"{split}_preprocessed.pkl"
    if preprocessed_path.exists():
        return (
            _load_document_sentences_from_preprocessed(preprocessed_path),
            preprocessed_path,
        )
    if strict:
        raise FileNotFoundError(
            f"Strict perplexity requires preprocessed corpus artifact: {preprocessed_path}"
        )

    artifact_split = resolve_artifact_split_config(
        exp_dir / f"doc_topic_{split}_soft.pkl",
        split=split,
        default_text_column=data_column,
        default_target_column=target_column,
    )
    return (
        _load_document_sentences_from_csv(
            dataset=dataset,
            category=category,
            split=split,
            data_column=artifact_split.text_column,
            target_column=artifact_split.target_column,
            language=language,
            delimiter=delimiter,
            segmenter=segmenter,
            split_csvs=artifact_split.split_csvs,
        ),
        None,
    )


def _encode_documents_raw(
    *,
    encoder: SentenceEncoder,
    documents: Sequence[Sequence[str]],
    batch_size: int,
    show_progress: bool,
) -> list[np.ndarray]:
    lengths = [len(document) for document in documents]
    flat_sentences = [
        sentence
        for document in documents
        for sentence in document
        if str(sentence).strip()
    ]
    if not flat_sentences:
        return [
            np.zeros((0, encoder.get_sentence_embedding_dimension())) for _ in documents
        ]

    encoded = np.asarray(
        encoder.encode(
            flat_sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        ),
        dtype=np.float64,
    )
    if encoded.ndim == 1:
        encoded = encoded.reshape(1, -1)

    encoded_docs: list[np.ndarray] = []
    cursor = 0
    embedding_dim = encoded.shape[1]
    for length in lengths:
        if length <= 0:
            encoded_docs.append(np.zeros((0, embedding_dim), dtype=np.float64))
            continue
        next_cursor = cursor + length
        encoded_docs.append(encoded[cursor:next_cursor])
        cursor = next_cursor
    return encoded_docs


def _load_vmf_params(
    exp_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    topic_means = np.asarray(
        load_artifact_pickle(exp_dir / "topic_means.pkl"), dtype=np.float64
    )
    kappa_per_topic = np.asarray(
        load_artifact_pickle(exp_dir / "kappa_per_topic.pkl"),
        dtype=np.float64,
    )
    mixture_weights = np.asarray(
        load_artifact_pickle(exp_dir / "mixture_weights.pkl"),
        dtype=np.float64,
    )
    component_means = np.asarray(
        load_artifact_pickle(exp_dir / "component_means.pkl"),
        dtype=np.float64,
    )
    return topic_means, kappa_per_topic, mixture_weights, component_means


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def _transform_and_normalize_documents(
    *,
    exp_dir: Path,
    raw_encoded_docs: Sequence[np.ndarray],
    params: dict[str, Any],
    strict: bool,
) -> list[np.ndarray]:
    mode = str(params.get("pre_normalize_transform") or "none").strip().lower()
    if mode == "meancenter":
        mode = "mean_center"
    if mode == "whiten":
        mode = "whitening"
    if mode not in {"none", "mean_center", "whitening"}:
        raise ValueError(f"Unsupported pre_normalize_transform: {mode}")

    mean: np.ndarray | None = None
    whitening: np.ndarray | None = None
    if mode in {"mean_center", "whitening"}:
        mean_path = exp_dir / "embedding_transform_mean.pkl"
        if not mean_path.exists():
            if strict:
                raise FileNotFoundError(
                    f"Embedding transform mean not found: {mean_path}"
                )
        else:
            mean = np.asarray(load_artifact_pickle(mean_path), dtype=np.float64)
    if mode == "whitening":
        whitening_path = exp_dir / "embedding_transform_whitening_matrix.pkl"
        if not whitening_path.exists():
            if strict:
                raise FileNotFoundError(
                    f"Embedding whitening matrix not found: {whitening_path}"
                )
        else:
            whitening = np.asarray(
                load_artifact_pickle(whitening_path), dtype=np.float64
            )

    encoded_docs: list[np.ndarray] = []
    for raw_doc in raw_encoded_docs:
        arr = np.asarray(raw_doc, dtype=np.float64)
        if arr.size == 0:
            encoded_docs.append(arr.reshape(0, arr.shape[-1] if arr.ndim == 2 else 0))
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if mean is not None:
            arr = arr - mean
        if whitening is not None:
            arr = arr @ whitening
        encoded_docs.append(_normalize_rows(arr))
    return encoded_docs


def _log_vmf_normalization_const(
    *, kappa_per_topic: np.ndarray, embedding_dim: int
) -> np.ndarray:
    kappa_safe = np.clip(np.asarray(kappa_per_topic, dtype=np.float64), 1e-12, None)
    v = float(embedding_dim) / 2.0 - 1.0
    ive_val = np.maximum(ive(v, kappa_safe), 1e-300)
    log_iv = np.log(ive_val) + kappa_safe
    return (
        v * np.log(kappa_safe)
        - (float(embedding_dim) / 2.0) * math.log(2.0 * math.pi)
        - log_iv
    )


def _vmf_log_density_matrix(
    *,
    embeddings: np.ndarray,
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, topic_means.shape[0]), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    embedding_dim = arr.shape[1]
    log_c = _log_vmf_normalization_const(
        kappa_per_topic=kappa_per_topic,
        embedding_dim=embedding_dim,
    )

    if component_means.ndim == 3 and component_means.shape[1] > 1:
        scaled_component_means = kappa_per_topic[:, None, None] * component_means
        scores = np.einsum("nd,kcd->nkc", arr, scaled_component_means, optimize=True)
        log_comp = scores + np.log(mixture_weights + 1e-12)[None, :, :]
        return log_c[None, :] + logsumexp(log_comp, axis=2)

    scaled_topic_means = kappa_per_topic[:, None] * topic_means
    return arr @ scaled_topic_means.T + log_c[None, :]


def _safe_exp(value: float) -> tuple[float, bool, bool]:
    try:
        result = math.exp(value)
    except OverflowError:
        return float("inf"), False, True
    return result, bool(result == 0.0 and value < 0.0), bool(math.isinf(result))


def _compute_predictive_perplexity(
    *,
    exp_dir: Path,
    theta_path: Path,
    documents: Sequence[Sequence[str]],
    raw_encoded_docs: Sequence[np.ndarray],
    strict: bool,
) -> dict[str, Any]:
    params = _metadata_mapping(exp_dir / VMF_PARAMS_FILENAME)
    theta = np.asarray(load_artifact_pickle(theta_path), dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError(f"Expected 2D theta at {theta_path}, got shape {theta.shape}.")
    if theta.shape[0] != len(documents):
        raise ValueError(
            f"Theta rows {theta.shape[0]} do not match document count {len(documents)} "
            f"for {theta_path}."
        )
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    theta = theta / row_sums

    topic_means, kappa_per_topic, mixture_weights, component_means = _load_vmf_params(
        exp_dir
    )
    if theta.shape[1] != topic_means.shape[0]:
        raise ValueError(
            f"Theta columns {theta.shape[1]} do not match topics {topic_means.shape[0]}."
        )

    encoded_docs = _transform_and_normalize_documents(
        exp_dir=exp_dir,
        raw_encoded_docs=raw_encoded_docs,
        params=params,
        strict=strict,
    )

    total_ll = 0.0
    total_sentences = 0
    nonempty_docs = 0
    for doc_index, encoded_doc in enumerate(encoded_docs):
        if encoded_doc.size == 0:
            continue
        log_lik = _vmf_log_density_matrix(
            embeddings=encoded_doc,
            topic_means=topic_means,
            kappa_per_topic=kappa_per_topic,
            mixture_weights=mixture_weights,
            component_means=component_means,
        )
        log_theta = np.log(theta[doc_index] + 1e-12)
        sentence_ll = logsumexp(log_lik + log_theta[None, :], axis=1)
        total_ll += float(sentence_ll.sum())
        total_sentences += int(sentence_ll.shape[0])
        nonempty_docs += 1

    if total_sentences <= 0:
        avg_ll = float("nan")
        log_perplexity = float("nan")
        perplexity = float("nan")
        underflow = False
        overflow = False
    else:
        avg_ll = total_ll / float(total_sentences)
        log_perplexity = -avg_ll
        perplexity, underflow, overflow = _safe_exp(log_perplexity)

    return {
        "avg_log_likelihood": avg_ll,
        "log_perplexity": log_perplexity,
        "perplexity": perplexity,
        "perplexity_underflow": underflow,
        "perplexity_overflow": overflow,
        "num_documents": int(len(documents)),
        "num_nonempty_documents": int(nonempty_docs),
        "num_sentences": int(total_sentences),
    }


def run_topic_count_perplexity_analysis(
    *,
    dataset: str,
    iterations: Sequence[int],
    topics: Sequence[int],
    categories: Sequence[str],
    data_runs: Sequence[str] = ("default",),
    source_condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    split: str = "test",
    eval_mode: str = "predictive_soft_theta",
    strict: bool = True,
    encoder_model: str | None = None,
    device: str | None = None,
    encode_batch_size: int = 64,
    show_progress: bool = True,
    data_column: str = "data",
    target_column: str = "target_str",
    language: str = "english",
    delimiter: str | None = " / ",
    segmenter: str = "delimiter",
    results_root: Path = EXPERIMENT_RESULTS_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be one of {'train', 'test'}.")
    normalized_eval_mode = str(eval_mode).strip().lower().replace("-", "_")
    if normalized_eval_mode not in SUPPORTED_EVAL_MODES:
        raise ValueError(
            f"eval_mode must be one of {sorted(SUPPORTED_EVAL_MODES)}, "
            f"got: {eval_mode}"
        )
    condition_count = len(data_runs) * len(categories) * len(iterations) * len(topics)
    if source_condition_id is not None and condition_count > 1:
        raise ValueError(
            "condition_id can only be used when selecting a single data_run, "
            "category, iteration, and topic count."
        )
    rows: list[dict[str, Any]] = []
    summary_provenance: list[dict[str, Any]] = []
    default_data_run = str(data_runs[0]) if data_runs else "default"
    default_category = str(categories[0]) if categories else "all"
    default_iteration = int(iterations[0]) if iterations else 0
    default_num_topics = int(topics[0]) if topics else 0
    summary_csv_path = (
        out_root
        / dataset
        / default_data_run
        / default_category
        / f"it{default_iteration}__k{default_num_topics}__perplexity_{split}__pending"
        / "perplexity_summary.csv"
    )
    encoders: dict[str, SentenceEncoder] = {}
    corpus_cache: dict[
        tuple[str, str, str, str], tuple[list[list[str]], Path | None]
    ] = {}
    raw_embedding_cache: dict[tuple[str, str, str, str, str], list[np.ndarray]] = {}
    for data_run in data_runs:
        for category in categories:
            for iteration in iterations:
                for num_topics in topics:
                    exp_dir = _experiment_dir(
                        dataset=dataset,
                        iteration=int(iteration),
                        num_topics=int(num_topics),
                        category=category,
                        data_run=data_run,
                        condition_id=source_condition_id,
                        num_components=num_components,
                        embedding_variant=embedding_variant,
                        results_root=results_root,
                    )
                    metrics_path = exp_dir / "metrics.json"
                    if not metrics_path.exists():
                        logger.warning(f"[skip] missing {metrics_path}")
                        continue
                    metrics = read_json(metrics_path)
                    theta_path = exp_dir / f"doc_topic_{split}_soft.pkl"
                    preprocessed_path: Path | None = None
                    eval_stats: dict[str, Any]
                    resolved_encoder_model: str | None = None
                    if normalized_eval_mode == "metrics":
                        metrics_perplexity = _safe_float(metrics.get("perplexity"))
                        eval_stats = {
                            "avg_log_likelihood": _safe_float(
                                metrics.get("avg_log_likelihood")
                            ),
                            "log_perplexity": (
                                None
                                if metrics_perplexity is None
                                else (
                                    float("-inf")
                                    if metrics_perplexity == 0.0
                                    else math.log(metrics_perplexity)
                                )
                            ),
                            "perplexity": metrics_perplexity,
                            "perplexity_underflow": False,
                            "perplexity_overflow": False,
                            "num_documents": None,
                            "num_nonempty_documents": None,
                            "num_sentences": None,
                        }
                    else:
                        if not theta_path.exists():
                            raise FileNotFoundError(
                                f"Soft doc-topic artifact not found: {theta_path}"
                            )
                        resolved_encoder_model = _resolve_encoder_model(
                            requested_encoder_model=encoder_model,
                            requested_embedding_variant=embedding_variant,
                            exp_dir=exp_dir,
                        )
                        corpus_key = (str(data_run), str(category), split, str(exp_dir))
                        corpus_pair = corpus_cache.get(corpus_key)
                        if corpus_pair is None:
                            corpus_pair = _load_evaluation_corpus(
                                exp_dir=exp_dir,
                                dataset=dataset,
                                category=category,
                                split=split,
                                data_column=data_column,
                                target_column=target_column,
                                language=language,
                                delimiter=delimiter,
                                segmenter=segmenter,
                                strict=strict,
                            )
                            corpus_cache[corpus_key] = corpus_pair
                        documents, preprocessed_path = corpus_pair
                        encoder = encoders.get(resolved_encoder_model)
                        if encoder is None:
                            encoder = build_sentence_encoder(
                                model_name=resolved_encoder_model,
                                device=device,
                            )
                            encoders[resolved_encoder_model] = encoder
                        embedding_key = (
                            str(data_run),
                            str(category),
                            split,
                            str(preprocessed_path or exp_dir),
                            resolved_encoder_model,
                        )
                        raw_encoded_docs = raw_embedding_cache.get(embedding_key)
                        if raw_encoded_docs is None:
                            raw_encoded_docs = _encode_documents_raw(
                                encoder=encoder,
                                documents=documents,
                                batch_size=encode_batch_size,
                                show_progress=show_progress,
                            )
                            raw_embedding_cache[embedding_key] = raw_encoded_docs
                        eval_stats = _compute_predictive_perplexity(
                            exp_dir=exp_dir,
                            theta_path=theta_path,
                            documents=documents,
                            raw_encoded_docs=raw_encoded_docs,
                            strict=strict,
                        )
                    summary_provenance.append(
                        {
                            "model": "vmf_sentence_lda",
                            "data_run": data_run,
                            "category": category,
                            "iteration": int(iteration),
                            "num_topics": int(num_topics),
                            "source_condition_id": source_condition_id,
                            "num_components": (
                                None if num_components is None else int(num_components)
                            ),
                            "embedding_variant": embedding_variant,
                            "eval_split": split,
                            "eval_mode": normalized_eval_mode,
                            "model_provenance": _resolve_metrics_provenance(
                                metrics_path
                            ),
                        }
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "data_run": data_run,
                            "category": category,
                            "iteration": int(iteration),
                            "num_topics": int(num_topics),
                            "source_condition_id": source_condition_id,
                            "num_components": (
                                None if num_components is None else int(num_components)
                            ),
                            "embedding_variant": embedding_variant,
                            "eval_split": split,
                            "eval_mode": normalized_eval_mode,
                            "strict": bool(strict),
                            "avg_log_likelihood": _safe_float(
                                eval_stats.get("avg_log_likelihood")
                            ),
                            "log_perplexity": _safe_float(
                                eval_stats.get("log_perplexity")
                            ),
                            "perplexity": _safe_float(eval_stats.get("perplexity")),
                            "perplexity_underflow": bool(
                                eval_stats.get("perplexity_underflow", False)
                            ),
                            "perplexity_overflow": bool(
                                eval_stats.get("perplexity_overflow", False)
                            ),
                            "num_documents": eval_stats.get("num_documents"),
                            "num_nonempty_documents": eval_stats.get(
                                "num_nonempty_documents"
                            ),
                            "num_sentences": eval_stats.get("num_sentences"),
                            "train_metrics_avg_log_likelihood": _safe_float(
                                metrics.get("avg_log_likelihood")
                            ),
                            "train_metrics_perplexity": _safe_float(
                                metrics.get("perplexity")
                            ),
                            "elapsed_sec": _safe_float(metrics.get("elapsed_sec")),
                            "metrics_path": str(metrics_path),
                            "condition_dir": str(exp_dir),
                            "theta_path": (
                                None
                                if normalized_eval_mode == "metrics"
                                else str(theta_path)
                            ),
                            "preprocessed_path": (
                                None
                                if preprocessed_path is None
                                else str(preprocessed_path)
                            ),
                            "encoder_model": resolved_encoder_model,
                        }
                    )

    fieldnames = [
        "dataset",
        "data_run",
        "category",
        "iteration",
        "num_topics",
        "source_condition_id",
        "num_components",
        "embedding_variant",
        "eval_split",
        "eval_mode",
        "strict",
        "avg_log_likelihood",
        "log_perplexity",
        "perplexity",
        "perplexity_underflow",
        "perplexity_overflow",
        "num_documents",
        "num_nonempty_documents",
        "num_sentences",
        "train_metrics_avg_log_likelihood",
        "train_metrics_perplexity",
        "elapsed_sec",
        "metrics_path",
        "condition_dir",
        "theta_path",
        "preprocessed_path",
        "encoder_model",
    ]
    if rows:
        first_written_path: Path | None = None
        for data_run in data_runs:
            for category in categories:
                category_rows = [
                    row
                    for row in rows
                    if str(row["data_run"]) == str(data_run)
                    and str(row["category"]) == str(category)
                ]
                if not category_rows:
                    continue
                category_provenance = [
                    item
                    for item in summary_provenance
                    if str(item["data_run"]) == str(data_run)
                    and str(item["category"]) == str(category)
                ]
                output_condition_id, condition_fingerprint = _build_output_condition_id(
                    dataset=dataset,
                    data_run=data_run,
                    category=category,
                    iterations=iterations,
                    topics=topics,
                    source_condition_id=source_condition_id,
                    num_components=num_components,
                    embedding_variant=embedding_variant,
                    split=split,
                    eval_mode=normalized_eval_mode,
                    strict=strict,
                )
                display_key = output_condition_id
                started_at, execution_id = _start_execution()
                out_dir = build_archive_result_dir(
                    base_root=out_root,
                    dataset=dataset,
                    data_run=data_run,
                    category=category,
                    display_key=display_key,
                    started_at=started_at,
                    execution_id=execution_id,
                )
                latest_dir = build_latest_result_dir(
                    base_root=out_root,
                    dataset=dataset,
                    data_run=data_run,
                    category=category,
                    display_key=display_key,
                )
                summary_csv_path = out_dir / "perplexity_summary.csv"
                write_csv_rows(
                    fieldnames=fieldnames,
                    rows=category_rows,
                    path=summary_csv_path,
                )
                write_tabular_report_json(
                    meta={
                        "task": "topic_count_diagnostics",
                        "dataset": dataset,
                        "data_run": data_run,
                        "category": category,
                        "condition_id": output_condition_id,
                        "display_key": display_key,
                        "condition_fingerprint": condition_fingerprint,
                        "started_at": started_at,
                        "execution_id": execution_id,
                        "archive_dir": str(out_dir),
                        "latest_dir": str(latest_dir),
                        "iterations": [int(i) for i in iterations],
                        "topics": [int(k) for k in topics],
                        "source_condition_id": source_condition_id,
                        "num_components": (
                            None if num_components is None else int(num_components)
                        ),
                        "embedding_variant": embedding_variant,
                        "eval_split": split,
                        "eval_mode": normalized_eval_mode,
                        "strict": bool(strict),
                        "encoder_model": encoder_model,
                        "device": device,
                        "encode_batch_size": int(encode_batch_size),
                        "data_column": data_column,
                        "target_column": target_column,
                        "language": language,
                        "delimiter": delimiter,
                        "segmenter": segmenter,
                        "categories": list(categories),
                        "data_runs": list(data_runs),
                        "results_root": str(results_root),
                        "model_provenance": category_provenance,
                    },
                    columns=fieldnames,
                    rows=category_rows,
                    path=out_dir / "perplexity_summary.json",
                )
                write_latest_result_pointer(
                    base_root=out_root,
                    task="topic_count_diagnostics",
                    dataset=dataset,
                    data_run=data_run,
                    category=category,
                    display_key=display_key,
                    archive_dir=out_dir,
                    started_at=started_at,
                    execution_id=execution_id,
                    condition_fingerprint=condition_fingerprint,
                    artifacts={
                        "csv": summary_csv_path.name,
                        "json": "perplexity_summary.json",
                    },
                )
                logger.info(f"perplexity summary CSV written to {summary_csv_path}")
                if first_written_path is None:
                    first_written_path = summary_csv_path
        if first_written_path is not None:
            return first_written_path
    return summary_csv_path


run_topic_count_diagnostics = run_topic_count_perplexity_analysis

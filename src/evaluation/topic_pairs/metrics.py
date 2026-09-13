"""Topic-pair metrics runner (task ``topic_pair_metrics``).

One condition per (model, num_topics, category) aggregates all iterations and
is written under ``results/topic_analysis/topic_pairs/`` with the
archive/latest layout of the other evaluation roots. A further condition per
(num_topics, category), stored under the pseudo-model ``cross_model``, holds
the expected co-assignment mass between every pair of models of the same
iteration, so that "topic j of model A is merged into topic b of model B" can
be decided downstream from the same posteriors.

Per category the reference corpus is loaded once, encoded once (cached) and
labelled once; per run the frozen model parameters give sentence log
likelihoods, the collapsed fold-in of the representative-word protocol turns
them into posteriors, and :mod:`.numerics` does the rest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.baselines.models.sentence_gaussian_helpers import (
    load_sentence_gaussianlda_model,
)
from src.baselines.params import format_prior_scale_variant
from src.core.artifacts import load_json, save_json
from src.core.errors import MissingArtifactError
from src.core.paths import (
    RESULTS_ROOT,
    build_archive_result_dir,
    build_latest_result_dir,
    resolve_project_path,
    write_latest_result_pointer,
)
from src.core.result_identity import build_condition_id, build_execution_id
from src.evaluation.reporting import (
    read_evaluation_json,
    write_csv_rows,
    write_evaluation_json,
    write_tabular_report_json,
)
from src.evaluation.schema import build_evaluation_meta
from src.evaluation.topic_pairs.inputs import (
    EMBEDDING_SPACE,
    SENTENCE_ENCODER_MODELS,
    CachedEmbeddings,
    SentenceAlignmentError,
    TrainCorpus,
    assert_same_sentences,
    encode_train_corpus,
    encoder_config_of,
    encoder_fingerprint,
    load_fine_labels,
    load_train_corpus,
    load_vmf_model_reference,
    model_embedding_variant,
    normalize_model_name,
    normalize_model_names,
    parameter_variant_for,
    provenance_for,
    reference_model,
    resolve_condition_dir,
    sentence_labels,
    unit_normalize,
)
from src.evaluation.topic_pairs.numerics import (
    KAPPA_ESTIMATOR,
    PER_PAIR_KEYS,
    PER_TOPIC_KEYS,
    PER_TOPIC_MATRIX_KEYS,
    SCALAR_SUMMARY_KEYS,
    TopicPairResult,
    compute_topic_pair_metrics,
    cross_model_overlap,
    model_reference_metrics,
)
from src.evaluation.word_based.sentence_encoding import (
    resolve_topic_word_encoder_device,
)
from src.evaluation.word_based.topic_assignment import (
    CollapsedFoldInConfig,
    run_collapsed_fold_in,
)
from src.evaluation.word_based.topic_word_runtime import (
    sentence_gaussian_log_likelihoods,
    sentlda_sentence_log_likelihoods,
    vmf_sentence_log_likelihoods,
)
from src.utils.encoder_profiles import embedding_variant_base, encoder_model_alias
from src.utils.logging import get_logger

TASK_NAME = "topic_pair_metrics"
CROSS_MODEL_KEY = "cross_model"
METRIC_SCHEMA_VERSION = 1
POSTERIOR_DEFINITION = "collapsed_fold_in"
JS_LOG_BASE = "e"
DEFAULT_SPLIT = "train"
ANALYSIS_ROOT = RESULTS_ROOT / "topic_analysis"
DEFAULT_OUT_ROOT = ANALYSIS_ROOT / "topic_pairs"
DEFAULT_CACHE_ROOT = DEFAULT_OUT_ROOT / ".cache" / "sentence_embeddings"
METRICS_FILENAME = "topic_pair_metrics_agg.json"
METADATA_FILENAME = "metadata.json"
FAILED_CONDITIONS_FILENAME = "failed_conditions.json"
CONDITION_FAILURE_POLICIES: tuple[str, ...] = ("fail-fast", "isolate")
# Protocol fields recorded on every condition; the summary requires the
# conditions of one sidecar to agree on all of them.
PROTOCOL_FIELDS: tuple[str, ...] = (
    "split",
    "embedding_space",
    "posterior_definition",
    "foldin_num_chains",
    "foldin_burn_in_sweeps",
    "foldin_retained_samples",
    "foldin_thinning",
    "foldin_random_seed",
    "kappa_estimator",
    "js_log_base",
    "target_column",
    "metric_schema_version",
)
SUMMARY_BASE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "data_run",
    "num_topics",
    "category",
    "model",
    "embedding_variant",
    "prior_scale",
    "split",
    "num_iterations",
)
logger = get_logger(__name__)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _uses_default_output_layout(out_root: Path) -> bool:
    return resolve_project_path(out_root) == DEFAULT_OUT_ROOT


def pair_key(model_a: str, model_b: str) -> str:
    return f"{model_a}|{model_b}"


def _resolve_requested_embedding_variant(
    *, embedding_variant: str | None, encoder_model: str | None
) -> str | None:
    requested = None if embedding_variant in {None, ""} else str(embedding_variant)
    if encoder_model in {None, ""}:
        return requested
    encoder_variant = encoder_model_alias(str(encoder_model))
    if requested is not None:
        if embedding_variant_base(requested) != embedding_variant_base(encoder_variant):
            raise ValueError(
                "encoder_model and embedding_variant mismatch: "
                f"encoder_model='{encoder_model}' resolves to "
                f"'{embedding_variant_base(encoder_variant)}', but "
                f"embedding_variant='{requested}' resolves to "
                f"'{embedding_variant_base(requested)}'."
            )
        return requested
    return encoder_variant


def _normalize_topics(num_topics: int | Sequence[int]) -> list[int]:
    if isinstance(num_topics, (int, np.integer)):
        return [int(num_topics)]
    values: list[int] = []
    for value in num_topics:
        value_int = int(value)
        if value_int not in values:
            values.append(value_int)
    if not values:
        raise ValueError("num_topics must contain at least one value.")
    return values


def foldin_protocol_fields(config: CollapsedFoldInConfig) -> dict[str, Any]:
    return {
        "foldin_num_chains": int(config.num_chains),
        "foldin_burn_in_sweeps": int(config.burn_in_sweeps),
        "foldin_retained_samples": int(config.retained_samples),
        "foldin_thinning": int(config.thinning),
        "foldin_random_seed": int(config.random_seed),
    }


def _protocol(
    *, split: str, foldin_config: CollapsedFoldInConfig, target_column: str
) -> dict[str, Any]:
    return {
        "split": split,
        "embedding_space": EMBEDDING_SPACE,
        "posterior_definition": POSTERIOR_DEFINITION,
        **foldin_protocol_fields(foldin_config),
        "kappa_estimator": KAPPA_ESTIMATOR,
        "js_log_base": JS_LOG_BASE,
        "target_column": target_column,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
    }


def _build_output_condition_id(
    *,
    model: str,
    dataset: str,
    data_run: str,
    category: str,
    iterations: Sequence[int],
    num_topics: int,
    protocol: dict[str, Any],
    embedding_variant: str | None,
    parameter_variant: str | None,
    prior_scale: float | None,
    encoder_space: str | None = None,
    models: Sequence[str] | None = None,
) -> tuple[str, str]:
    extra_labels: list[str] = ["cross" if model == CROSS_MODEL_KEY else model]
    if embedding_variant not in {None, ""}:
        extra_labels.append(str(embedding_variant))
    if parameter_variant not in {None, ""}:
        extra_labels.append(str(parameter_variant))
    payload = {
        "task": TASK_NAME,
        "model": model,
        "models": None if models is None else list(models),
        "dataset": dataset,
        "data_run": data_run,
        "category": category,
        "iterations": [int(value) for value in iterations],
        "num_topics": int(num_topics),
        "embedding_variant": embedding_variant,
        "prior_scale": None if prior_scale is None else float(prior_scale),
        **protocol,
    }
    # Every metric is computed in the requested sentence-encoder space, so a
    # model that stores no variant of its own (SentLDA reads no embeddings)
    # would otherwise get one condition id for all encoders. Models whose own
    # ``embedding_variant`` already names the encoder keep their historical id.
    if embedding_variant in {None, ""} and encoder_space not in {None, ""}:
        payload["encoder_space"] = str(encoder_space)
        extra_labels.append(str(encoder_space))
    return build_condition_id(
        iteration=int(min(iterations)),
        num_topics=int(num_topics),
        fingerprint_payload=payload,
        extra_labels=extra_labels,
    )


def _existing_output(
    *,
    out_root: Path,
    uses_default_output_layout: bool,
    dataset: str,
    data_run: str,
    category: str,
    condition_id: str,
) -> Path | None:
    if uses_default_output_layout:
        latest_dir = build_latest_result_dir(
            base_root=out_root,
            dataset=dataset,
            data_run=data_run,
            category=category,
            display_key=condition_id,
        )
        pointer_path = latest_dir / "CURRENT.json"
        if not pointer_path.exists():
            return None
        payload = load_json(pointer_path)
        if not isinstance(payload, dict) or not payload.get("archive_dir"):
            return None
        candidate = resolve_project_path(str(payload["archive_dir"])) / METRICS_FILENAME
        return candidate if candidate.exists() else None
    candidate = (
        out_root / dataset / data_run / category / condition_id / METRICS_FILENAME
    )
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Per-category context and per-run computation
# ---------------------------------------------------------------------------


class _DimensionOnlyEncoder:
    """Stands in for the encoder when only its output dimension is needed."""

    def __init__(self, dimension: int) -> None:
        self._dimension = int(dimension)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension


@dataclass
class CategoryContext:
    dataset: str
    data_run: str
    category: str
    split: str
    reference_model: str
    reference_condition_dir: Path
    corpus: TrainCorpus
    embeddings: CachedEmbeddings
    embeddings_unit: np.ndarray
    encoder_config: dict[str, Any]
    encoder_fingerprint: str
    doc_labels: np.ndarray
    labels: np.ndarray
    label_names: list[str]


def _encoder_config_for_category(
    *,
    models: Sequence[str],
    reference: str,
    reference_dir: Path,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    embedding_variant: str | None,
    encoder_model: str | None,
    prior_scale: float | None,
) -> dict[str, Any]:
    if reference in SENTENCE_ENCODER_MODELS:
        return encoder_config_of(reference_dir)
    for model in models:
        if model in SENTENCE_ENCODER_MODELS:
            condition_dir = resolve_condition_dir(
                model=model,
                dataset=dataset,
                data_run=data_run,
                iteration=iteration,
                num_topics=num_topics,
                category=category,
                embedding_variant=embedding_variant,
                prior_scale=prior_scale,
            )
            return encoder_config_of(condition_dir)
    if encoder_model in {None, ""}:
        raise ValueError(
            "no selected model records an encoder_config; pass encoder_model"
        )
    return {"model_name": str(encoder_model)}


def load_category_context(
    *,
    models: Sequence[str],
    dataset: str,
    data_run: str,
    category: str,
    split: str,
    iteration: int,
    num_topics: int,
    embedding_variant: str | None,
    encoder_model: str | None,
    prior_scale: float | None,
    cache_root: Path,
    encoder_device: str,
    encode_batch_size: int | None,
    target_column: str,
    label_schema: str,
) -> CategoryContext:
    reference = reference_model(models)
    reference_dir = resolve_condition_dir(
        model=reference,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        embedding_variant=embedding_variant,
        prior_scale=prior_scale,
    )
    corpus = load_train_corpus(reference_dir, model=reference, split=split)
    if reference == "sentlda":
        logger.warning(
            "[%s/%s] the reference corpus is a SentLDA run; its raw document "
            "indices may be category-frame positions, so fine labels can be wrong",
            dataset,
            category,
        )
    encoder_config = _encoder_config_for_category(
        models=models,
        reference=reference,
        reference_dir=reference_dir,
        dataset=dataset,
        data_run=data_run,
        category=category,
        iteration=iteration,
        num_topics=num_topics,
        embedding_variant=embedding_variant,
        encoder_model=encoder_model,
        prior_scale=prior_scale,
    )
    embeddings = encode_train_corpus(
        corpus,
        encoder_config=encoder_config,
        cache_root=cache_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        split=split,
        device=encoder_device,
        encode_batch_size=encode_batch_size,
    )
    logger.info(
        "[%s/%s] sentence embeddings %s (%d sentences x %d) from %s",
        dataset,
        category,
        "read from cache" if embeddings.cache_hit else "encoded and cached",
        embeddings.embeddings.shape[0],
        embeddings.embedding_dim,
        embeddings.cache_dir,
    )
    doc_labels, label_names = load_fine_labels(
        dataset,
        split=split,
        raw_doc_indices=corpus.raw_doc_indices,
        category=category,
        target_column=target_column,
        label_schema=label_schema,
    )
    return CategoryContext(
        dataset=dataset,
        data_run=data_run,
        category=category,
        split=split,
        reference_model=reference,
        reference_condition_dir=reference_dir,
        corpus=corpus,
        embeddings=embeddings,
        embeddings_unit=unit_normalize(embeddings.embeddings),
        encoder_config=encoder_config,
        encoder_fingerprint=encoder_fingerprint(encoder_config),
        doc_labels=doc_labels,
        labels=sentence_labels(doc_labels, corpus.doc_offsets),
        label_names=label_names,
    )


@dataclass(frozen=True)
class ModelPosterior:
    model: str
    condition_dir: Path
    probs: np.ndarray  # (S, K) collapsed fold-in posterior means
    alpha: np.ndarray
    posterior_metadata: dict[str, Any]


def compute_model_posterior(
    *,
    model: str,
    condition_dir: Path,
    context: CategoryContext,
    foldin_config: CollapsedFoldInConfig,
) -> ModelPosterior:
    """Sentence-topic posteriors of one run under the shared fold-in protocol."""

    key = normalize_model_name(model)
    corpus = load_train_corpus(condition_dir, model=key, split=context.split)
    assert_same_sentences(context.corpus, corpus)
    if key == "vmf":
        likelihoods, alpha = vmf_sentence_log_likelihoods(
            condition_dir=condition_dir,
            encoded_documents=context.embeddings.iter_documents(),
        )
    elif key == "sentlda":
        likelihoods, alpha, _ = sentlda_sentence_log_likelihoods(
            condition_dir=condition_dir, documents=corpus.documents
        )
    else:
        persisted = load_sentence_gaussianlda_model(
            param_dir=Path(condition_dir) / "params",
            encoder=_DimensionOnlyEncoder(context.embeddings.embedding_dim),
        )
        likelihoods, alpha = sentence_gaussian_log_likelihoods(
            persisted_model=persisted,
            encoded_documents=context.embeddings.iter_documents(),
        )
    posterior = run_collapsed_fold_in(
        alpha=alpha,
        assignment_unit_type="sentence",
        config=foldin_config,
        log_likelihood_by_doc=likelihoods,
        corpus_fingerprint=corpus.sentence_sha1,
    )
    blocks = [
        np.asarray(block, dtype=np.float64) for block in posterior.posterior_mean_by_doc
    ]
    probs = (
        np.vstack(blocks)
        if blocks
        else np.empty((0, int(alpha.shape[0])), dtype=np.float64)
    )
    if probs.shape[0] != context.corpus.num_sentences:
        raise SentenceAlignmentError(
            f"{key}: fold-in produced {probs.shape[0]} sentence rows for "
            f"{context.corpus.num_sentences} reference sentences"
        )
    return ModelPosterior(
        model=key,
        condition_dir=Path(condition_dir),
        probs=probs,
        alpha=np.asarray(alpha, dtype=np.float64),
        posterior_metadata=dict(posterior.metadata),
    )


def _as_list(values: np.ndarray) -> list[Any]:
    return np.asarray(values, dtype=np.float64).tolist()


def _iteration_entry(
    *,
    iteration: int,
    context: CategoryContext,
    posterior: ModelPosterior,
    result: TopicPairResult,
    model_own: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "iteration": int(iteration),
        "num_documents": int(context.corpus.num_documents),
    }
    entry.update(result.scalar_summary())
    entry["per_topic"] = {
        key: _as_list(result.per_topic[key]) for key in PER_TOPIC_KEYS
    }
    for key in PER_TOPIC_MATRIX_KEYS:
        entry["per_topic"][key] = _as_list(result.per_topic[key])
    entry["per_pair"] = {key: _as_list(result.per_pair[key]) for key in PER_PAIR_KEYS}
    entry["model_own"] = (
        None
        if model_own is None
        else {key: _as_list(value) for key, value in model_own.items()}
    )
    entry["source_condition_dir"] = str(posterior.condition_dir)
    return entry


def _write_matrix_csv(
    path: Path, matrix: np.ndarray, *, columns: Sequence[str] | None = None
) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix for {path}")
    names = (
        [str(name) for name in columns]
        if columns is not None
        else [str(index) for index in range(values.shape[1])]
    )
    rows = [
        {
            "topic": row_index,
            **{name: float(values[row_index, col]) for col, name in enumerate(names)},
        }
        for row_index in range(values.shape[0])
    ]
    write_csv_rows(fieldnames=["topic", *names], rows=rows, path=path)


def _write_model_iteration_artifacts(
    *,
    iter_out_dir: Path,
    result: TopicPairResult,
    model_own: dict[str, np.ndarray] | None,
    label_names: Sequence[str],
) -> dict[str, str]:
    ensure_directory(iter_out_dir)
    written: dict[str, str] = {}
    topic_rows = []
    for topic in range(result.num_topics):
        row: dict[str, Any] = {"topic": topic}
        for key in PER_TOPIC_KEYS:
            row[key] = float(result.per_topic[key][topic])
        if model_own is not None:
            row["kappa_model"] = float(model_own["kappa_model"][topic])
        topic_rows.append(row)
    fieldnames = ["topic", *PER_TOPIC_KEYS]
    if model_own is not None:
        fieldnames.append("kappa_model")
    path = iter_out_dir / "topic_metrics.csv"
    write_csv_rows(fieldnames=fieldnames, rows=topic_rows, path=path)
    written["topic_metrics_csv"] = f"{iter_out_dir.name}/{path.name}"
    path = iter_out_dir / "label_mass.csv"
    _write_matrix_csv(path, result.per_topic["label_mass"], columns=label_names)
    written["label_mass_csv"] = f"{iter_out_dir.name}/{path.name}"
    for key in PER_PAIR_KEYS:
        path = iter_out_dir / f"{key}.csv"
        _write_matrix_csv(path, result.per_pair[key])
        written[f"{key}_csv"] = f"{iter_out_dir.name}/{path.name}"
    if model_own is not None:
        path = iter_out_dir / "centroid_cosine_model.csv"
        _write_matrix_csv(path, model_own["centroid_cosine_model"])
        written["centroid_cosine_model_csv"] = f"{iter_out_dir.name}/{path.name}"
    return written


def _write_cross_iteration_artifacts(
    *, iter_out_dir: Path, overlaps: dict[str, np.ndarray]
) -> dict[str, str]:
    ensure_directory(iter_out_dir)
    written: dict[str, str] = {}
    for key, matrix in overlaps.items():
        name = key.replace("|", "__")
        path = iter_out_dir / f"overlap_{name}.csv"
        _write_matrix_csv(path, matrix)
        written[f"overlap_{name}_csv"] = f"{iter_out_dir.name}/{path.name}"
    return written


def _aggregate(
    entries: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for key in keys:
        values = np.asarray(
            [float(entry.get(key, np.nan)) for entry in entries], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        aggregate[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": (
                float(finite.std(ddof=1))
                if finite.size > 1
                else (0.0 if finite.size == 1 else float("nan"))
            ),
            "n": int(finite.size),
        }
    return aggregate


def round_sigfigs(value: float, sig: int = 4) -> float:
    if value is None or not np.isfinite(value):
        return float("nan") if value is None else float(value)
    return float(f"{value:.{sig}g}")


def _summary_row(*, meta: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    aggregate = results.get("aggregate", {})
    per_iteration = results.get("per_iteration", [])
    row: dict[str, Any] = {
        "dataset": meta.get("dataset"),
        "data_run": meta.get("data_run"),
        "num_topics": meta.get("num_topics"),
        "category": meta.get("category"),
        "model": meta.get("model"),
        "embedding_variant": meta.get("effective_embedding_variant"),
        "prior_scale": meta.get("prior_scale"),
        "split": meta.get("split"),
        "num_iterations": len(per_iteration),
    }
    for key in SCALAR_SUMMARY_KEYS:
        stats = aggregate.get(key, {})
        row[f"{key}_mean"] = round_sigfigs(float(stats.get("mean", np.nan)))
        row[f"{key}_std"] = round_sigfigs(float(stats.get("std", np.nan)))
    return row


def summary_fieldnames() -> list[str]:
    fieldnames = list(SUMMARY_BASE_COLUMNS)
    for key in SCALAR_SUMMARY_KEYS:
        fieldnames.append(f"{key}_mean")
        fieldnames.append(f"{key}_std")
    return fieldnames


@dataclass
class _ConditionOutput:
    condition_id: str
    condition_fingerprint: str
    out_dir: Path
    archive_out_dir: Path | None
    latest_out_dir: Path | None
    started_at: str
    execution_id: str


def _open_condition_output(
    *,
    out_root: Path,
    uses_default_output_layout: bool,
    dataset: str,
    data_run: str,
    category: str,
    condition_id: str,
    condition_fingerprint: str,
) -> _ConditionOutput:
    started_at = datetime.now(UTC).isoformat()
    execution_id = build_execution_id(prefix="exec", started_at=started_at)
    if uses_default_output_layout:
        archive_out_dir = build_archive_result_dir(
            base_root=out_root,
            dataset=dataset,
            data_run=data_run,
            category=category,
            display_key=condition_id,
            started_at=started_at,
            execution_id=execution_id,
        )
        latest_out_dir = build_latest_result_dir(
            base_root=out_root,
            dataset=dataset,
            data_run=data_run,
            category=category,
            display_key=condition_id,
        )
        out_dir = archive_out_dir
    else:
        archive_out_dir = None
        latest_out_dir = None
        out_dir = out_root / dataset / data_run / category / condition_id
    ensure_directory(out_dir)
    return _ConditionOutput(
        condition_id=condition_id,
        condition_fingerprint=condition_fingerprint,
        out_dir=out_dir,
        archive_out_dir=archive_out_dir,
        latest_out_dir=latest_out_dir,
        started_at=started_at,
        execution_id=execution_id,
    )


def _finish_condition_output(
    *,
    output: _ConditionOutput,
    out_root: Path,
    dataset: str,
    data_run: str,
    category: str,
    meta: dict[str, Any],
    results: dict[str, Any],
    artifacts: dict[str, str],
    label: str,
) -> None:
    out_path = output.out_dir / METRICS_FILENAME
    write_evaluation_json(meta=meta, results=results, path=out_path)
    save_json(meta, output.out_dir / METADATA_FILENAME)
    logger.info("[%s] topic-pair metrics saved to %s", label, out_path)
    if output.archive_out_dir is not None:
        pointer_path = write_latest_result_pointer(
            base_root=out_root,
            task=TASK_NAME,
            dataset=dataset,
            data_run=data_run,
            category=category,
            display_key=output.condition_id,
            archive_dir=output.archive_out_dir,
            started_at=output.started_at,
            execution_id=output.execution_id,
            condition_fingerprint=output.condition_fingerprint,
            artifacts=artifacts,
        )
        logger.info("[%s] updated latest pointer at %s", label, pointer_path)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass
class _KConditions:
    """Which (model | cross) conditions of one (K, category) still need writing."""

    model_ids: dict[str, tuple[str, str]] = field(default_factory=dict)
    cross_id: tuple[str, str] | None = None
    models_to_write: list[str] = field(default_factory=list)
    write_cross: bool = False
    existing_rows: list[dict[str, Any]] = field(default_factory=list)


def run_topic_pair_metrics(
    *,
    models: Sequence[str],
    dataset: str,
    iterations: Sequence[int],
    num_topics: int | Sequence[int],
    categories: Sequence[str],
    data_runs: Sequence[str] = ("default",),
    split: str = DEFAULT_SPLIT,
    embedding_variant: str | None = None,
    encoder_model: str | None = None,
    prior_scale: float | None = None,
    out_root: Path = DEFAULT_OUT_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    encoder_device: str = "auto",
    encode_batch_size: int | None = None,
    target_column: str = "target_str",
    label_schema: str = "identity",
    foldin_config: CollapsedFoldInConfig | None = None,
    include_model_reference: bool = True,
    save_per_iter_artifacts: bool = True,
    skip_existing: bool = False,
    condition_failure_policy: str = "fail-fast",
) -> Path:
    if condition_failure_policy not in CONDITION_FAILURE_POLICIES:
        raise ValueError(
            f"Unsupported condition_failure_policy '{condition_failure_policy}'. "
            f"Use one of {CONDITION_FAILURE_POLICIES}."
        )
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if prior_scale is not None:
        format_prior_scale_variant(prior_scale)
    resolved_models = normalize_model_names(models)
    topic_values = _normalize_topics(num_topics)
    iteration_values = [int(value) for value in iterations]
    if not iteration_values:
        raise ValueError("iterations must contain at least one value.")
    if not categories:
        raise ValueError("categories must contain at least one value.")
    resolved_foldin = foldin_config or CollapsedFoldInConfig()
    resolved_foldin.validate()
    out_root = Path(out_root)
    cache_root = Path(cache_root)
    uses_default_output_layout = _uses_default_output_layout(out_root)
    requested_variant = _resolve_requested_embedding_variant(
        embedding_variant=embedding_variant, encoder_model=encoder_model
    )
    resolved_device = resolve_topic_word_encoder_device(encoder_device)
    protocol = _protocol(
        split=split, foldin_config=resolved_foldin, target_column=target_column
    )
    failure_exceptions = (MissingArtifactError, FileNotFoundError, ValueError)

    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []

    def _record_failure(exc: BaseException, **where: Any) -> None:
        if condition_failure_policy == "fail-fast":
            raise exc
        failure = {
            **where,
            "dataset": dataset,
            "iterations": iteration_values,
            "prior_scale": prior_scale,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failures.append(failure)
        logger.warning("skipping %s after %s: %s", where, type(exc).__name__, exc)

    def _condition_ids(
        data_run: str, category: str, topic_count: int
    ) -> tuple[dict[str, tuple[str, str]], tuple[str, str] | None]:
        model_ids: dict[str, tuple[str, str]] = {}
        for model in resolved_models:
            model_ids[model] = _build_output_condition_id(
                model=model,
                dataset=dataset,
                data_run=data_run,
                category=category,
                iterations=iteration_values,
                num_topics=topic_count,
                protocol=protocol,
                embedding_variant=model_embedding_variant(model, requested_variant),
                parameter_variant=parameter_variant_for(model, prior_scale),
                prior_scale=prior_scale,
                encoder_space=requested_variant,
            )
        cross_id = None
        if len(resolved_models) >= 2:
            cross_id = _build_output_condition_id(
                model=CROSS_MODEL_KEY,
                dataset=dataset,
                data_run=data_run,
                category=category,
                iterations=iteration_values,
                num_topics=topic_count,
                protocol=protocol,
                embedding_variant=requested_variant,
                parameter_variant=None,
                prior_scale=prior_scale,
                models=resolved_models,
            )
        return model_ids, cross_id

    def _plan(data_run: str, category: str, topic_count: int) -> _KConditions:
        model_ids, cross_id = _condition_ids(data_run, category, topic_count)
        plan = _KConditions(model_ids=model_ids, cross_id=cross_id)
        for model in resolved_models:
            existing = (
                _existing_output(
                    out_root=out_root,
                    uses_default_output_layout=uses_default_output_layout,
                    dataset=dataset,
                    data_run=data_run,
                    category=category,
                    condition_id=model_ids[model][0],
                )
                if skip_existing
                else None
            )
            if existing is None:
                plan.models_to_write.append(model)
            else:
                meta, results = read_evaluation_json(existing)
                plan.existing_rows.append(_summary_row(meta=meta, results=results))
                logger.info(
                    "[%s] skip existing condition %s (%s)",
                    model,
                    model_ids[model][0],
                    existing,
                )
        if cross_id is not None:
            existing = (
                _existing_output(
                    out_root=out_root,
                    uses_default_output_layout=uses_default_output_layout,
                    dataset=dataset,
                    data_run=data_run,
                    category=category,
                    condition_id=cross_id[0],
                )
                if skip_existing
                else None
            )
            if existing is None:
                plan.write_cross = True
            else:
                meta, results = read_evaluation_json(existing)
                plan.existing_rows.append(_summary_row(meta=meta, results=results))
                logger.info(
                    "[cross] skip existing condition %s (%s)", cross_id[0], existing
                )
        return plan

    for data_run in data_runs:
        for category in categories:
            context: CategoryContext | None = None
            context_error: BaseException | None = None
            for topic_count in topic_values:
                plan = _plan(data_run, category, topic_count)
                summary_rows.extend(plan.existing_rows)
                if not plan.models_to_write and not plan.write_cross:
                    continue
                # Every model's posterior is needed when the cross condition is
                # (re)built; otherwise only the missing model conditions.
                models_to_compute = (
                    list(resolved_models)
                    if plan.write_cross
                    else list(plan.models_to_write)
                )
                if context is None and context_error is None:
                    try:
                        context = load_category_context(
                            models=resolved_models,
                            dataset=dataset,
                            data_run=data_run,
                            category=category,
                            split=split,
                            iteration=iteration_values[0],
                            num_topics=topic_count,
                            embedding_variant=requested_variant,
                            encoder_model=encoder_model,
                            prior_scale=prior_scale,
                            cache_root=cache_root,
                            encoder_device=resolved_device,
                            encode_batch_size=encode_batch_size,
                            target_column=target_column,
                            label_schema=label_schema,
                        )
                    except failure_exceptions as exc:
                        context_error = exc
                        _record_failure(
                            exc,
                            model="category_context",
                            data_run=data_run,
                            category=category,
                            num_topics=topic_count,
                            embedding_variant=requested_variant,
                            condition_id=None,
                        )
                if context is None:
                    for model in models_to_compute:
                        failures.append(
                            {
                                "model": model,
                                "dataset": dataset,
                                "data_run": data_run,
                                "category": category,
                                "num_topics": topic_count,
                                "iterations": iteration_values,
                                "embedding_variant": model_embedding_variant(
                                    model, requested_variant
                                ),
                                "prior_scale": prior_scale,
                                "condition_id": plan.model_ids[model][0],
                                "error_type": type(context_error).__name__,
                                "error": f"category context unavailable: {context_error}",
                            }
                        )
                    continue

                per_iteration: dict[str, list[dict[str, Any]]] = {
                    model: [] for model in models_to_compute
                }
                artifacts_written: dict[str, dict[str, str]] = {
                    model: {} for model in models_to_compute
                }
                condition_dirs: dict[str, list[str]] = {
                    model: [] for model in models_to_compute
                }
                posterior_meta: dict[str, dict[str, Any]] = {}
                failed_models: set[str] = set()
                cross_entries: list[dict[str, Any]] = []
                cross_artifacts: dict[str, str] = {}
                cross_pairs: list[str] = []
                cross_output: _ConditionOutput | None = None
                model_outputs: dict[str, _ConditionOutput] = {}

                for iteration in iteration_values:
                    posteriors: dict[str, ModelPosterior] = {}
                    for model in models_to_compute:
                        if model in failed_models:
                            continue
                        try:
                            condition_dir = resolve_condition_dir(
                                model=model,
                                dataset=dataset,
                                data_run=data_run,
                                iteration=iteration,
                                num_topics=topic_count,
                                category=category,
                                embedding_variant=requested_variant,
                                prior_scale=prior_scale,
                            )
                            posterior = compute_model_posterior(
                                model=model,
                                condition_dir=condition_dir,
                                context=context,
                                foldin_config=resolved_foldin,
                            )
                            result = compute_topic_pair_metrics(
                                context.embeddings_unit,
                                posterior.probs,
                                context.labels,
                                num_labels=len(context.label_names),
                            )
                            model_own = None
                            if include_model_reference and model == "vmf":
                                reference = load_vmf_model_reference(condition_dir)
                                model_own = model_reference_metrics(
                                    reference["topic_means"], reference["kappa_model"]
                                )
                        except failure_exceptions as exc:
                            failed_models.add(model)
                            _record_failure(
                                exc,
                                model=model,
                                data_run=data_run,
                                category=category,
                                num_topics=topic_count,
                                embedding_variant=model_embedding_variant(
                                    model, requested_variant
                                ),
                                condition_id=plan.model_ids[model][0],
                            )
                            continue
                        posteriors[model] = posterior
                        posterior_meta.setdefault(model, posterior.posterior_metadata)
                        condition_dirs[model].append(str(condition_dir))
                        if model in plan.models_to_write:
                            if model not in model_outputs:
                                model_outputs[model] = _open_condition_output(
                                    out_root=out_root,
                                    uses_default_output_layout=uses_default_output_layout,
                                    dataset=dataset,
                                    data_run=data_run,
                                    category=category,
                                    condition_id=plan.model_ids[model][0],
                                    condition_fingerprint=plan.model_ids[model][1],
                                )
                            per_iteration[model].append(
                                _iteration_entry(
                                    iteration=iteration,
                                    context=context,
                                    posterior=posterior,
                                    result=result,
                                    model_own=model_own,
                                )
                            )
                            if save_per_iter_artifacts:
                                written = _write_model_iteration_artifacts(
                                    iter_out_dir=model_outputs[model].out_dir
                                    / f"iter{iteration}",
                                    result=result,
                                    model_own=model_own,
                                    label_names=context.label_names,
                                )
                                for name, relative in written.items():
                                    artifacts_written[model][
                                        f"{name}_iter{iteration}"
                                    ] = relative
                        logger.info(
                            "[%s] %s/%s K=%d it=%d: %d sentences, %d empty topics",
                            model,
                            dataset,
                            category,
                            topic_count,
                            iteration,
                            result.num_sentences,
                            result.num_empty_topics,
                        )

                    if plan.write_cross and plan.cross_id is not None:
                        available = [m for m in resolved_models if m in posteriors]
                        overlaps: dict[str, np.ndarray] = {}
                        for index_a, model_a in enumerate(available):
                            for model_b in available[index_a + 1 :]:
                                overlaps[pair_key(model_a, model_b)] = (
                                    cross_model_overlap(
                                        posteriors[model_a].probs,
                                        posteriors[model_b].probs,
                                    )
                                )
                        if overlaps:
                            if cross_output is None:
                                cross_output = _open_condition_output(
                                    out_root=out_root,
                                    uses_default_output_layout=uses_default_output_layout,
                                    dataset=dataset,
                                    data_run=data_run,
                                    category=category,
                                    condition_id=plan.cross_id[0],
                                    condition_fingerprint=plan.cross_id[1],
                                )
                            cross_pairs = sorted(set(cross_pairs) | set(overlaps))
                            cross_entries.append(
                                {
                                    "iteration": int(iteration),
                                    "num_sentences": int(context.corpus.num_sentences),
                                    "num_topics": {
                                        model: int(posteriors[model].probs.shape[1])
                                        for model in available
                                    },
                                    "overlap": {
                                        key: _as_list(value)
                                        for key, value in overlaps.items()
                                    },
                                }
                            )
                            if save_per_iter_artifacts:
                                written = _write_cross_iteration_artifacts(
                                    iter_out_dir=cross_output.out_dir
                                    / f"iter{iteration}",
                                    overlaps=overlaps,
                                )
                                for name, relative in written.items():
                                    cross_artifacts[f"{name}_iter{iteration}"] = (
                                        relative
                                    )

                common_meta = {
                    "dataset": dataset,
                    "data_run": data_run,
                    "num_topics": topic_count,
                    "category": category,
                    "iterations": iteration_values,
                    **protocol,
                    "embedding_variant": requested_variant,
                    "encoder_model": encoder_model,
                    "encoder_model_name": context.encoder_config.get("model_name"),
                    "encoder_config_fingerprint": context.encoder_fingerprint,
                    "encoder_device": resolved_device,
                    "embedding_cache_dir": str(context.embeddings.cache_dir),
                    "sentence_sha1": context.corpus.sentence_sha1,
                    "num_sentences": int(context.corpus.num_sentences),
                    "num_documents": int(context.corpus.num_documents),
                    "label_names": list(context.label_names),
                    "label_source_model": context.reference_model,
                    "label_source_condition_dir": str(context.reference_condition_dir),
                    "reference_model": context.reference_model,
                    "prior_scale": None if prior_scale is None else float(prior_scale),
                }

                for model in plan.models_to_write:
                    if model in failed_models or model not in model_outputs:
                        continue
                    entries = per_iteration[model]
                    if len(entries) != len(iteration_values):
                        continue
                    aggregate = _aggregate(entries, SCALAR_SUMMARY_KEYS)
                    output = model_outputs[model]
                    provenance = provenance_for(
                        Path(condition_dirs[model][0]), model=model
                    )
                    meta = build_evaluation_meta(
                        task=TASK_NAME,
                        model=model,
                        condition_id=output.condition_id,
                        display_key=output.condition_id,
                        condition_fingerprint=output.condition_fingerprint,
                        started_at=output.started_at,
                        execution_id=output.execution_id,
                        archive_dir=str(output.out_dir),
                        latest_dir=(
                            None
                            if output.latest_out_dir is None
                            else str(output.latest_out_dir)
                        ),
                        **common_meta,
                        effective_embedding_variant=model_embedding_variant(
                            model, requested_variant
                        ),
                        parameter_variant=parameter_variant_for(model, prior_scale),
                        max_kappa=None,
                        model_reference_included=bool(
                            include_model_reference and model == "vmf"
                        ),
                        posterior_metadata=posterior_meta.get(model),
                        source_condition_dirs=condition_dirs[model],
                        model_provenance=provenance,
                    )
                    artifacts = {
                        "metrics": METRICS_FILENAME,
                        "metadata": METADATA_FILENAME,
                    }
                    artifacts.update(artifacts_written[model])
                    results = {"aggregate": aggregate, "per_iteration": entries}
                    _finish_condition_output(
                        output=output,
                        out_root=out_root,
                        dataset=dataset,
                        data_run=data_run,
                        category=category,
                        meta=meta,
                        results=results,
                        artifacts=artifacts,
                        label=model,
                    )
                    summary_rows.append(_summary_row(meta=meta, results=results))

                if plan.write_cross and plan.cross_id is not None:
                    complete_pairs = [
                        key
                        for key in cross_pairs
                        if all(key in entry["overlap"] for entry in cross_entries)
                    ]
                    if (
                        cross_output is not None
                        and len(cross_entries) == len(iteration_values)
                        and complete_pairs
                    ):
                        cross_models = sorted(
                            {
                                model
                                for key in complete_pairs
                                for model in key.split("|")
                            },
                            key=resolved_models.index,
                        )
                        entries = [
                            {
                                **entry,
                                "overlap": {
                                    key: entry["overlap"][key] for key in complete_pairs
                                },
                            }
                            for entry in cross_entries
                        ]
                        meta = build_evaluation_meta(
                            task=TASK_NAME,
                            model=CROSS_MODEL_KEY,
                            models=cross_models,
                            pairs=complete_pairs,
                            condition_id=cross_output.condition_id,
                            display_key=cross_output.condition_id,
                            condition_fingerprint=cross_output.condition_fingerprint,
                            started_at=cross_output.started_at,
                            execution_id=cross_output.execution_id,
                            archive_dir=str(cross_output.out_dir),
                            latest_dir=(
                                None
                                if cross_output.latest_out_dir is None
                                else str(cross_output.latest_out_dir)
                            ),
                            **common_meta,
                            effective_embedding_variant=requested_variant,
                            parameter_variant=None,
                            posterior_metadata={
                                m: posterior_meta.get(m) for m in cross_models
                            },
                            source_condition_dirs={
                                m: condition_dirs[m] for m in cross_models
                            },
                            model_provenance={
                                m: provenance_for(Path(condition_dirs[m][0]), model=m)
                                for m in cross_models
                            },
                        )
                        artifacts = {
                            "metrics": METRICS_FILENAME,
                            "metadata": METADATA_FILENAME,
                        }
                        artifacts.update(cross_artifacts)
                        results = {"aggregate": {}, "per_iteration": entries}
                        _finish_condition_output(
                            output=cross_output,
                            out_root=out_root,
                            dataset=dataset,
                            data_run=data_run,
                            category=category,
                            meta=meta,
                            results=results,
                            artifacts=artifacts,
                            label=CROSS_MODEL_KEY,
                        )
                        summary_rows.append(_summary_row(meta=meta, results=results))
                    else:
                        _record_failure(
                            RuntimeError(
                                "cross-model overlap incomplete: "
                                f"{len(cross_entries)}/{len(iteration_values)} iterations, "
                                f"pairs={complete_pairs}"
                            ),
                            model=CROSS_MODEL_KEY,
                            data_run=data_run,
                            category=category,
                            num_topics=topic_count,
                            embedding_variant=requested_variant,
                            condition_id=plan.cross_id[0],
                        )

    ensure_directory(out_root)
    summary_path = out_root / "summary.csv"
    if summary_rows:
        fieldnames = summary_fieldnames()
        write_csv_rows(fieldnames=fieldnames, rows=summary_rows, path=summary_path)
        write_tabular_report_json(
            meta={
                "task": f"{TASK_NAME}_summary",
                "dataset": dataset,
                "data_runs": list(data_runs),
                **protocol,
                "embedding_variant": requested_variant,
                "encoder_model": encoder_model,
                "prior_scale": prior_scale,
                "out_root": str(out_root),
                "failed_conditions": failures,
            },
            columns=fieldnames,
            rows=summary_rows,
            path=out_root / "summary.json",
        )
        logger.info("summary CSV written to %s", summary_path)
    if failures:
        failures_path = out_root / FAILED_CONDITIONS_FILENAME
        save_json({"task": TASK_NAME, "failures": failures}, failures_path)
        logger.warning(
            "%d condition(s) failed; details written to %s",
            len(failures),
            failures_path,
        )
    return summary_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Topic-pair metrics of sentence-level topic models in the shared "
            "sentence-embedding space (centroid cosine, assignment confusion, "
            "fine-label divergence, empirical concentration, cross-model overlap)."
        )
    )
    parser.add_argument(
        "--model", nargs="+", default=["vmf", "sentlda", "sentence_gaussianlda"]
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_run", "--data-run", nargs="+", default=["default"])
    parser.add_argument("--iteration", type=int, nargs="+", required=True)
    parser.add_argument(
        "--num_topics", "--num-topics", type=int, nargs="+", required=True
    )
    parser.add_argument("--category", nargs="+", required=True)
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "test"])
    parser.add_argument("--embedding_variant", "--embedding-variant", default=None)
    parser.add_argument("--encoder_model", "--encoder-model", default=None)
    parser.add_argument("--prior-scale", "--prior_scale", type=float, default=None)
    parser.add_argument("--out_root", "--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--cache_root", "--cache-root", type=Path, default=DEFAULT_CACHE_ROOT
    )
    parser.add_argument("--encoder_device", "--encoder-device", default="auto")
    parser.add_argument(
        "--encode_batch_size", "--encode-batch-size", type=int, default=None
    )
    parser.add_argument("--target_column", "--target-column", default="target_str")
    parser.add_argument("--foldin_burn_in", "--foldin-burn-in", type=int, default=None)
    parser.add_argument(
        "--foldin_retained", "--foldin-retained", type=int, default=None
    )
    parser.add_argument("--foldin_seed", "--foldin-seed", type=int, default=None)
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true")
    parser.add_argument(
        "--no_per_iter_artifacts",
        "--no-per-iter-artifacts",
        dest="save_per_iter_artifacts",
        action="store_false",
    )
    parser.add_argument(
        "--no_model_reference",
        "--no-model-reference",
        dest="include_model_reference",
        action="store_false",
    )
    parser.add_argument(
        "--condition-failure-policy",
        "--condition_failure_policy",
        dest="condition_failure_policy",
        choices=list(CONDITION_FAILURE_POLICIES),
        default="fail-fast",
    )
    return parser.parse_args(argv)


def foldin_config_from_options(
    *, burn_in: int | None, retained: int | None, seed: int | None
) -> CollapsedFoldInConfig:
    default = CollapsedFoldInConfig()
    return CollapsedFoldInConfig(
        num_chains=default.num_chains,
        burn_in_sweeps=default.burn_in_sweeps if burn_in is None else int(burn_in),
        retained_samples=(
            default.retained_samples if retained is None else int(retained)
        ),
        thinning=default.thinning,
        random_seed=default.random_seed if seed is None else int(seed),
        backend=default.backend,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_topic_pair_metrics(
        models=args.model,
        dataset=args.dataset,
        data_runs=args.data_run,
        iterations=args.iteration,
        num_topics=args.num_topics,
        categories=args.category,
        split=args.split,
        embedding_variant=args.embedding_variant,
        encoder_model=args.encoder_model,
        prior_scale=args.prior_scale,
        out_root=args.out_root,
        cache_root=args.cache_root,
        encoder_device=args.encoder_device,
        encode_batch_size=args.encode_batch_size,
        target_column=args.target_column,
        foldin_config=foldin_config_from_options(
            burn_in=args.foldin_burn_in,
            retained=args.foldin_retained,
            seed=args.foldin_seed,
        ),
        include_model_reference=args.include_model_reference,
        save_per_iter_artifacts=args.save_per_iter_artifacts,
        skip_existing=args.skip_existing,
        condition_failure_policy=args.condition_failure_policy,
    )


__all__ = [
    "CROSS_MODEL_KEY",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_OUT_ROOT",
    "METRICS_FILENAME",
    "PROTOCOL_FIELDS",
    "TASK_NAME",
    "CategoryContext",
    "ModelPosterior",
    "compute_model_posterior",
    "foldin_config_from_options",
    "load_category_context",
    "main",
    "pair_key",
    "run_topic_pair_metrics",
    "summary_fieldnames",
]


if __name__ == "__main__":
    main()

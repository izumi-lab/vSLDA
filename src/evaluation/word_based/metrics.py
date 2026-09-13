from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from gensim.corpora import Dictionary

from src.baselines.params import (
    format_prior_scale_variant,
    normalize_covariance_type,
)
from src.core.artifacts import save_json, save_pickle
from src.core.paths import (
    build_archive_result_dir,
    build_latest_result_dir,
    resolve_project_path,
    write_latest_result_pointer,
)
from src.core.result_identity import build_execution_id
from src.core.vmf_variant import normalize_vmf_parameter_variant
from src.evaluation.reporting import write_evaluation_json
from src.evaluation.schema import build_evaluation_meta
from src.evaluation.word_based import cli as cli_module
from src.evaluation.word_based import corpus_bundle as corpus_bundle_module
from src.evaluation.word_based import model_inputs as model_inputs_module
from src.evaluation.word_based import reporting as reporting_module
from src.evaluation.word_based.reference_cache import (
    load_reference_count_cache_v2,
    save_reference_count_cache_v2,
)
from src.evaluation.word_based.reference_counts import (
    DEFAULT_REFERENCE_COUNT_CHUNK_SIZE,
    DEFAULT_REFERENCE_COUNT_WORKERS,
    ReferenceCountBackend,
    SharedReferenceCounts,
    build_shared_reference_counts,
    collect_target_words,
    compute_shared_reference_coherence_scores,
    effective_window_sizes_for_coherences,
)
from src.evaluation.word_based.reference_df import (
    load_reference_document_frequencies,
)
from src.evaluation.word_based.reference_query import build_reference_count_query
from src.evaluation.word_based.resumability import (
    fingerprint_payload,
    load_topic_word_checkpoint,
    reference_corpus_identity,
    save_failure_record,
    save_topic_word_checkpoint,
    topic_word_checkpoint_dir,
    write_completion_marker,
)
from src.evaluation.word_based.sentence_encoding import (
    resolve_topic_word_encoder_device,
)
from src.evaluation.word_based.topic_assignment import (
    COLLAPSED_MODELS,
    CollapsedFoldInConfig,
    ConditionEvaluationError,
    EmptyTopicError,
    fingerprint_jsonable,
)
from src.evaluation.word_based.topic_word_metrics import (
    DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT,
    EPSILON_SMOOTHED_COHERENCES,
    MULTI_COHERENCE_CHOICES,
    PALMETTO_CV_IMPLEMENTATION,
    STREAMING_REFERENCE_COHERENCES,
    aggregate_metrics,
    apply_fixed_k_empty_topic_policy,
    coherence_metric_key,
    compute_streaming_reference_coherence_scores,
    compute_topic_diversity,
    describe_coherence_metric,
    evaluate_topic_words,
    normalize_coherences,
    truncate_topic_words,
)
from src.evaluation.word_based.topic_word_runtime import (
    DEFAULT_TOPIC_WORD_SCORE_MODE,
    TOPIC_WORD_RANKING_SCHEMA_VERSION,
    RuntimeTopicWords,
    resolve_runtime_topic_words,
    select_metric_topic_words,
)
from src.evaluation.word_based.topic_words import (
    TopicWords,
    TopicWordsResult,
    serialize_topic_words,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

ENCODER_TOPIC_WORD_MODELS = {"vmf", "vmf_sentence_lda", "sentence_gaussianlda"}

ModelType = Literal[
    "vmf",
    "gaussian",
    "sentence_gaussianlda",
    "sentlda",
    "bertopic_kmeans",
    "bleilda",
    "gaussianlda",
    "etm",
    "mvtm",
    "ctm",
    "senclu",
    "spherical_kmeans",
    "gaussian_kmeans",
    "movmf",
    "gaussian_mixture",
    "sam",
    "sam_tf",
]

ANALYSIS_ROOT = model_inputs_module.ANALYSIS_ROOT
DEFAULT_OUT_ROOT = model_inputs_module.DEFAULT_OUT_ROOT
DEFAULT_EMBEDDING_VARIANT = model_inputs_module.DEFAULT_EMBEDDING_VARIANT
MODEL_ALIASES = model_inputs_module.MODEL_ALIASES
MODEL_CHOICES = model_inputs_module.MODEL_CHOICES
RUNTIME_TOPIC_WORD_MODELS = COLLAPSED_MODELS | {"etm", "ctm", "sam", "sam_tf"}


def _topic_word_score_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "topic_word_score_mode", DEFAULT_TOPIC_WORD_SCORE_MODE))


def _metric_topic_words_result(
    *, args: argparse.Namespace, runtime: RuntimeTopicWords
) -> TopicWordsResult:
    selected = select_metric_topic_words(
        runtime,
        score_mode=_topic_word_score_mode(args),
    )
    return TopicWordsResult(
        topic_words=selected.topic_words,
        topic_word_source=selected.topic_word_source,
        score_mode=selected.score_mode,
        score_definition=selected.score_definition,
        runtime_payload=runtime,
    )


@dataclass
class PendingWordBasedIteration:
    iteration: int
    topic_words: TopicWords
    runtime_payload: RuntimeTopicWords | None
    checkpoint_path: Path | None = None
    checkpoint_identity: dict[str, object] | None = None


@dataclass
class PendingWordBasedGroup:
    data_run: str
    model: str
    num_topics: int
    category: str
    iterations: list[PendingWordBasedIteration]
    topic_word_source: str
    topic_word_score_mode: str
    topic_word_score_definition: str


@dataclass
class PendingWordBasedGroupTask:
    sort_index: int
    data_run: str
    model: str
    num_topics: int
    category: str
    progress_start: int


@dataclass
class ScoredWordBasedGroup:
    group: PendingWordBasedGroup
    per_iter_metrics: list[dict[str, float]]
    per_iter_topic_words: list[dict[str, object]]
    used_iterations: list[int]


class WordBasedConditionFailures(RuntimeError):
    def __init__(self, failures: list[dict[str, object]]):
        self.failures = failures
        super().__init__(
            f"{len(failures)} word-based condition(s) were isolated; "
            "successful conditions were saved (continue-and-fail policy)"
        )


def ensure_directory(path: Path) -> None:
    reporting_module.ensure_directory(path)


def normalize_model_name(model: str) -> str:
    return model_inputs_module.normalize_model_name(model)


def build_result_dir(
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int | list[int] | tuple[int, ...],
    category: str,
    data_run: str = "default",
    vmf_variant: str | None = None,
) -> Path:
    return model_inputs_module.build_result_dir(
        model=model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        vmf_variant=vmf_variant,
    )


def build_baseline_param_dir(
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
) -> Path:
    return model_inputs_module.build_baseline_param_dir(
        model=model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
    )


def resolve_model_provenance(
    *,
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    embedding_variant: str | None = None,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
) -> dict[str, object]:
    return model_inputs_module.resolve_model_provenance(
        model=model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
        prior_scale=prior_scale,
        covariance_type=covariance_type,
        vmf_variant=vmf_variant,
    )


def _build_output_condition_id(
    *,
    model: str,
    dataset: str,
    data_run: str,
    category: str,
    iterations: list[int],
    num_topics: int,
    coherence: str,
    coherences: list[str] | None = None,
    coherence_topn: int,
    coherence_window_size: int | dict[str, int | None] | None,
    coherence_implementation: str | dict[str, str] | None,
    coherence_min_window_count: int | dict[str, int | None] | None,
    coherence_reference: str,
    coherence_reference_path: str | None,
    coherence_reference_format: str | None,
    coherence_reference_max_docs: int | None,
    coherence_reference_min_doc_tokens: int,
    coherence_reference_streaming: bool,
    diversity_topn: int,
    coherence_split: str,
    topic_word_source: str,
    embedding_variant: str | None,
    metric_names: list[str],
    dict_exclude_tokens: frozenset[str] = frozenset(),
    posterior_settings: dict[str, object] | None = None,
    topic_word_score_mode: str | None = None,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    source_condition_id: str | None = None,
    source_condition_fingerprint: str | None = None,
    parameter_variant: str | None = None,
    dict_no_above: float | None = None,
    reference_min_df: int = 0,
    reference_max_df_ratio: float = 1.0,
) -> tuple[str, str]:
    return reporting_module.build_output_condition_id(
        model=model,
        dataset=dataset,
        data_run=data_run,
        category=category,
        iterations=iterations,
        num_topics=num_topics,
        coherence=coherence,
        coherences=coherences,
        coherence_topn=coherence_topn,
        coherence_window_size=coherence_window_size,
        coherence_implementation=coherence_implementation,
        coherence_min_window_count=coherence_min_window_count,
        coherence_reference=coherence_reference,
        coherence_reference_path=coherence_reference_path,
        coherence_reference_format=coherence_reference_format,
        coherence_reference_max_docs=coherence_reference_max_docs,
        coherence_reference_min_doc_tokens=coherence_reference_min_doc_tokens,
        coherence_reference_streaming=coherence_reference_streaming,
        diversity_topn=diversity_topn,
        coherence_split=coherence_split,
        topic_word_source=topic_word_source,
        embedding_variant=embedding_variant,
        metric_names=metric_names,
        dict_exclude_tokens=dict_exclude_tokens,
        posterior_settings=posterior_settings,
        topic_word_score_mode=topic_word_score_mode,
        topic_word_ranking_schema_version=TOPIC_WORD_RANKING_SCHEMA_VERSION,
        prior_scale=prior_scale,
        covariance_type=covariance_type,
        source_condition_id=source_condition_id,
        source_condition_fingerprint=source_condition_fingerprint,
        parameter_variant=parameter_variant,
        dict_no_above=dict_no_above,
        reference_min_df=reference_min_df,
        reference_max_df_ratio=reference_max_df_ratio,
    )


def _posterior_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "posterior_num_chains": int(getattr(args, "posterior_num_chains", 1)),
        "posterior_burn_in_sweeps": int(getattr(args, "posterior_burn_in_sweeps", 20)),
        "posterior_retained_samples": int(
            getattr(args, "posterior_retained_samples", 20)
        ),
        "posterior_thinning": int(getattr(args, "posterior_thinning", 1)),
        "posterior_seed": int(getattr(args, "posterior_seed", 0)),
        "posterior_backend": str(getattr(args, "posterior_backend", "numba")),
        "etm_theta_samples": int(getattr(args, "etm_theta_samples", 100)),
        "etm_posterior_seed": int(getattr(args, "etm_posterior_seed", 0)),
        "npmi_min_expected_count": getattr(args, "npmi_min_expected_count", None),
    }


def _resolve_split_csvs_and_target_column(
    *,
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
) -> tuple[tuple[str, ...] | None, str]:
    return model_inputs_module.resolve_split_csvs_and_target_column(
        model=model,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        embedding_variant=embedding_variant,
        prior_scale=prior_scale,
        covariance_type=covariance_type,
        vmf_variant=vmf_variant,
    )


def _load_filtered_split_texts_from_csvs(
    *,
    dataset: str,
    csv_paths: tuple[str, ...],
    category: str,
    data_column: str = "data",
    target_column: str = "target_str",
    exclude_labels: set[str] | None = None,
) -> list[str]:
    return corpus_bundle_module.load_filtered_split_texts_from_csvs(
        dataset=dataset,
        csv_paths=csv_paths,
        category=category,
        data_column=data_column,
        target_column=target_column,
        exclude_labels=exclude_labels,
    )


def load_documents(
    dataset: str,
    category: str,
    split: str,
    exclude_labels: set[str] | None = None,
    split_csvs: tuple[str, ...] | None = None,
    target_column: str = "target_str",
) -> list[str]:
    return corpus_bundle_module.load_documents(
        dataset=dataset,
        category=category,
        split=split,
        exclude_labels=exclude_labels,
        split_csvs=split_csvs,
        target_column=target_column,
    )


def tokenize_documents(
    documents: list[str],
    min_token_len: int,
    language: str,
    delimiter: str | None = " / ",
    segmenter: str = "delimiter",
    tokenizer: str = "default",
    ja_replace_num: bool = True,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
) -> list[list[str]]:
    return corpus_bundle_module.tokenize_documents(
        documents=documents,
        min_token_len=min_token_len,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )


def tokenize_document_sentences(
    text: str,
    min_token_len: int,
    language: str,
    delimiter: str | None = " / ",
    segmenter: str = "delimiter",
    tokenizer: str = "default",
    ja_replace_num: bool = True,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
) -> list[list[str]]:
    return corpus_bundle_module.tokenize_document_sentences(
        text=text,
        min_token_len=min_token_len,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )


def tokenize_sentence_documents(
    documents: list[str],
    min_token_len: int,
    language: str,
    delimiter: str | None = " / ",
    segmenter: str = "delimiter",
    tokenizer: str = "default",
    ja_replace_num: bool = True,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
) -> list[list[list[str]]]:
    return corpus_bundle_module.tokenize_sentence_documents(
        documents=documents,
        min_token_len=min_token_len,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )


def build_sentence_bow_by_document(
    sentence_tokens_by_doc: list[list[list[str]]],
    dictionary: Dictionary,
) -> list[list[list[tuple[int, int]]]]:
    return corpus_bundle_module.build_sentence_bow_by_document(
        sentence_tokens_by_doc=sentence_tokens_by_doc,
        dictionary=dictionary,
    )


def build_corpus_bundle(
    dataset: str,
    category: str,
    split: str,
    min_token_len: int,
    language: str,
    delimiter: str | None = " / ",
    ja_replace_num: bool = True,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
    dict_no_below: int = 3,
    dict_no_above: float = 0.7,
    dict_exclude_tokens: frozenset[str] = frozenset(),
    dict_exclude_single_alpha: bool = False,
    dict_exclude_with_digit: bool = False,
    dict_exclude_hiragana_only: bool = False,
    exclude_labels: set[str] | None = None,
    split_csvs: tuple[str, ...] | None = None,
    target_column: str = "target_str",
    reference_document_frequencies: Mapping[str, int] | None = None,
    reference_num_docs: int | None = None,
    reference_min_df: int = 0,
    reference_max_df_ratio: float = 1.0,
) -> tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    return corpus_bundle_module.build_corpus_bundle(
        dataset=dataset,
        category=category,
        split=split,
        min_token_len=min_token_len,
        language=language,
        delimiter=delimiter,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
        dict_no_below=dict_no_below,
        dict_no_above=dict_no_above,
        dict_exclude_tokens=dict_exclude_tokens,
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
        exclude_labels=exclude_labels,
        split_csvs=split_csvs,
        target_column=target_column,
        reference_document_frequencies=reference_document_frequencies,
        reference_num_docs=reference_num_docs,
        reference_min_df=reference_min_df,
        reference_max_df_ratio=reference_max_df_ratio,
    )


def build_reference_corpus_bundle(
    *,
    path: Path,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
    dict_no_below: int = 3,
    dict_no_above: float = 0.7,
    dict_exclude_tokens: frozenset[str] = frozenset(),
    dict_exclude_single_alpha: bool = False,
    dict_exclude_with_digit: bool = False,
    dict_exclude_hiragana_only: bool = False,
) -> tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    return corpus_bundle_module.build_reference_corpus_bundle(
        path=path,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
        dict_no_below=dict_no_below,
        dict_no_above=dict_no_above,
        dict_exclude_tokens=dict_exclude_tokens,
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
    )


def load_doc_topics(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    prefer_soft: bool = False,
    embedding_variant: str | None = None,
    vmf_variant: str | None = None,
):
    return model_inputs_module.load_doc_topics(
        model=model,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        prefer_soft=prefer_soft,
        embedding_variant=embedding_variant,
        vmf_variant=vmf_variant,
    )


def aggregate_doc_topics_from_sentence_topics(
    sentence_topics_by_doc,
    num_topics: int,
):
    return model_inputs_module.aggregate_doc_topics_from_sentence_topics(
        sentence_topics_by_doc=sentence_topics_by_doc,
        num_topics=num_topics,
    )


def resolve_sentence_topics_path(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
    vmf_variant: str | None = None,
) -> Path:
    return model_inputs_module.resolve_sentence_topics_path(
        model=model,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        embedding_variant=embedding_variant,
        vmf_variant=vmf_variant,
    )


def resolve_preprocessed_corpus_path(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
) -> Path:
    return model_inputs_module.resolve_preprocessed_corpus_path(
        model=model,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        embedding_variant=embedding_variant,
    )


def round_sigfigs(value: float, sig: int = 4) -> float:
    return reporting_module.round_sigfigs(value=value, sig=sig)


def parse_args() -> argparse.Namespace:
    return cli_module.parse_args()


def _requested_topic_word_topn(args: argparse.Namespace) -> int:
    return max(int(args.coherence_topn), int(args.diversity_topn))


def _dict_exclude_tokens(args: argparse.Namespace) -> frozenset[str]:
    raw = getattr(args, "dict_exclude_tokens", frozenset())
    if isinstance(raw, str):
        return frozenset(token.strip() for token in raw.split(",") if token.strip())
    return frozenset(str(token) for token in raw)


def _reference_min_df(args: argparse.Namespace) -> int:
    return int(getattr(args, "reference_min_df", 0) or 0)


def _reference_max_df_ratio(args: argparse.Namespace) -> float:
    value = getattr(args, "reference_max_df_ratio", 1.0)
    return 1.0 if value is None else float(value)


def _reference_band_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Arguments that restrict V_eval by reference-corpus document frequency.

    Returns an empty mapping when the band is inactive so the (expensive)
    document frequency table is never built for default runs.
    """

    if not _restricts_evaluation_vocabulary(args):
        return {}
    reference_path = getattr(args, "coherence_reference_path", None)
    if reference_path is None:
        raise ValueError(
            "--coherence_reference_path is required when --reference-min-df or "
            "--reference-max-df-ratio restricts the evaluation vocabulary"
        )
    table = load_reference_document_frequencies(
        resolve_project_path(reference_path),
        max_docs=getattr(args, "coherence_reference_max_docs", None),
        min_doc_tokens=int(getattr(args, "coherence_reference_min_doc_tokens", 1)),
        cache_root=Path(args.out_root) / ".cache",
    )
    return {
        "reference_document_frequencies": table.document_frequencies,
        "reference_num_docs": table.num_docs,
        "reference_min_df": _reference_min_df(args),
        "reference_max_df_ratio": _reference_max_df_ratio(args),
    }


def _restricts_evaluation_vocabulary(args: argparse.Namespace) -> bool:
    return _reference_min_df(args) > 0 or _reference_max_df_ratio(args) < 1.0


def _isolates_condition_failures(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "condition_failure_policy", "exclude-condition"))
        != "fail-fast"
    )


def _raises_after_isolated_failures(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "condition_failure_policy", "exclude-condition"))
        == "continue-and-fail"
    )


def _checkpoint_root(args: argparse.Namespace) -> Path:
    configured = getattr(args, "checkpoint_root", None)
    return (
        resolve_project_path(Path(configured))
        if configured is not None
        else args.out_root / ".checkpoints"
    )


def _topic_word_checkpoint_identity(
    *,
    args: argparse.Namespace,
    model: str,
    data_run: str,
    category: str,
    iteration: int,
    vocabulary_fingerprint: str,
) -> dict[str, object]:
    provenance = resolve_model_provenance(
        model=model,  # type: ignore[arg-type]
        dataset=args.dataset,
        iteration=iteration,
        num_topics=args.num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=_effective_embedding_variant_for_model(model, args),
        prior_scale=_effective_prior_scale_for_model(model, args),
        covariance_type=_effective_covariance_type_for_model(model, args),
        vmf_variant=_effective_vmf_variant_for_model(model, args),
    )
    identity = {
        "schema_version": 1,
        "topic_word_ranking_schema_version": TOPIC_WORD_RANKING_SCHEMA_VERSION,
        "dataset": args.dataset,
        "data_run": data_run,
        "model": model,
        "category": category,
        "num_topics": int(args.num_topics),
        "iteration": int(iteration),
        "embedding_variant": _effective_embedding_variant_for_model(model, args),
        "prior_scale": _effective_prior_scale_for_model(model, args),
        "covariance_type": _effective_covariance_type_for_model(model, args),
        "vmf_variant": _effective_vmf_variant_for_model(model, args),
        "split": args.coherence_split,
        "topic_word_topn": _requested_topic_word_topn(args),
        "language": args.language,
        "delimiter": args.delimiter,
        "min_token_len": int(args.coherence_min_token_len),
        "dict_no_below": int(args.dict_no_below),
        "dict_no_above": float(args.dict_no_above),
        "dict_exclude_tokens": sorted(_dict_exclude_tokens(args)),
        "dict_exclude_single_alpha": bool(args.dict_exclude_single_alpha),
        "dict_exclude_with_digit": bool(args.dict_exclude_with_digit),
        "dict_exclude_hiragana_only": bool(args.dict_exclude_hiragana_only),
        "ja_replace_num": bool(args.ja_replace_num),
        "posterior_settings": _posterior_settings(args),
        "etm_theta_samples": int(getattr(args, "etm_theta_samples", 100)),
        "etm_posterior_seed": int(getattr(args, "etm_posterior_seed", 0)),
        "npmi_min_expected_count": getattr(args, "npmi_min_expected_count", None),
        "vocabulary_fingerprint": vocabulary_fingerprint,
        "model_provenance": provenance,
    }
    if model == "ctm":
        identity["ctm_responsibility_schema_version"] = 1
    return identity


def _condition_failure_payload(
    *,
    task: PendingWordBasedGroupTask,
    args: argparse.Namespace,
    exc: ConditionEvaluationError,
    stage: str,
    iteration: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "failed",
        "stage": stage,
        "dataset": args.dataset,
        "data_run": task.data_run,
        "model": task.model,
        "category": task.category,
        "num_topics": int(task.num_topics),
        "iterations": [int(value) for value in args.iteration],
        "embedding_variant": _effective_embedding_variant_for_model(task.model, args),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "retryable": False,
    }
    if iteration is not None:
        payload["iteration"] = int(iteration)
    for attribute in (
        "topic_id",
        "topic_ids",
        "eligible_word_count",
        "required_topn",
        "reason",
        "eligible_word_counts",
    ):
        if hasattr(exc, attribute):
            value = getattr(exc, attribute)
            payload[attribute] = value.tolist() if hasattr(value, "tolist") else value
    if hasattr(exc, "eligible_words"):
        eligible_words = getattr(exc, "eligible_words")
        if isinstance(eligible_words, list):
            payload["special_words"] = list(eligible_words)
    if type(exc).__name__ == "DegenerateTopicError":
        payload["status"] = "insufficient_topic_words"
    return payload


def _persist_partial_topic_words(
    *,
    args: argparse.Namespace,
    task: PendingWordBasedGroupTask,
    iteration: int,
    runtime: RuntimeTopicWords,
    checkpoint_identity: dict[str, object],
) -> Path:
    """Persist qualitative topic words without publishing quantitative metrics."""

    if not runtime.empty_topic_ids:
        raise ValueError("partial topic-word output requires at least one empty topic")
    started_at = datetime.now(UTC).isoformat()
    execution_id = build_execution_id(prefix="exec", started_at=started_at)
    identity_fingerprint = fingerprint_payload(checkpoint_identity)
    display_key = (
        f"{task.model}__k{int(task.num_topics)}__iter{int(iteration)}"
        f"__partial__{identity_fingerprint}"
    )
    partial_root = args.out_root / "partial"
    archive_dir = build_archive_result_dir(
        base_root=partial_root,
        dataset=args.dataset,
        data_run=task.data_run,
        category=task.category,
        display_key=display_key,
        started_at=started_at,
        execution_id=execution_id,
    )
    ensure_directory(archive_dir)
    provenance = checkpoint_identity.get("model_provenance")
    if not isinstance(provenance, dict):
        provenance = resolve_model_provenance(
            model=task.model,  # type: ignore[arg-type]
            dataset=args.dataset,
            iteration=int(iteration),
            num_topics=int(task.num_topics),
            category=task.category,
            data_run=task.data_run,
            embedding_variant=_effective_embedding_variant_for_model(task.model, args),
            prior_scale=_effective_prior_scale_for_model(task.model, args),
            covariance_type=_effective_covariance_type_for_model(task.model, args),
            vmf_variant=_effective_vmf_variant_for_model(task.model, args),
        )
    empty_topic_ids = [int(topic_id) for topic_id in runtime.empty_topic_ids]
    common_meta: dict[str, object] = {
        "task": "word_based_topic_words_partial",
        "status": "partial",
        "metrics_status": "excluded",
        "failure_type": "EmptyTopicError",
        "empty_topic_ids": empty_topic_ids,
        "num_nonempty_topics": int(task.num_topics) - len(empty_topic_ids),
        "dataset": args.dataset,
        "data_run": task.data_run,
        "category": task.category,
        "model": task.model,
        "num_topics": int(task.num_topics),
        "iteration": int(iteration),
        "iterations": [int(iteration)],
        "split": args.coherence_split,
        "condition_id": display_key,
        "display_key": display_key,
        "condition_fingerprint": identity_fingerprint,
        "model_provenance": provenance,
        "evaluation_vocabulary_fingerprint": runtime.vocabulary_fingerprint,
        "source_condition_fingerprint": runtime.source_condition_fingerprint,
        "started_at": started_at,
        "execution_id": execution_id,
    }
    display_path = archive_dir / "topic_words_display_topk.json"
    probability_path = archive_dir / "topic_words_probability_topk.json"
    rows_display = [
        {
            "iteration": int(iteration),
            "topics": serialize_topic_words(runtime.display_topic_words),
        }
    ]
    rows_probability = [
        {
            "iteration": int(iteration),
            "topics": serialize_topic_words(runtime.evaluation.topic_words),
        }
    ]
    write_evaluation_json(
        meta={
            **common_meta,
            "topic_word_role": "display",
            "source": runtime.display_source,
            "score_mode": runtime.display_score_mode,
        },
        results={"per_iteration": rows_display},
        path=display_path,
    )
    write_evaluation_json(
        meta={
            **common_meta,
            "topic_word_role": "diagnostic_probability",
            "source": runtime.evaluation.topic_word_source,
            "score_mode": runtime.evaluation.score_mode,
            "score_definition": runtime.evaluation.score_definition,
        },
        results={"per_iteration": rows_probability},
        path=probability_path,
    )
    artifacts: dict[str, str] = {
        "topic_words_display_topk": display_path.name,
        "topic_words_probability_topk": probability_path.name,
    }
    if runtime.expected_counts is not None:
        # Name the split the counts actually came from; the complete writer uses
        # the same f"topic_word_{split}_..." form.
        split = str(args.coherence_split)
        expected_counts_path = archive_dir / f"topic_word_{split}_expected_counts.pkl"
        save_pickle(runtime.expected_counts, expected_counts_path)
        artifacts[f"topic_word_{split}_expected_counts"] = expected_counts_path.name
    metadata_path = archive_dir / "metadata.json"
    save_json(common_meta, metadata_path)
    artifacts["metadata"] = metadata_path.name
    pointer_path = write_latest_result_pointer(
        base_root=partial_root,
        task="word_based_topic_words_partial",
        dataset=args.dataset,
        data_run=task.data_run,
        category=task.category,
        display_key=display_key,
        archive_dir=archive_dir,
        started_at=started_at,
        execution_id=execution_id,
        condition_fingerprint=identity_fingerprint,
        artifacts=artifacts,
    )
    return pointer_path


def _record_condition_failure(
    *,
    args: argparse.Namespace,
    task: PendingWordBasedGroupTask,
    exc: ConditionEvaluationError,
    stage: str,
    failures: list[dict[str, object]],
    iteration: int | None = None,
) -> dict[str, object]:
    failure = _condition_failure_payload(
        task=task,
        args=args,
        exc=exc,
        stage=stage,
        iteration=iteration,
    )
    failures.append(failure)
    save_failure_record(
        checkpoint_root=_checkpoint_root(args),
        identity=failure,
        payload=failure,
    )
    logger.error("word_based condition excluded: %s", failure)
    return failure


def _requested_coherences(args: argparse.Namespace) -> list[str]:
    coherences = normalize_coherences(getattr(args, "coherence", "c_v"))
    if len(coherences) > 1:
        unsupported = [
            coherence
            for coherence in coherences
            if coherence not in MULTI_COHERENCE_CHOICES
        ]
        if unsupported:
            raise ValueError(
                "Multiple coherence metrics currently support only "
                f"{list(MULTI_COHERENCE_CHOICES)}, got {unsupported!r}."
            )
    return coherences


def _requested_num_topics_values(args: argparse.Namespace) -> list[int]:
    raw = getattr(args, "num_topics")
    if isinstance(raw, (list, tuple, set)):
        values = [int(value) for value in raw]
    else:
        values = [int(raw)]
    if not values:
        raise ValueError("num_topics must not be empty.")
    if any(value < 1 for value in values):
        raise ValueError(f"num_topics values must be >= 1, got {values!r}")
    return values


def _primary_coherence(coherences: list[str]) -> str:
    return coherences[0]


def _uses_multiple_coherences(coherences: list[str]) -> bool:
    return len(coherences) > 1


def _metric_names_for_coherences(coherences: list[str]) -> list[str]:
    multiple = _uses_multiple_coherences(coherences)
    return [
        coherence_metric_key(coherence, multiple=multiple) for coherence in coherences
    ] + ["diversity"]


def _uses_mvtm_fixed_k_policy(args: argparse.Namespace, *, model: str) -> bool:
    return (
        model == "mvtm"
        and str(getattr(args, "mvtm_empty_topic_policy", "exclude")) == "fixed-k"
    )


def _uses_fixed_k_reporting(args: argparse.Namespace) -> bool:
    return str(getattr(args, "mvtm_empty_topic_policy", "exclude")) == "fixed-k"


def _fixed_k_metric_names(coherences: list[str]) -> list[str]:
    base_names = _metric_names_for_coherences(coherences)
    coherence_names = base_names[:-1]
    return [
        *base_names,
        *(f"{name}_active_only" for name in coherence_names),
        "diversity_active_only",
        "topic_utilization",
        "num_active_topics",
        "num_empty_topics",
        "complete_run_rate",
    ]


def _apply_fixed_k_reporting(
    *,
    metrics: dict[str, float],
    topic_words: TopicWords,
    coherences: list[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    return apply_fixed_k_empty_topic_policy(
        active_metrics=metrics,
        topic_words=topic_words,
        coherences=coherences,
        diversity_topn=int(args.diversity_topn),
    )


def _empty_topic_policy_meta(
    *, args: argparse.Namespace, model: str
) -> dict[str, object]:
    requested_policy = str(getattr(args, "mvtm_empty_topic_policy", "exclude"))
    applied_policy = (
        "fixed-k" if _uses_mvtm_fixed_k_policy(args, model=model) else "exclude"
    )
    return {
        "requested_mvtm_policy": requested_policy,
        "applied_policy": applied_policy,
        "scope": "mvtm",
        "standard_metric_keys": (
            "fixed_k_for_c_v_c_npmi_doc_npmi; active_only_for_unbounded_coherence"
            if requested_policy == "fixed-k"
            else "complete_topics_only"
        ),
        "empty_topic_values": {
            "c_v": 0.0,
            "c_npmi": -1.0,
            "doc_npmi": -1.0,
            "c_uci": "active_only_unbounded",
            "u_mass": "active_only_unbounded",
        },
        "diversity_denominator": (
            "requested_num_topics_times_diversity_topn"
            if requested_policy == "fixed-k"
            else "emitted_topic_word_slots"
        ),
    }


def _coherence_from_metric_name(
    metric_name: str,
    *,
    coherences: list[str],
) -> str | None:
    if metric_name == "coherence":
        return _primary_coherence(coherences)
    prefix = "coherence_"
    if metric_name.startswith(prefix):
        candidate = metric_name[len(prefix) :]
        if candidate in coherences:
            return candidate
    return None


def _effective_coherence_window_sizes(
    coherences: list[str],
    requested: int | None,
) -> dict[str, int | None]:
    return {
        coherence: _effective_coherence_window_size(coherence, requested)
        for coherence in coherences
    }


def _coherence_window_size_sources(
    coherences: list[str],
    requested: int | None,
) -> dict[str, str]:
    return {
        coherence: _coherence_window_size_source(coherence, requested)
        for coherence in coherences
    }


def _effective_coherence_min_window_counts(
    coherences: list[str],
    requested: int | None,
) -> dict[str, int | None]:
    return {
        coherence: _effective_coherence_min_window_count(coherence, requested)
        for coherence in coherences
    }


def _coherence_implementations(coherences: list[str]) -> dict[str, str]:
    return {coherence: _coherence_implementation(coherence) for coherence in coherences}


def _single_coherence_meta(
    *,
    coherence: str,
    args: argparse.Namespace,
    model: str,
    topic_word_source: str,
    topic_word_score_mode: str,
    topic_word_score_definition: str,
    reference_meta: dict[str, object],
    window_size: int | None,
    window_size_source: str,
    min_window_count: int | None,
) -> dict[str, object]:
    details = describe_coherence_metric(coherence)
    return {
        "metric": coherence,
        "implementation": _coherence_implementation(coherence),
        "definition": details["definition"],
        "cooccurrence_unit": details["cooccurrence_unit"],
        "zero_cooccurrence_policy": details["zero_cooccurrence_policy"],
        "pmi_smoothing_epsilon": details.get("pmi_smoothing_epsilon"),
        "probability_estimation": (
            "boolean_sliding_window" if coherence == "c_v" else None
        ),
        "confirmation_measure": (
            "normalized_log_ratio_npmi" if coherence == "c_v" else None
        ),
        "vector_space": "top_word_npmi" if coherence == "c_v" else None,
        "segmentation": "one_set" if coherence == "c_v" else None,
        "similarity": "cosine" if coherence == "c_v" else None,
        "aggregation": "arithmetic_mean" if coherence == "c_v" else None,
        "undefined_npmi_policy": "zero" if coherence == "c_v" else None,
        "zero_vector_similarity_policy": "zero" if coherence == "c_v" else None,
        **reference_meta,
        "coherence_window_size": window_size,
        "coherence_window_size_source": window_size_source,
        "coherence_min_window_count": min_window_count,
        "topn": int(args.coherence_topn),
        "split": args.coherence_split,
        "min_token_len": int(args.coherence_min_token_len),
        "dict_no_below": int(args.dict_no_below),
        "dict_no_above": float(args.dict_no_above),
        "dict_exclude_tokens": sorted(_dict_exclude_tokens(args)),
        "dict_exclude_single_alpha": bool(args.dict_exclude_single_alpha),
        "dict_exclude_with_digit": bool(args.dict_exclude_with_digit),
        "dict_exclude_hiragana_only": bool(args.dict_exclude_hiragana_only),
        "reference_min_df": _reference_min_df(args),
        "reference_max_df_ratio": _reference_max_df_ratio(args),
        "language": args.language,
        "delimiter": args.delimiter,
        "ja_replace_num": bool(args.ja_replace_num),
        "ja_dicdir": args.ja_dicdir,
        "ja_require_unidic": bool(args.ja_require_unidic),
        "topic_word_score_mode": topic_word_score_mode,
        "topic_word_score_definition": topic_word_score_definition,
        "topic_word_source": topic_word_source,
        "gaussian_word2vec": args.gaussian_word2vec,
    }


def _coherence_meta(
    *,
    coherences: list[str],
    args: argparse.Namespace,
    model: str,
    topic_word_source: str,
    topic_word_score_mode: str,
    topic_word_score_definition: str,
    reference_meta: dict[str, object],
    window_sizes: dict[str, int | None],
    window_size_sources: dict[str, str],
    min_window_counts: dict[str, int | None],
) -> dict[str, object]:
    by_metric = {
        coherence: _single_coherence_meta(
            coherence=coherence,
            args=args,
            model=model,
            topic_word_source=topic_word_source,
            topic_word_score_mode=topic_word_score_mode,
            topic_word_score_definition=topic_word_score_definition,
            reference_meta=reference_meta,
            window_size=window_sizes[coherence],
            window_size_source=window_size_sources[coherence],
            min_window_count=min_window_counts[coherence],
        )
        for coherence in coherences
    }
    if len(coherences) == 1:
        return by_metric[coherences[0]]
    return {
        "metrics": list(coherences),
        "primary_metric": _primary_coherence(coherences),
        "by_metric": by_metric,
        **reference_meta,
        "topn": int(args.coherence_topn),
        "split": args.coherence_split,
        "min_token_len": int(args.coherence_min_token_len),
    }


def _requested_embedding_variant(args: argparse.Namespace) -> str | None:
    value = getattr(args, "embedding_variant", DEFAULT_EMBEDDING_VARIANT)
    if value is None:
        return None
    variant = str(value).strip()
    return variant or None


def _effective_embedding_variant_for_model(
    model: str,
    args: argparse.Namespace,
) -> str | None:
    return model_inputs_module.effective_embedding_variant(
        model,
        _requested_embedding_variant(args),
    )


def _effective_vmf_variant_for_model(
    model: str,
    args: argparse.Namespace,
) -> str | None:
    """Requested hyperparameter-sweep label of the vMF runs (None = default runs)."""
    if model not in {"vmf", "vmf_sentence_lda"}:
        return None
    return normalize_vmf_parameter_variant(getattr(args, "vmf_variant", None))


def _effective_prior_scale_for_model(
    model: str,
    args: argparse.Namespace,
) -> float | None:
    if model not in {"gaussianlda", "sentence_gaussianlda", "gaussian"}:
        return None
    value = getattr(args, "prior_scale", None)
    return None if value is None else float(value)


def _effective_covariance_type_for_model(
    model: str,
    args: argparse.Namespace,
) -> str | None:
    """Requested covariance type of the sentence Gaussian LDA (None = full / unset)."""
    if model not in {"sentence_gaussianlda", "gaussian"}:
        return None
    value = getattr(args, "covariance_type", None)
    if value is None:
        return None
    return normalize_covariance_type(value)


def _uses_default_output_layout(out_root: Path) -> bool:
    return resolve_project_path(out_root) == DEFAULT_OUT_ROOT


def _expected_topic_word_identity(
    *,
    model: str,
    args: argparse.Namespace,
) -> str:
    if _topic_word_score_mode(args) == "word_topic_npmi":
        if model in COLLAPSED_MODELS:
            return "posthoc_word_topic_npmi"
        if model == "etm":
            return "variational_word_topic_npmi"
        if model == "ctm":
            return "ctm_variational_word_topic_npmi"
        if model in {"sam", "sam_tf"}:
            return "sam_positive_part_word_topic_npmi"
    if model in COLLAPSED_MODELS:
        return "posthoc_expected_p_w_given_topic"
    if model == "etm":
        return "native_etm_beta"
    if model == "ctm":
        return "native_ctm_decoder_topic_word_distribution"
    if model in {"sam", "sam_tf"}:
        return "native_sam_signed_topic_direction"
    raise ValueError(f"Unsupported model for coherence analysis: {model}")


def _expected_output_condition_id(
    *,
    model: str,
    data_run: str,
    category: str,
    args: argparse.Namespace,
    coherences: list[str],
    coherence_window_sizes: dict[str, int | None],
    coherence_implementations: dict[str, str],
    coherence_min_window_counts: dict[str, int | None],
    metric_names: list[str],
) -> tuple[str, str]:
    topic_word_source = _expected_topic_word_identity(model=model, args=args)
    provenance = resolve_model_provenance(
        model=model,  # type: ignore[arg-type]
        dataset=args.dataset,
        iteration=int(min(args.iteration)),
        num_topics=args.num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=_effective_embedding_variant_for_model(model, args),
        prior_scale=_effective_prior_scale_for_model(model, args),
        covariance_type=_effective_covariance_type_for_model(model, args),
        vmf_variant=_effective_vmf_variant_for_model(model, args),
    )
    return _build_output_condition_id(
        model=model,
        dataset=args.dataset,
        data_run=data_run,
        category=category,
        iterations=list(args.iteration),
        num_topics=args.num_topics,
        coherence=_primary_coherence(coherences),
        coherences=coherences if _uses_multiple_coherences(coherences) else None,
        coherence_topn=args.coherence_topn,
        coherence_window_size=(
            coherence_window_sizes
            if _uses_multiple_coherences(coherences)
            else coherence_window_sizes[_primary_coherence(coherences)]
        ),
        coherence_implementation=(
            coherence_implementations
            if _uses_multiple_coherences(coherences)
            else coherence_implementations[_primary_coherence(coherences)]
        ),
        coherence_min_window_count=(
            coherence_min_window_counts
            if _uses_multiple_coherences(coherences)
            else coherence_min_window_counts[_primary_coherence(coherences)]
        ),
        coherence_reference=args.coherence_reference,
        coherence_reference_path=(
            None
            if args.coherence_reference_path is None
            else str(resolve_project_path(args.coherence_reference_path))
        ),
        coherence_reference_format=(
            args.coherence_reference_format
            if args.coherence_reference == "wikipedia"
            else None
        ),
        coherence_reference_max_docs=args.coherence_reference_max_docs,
        coherence_reference_min_doc_tokens=args.coherence_reference_min_doc_tokens,
        coherence_reference_streaming=_expected_reference_streaming_flag(
            args=args,
            coherences=coherences,
        ),
        diversity_topn=args.diversity_topn,
        coherence_split=args.coherence_split,
        topic_word_source=topic_word_source,
        embedding_variant=_effective_embedding_variant_for_model(model, args),
        prior_scale=_effective_prior_scale_for_model(model, args),
        covariance_type=_effective_covariance_type_for_model(model, args),
        source_condition_id=(
            None
            if provenance.get("condition_id") is None
            else str(provenance["condition_id"])
        ),
        source_condition_fingerprint=(
            None
            if provenance.get("condition_fingerprint") is None
            else str(provenance["condition_fingerprint"])
        ),
        parameter_variant=(
            None
            if provenance.get("parameter_variant") is None
            else str(provenance["parameter_variant"])
        ),
        metric_names=metric_names,
        dict_exclude_tokens=_dict_exclude_tokens(args),
        posterior_settings=_posterior_settings(args),
        topic_word_score_mode=_topic_word_score_mode(args),
        dict_no_above=float(args.dict_no_above),
        reference_min_df=_reference_min_df(args),
        reference_max_df_ratio=_reference_max_df_ratio(args),
    )


def _expected_output_path(
    *,
    model: str,
    data_run: str,
    category: str,
    args: argparse.Namespace,
    coherences: list[str],
    coherence_window_sizes: dict[str, int | None],
    coherence_implementations: dict[str, str],
    coherence_min_window_counts: dict[str, int | None],
    metric_names: list[str],
) -> Path:
    condition_id, _condition_fingerprint = _expected_output_condition_id(
        model=model,
        data_run=data_run,
        category=category,
        args=args,
        coherences=coherences,
        coherence_window_sizes=coherence_window_sizes,
        coherence_implementations=coherence_implementations,
        coherence_min_window_counts=coherence_min_window_counts,
        metric_names=metric_names,
    )
    if _uses_default_output_layout(args.out_root):
        return (
            build_latest_result_dir(
                base_root=args.out_root,
                dataset=args.dataset,
                data_run=data_run,
                category=category,
                display_key=condition_id,
            )
            / "CURRENT.json"
        )
    return (
        args.out_root
        / args.dataset
        / data_run
        / category
        / condition_id
        / "metrics_agg.json"
    )


def _should_skip_existing_output(
    *,
    model: str,
    data_run: str,
    category: str,
    args: argparse.Namespace,
    coherences: list[str],
    coherence_window_sizes: dict[str, int | None],
    coherence_implementations: dict[str, str],
    coherence_min_window_counts: dict[str, int | None],
    metric_names: list[str],
) -> bool:
    if not bool(getattr(args, "skip_existing", False)):
        return False
    output_path = _expected_output_path(
        model=model,
        data_run=data_run,
        category=category,
        args=args,
        coherences=coherences,
        coherence_window_sizes=coherence_window_sizes,
        coherence_implementations=coherence_implementations,
        coherence_min_window_counts=coherence_min_window_counts,
        metric_names=metric_names,
    )
    complete = False
    if output_path.name == "CURRENT.json":
        complete = output_path.exists()
    else:
        completion_path = output_path.parent / "COMPLETE.json"
        if completion_path.exists():
            try:
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                _condition_id, expected_fingerprint = _expected_output_condition_id(
                    model=model,
                    data_run=data_run,
                    category=category,
                    args=args,
                    coherences=coherences,
                    coherence_window_sizes=coherence_window_sizes,
                    coherence_implementations=coherence_implementations,
                    coherence_min_window_counts=coherence_min_window_counts,
                    metric_names=metric_names,
                )
                required = [
                    output_path,
                    output_path.parent / "metadata.json",
                    output_path.parent / "topic_words_evaluation_topk.json",
                    output_path.parent / "topic_words_display_topk.json",
                    output_path.parent / "topic_words_probability_topk.json",
                ]
                complete = completion.get(
                    "condition_fingerprint"
                ) == expected_fingerprint and all(path.exists() for path in required)
            except (OSError, ValueError, TypeError):
                complete = False
        elif output_path.exists():
            # Backward-compatible recognition of complete pre-marker outputs.
            required = [
                output_path,
                output_path.parent / "metadata.json",
                output_path.parent / "topic_words_evaluation_topk.json",
                output_path.parent / "topic_words_display_topk.json",
                output_path.parent / "topic_words_probability_topk.json",
            ]
            complete = all(path.exists() for path in required)
    if complete:
        logger.info(
            "word_based_metrics skipping existing output data_run=%s model=%s "
            "category=%s num_topics=%s iterations=%s path=%s",
            data_run,
            model,
            category,
            args.num_topics,
            list(args.iteration),
            output_path,
        )
        return True
    return False


def _effective_coherence_window_size(
    coherence: str, requested: int | None
) -> int | None:
    if requested is not None:
        return int(requested)
    defaults = {
        "c_v": 110,
        "c_uci": 10,
        "c_npmi": 10,
    }
    return defaults.get(coherence)


def _coherence_window_size_source(coherence: str, requested: int | None) -> str:
    if requested is not None:
        return "user"
    if _effective_coherence_window_size(coherence, requested) is None:
        return "not_applicable"
    if coherence == "c_v":
        return "palmetto_compatible_default_c_v"
    if coherence in EPSILON_SMOOTHED_COHERENCES:
        return f"epsilon_smoothed_default_{coherence}"
    return f"gensim_default_{coherence}"


def _coherence_implementation(coherence: str) -> str:
    if coherence == "c_v":
        return PALMETTO_CV_IMPLEMENTATION
    if coherence in EPSILON_SMOOTHED_COHERENCES:
        return "project_epsilon_smoothed"
    if coherence == "doc_npmi":
        return "project_doc_npmi"
    return "gensim"


def _effective_coherence_min_window_count(
    coherence: str,
    requested: int | None,
) -> int | None:
    if coherence != "c_v":
        return None
    if requested is None:
        return DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT
    return int(requested)


def _validate_reference_args(args: argparse.Namespace) -> None:
    if getattr(args, "prior_scale", None) is not None:
        format_prior_scale_variant(float(args.prior_scale))
    if getattr(args, "covariance_type", None) is not None:
        normalize_covariance_type(args.covariance_type)
    normalize_vmf_parameter_variant(getattr(args, "vmf_variant", None))
    _coherence_count_backend(args)
    _coherence_count_workers(args)
    _coherence_count_chunk_size(args)
    _coherence_topic_word_workers(args)
    _coherence_score_workers(args)
    reference = str(getattr(args, "coherence_reference", "dataset"))
    coherences = _requested_coherences(args)
    if reference not in {"dataset", "wikipedia"}:
        raise ValueError(
            "coherence_reference must be one of: 'dataset', 'wikipedia' "
            f"(got {reference!r})"
        )
    if getattr(args, "coherence_window_size", None) is not None:
        if int(args.coherence_window_size) < 1:
            raise ValueError(
                f"coherence_window_size must be >= 1, got {args.coherence_window_size}"
            )
    if getattr(args, "coherence_min_window_count", None) is not None:
        if int(args.coherence_min_window_count) < 1:
            raise ValueError(
                "coherence_min_window_count must be >= 1 when provided, "
                f"got {args.coherence_min_window_count}"
            )
    if getattr(args, "coherence_reference_max_docs", None) is not None:
        if int(args.coherence_reference_max_docs) < 1:
            raise ValueError(
                "coherence_reference_max_docs must be >= 1 when provided, "
                f"got {args.coherence_reference_max_docs}"
            )
    if int(getattr(args, "coherence_reference_min_doc_tokens", 1)) < 1:
        raise ValueError(
            "coherence_reference_min_doc_tokens must be >= 1, "
            f"got {args.coherence_reference_min_doc_tokens}"
        )
    if reference == "wikipedia":
        if getattr(args, "coherence_reference_path", None) is None:
            raise ValueError(
                "coherence_reference_path is required when coherence_reference='wikipedia'."
            )
        if str(getattr(args, "coherence_reference_format", "tokenized_jsonl")) != (
            "tokenized_jsonl"
        ):
            raise ValueError("Only tokenized_jsonl reference corpora are supported.")
        if str(getattr(args, "language", "english")).lower() not in {"english", "en"}:
            raise ValueError(
                "Wikipedia-reference coherence is currently supported only for "
                "English tokenization."
            )
        unsupported_streaming = [
            coherence
            for coherence in coherences
            if coherence not in STREAMING_REFERENCE_COHERENCES
        ]
        if _uses_streaming_reference(args) and unsupported_streaming:
            raise ValueError(
                "Streaming Wikipedia-reference coherence supports only "
                f"{sorted(STREAMING_REFERENCE_COHERENCES)}, "
                f"got {unsupported_streaming!r}."
            )


def _uses_streaming_reference(args: argparse.Namespace) -> bool:
    return str(getattr(args, "coherence_reference", "dataset")) == "wikipedia" and (
        bool(getattr(args, "coherence_reference_streaming", False))
        or getattr(args, "coherence_reference_max_docs", None) is None
    )


def _coherence_count_backend(args: argparse.Namespace) -> ReferenceCountBackend:
    backend = str(getattr(args, "coherence_count_backend", "numba")).strip()
    if backend not in {"python", "numba", "numba_interval"}:
        raise ValueError(
            "coherence_count_backend must be one of "
            "{'python', 'numba', 'numba_interval'}, "
            f"got {backend!r}."
        )
    return backend  # type: ignore[return-value]


def _coherence_count_workers(args: argparse.Namespace) -> int:
    workers = int(
        getattr(args, "coherence_count_workers", DEFAULT_REFERENCE_COUNT_WORKERS)
    )
    if workers < 1:
        raise ValueError(f"coherence_count_workers must be >= 1, got {workers}")
    return workers


def _coherence_count_chunk_size(args: argparse.Namespace) -> int:
    chunk_size = int(
        getattr(args, "coherence_count_chunk_size", DEFAULT_REFERENCE_COUNT_CHUNK_SIZE)
    )
    if chunk_size < 1:
        raise ValueError(f"coherence_count_chunk_size must be >= 1, got {chunk_size}")
    return chunk_size


def _reference_count_max_pending(args: argparse.Namespace) -> int | None:
    value = getattr(args, "reference_count_max_pending", None)
    if value is None:
        return None
    max_pending = int(value)
    if max_pending < 1:
        raise ValueError(f"reference_count_max_pending must be >= 1, got {max_pending}")
    return max_pending


def _coherence_topic_word_workers(args: argparse.Namespace) -> int:
    workers = int(getattr(args, "coherence_topic_word_workers", 1))
    if workers < 1:
        raise ValueError(f"coherence_topic_word_workers must be >= 1, got {workers}")
    return workers


def _coherence_score_workers(args: argparse.Namespace) -> int:
    workers = int(getattr(args, "coherence_score_workers", 1))
    if workers < 1:
        raise ValueError(f"coherence_score_workers must be >= 1, got {workers}")
    return workers


def _uses_shared_reference_counts(
    *,
    args: argparse.Namespace,
    coherences: list[str],
) -> bool:
    return (
        str(getattr(args, "coherence_reference", "dataset")) == "wikipedia"
        and getattr(args, "coherence_reference_path", None) is not None
        and all(coherence in STREAMING_REFERENCE_COHERENCES for coherence in coherences)
    )


def _expected_reference_streaming_flag(
    *,
    args: argparse.Namespace,
    coherences: list[str],
) -> bool:
    if _uses_shared_reference_counts(args=args, coherences=coherences):
        return True
    return _uses_streaming_reference(args)


def _get_corpus_bundle_cached(
    *,
    cache: dict[
        tuple[
            str,
            str,
            str,
            str,
            int,
            str,
            str | None,
            bool,
            str | None,
            bool,
            int,
            float,
            frozenset[str],
            bool,
            bool,
            bool,
            tuple[str, ...] | None,
            tuple[str, ...] | None,
            str,
        ],
        tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]],
    ],
    dataset: str,
    data_run: str,
    category: str,
    split: str,
    min_token_len: int,
    language: str,
    delimiter: str | None,
    ja_replace_num: bool,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
    dict_no_below: int,
    dict_no_above: float,
    dict_exclude_tokens: frozenset[str],
    dict_exclude_single_alpha: bool,
    dict_exclude_with_digit: bool,
    dict_exclude_hiragana_only: bool,
    exclude_labels: set[str] | None = None,
    split_csvs: tuple[str, ...] | None = None,
    target_column: str = "target_str",
    reference_document_frequencies: Mapping[str, int] | None = None,
    reference_num_docs: int | None = None,
    reference_min_df: int = 0,
    reference_max_df_ratio: float = 1.0,
) -> tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    exclude_key = tuple(sorted(exclude_labels)) if exclude_labels else None
    cache_key = (
        dataset,
        data_run,
        category,
        split,
        min_token_len,
        language,
        delimiter,
        ja_replace_num,
        ja_dicdir,
        ja_require_unidic,
        dict_no_below,
        dict_no_above,
        dict_exclude_tokens,
        dict_exclude_single_alpha,
        dict_exclude_with_digit,
        dict_exclude_hiragana_only,
        exclude_key,
        split_csvs,
        target_column,
        int(reference_min_df),
        float(reference_max_df_ratio),
    )
    if cache_key in cache:
        return cache[cache_key]
    bundle = build_corpus_bundle(
        dataset=dataset,
        category=category,
        split=split,
        min_token_len=min_token_len,
        language=language,
        delimiter=delimiter,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
        dict_no_below=dict_no_below,
        dict_no_above=dict_no_above,
        dict_exclude_tokens=dict_exclude_tokens,
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
        exclude_labels=exclude_labels,
        split_csvs=split_csvs,
        target_column=target_column,
        reference_document_frequencies=reference_document_frequencies,
        reference_num_docs=reference_num_docs,
        reference_min_df=reference_min_df,
        reference_max_df_ratio=reference_max_df_ratio,
    )
    cache[cache_key] = bundle
    return bundle


def _get_reference_corpus_bundle_cached(
    *,
    cache: dict[
        tuple[
            str,
            str,
            int | None,
            int,
            int,
            float,
            frozenset[str],
            bool,
            bool,
            bool,
        ],
        tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]],
    ],
    path: Path,
    max_docs: int | None,
    min_doc_tokens: int,
    dict_no_below: int,
    dict_no_above: float,
    dict_exclude_tokens: frozenset[str],
    dict_exclude_single_alpha: bool,
    dict_exclude_with_digit: bool,
    dict_exclude_hiragana_only: bool,
) -> tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    resolved_path = resolve_project_path(path)
    cache_key = (
        "tokenized_jsonl",
        str(resolved_path),
        max_docs,
        min_doc_tokens,
        dict_no_below,
        dict_no_above,
        dict_exclude_tokens,
        dict_exclude_single_alpha,
        dict_exclude_with_digit,
        dict_exclude_hiragana_only,
    )
    if cache_key in cache:
        return cache[cache_key]
    bundle = build_reference_corpus_bundle(
        path=resolved_path,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
        dict_no_below=dict_no_below,
        dict_no_above=dict_no_above,
        dict_exclude_tokens=dict_exclude_tokens,
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
    )
    cache[cache_key] = bundle
    return bundle


def _resolve_topic_words_result(
    *,
    args: argparse.Namespace,
    cache,
    model: str,
    data_run: str,
    category: str,
    iteration: int,
    split_csvs: tuple[str, ...] | None,
    target_column: str,
    texts: list[list[str]],
    dictionary: Dictionary,
    corpus_bow: list[list[tuple[int, int]]],
) -> tuple[TopicWordsResult, list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    if model not in RUNTIME_TOPIC_WORD_MODELS:
        raise ValueError(f"Unsupported model for coherence analysis: {model}")
    posterior_config = CollapsedFoldInConfig(
        num_chains=int(getattr(args, "posterior_num_chains", 1)),
        burn_in_sweeps=int(getattr(args, "posterior_burn_in_sweeps", 20)),
        retained_samples=int(getattr(args, "posterior_retained_samples", 20)),
        thinning=int(getattr(args, "posterior_thinning", 1)),
        random_seed=int(getattr(args, "posterior_seed", 0)),
        backend=str(getattr(args, "posterior_backend", "numba")),
    )
    runtime = resolve_runtime_topic_words(
        model=model,
        dataset=args.dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=args.num_topics,
        category=category,
        split=args.coherence_split,
        dictionary=dictionary,
        topn=_requested_topic_word_topn(args),
        embedding_variant=_effective_embedding_variant_for_model(model, args),
        posterior_config=posterior_config,
        etm_theta_samples=int(getattr(args, "etm_theta_samples", 100)),
        etm_posterior_seed=int(getattr(args, "etm_posterior_seed", 0)),
        npmi_min_expected_count=getattr(args, "npmi_min_expected_count", None),
        encoder_device=str(getattr(args, "topic_word_encoder_effective_device", "cpu")),
        encoder_device_requested=str(
            getattr(args, "topic_word_encoder_device", "auto")
        ),
        encoder_batch_size_override=getattr(args, "topic_word_encode_batch_size", None),
        prior_scale=_effective_prior_scale_for_model(model, args),
        covariance_type=_effective_covariance_type_for_model(model, args),
        vmf_variant=_effective_vmf_variant_for_model(model, args),
    )
    result = _metric_topic_words_result(args=args, runtime=runtime)
    return result, texts, dictionary, corpus_bow


def _persist_runtime_topic_word_artifacts(
    *,
    out_dir: Path,
    model: str,
    split: str,
    runtimes_by_iteration: list[tuple[int, RuntimeTopicWords]],
    common_meta: dict[str, object],
    evaluation_score_mode: str = "topic_word_probability",
    write_posterior_mean: bool = False,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    if not runtimes_by_iteration:
        raise ValueError("No runtime topic-word results to persist.")
    first = runtimes_by_iteration[0][1]
    for iteration, runtime in runtimes_by_iteration[1:]:
        if runtime.vocabulary_fingerprint != first.vocabulary_fingerprint:
            raise ValueError(f"Evaluation vocabulary changed at iteration {iteration}.")
        if runtime.protocol != first.protocol:
            raise ValueError(f"Topic-word protocol changed at iteration {iteration}.")

    selected_by_iteration = [
        (
            iteration,
            select_metric_topic_words(runtime, score_mode=evaluation_score_mode),
        )
        for iteration, runtime in runtimes_by_iteration
    ]
    first_selected = selected_by_iteration[0][1]
    evaluation_rows = [
        {
            "iteration": int(iteration),
            "topics": serialize_topic_words(selected.topic_words),
        }
        for iteration, selected in selected_by_iteration
    ]
    probability_rows = [
        {
            "iteration": int(iteration),
            "topics": serialize_topic_words(runtime.evaluation.topic_words),
        }
        for iteration, runtime in runtimes_by_iteration
    ]
    display_rows = [
        {
            "iteration": int(iteration),
            "topics": serialize_topic_words(runtime.display_topic_words),
        }
        for iteration, runtime in runtimes_by_iteration
    ]
    evaluation_path = out_dir / "topic_words_evaluation_topk.json"
    display_path = out_dir / "topic_words_display_topk.json"
    probability_path = out_dir / "topic_words_probability_topk.json"
    write_evaluation_json(
        meta={
            **common_meta,
            "task": "word_based_topic_words",
            "model": model,
            "split": split,
            "topic_word_role": "evaluation",
            "source": first_selected.topic_word_source,
            "score_mode": first_selected.score_mode,
            "score_definition": first_selected.score_definition,
            "evaluation_vocabulary_fingerprint": first.vocabulary_fingerprint,
        },
        results={"per_iteration": evaluation_rows},
        path=evaluation_path,
    )
    write_evaluation_json(
        meta={
            **common_meta,
            "task": "word_based_topic_words",
            "model": model,
            "split": split,
            "topic_word_role": "display",
            "source": first.display_source,
            "score_mode": first.display_score_mode,
            "evaluation_vocabulary_fingerprint": first.vocabulary_fingerprint,
        },
        results={"per_iteration": display_rows},
        path=display_path,
    )
    write_evaluation_json(
        meta={
            **common_meta,
            "task": "word_based_topic_words",
            "model": model,
            "split": split,
            "topic_word_role": "diagnostic_probability",
            "source": first.evaluation.topic_word_source,
            "score_mode": first.evaluation.score_mode,
            "score_definition": first.evaluation.score_definition,
            "evaluation_vocabulary_fingerprint": first.vocabulary_fingerprint,
        },
        results={"per_iteration": probability_rows},
        path=probability_path,
    )

    iteration_artifacts: dict[str, object] = {}
    coverage_by_iteration: dict[str, object] = {}
    for iteration, runtime in runtimes_by_iteration:
        iteration_dir = out_dir / "iterations" / f"iteration_{iteration}"
        ensure_directory(iteration_dir)
        artifacts: dict[str, str] = {}
        if write_posterior_mean and runtime.posterior_mean_by_doc is not None:
            prefix = (
                f"etm_token_topic_{split}_posterior_mean"
                if model == "etm"
                else f"topic_assignment_{split}_posterior_mean"
            )
            posterior_pickle_path = iteration_dir / f"{prefix}.pkl"
            posterior_json_path = iteration_dir / f"{prefix}.json"
            save_pickle(runtime.posterior_mean_by_doc, posterior_pickle_path)
            # The JSON file is a metadata-only sidecar; the posterior arrays live
            # exclusively in the pickle.
            save_json(
                {
                    "iteration": int(iteration),
                    **(runtime.posterior_metadata or {}),
                },
                posterior_json_path,
            )
            artifacts["posterior_mean_pickle"] = str(
                posterior_pickle_path.relative_to(out_dir)
            )
            artifacts["posterior_mean_json"] = str(
                posterior_json_path.relative_to(out_dir)
            )
        if runtime.expected_counts is not None:
            counts_path = iteration_dir / f"topic_word_{split}_expected_counts.pkl"
            save_pickle(runtime.expected_counts, counts_path)
            artifacts["expected_counts"] = str(counts_path.relative_to(out_dir))
        coverage_path = iteration_dir / f"topic_word_{split}_coverage.json"
        coverage_payload = {
            "iteration": int(iteration),
            "protocol": runtime.protocol,
            "coverage": runtime.coverage,
            "empty_topic_ids": [int(topic_id) for topic_id in runtime.empty_topic_ids],
            "num_active_topics": int(len(runtime.display_topic_words))
            - len(runtime.empty_topic_ids),
            "num_empty_topics": len(runtime.empty_topic_ids),
            "topic_utilization": (
                (len(runtime.display_topic_words) - len(runtime.empty_topic_ids))
                / len(runtime.display_topic_words)
                if runtime.display_topic_words
                else float("nan")
            ),
            "source_condition_dir": str(runtime.condition_dir),
            "source_condition_fingerprint": runtime.source_condition_fingerprint,
            "evaluation_vocabulary_fingerprint": runtime.vocabulary_fingerprint,
            "corpus_fingerprint": runtime.corpus_fingerprint,
            "execution_metadata": runtime.execution_metadata,
        }
        save_json(coverage_payload, coverage_path)
        artifacts["coverage"] = str(coverage_path.relative_to(out_dir))
        iteration_artifacts[str(iteration)] = artifacts
        coverage_by_iteration[str(iteration)] = coverage_payload

    runtime_meta = {
        "topic_word_protocol": first.protocol,
        "evaluation_topic_word_source": first_selected.topic_word_source,
        "display_topic_word_source": first.display_source,
        "probability_topic_word_source": first.evaluation.topic_word_source,
        "topic_words_probability_topk": probability_path.name,
        "evaluation_vocabulary_fingerprint": first.vocabulary_fingerprint,
        "corpus_fingerprint_by_iteration": {
            str(iteration): runtime.corpus_fingerprint
            for iteration, runtime in runtimes_by_iteration
        },
        "coverage_by_iteration": coverage_by_iteration,
        "execution_metadata_by_iteration": {
            str(iteration): runtime.execution_metadata
            for iteration, runtime in runtimes_by_iteration
            if runtime.execution_metadata is not None
        },
        "empty_topic_ids_by_iteration": {
            str(iteration): [int(topic_id) for topic_id in runtime.empty_topic_ids]
            for iteration, runtime in runtimes_by_iteration
        },
        "topic_utilization_by_iteration": {
            str(iteration): (
                (len(runtime.display_topic_words) - len(runtime.empty_topic_ids))
                / len(runtime.display_topic_words)
                if runtime.display_topic_words
                else float("nan")
            )
            for iteration, runtime in runtimes_by_iteration
        },
    }
    return evaluation_path, display_path, iteration_artifacts, runtime_meta


def _write_word_based_group_outputs(
    *,
    args: argparse.Namespace,
    model: str,
    data_run: str,
    category: str,
    coherences: list[str],
    primary_coherence: str,
    multiple_coherences: bool,
    coherence_window_sizes: dict[str, int | None],
    coherence_window_size_sources: dict[str, str],
    coherence_implementations: dict[str, str],
    coherence_min_window_counts: dict[str, int | None],
    metric_names: list[str],
    per_iter_metrics: list[dict[str, float]],
    per_iter_topic_words: list[dict[str, object]],
    used_iterations: list[int],
    topic_word_source: str,
    topic_word_score_mode: str,
    topic_word_score_definition: str,
    coherence_reference_num_docs: int,
    coherence_reference_vocab_size: int,
    coherence_reference_streaming: bool,
    summary_rows: list[dict[str, str | float]],
    summary_provenance: list[dict[str, object]],
    runtime_iterations: list[tuple[int, RuntimeTopicWords]],
) -> None:
    write_started = perf_counter()
    logger.info(
        "wb write start data_run=%s model=%s category=%s iterations=%s",
        data_run,
        model,
        category,
        list(used_iterations),
    )
    agg = aggregate_metrics(per_iter_metrics, metric_names=metric_names)
    provenance = resolve_model_provenance(
        model=model,
        dataset=args.dataset,
        # Must match _expected_output_condition_id: the skip check runs before
        # any iteration is evaluated, so it can only key on the requested
        # iterations. Using used_iterations[0] would write the result where the
        # resume check never looks whenever the first requested iteration is
        # excluded or the iterations are passed unsorted.
        iteration=int(min(args.iteration)),
        num_topics=args.num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=_effective_embedding_variant_for_model(model, args),
        prior_scale=_effective_prior_scale_for_model(model, args),
        covariance_type=_effective_covariance_type_for_model(model, args),
        vmf_variant=_effective_vmf_variant_for_model(model, args),
    )
    condition_id, condition_fingerprint = _build_output_condition_id(
        model=model,
        dataset=args.dataset,
        data_run=data_run,
        category=category,
        iterations=[int(value) for value in args.iteration],
        num_topics=args.num_topics,
        coherence=primary_coherence,
        coherences=coherences if multiple_coherences else None,
        coherence_topn=args.coherence_topn,
        coherence_window_size=(
            coherence_window_sizes
            if multiple_coherences
            else coherence_window_sizes[primary_coherence]
        ),
        coherence_implementation=(
            coherence_implementations
            if multiple_coherences
            else coherence_implementations[primary_coherence]
        ),
        coherence_min_window_count=(
            coherence_min_window_counts
            if multiple_coherences
            else coherence_min_window_counts[primary_coherence]
        ),
        coherence_reference=args.coherence_reference,
        coherence_reference_path=(
            None
            if args.coherence_reference_path is None
            else str(resolve_project_path(args.coherence_reference_path))
        ),
        coherence_reference_format=(
            args.coherence_reference_format
            if args.coherence_reference == "wikipedia"
            else None
        ),
        coherence_reference_max_docs=args.coherence_reference_max_docs,
        coherence_reference_min_doc_tokens=args.coherence_reference_min_doc_tokens,
        coherence_reference_streaming=coherence_reference_streaming,
        diversity_topn=args.diversity_topn,
        coherence_split=args.coherence_split,
        topic_word_source=topic_word_source,
        embedding_variant=_effective_embedding_variant_for_model(model, args),
        prior_scale=_effective_prior_scale_for_model(model, args),
        covariance_type=_effective_covariance_type_for_model(model, args),
        source_condition_id=(
            None
            if provenance.get("condition_id") is None
            else str(provenance["condition_id"])
        ),
        source_condition_fingerprint=(
            None
            if provenance.get("condition_fingerprint") is None
            else str(provenance["condition_fingerprint"])
        ),
        parameter_variant=(
            None
            if provenance.get("parameter_variant") is None
            else str(provenance["parameter_variant"])
        ),
        metric_names=metric_names,
        dict_exclude_tokens=_dict_exclude_tokens(args),
        posterior_settings=_posterior_settings(args),
        topic_word_score_mode=_topic_word_score_mode(args),
        dict_no_above=float(args.dict_no_above),
        reference_min_df=_reference_min_df(args),
        reference_max_df_ratio=_reference_max_df_ratio(args),
    )
    display_key = condition_id
    started_at = datetime.now(UTC).isoformat()
    execution_id = build_execution_id(prefix="exec", started_at=started_at)
    uses_default_output_layout = _uses_default_output_layout(args.out_root)
    if uses_default_output_layout:
        archive_out_dir = build_archive_result_dir(
            base_root=args.out_root,
            dataset=args.dataset,
            data_run=data_run,
            category=category,
            display_key=display_key,
            started_at=started_at,
            execution_id=execution_id,
        )
        latest_out_dir = build_latest_result_dir(
            base_root=args.out_root,
            dataset=args.dataset,
            data_run=data_run,
            category=category,
            display_key=display_key,
        )
        out_dir = archive_out_dir
    else:
        archive_out_dir = None
        latest_out_dir = None
        out_dir = args.out_root / args.dataset / data_run / category / condition_id
    ensure_directory(out_dir)
    coherence_reference_path = (
        None
        if args.coherence_reference_path is None
        else str(resolve_project_path(args.coherence_reference_path))
    )
    coherence_reference_meta = {
        "coherence_reference": args.coherence_reference,
        "coherence_reference_path": coherence_reference_path,
        "coherence_reference_format": (
            args.coherence_reference_format
            if args.coherence_reference == "wikipedia"
            else None
        ),
        "coherence_reference_num_docs": int(coherence_reference_num_docs),
        "coherence_reference_vocab_size": int(coherence_reference_vocab_size),
        "coherence_reference_max_docs": args.coherence_reference_max_docs,
        "coherence_reference_min_doc_tokens": int(
            args.coherence_reference_min_doc_tokens
        ),
        "coherence_reference_streaming": bool(coherence_reference_streaming),
        "coherence_reference_language": (
            "en" if args.coherence_reference == "wikipedia" else None
        ),
    }
    requested_embedding_variant = _requested_embedding_variant(args)
    effective_embedding_variant = model_inputs_module.effective_embedding_variant(
        model,
        requested_embedding_variant,
    )
    metrics_meta = build_evaluation_meta(
        task="word_based_metrics",
        model=model,
        dataset=args.dataset,
        data_run=data_run,
        num_topics=args.num_topics,
        category=category,
        condition_id=condition_id,
        display_key=display_key,
        condition_fingerprint=condition_fingerprint,
        embedding_variant=requested_embedding_variant,
        effective_embedding_variant=effective_embedding_variant,
        prior_scale=_effective_prior_scale_for_model(model, args),
        covariance_type=_effective_covariance_type_for_model(model, args),
        iterations=used_iterations,
        started_at=started_at,
        execution_id=execution_id,
        archive_dir=str(out_dir),
        latest_dir=None if latest_out_dir is None else str(latest_out_dir),
        model_provenance=provenance,
        source_condition_id=provenance.get("condition_id"),
        source_condition_fingerprint=provenance.get("condition_fingerprint"),
        parameter_variant=provenance.get("parameter_variant"),
        metric_names=metric_names,
        topic_words={
            "topn": int(_requested_topic_word_topn(args)),
            "coherence_topn": int(args.coherence_topn),
            "diversity_topn": int(args.diversity_topn),
            "source": topic_word_source,
            "score_mode": topic_word_score_mode,
            "score_definition": topic_word_score_definition,
        },
        coherence=_coherence_meta(
            coherences=coherences,
            args=args,
            model=model,
            topic_word_source=topic_word_source,
            topic_word_score_mode=topic_word_score_mode,
            topic_word_score_definition=topic_word_score_definition,
            reference_meta=coherence_reference_meta,
            window_sizes=coherence_window_sizes,
            window_size_sources=coherence_window_size_sources,
            min_window_counts=coherence_min_window_counts,
        ),
        diversity={
            "topn": int(args.diversity_topn),
            "topic_word_source": topic_word_source,
        },
    )
    (
        evaluation_topic_words_path,
        display_topic_words_path,
        iteration_artifacts,
        runtime_meta,
    ) = _persist_runtime_topic_word_artifacts(
        out_dir=out_dir,
        model=model,
        split=args.coherence_split,
        runtimes_by_iteration=runtime_iterations,
        write_posterior_mean=bool(
            getattr(args, "write_posterior_mean_artifact", False)
        ),
        common_meta={
            "dataset": args.dataset,
            "data_run": data_run,
            "category": category,
            "num_topics": args.num_topics,
            "condition_id": condition_id,
            "condition_fingerprint": condition_fingerprint,
            "iterations": used_iterations,
            "model_provenance": provenance,
        },
        evaluation_score_mode=_topic_word_score_mode(args),
    )
    metrics_meta.update(runtime_meta)
    metrics_meta["topic_word_score_mode"] = _topic_word_score_mode(args)
    metrics_meta["topic_word_ranking_schema_version"] = (
        TOPIC_WORD_RANKING_SCHEMA_VERSION
    )
    metrics_meta["posterior_settings"] = _posterior_settings(args)
    metrics_meta["empty_topic_evaluation"] = _empty_topic_policy_meta(
        args=args,
        model=model,
    )
    metrics_meta["requested_iterations"] = [int(value) for value in args.iteration]
    metrics_meta["evaluated_iterations"] = [int(value) for value in used_iterations]
    metrics_meta["degenerate_iterations"] = sorted(
        set(int(value) for value in args.iteration) - set(used_iterations)
    )
    metrics_results = {
        "aggregate": agg,
        "per_iteration": per_iter_metrics,
        "topic_words_evaluation_topk": {
            "topn": int(_requested_topic_word_topn(args)),
            "coherence_topn": int(args.coherence_topn),
            "diversity_topn": int(args.diversity_topn),
            "per_iteration": per_iter_topic_words,
        },
        **runtime_meta,
    }
    out_path = out_dir / "metrics_agg.json"
    write_evaluation_json(meta=metrics_meta, results=metrics_results, path=out_path)
    logger.info(f"[{model}] aggregated metrics saved to {out_path}")
    logger.info(
        "[%s] evaluation/display top words saved to %s and %s",
        model,
        evaluation_topic_words_path,
        display_topic_words_path,
    )
    metadata_path = out_dir / "metadata.json"
    save_json(metrics_meta, metadata_path)
    logger.info(f"[{model}] metadata saved to {metadata_path}")
    completion_artifacts = {
        "metrics_agg": out_path.name,
        "topic_words_evaluation_topk": evaluation_topic_words_path.name,
        "topic_words_display_topk": display_topic_words_path.name,
        "topic_words_probability_topk": runtime_meta["topic_words_probability_topk"],
        "iteration_artifacts": iteration_artifacts,
        "metadata": metadata_path.name,
    }
    completion_path = write_completion_marker(
        output_dir=out_dir,
        condition_fingerprint=condition_fingerprint,
        artifacts=completion_artifacts,
    )
    logger.info("[%s] completion marker saved to %s", model, completion_path)
    if uses_default_output_layout and archive_out_dir is not None:
        pointer_path = write_latest_result_pointer(
            base_root=args.out_root,
            task="word_based_metrics",
            dataset=args.dataset,
            data_run=data_run,
            category=category,
            display_key=display_key,
            archive_dir=archive_out_dir,
            started_at=started_at,
            execution_id=execution_id,
            condition_fingerprint=condition_fingerprint,
            artifacts=completion_artifacts,
        )
        logger.info("[%s] updated latest pointer at %s", model, pointer_path)

    logger.info(
        "wb write done data_run=%s model=%s category=%s out_dir=%s sec=%.1f",
        data_run,
        model,
        category,
        out_dir,
        perf_counter() - write_started,
    )
    for metric_name, stats in agg.items():
        row_coherence = _coherence_from_metric_name(metric_name, coherences=coherences)
        row_coherence_details = (
            describe_coherence_metric(row_coherence)
            if row_coherence is not None
            else None
        )
        summary_rows.append(
            {
                "dataset": args.dataset,
                "data_run": data_run,
                "num_topics": args.num_topics,
                "category": category,
                "model": model,
                "metric": metric_name,
                "mean": round_sigfigs(stats.get("mean", float("nan"))),
                "std": round_sigfigs(stats.get("std", float("nan"))),
                "coherence_metric": row_coherence if row_coherence is not None else "",
                "coherence_implementation": (
                    coherence_implementations[row_coherence]
                    if row_coherence is not None
                    else ""
                ),
                "coherence_definition": (
                    row_coherence_details["definition"]
                    if row_coherence_details is not None
                    else ""
                ),
                "coherence_cooccurrence_unit": (
                    row_coherence_details["cooccurrence_unit"]
                    if row_coherence_details is not None
                    else ""
                ),
                "coherence_zero_cooccurrence_policy": (
                    row_coherence_details["zero_cooccurrence_policy"]
                    if row_coherence_details is not None
                    else ""
                ),
                "coherence_split": (
                    args.coherence_split if row_coherence is not None else ""
                ),
                "coherence_topn": (
                    args.coherence_topn if row_coherence is not None else ""
                ),
                "coherence_window_size": (
                    coherence_window_sizes[row_coherence]
                    if row_coherence is not None
                    else ""
                ),
                "coherence_window_size_source": (
                    coherence_window_size_sources[row_coherence]
                    if row_coherence is not None
                    else ""
                ),
                "coherence_min_window_count": (
                    coherence_min_window_counts[row_coherence]
                    if row_coherence is not None
                    else ""
                ),
                "coherence_reference": (
                    args.coherence_reference if row_coherence is not None else ""
                ),
                "coherence_reference_path": (
                    coherence_reference_path if row_coherence is not None else ""
                ),
                "coherence_reference_format": (
                    (
                        args.coherence_reference_format
                        if args.coherence_reference == "wikipedia"
                        else ""
                    )
                    if row_coherence is not None
                    else ""
                ),
                "coherence_reference_num_docs": (
                    coherence_reference_num_docs if row_coherence is not None else ""
                ),
                "coherence_reference_vocab_size": (
                    coherence_reference_vocab_size if row_coherence is not None else ""
                ),
                "coherence_reference_max_docs": (
                    args.coherence_reference_max_docs
                    if row_coherence is not None
                    else ""
                ),
                "coherence_reference_min_doc_tokens": (
                    args.coherence_reference_min_doc_tokens
                    if row_coherence is not None
                    else ""
                ),
                "coherence_reference_streaming": (
                    coherence_reference_streaming if row_coherence is not None else ""
                ),
                "diversity_topn": (
                    args.diversity_topn if metric_name == "diversity" else ""
                ),
                "topic_word_topn": _requested_topic_word_topn(args),
                "dict_no_below": (
                    args.dict_no_below if row_coherence is not None else ""
                ),
                "dict_no_above": (
                    args.dict_no_above if row_coherence is not None else ""
                ),
                "dict_exclude_tokens": (
                    ",".join(sorted(_dict_exclude_tokens(args)))
                    if row_coherence is not None
                    else ""
                ),
                "dict_exclude_single_alpha": (
                    args.dict_exclude_single_alpha if row_coherence is not None else ""
                ),
                "dict_exclude_with_digit": (
                    args.dict_exclude_with_digit if row_coherence is not None else ""
                ),
                "dict_exclude_hiragana_only": (
                    args.dict_exclude_hiragana_only if row_coherence is not None else ""
                ),
                "language": args.language if row_coherence is not None else "",
                "embedding_variant": requested_embedding_variant,
                "effective_embedding_variant": effective_embedding_variant,
                "prior_scale": _effective_prior_scale_for_model(model, args),
                "covariance_type": _effective_covariance_type_for_model(model, args),
                "vmf_variant": _effective_vmf_variant_for_model(model, args),
                "topic_word_source": topic_word_source,
                "topic_word_score_mode": topic_word_score_mode,
            }
        )
    summary_provenance.append(
        {
            "model": model,
            "data_run": data_run,
            "category": category,
            "model_provenance": provenance,
        }
    )


def _collect_pending_word_based_group(
    *,
    args: argparse.Namespace,
    task: PendingWordBasedGroupTask,
    total_conditions: int,
    failure_sink: list[dict[str, object]] | None = None,
) -> PendingWordBasedGroup | None:
    data_run = task.data_run
    model = task.model
    local_args = argparse.Namespace(**vars(args))
    local_args.num_topics = int(task.num_topics)
    category = task.category
    cache: dict[
        tuple[
            str,
            str,
            str,
            str,
            int,
            str,
            str | None,
            bool,
            str | None,
            bool,
            int,
            float,
            frozenset[str],
            bool,
            bool,
            bool,
            tuple[str, ...] | None,
            tuple[str, ...] | None,
            str,
        ],
        tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]],
    ] = {}
    logger.info(
        "word_based_metrics condition_group data_run=%s model=%s category=%s "
        "iterations=%s",
        data_run,
        model,
        category,
        list(args.iteration),
    )
    pending_iterations: list[PendingWordBasedIteration] = []
    topic_word_source: str | None = None
    topic_word_score_mode = ""
    topic_word_score_definition = ""
    for offset, iteration in enumerate(args.iteration, start=1):
        condition_progress = f"{task.progress_start + offset}/{total_conditions}"
        iteration_started = perf_counter()
        logger.info(
            "wb %s start data_run=%s model=%s category=%s iteration=%s",
            condition_progress,
            data_run,
            model,
            category,
            iteration,
        )
        split_csvs, resolved_target_column = _resolve_split_csvs_and_target_column(
            model=model,
            dataset=local_args.dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=local_args.num_topics,
            category=category,
            split=local_args.coherence_split,
            embedding_variant=_effective_embedding_variant_for_model(model, local_args),
            prior_scale=_effective_prior_scale_for_model(model, local_args),
            vmf_variant=_effective_vmf_variant_for_model(model, local_args),
        )
        (
            topic_word_texts,
            topic_word_dictionary,
            topic_word_corpus_bow,
        ) = _get_corpus_bundle_cached(
            cache=cache,
            dataset=local_args.dataset,
            data_run=data_run,
            category=category,
            split=local_args.coherence_split,
            min_token_len=local_args.coherence_min_token_len,
            language=local_args.language,
            delimiter=local_args.delimiter,
            ja_replace_num=local_args.ja_replace_num,
            ja_dicdir=local_args.ja_dicdir,
            ja_require_unidic=local_args.ja_require_unidic,
            dict_no_below=local_args.dict_no_below,
            dict_no_above=local_args.dict_no_above,
            dict_exclude_tokens=_dict_exclude_tokens(local_args),
            dict_exclude_single_alpha=local_args.dict_exclude_single_alpha,
            dict_exclude_with_digit=local_args.dict_exclude_with_digit,
            dict_exclude_hiragana_only=local_args.dict_exclude_hiragana_only,
            exclude_labels=None,
            split_csvs=split_csvs,
            target_column=resolved_target_column,
            **_reference_band_kwargs(local_args),
        )
        ordered_vocabulary = [
            str(topic_word_dictionary[index])
            for index in range(len(topic_word_dictionary))
        ]
        checkpoint_identity = _topic_word_checkpoint_identity(
            args=local_args,
            model=model,
            data_run=data_run,
            category=category,
            iteration=iteration,
            vocabulary_fingerprint=fingerprint_jsonable(
                {"ordered_vocabulary": ordered_vocabulary}
            ),
        )
        checkpoint_dir = topic_word_checkpoint_dir(
            checkpoint_root=_checkpoint_root(local_args),
            identity=checkpoint_identity,
        )
        checkpoint_mode = str(getattr(local_args, "checkpoint_mode", "auto"))
        runtime = (
            load_topic_word_checkpoint(
                checkpoint_dir=checkpoint_dir,
                expected_identity=checkpoint_identity,
            )
            if checkpoint_mode == "auto"
            else None
        )
        if runtime is not None:
            logger.info(
                "wb %s topic_words checkpoint hit path=%s",
                condition_progress,
                checkpoint_dir,
            )
            topic_words_result = _metric_topic_words_result(
                args=local_args,
                runtime=runtime,
            )
        else:
            try:
                (
                    topic_words_result,
                    _topic_word_texts,
                    _topic_word_dictionary,
                    _topic_word_corpus_bow,
                ) = _resolve_topic_words_result(
                    args=local_args,
                    cache=cache,
                    model=model,
                    data_run=data_run,
                    category=category,
                    iteration=iteration,
                    split_csvs=split_csvs,
                    target_column=resolved_target_column,
                    texts=topic_word_texts,
                    dictionary=topic_word_dictionary,
                    corpus_bow=topic_word_corpus_bow,
                )
            except ConditionEvaluationError as exc:
                if not _isolates_condition_failures(local_args):
                    raise
                failure = _condition_failure_payload(
                    task=task,
                    args=local_args,
                    exc=exc,
                    stage="topic_words",
                    iteration=int(iteration),
                )
                if failure_sink is not None:
                    failure_sink.append(failure)
                save_failure_record(
                    checkpoint_root=_checkpoint_root(local_args),
                    identity=failure,
                    payload=failure,
                )
                logger.error("word_based iteration excluded: %s", failure)
                continue
            assert isinstance(topic_words_result.runtime_payload, RuntimeTopicWords)
            if checkpoint_mode != "off":
                save_topic_word_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    identity=checkpoint_identity,
                    runtime=topic_words_result.runtime_payload,
                    serialized_evaluation_words=serialize_topic_words(
                        topic_words_result.runtime_payload.evaluation.topic_words
                    ),
                    serialized_display_words=serialize_topic_words(
                        topic_words_result.runtime_payload.display_topic_words
                    ),
                )
                logger.info(
                    "wb %s topic_words checkpoint saved path=%s",
                    condition_progress,
                    checkpoint_dir,
                )
        assert isinstance(topic_words_result.runtime_payload, RuntimeTopicWords)
        runtime = topic_words_result.runtime_payload
        if runtime.empty_topic_ids and not _uses_mvtm_fixed_k_policy(
            local_args,
            model=model,
        ):
            partial_pointer = _persist_partial_topic_words(
                args=local_args,
                task=task,
                iteration=int(iteration),
                runtime=runtime,
                checkpoint_identity=checkpoint_identity,
            )
            exc = EmptyTopicError(topic_ids=list(runtime.empty_topic_ids))
            failure = _condition_failure_payload(
                task=task,
                args=local_args,
                exc=exc,
                stage="topic_words",
                iteration=int(iteration),
            )
            failure["partial_artifact_current"] = str(partial_pointer)
            if failure_sink is not None:
                failure_sink.append(failure)
            save_failure_record(
                checkpoint_root=_checkpoint_root(local_args),
                identity={
                    key: value
                    for key, value in failure.items()
                    if key != "partial_artifact_current"
                },
                payload=failure,
            )
            logger.error("word_based iteration excluded: %s", failure)
            logger.info(
                "wb %s partial topic_words saved path=%s empty_topic_ids=%s",
                condition_progress,
                partial_pointer,
                list(runtime.empty_topic_ids),
            )
            continue
        if runtime.empty_topic_ids:
            logger.warning(
                "wb %s MvTM fixed-K evaluation includes empty topics ids=%s",
                condition_progress,
                list(runtime.empty_topic_ids),
            )
        topic_word_source = topic_words_result.topic_word_source
        topic_word_score_mode = topic_words_result.score_mode or ""
        topic_word_score_definition = topic_words_result.score_definition or ""
        checkpoint_enabled = checkpoint_mode != "off"
        pending_iterations.append(
            PendingWordBasedIteration(
                iteration=int(iteration),
                topic_words=topic_words_result.topic_words,
                runtime_payload=(
                    None if checkpoint_enabled else topic_words_result.runtime_payload
                ),
                checkpoint_path=checkpoint_dir if checkpoint_enabled else None,
                checkpoint_identity=(
                    checkpoint_identity if checkpoint_enabled else None
                ),
            )
        )
        logger.info(
            "wb %s topic_words done data_run=%s model=%s category=%s "
            "iteration=%s source=%s topics=%s total_sec=%.1f",
            condition_progress,
            data_run,
            model,
            category,
            iteration,
            topic_word_source,
            len(topic_words_result.topic_words),
            perf_counter() - iteration_started,
        )
    if topic_word_source is None:
        return None
    return PendingWordBasedGroup(
        data_run=data_run,
        model=model,
        num_topics=local_args.num_topics,
        category=category,
        iterations=pending_iterations,
        topic_word_source=topic_word_source,
        topic_word_score_mode=topic_word_score_mode,
        topic_word_score_definition=topic_word_score_definition,
    )


def _score_pending_word_based_group(
    *,
    group: PendingWordBasedGroup,
    args: argparse.Namespace,
    metric_names: list[str],
    coherences: list[str],
    shared_counts,
) -> ScoredWordBasedGroup:
    local_args = argparse.Namespace(**vars(args))
    local_args.num_topics = int(group.num_topics)
    per_iter_metrics: list[dict[str, float]] = []
    per_iter_topic_words: list[dict[str, object]] = []
    used_iterations: list[int] = []
    for pending_iteration in group.iterations:
        fixed_k_reporting = _uses_fixed_k_reporting(local_args)
        scoring_topic_words = (
            [topic for topic in pending_iteration.topic_words if topic]
            if fixed_k_reporting
            else pending_iteration.topic_words
        )
        metrics = compute_shared_reference_coherence_scores(
            topic_words=scoring_topic_words,
            metric_names=(
                _metric_names_for_coherences(coherences)
                if fixed_k_reporting
                else metric_names
            ),
            coherences=coherences,
            counts=shared_counts,
            coherence_topn=local_args.coherence_topn,
            diversity_topn=local_args.diversity_topn,
            window_size=local_args.coherence_window_size,
            min_window_count=getattr(
                local_args,
                "coherence_min_window_count",
                None,
            ),
        )
        if fixed_k_reporting:
            metrics = _apply_fixed_k_reporting(
                metrics=metrics,
                topic_words=pending_iteration.topic_words,
                coherences=coherences,
                args=local_args,
            )
        metrics["num_topics"] = float(local_args.num_topics)
        per_iter_metrics.append(metrics)
        per_iter_topic_words.append(
            {
                "iteration": int(pending_iteration.iteration),
                "topics": serialize_topic_words(pending_iteration.topic_words),
            }
        )
        used_iterations.append(pending_iteration.iteration)
    return ScoredWordBasedGroup(
        group=group,
        per_iter_metrics=per_iter_metrics,
        per_iter_topic_words=per_iter_topic_words,
        used_iterations=used_iterations,
    )


def _load_pending_runtime(
    pending: PendingWordBasedIteration,
) -> RuntimeTopicWords:
    if pending.runtime_payload is not None:
        return pending.runtime_payload
    if pending.checkpoint_path is None or pending.checkpoint_identity is None:
        raise RuntimeError(
            f"iteration {pending.iteration} has neither runtime payload nor checkpoint"
        )
    runtime = load_topic_word_checkpoint(
        checkpoint_dir=pending.checkpoint_path,
        expected_identity=pending.checkpoint_identity,
    )
    if runtime is None:
        raise RuntimeError(
            f"topic-word checkpoint became unavailable: {pending.checkpoint_path}"
        )
    return runtime


def run_topic_coherence_analysis_from_args(args: argparse.Namespace) -> Path:
    args.out_root = resolve_project_path(args.out_root)
    batch_size_override = getattr(args, "topic_word_encode_batch_size", None)
    if batch_size_override is not None and int(batch_size_override) <= 0:
        raise ValueError("topic_word_encode_batch_size must be > 0")
    requested_encoder_device = str(getattr(args, "topic_word_encoder_device", "auto"))
    uses_topic_word_encoder = any(
        str(model) in ENCODER_TOPIC_WORD_MODELS for model in getattr(args, "model", ())
    )
    if uses_topic_word_encoder:
        effective_encoder_device = resolve_topic_word_encoder_device(
            requested_encoder_device
        )
        if (
            effective_encoder_device.startswith("cuda")
            and _coherence_topic_word_workers(args) != 1
        ):
            raise ValueError(
                "coherence_topic_word_workers must be 1 when the topic-word "
                "encoder uses CUDA; multiple encoder models can exhaust GPU memory"
            )
        if requested_encoder_device.strip().lower() == "auto":
            logger.info(
                "topic-word encoder device auto resolved to %s",
                effective_encoder_device,
            )
    else:
        effective_encoder_device = "cpu"
    args.topic_word_encoder_effective_device = effective_encoder_device
    if _topic_word_score_mode(args) not in {
        "word_topic_npmi",
        "topic_word_probability",
    }:
        raise ValueError(
            "topic_word_score_mode must be word_topic_npmi or " "topic_word_probability"
        )
    if getattr(args, "checkpoint_mode", "auto") not in {"auto", "off", "refresh"}:
        raise ValueError("checkpoint_mode must be auto, off, or refresh")
    if getattr(args, "reference_count_cache_mode", "auto") not in {
        "auto",
        "off",
        "refresh",
    }:
        raise ValueError("reference_count_cache_mode must be auto, off, or refresh")
    if getattr(args, "reference_index_mode", "off") not in {
        "off",
        "auto",
        "build",
        "refresh",
    }:
        raise ValueError("reference_index_mode must be off, auto, build, or refresh")
    if getattr(args, "condition_failure_policy", "exclude-condition") not in {
        "fail-fast",
        "exclude-condition",
        "isolate",
        "continue-and-fail",
    }:
        raise ValueError(
            "condition_failure_policy must be fail-fast, exclude-condition, "
            "isolate, or continue-and-fail"
        )
    if getattr(args, "mvtm_empty_topic_policy", "exclude") not in {
        "exclude",
        "fixed-k",
    }:
        raise ValueError("mvtm_empty_topic_policy must be exclude or fixed-k")
    CollapsedFoldInConfig(
        num_chains=int(getattr(args, "posterior_num_chains", 1)),
        burn_in_sweeps=int(getattr(args, "posterior_burn_in_sweeps", 20)),
        retained_samples=int(getattr(args, "posterior_retained_samples", 20)),
        thinning=int(getattr(args, "posterior_thinning", 1)),
        random_seed=int(getattr(args, "posterior_seed", 0)),
        backend=str(getattr(args, "posterior_backend", "numba")),  # type: ignore[arg-type]
    ).validate()
    if int(getattr(args, "etm_theta_samples", 100)) <= 0:
        raise ValueError("etm_theta_samples must be positive")
    if (
        getattr(args, "npmi_min_expected_count", None) is not None
        and float(args.npmi_min_expected_count) < 0.0
    ):
        raise ValueError("npmi_min_expected_count must be non-negative")
    _validate_reference_args(args)
    coherences = _requested_coherences(args)
    primary_coherence = _primary_coherence(coherences)
    multiple_coherences = _uses_multiple_coherences(coherences)
    coherence_window_sizes = _effective_coherence_window_sizes(
        coherences,
        args.coherence_window_size,
    )
    coherence_window_size_sources = _coherence_window_size_sources(
        coherences,
        args.coherence_window_size,
    )
    coherence_implementations = _coherence_implementations(coherences)
    coherence_min_window_counts = _effective_coherence_min_window_counts(
        coherences,
        getattr(args, "coherence_min_window_count", None),
    )
    if args.dict_no_below < 1:
        raise ValueError(f"dict_no_below must be >= 1, got {args.dict_no_below}")
    if not (0.0 < args.dict_no_above <= 1.0):
        raise ValueError(f"dict_no_above must be in (0, 1], got {args.dict_no_above}")
    if args.coherence_topn < 1:
        raise ValueError(f"coherence_topn must be >= 1, got {args.coherence_topn}")
    if args.diversity_topn < 1:
        raise ValueError(f"diversity_topn must be >= 1, got {args.diversity_topn}")

    requested_models = [str(model_name) for model_name in args.model]
    unsupported_models = [
        model_name for model_name in requested_models if model_name not in MODEL_CHOICES
    ]
    if unsupported_models:
        raise ValueError(
            "Unsupported word-based model(s): "
            f"{unsupported_models}. Available models: {MODEL_CHOICES}"
        )
    models = [normalize_model_name(model_name) for model_name in requested_models]
    num_topics_values = _requested_num_topics_values(args)
    args.num_topics = num_topics_values[0]
    display_num_topics: int | list[int] = (
        num_topics_values if len(num_topics_values) > 1 else num_topics_values[0]
    )
    coherence_window_size = coherence_window_sizes[primary_coherence]
    coherence_implementation = coherence_implementations[primary_coherence]
    coherence_min_window_count = coherence_min_window_counts[primary_coherence]
    metric_names = (
        _fixed_k_metric_names(coherences)
        if _uses_fixed_k_reporting(args)
        else _metric_names_for_coherences(coherences)
    )
    summary_rows: list[dict[str, str | float]] = []
    summary_provenance: list[dict[str, object]] = []
    condition_failures: list[dict[str, object]] = []
    total_conditions = (
        len(args.data_run)
        * len(models)
        * len(num_topics_values)
        * len(args.category)
        * len(args.iteration)
    )
    processed_conditions = 0
    coherence_cache: dict[
        tuple[
            str,
            str,
            str,
            str,
            int,
            str,
            str | None,
            bool,
            str | None,
            bool,
            int,
            float,
            bool,
            bool,
            bool,
            tuple[str, ...] | None,
            tuple[str, ...] | None,
            str,
        ],
        tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]],
    ] = {}
    reference_corpus_cache: dict[
        tuple[str, str, int | None, int, int, float, bool, bool, bool],
        tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]],
    ] = {}

    logger.info(
        "word_based_metrics start dataset=%s num_topics=%s models=%s data_runs=%s "
        "categories=%s iterations=%s total_conditions=%s",
        args.dataset,
        display_num_topics,
        models,
        list(args.data_run),
        list(args.category),
        list(args.iteration),
        total_conditions,
    )

    if _uses_shared_reference_counts(args=args, coherences=coherences):
        assert args.coherence_reference_path is not None
        group_tasks: list[PendingWordBasedGroupTask] = []
        for data_run in args.data_run:
            for model in models:
                for num_topics in num_topics_values:
                    topic_args = argparse.Namespace(**vars(args))
                    topic_args.num_topics = int(num_topics)
                    for category in args.category:
                        if _should_skip_existing_output(
                            model=model,
                            data_run=data_run,
                            category=category,
                            args=topic_args,
                            coherences=coherences,
                            coherence_window_sizes=coherence_window_sizes,
                            coherence_implementations=coherence_implementations,
                            coherence_min_window_counts=coherence_min_window_counts,
                            metric_names=metric_names,
                        ):
                            continue
                        progress_start = processed_conditions
                        processed_conditions += len(args.iteration)
                        group_tasks.append(
                            PendingWordBasedGroupTask(
                                sort_index=len(group_tasks),
                                data_run=data_run,
                                model=model,
                                num_topics=int(num_topics),
                                category=category,
                                progress_start=progress_start,
                            )
                        )

        topic_word_workers = min(_coherence_topic_word_workers(args), len(group_tasks))
        pending_groups: list[PendingWordBasedGroup] = []
        if group_tasks and topic_word_workers > 1:
            logger.info(
                "word_based_metrics topic_words parallel start workers=%s groups=%s",
                topic_word_workers,
                len(group_tasks),
            )
            grouped_results: list[tuple[int, PendingWordBasedGroup]] = []
            with ThreadPoolExecutor(max_workers=topic_word_workers) as executor:
                futures = {
                    executor.submit(
                        _collect_pending_word_based_group,
                        args=args,
                        task=task,
                        total_conditions=total_conditions,
                        failure_sink=condition_failures,
                    ): task
                    for task in group_tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except ConditionEvaluationError as exc:
                        if not _isolates_condition_failures(args):
                            raise
                        failure = _condition_failure_payload(
                            task=task,
                            args=args,
                            exc=exc,
                            stage="topic_words",
                        )
                        condition_failures.append(failure)
                        save_failure_record(
                            checkpoint_root=_checkpoint_root(args),
                            identity=failure,
                            payload=failure,
                        )
                        logger.error("word_based condition failed: %s", failure)
                        continue
                    if result is not None:
                        grouped_results.append((task.sort_index, result))
            pending_groups = [group for _sort_index, group in sorted(grouped_results)]
            logger.info(
                "word_based_metrics topic_words parallel done workers=%s groups=%s",
                topic_word_workers,
                len(pending_groups),
            )
        else:
            for task in group_tasks:
                try:
                    result = _collect_pending_word_based_group(
                        args=args,
                        task=task,
                        total_conditions=total_conditions,
                        failure_sink=condition_failures,
                    )
                except ConditionEvaluationError as exc:
                    if not _isolates_condition_failures(args):
                        raise
                    failure = _condition_failure_payload(
                        task=task,
                        args=args,
                        exc=exc,
                        stage="topic_words",
                    )
                    condition_failures.append(failure)
                    save_failure_record(
                        checkpoint_root=_checkpoint_root(args),
                        identity=failure,
                        payload=failure,
                    )
                    logger.error("word_based condition failed: %s", failure)
                    continue
                if result is not None:
                    pending_groups.append(result)

        if pending_groups:
            coherence_topic_words = [
                (
                    truncate_topic_words(
                        pending_iteration.topic_words,
                        args.coherence_topn,
                    )
                    if args.coherence_topn is not None
                    else pending_iteration.topic_words
                )
                for group in pending_groups
                for pending_iteration in group.iterations
            ]
            reference_path = resolve_project_path(args.coherence_reference_path)
            target_words = collect_target_words(coherence_topic_words)
            window_sizes = effective_window_sizes_for_coherences(
                coherences,
                window_size=args.coherence_window_size,
            )
            reference_query = build_reference_count_query(
                topic_words_by_condition=coherence_topic_words,
                window_sizes=window_sizes,
                need_document_counts="doc_npmi" in coherences,
            )
            reference_identity = reference_corpus_identity(reference_path)
            reference_index_root = getattr(args, "reference_index_root", None)
            if reference_index_root is None:
                # build_reference_index rejects an index built with a different
                # min_doc_tokens as stale, so the default root has to be keyed on
                # it too. Otherwise changing the threshold either aborts the run
                # ("build") or misses the cache forever ("auto").
                reference_index_root = (
                    args.out_root
                    / ".cache"
                    / "reference_index"
                    / "v1"
                    / fingerprint_payload(
                        {
                            **reference_identity,
                            "min_doc_tokens": int(
                                args.coherence_reference_min_doc_tokens
                            ),
                        }
                    )
                )
            else:
                reference_index_root = resolve_project_path(reference_index_root)
            reference_cache_mode = str(
                getattr(args, "reference_count_cache_mode", "auto")
            )
            # Legacy (v1) cache entries are never read: they do not record the
            # requested pair set, so a hit could silently treat missing pair
            # co-occurrences as zero. Only the v2 cache validates pair coverage.
            cache_v2_hit = (
                load_reference_count_cache_v2(
                    cache_root=args.out_root / ".cache",
                    reference_identity=reference_identity,
                    max_docs=args.coherence_reference_max_docs,
                    min_doc_tokens=int(args.coherence_reference_min_doc_tokens),
                    query=reference_query,
                )
                if reference_cache_mode == "auto"
                else None
            )
            shared_counts = cache_v2_hit[0] if cache_v2_hit is not None else None
            if shared_counts is not None:
                logger.info(
                    "wb reference_counts cache hit path=%s",
                    cache_v2_hit[1],
                )
            else:
                shared_counts = build_shared_reference_counts(
                    reference_path=reference_path,
                    target_words=target_words,
                    window_sizes=window_sizes,
                    max_docs=args.coherence_reference_max_docs,
                    min_doc_tokens=args.coherence_reference_min_doc_tokens,
                    backend=_coherence_count_backend(args),
                    workers=_coherence_count_workers(args),
                    chunk_size=_coherence_count_chunk_size(args),
                    progress_label="wb reference_counts",
                    query=reference_query,
                    reference_index_mode=str(
                        getattr(args, "reference_index_mode", "off")
                    ),
                    reference_index_root=reference_index_root,
                    max_pending=_reference_count_max_pending(args),
                )
                if reference_cache_mode != "off" and isinstance(
                    shared_counts, SharedReferenceCounts
                ):
                    reference_cache_path = save_reference_count_cache_v2(
                        cache_root=args.out_root / ".cache",
                        reference_identity=reference_identity,
                        max_docs=args.coherence_reference_max_docs,
                        min_doc_tokens=int(args.coherence_reference_min_doc_tokens),
                        query=reference_query,
                        counts=shared_counts,
                    )
                    logger.info(
                        "wb reference_counts cache saved path=%s",
                        reference_cache_path,
                    )

            def write_scored_group(scored_group: ScoredWordBasedGroup) -> None:
                group = scored_group.group
                group_args = argparse.Namespace(**vars(args))
                group_args.num_topics = int(group.num_topics)
                _write_word_based_group_outputs(
                    args=group_args,
                    model=group.model,
                    data_run=group.data_run,
                    category=group.category,
                    coherences=coherences,
                    primary_coherence=primary_coherence,
                    multiple_coherences=multiple_coherences,
                    coherence_window_sizes=coherence_window_sizes,
                    coherence_window_size_sources=coherence_window_size_sources,
                    coherence_implementations=coherence_implementations,
                    coherence_min_window_counts=coherence_min_window_counts,
                    metric_names=metric_names,
                    per_iter_metrics=scored_group.per_iter_metrics,
                    per_iter_topic_words=scored_group.per_iter_topic_words,
                    used_iterations=scored_group.used_iterations,
                    topic_word_source=group.topic_word_source,
                    topic_word_score_mode=group.topic_word_score_mode,
                    topic_word_score_definition=group.topic_word_score_definition,
                    coherence_reference_num_docs=shared_counts.num_docs,
                    coherence_reference_vocab_size=shared_counts.vocab_size,
                    coherence_reference_streaming=True,
                    summary_rows=summary_rows,
                    summary_provenance=summary_provenance,
                    runtime_iterations=[
                        (item.iteration, _load_pending_runtime(item))
                        for item in group.iterations
                    ],
                )

            score_workers = min(_coherence_score_workers(args), len(pending_groups))
            if score_workers > 1:
                logger.info(
                    "word_based_metrics scoring parallel start workers=%s groups=%s",
                    score_workers,
                    len(pending_groups),
                )
                with ThreadPoolExecutor(max_workers=score_workers) as executor:
                    futures = {
                        executor.submit(
                            _score_pending_word_based_group,
                            group=group,
                            args=args,
                            metric_names=metric_names,
                            coherences=coherences,
                            shared_counts=shared_counts,
                        ): index
                        for index, group in enumerate(pending_groups)
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            write_scored_group(future.result())
                        except ConditionEvaluationError as exc:
                            if not _isolates_condition_failures(args):
                                raise
                            group = pending_groups[index]
                            task = PendingWordBasedGroupTask(
                                sort_index=index,
                                data_run=group.data_run,
                                model=group.model,
                                num_topics=group.num_topics,
                                category=group.category,
                                progress_start=0,
                            )
                            failure = _condition_failure_payload(
                                task=task,
                                args=args,
                                exc=exc,
                                stage="scoring",
                            )
                            condition_failures.append(failure)
                            save_failure_record(
                                checkpoint_root=_checkpoint_root(args),
                                identity=failure,
                                payload=failure,
                            )
                logger.info(
                    "word_based_metrics scoring parallel done workers=%s groups=%s",
                    score_workers,
                    len(pending_groups),
                )
            else:
                for group in pending_groups:
                    try:
                        scored_group = _score_pending_word_based_group(
                            group=group,
                            args=args,
                            metric_names=metric_names,
                            coherences=coherences,
                            shared_counts=shared_counts,
                        )
                        write_scored_group(scored_group)
                    except ConditionEvaluationError as exc:
                        if not _isolates_condition_failures(args):
                            raise
                        task = PendingWordBasedGroupTask(
                            sort_index=0,
                            data_run=group.data_run,
                            model=group.model,
                            num_topics=group.num_topics,
                            category=group.category,
                            progress_start=0,
                        )
                        failure = _condition_failure_payload(
                            task=task,
                            args=args,
                            exc=exc,
                            stage="scoring",
                        )
                        condition_failures.append(failure)
                        save_failure_record(
                            checkpoint_root=_checkpoint_root(args),
                            identity=failure,
                            payload=failure,
                        )
        output_root = reporting_module.write_summary_outputs(
            out_root=args.out_root,
            summary_rows=summary_rows,
            dataset=args.dataset,
            data_runs=list(args.data_run),
            num_topics=display_num_topics,
            iterations=list(args.iteration),
            coherence_metric=",".join(coherences),
            metric_names=metric_names,
            summary_provenance=summary_provenance,
            failure_records=condition_failures,
            failure_checkpoint_root=_checkpoint_root(args),
        )
        logger.info(
            "word_based_metrics done dataset=%s num_topics=%s total_conditions=%s",
            args.dataset,
            display_num_topics,
            total_conditions,
        )
        if condition_failures and _raises_after_isolated_failures(args):
            raise WordBasedConditionFailures(condition_failures)
        return output_root

    if len(num_topics_values) > 1:
        output_root = args.out_root
        for num_topics in num_topics_values:
            topic_args = argparse.Namespace(**vars(args))
            topic_args.num_topics = int(num_topics)
            output_root = run_topic_coherence_analysis_from_args(topic_args)
        return output_root

    for data_run in args.data_run:
        for model in models:
            for category in args.category:
                if _should_skip_existing_output(
                    model=model,
                    data_run=data_run,
                    category=category,
                    args=args,
                    coherences=coherences,
                    coherence_window_sizes=coherence_window_sizes,
                    coherence_implementations=coherence_implementations,
                    coherence_min_window_counts=coherence_min_window_counts,
                    metric_names=metric_names,
                ):
                    continue
                logger.info(
                    "word_based_metrics condition_group data_run=%s model=%s category=%s "
                    "iterations=%s",
                    data_run,
                    model,
                    category,
                    list(args.iteration),
                )
                per_iter_metrics: list[dict[str, float]] = []
                per_iter_topic_words: list[dict[str, object]] = []
                runtime_iterations: list[tuple[int, RuntimeTopicWords]] = []
                used_iterations: list[int] = []
                topic_word_source: str | None = None
                topic_word_score_mode: str = ""
                topic_word_score_definition: str = ""
                coherence_reference_num_docs = 0
                coherence_reference_vocab_size = 0

                for iteration in args.iteration:
                    processed_conditions += 1
                    condition_progress = f"{processed_conditions}/{total_conditions}"
                    iteration_started = perf_counter()
                    logger.info(
                        "wb %s start data_run=%s model=%s category=%s iteration=%s",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                    )
                    stage_started = perf_counter()
                    logger.info(
                        "wb %s inputs start data_run=%s model=%s category=%s "
                        "iteration=%s split=%s",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                        args.coherence_split,
                    )
                    split_csvs, resolved_target_column = (
                        _resolve_split_csvs_and_target_column(
                            model=model,
                            dataset=args.dataset,
                            data_run=data_run,
                            iteration=iteration,
                            num_topics=args.num_topics,
                            category=category,
                            split=args.coherence_split,
                            embedding_variant=_effective_embedding_variant_for_model(
                                model, args
                            ),
                            prior_scale=_effective_prior_scale_for_model(model, args),
                            covariance_type=_effective_covariance_type_for_model(
                                model, args
                            ),
                            vmf_variant=_effective_vmf_variant_for_model(model, args),
                        )
                    )
                    logger.info(
                        "wb %s inputs done data_run=%s model=%s category=%s "
                        "iteration=%s target=%s csvs=%s sec=%.1f",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                        resolved_target_column,
                        0 if split_csvs is None else len(split_csvs),
                        perf_counter() - stage_started,
                    )
                    stage_started = perf_counter()
                    logger.info(
                        "wb %s corpus start data_run=%s model=%s category=%s "
                        "iteration=%s",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                    )
                    (
                        topic_word_texts,
                        topic_word_dictionary,
                        topic_word_corpus_bow,
                    ) = _get_corpus_bundle_cached(
                        cache=coherence_cache,
                        dataset=args.dataset,
                        data_run=data_run,
                        category=category,
                        split=args.coherence_split,
                        min_token_len=args.coherence_min_token_len,
                        language=args.language,
                        delimiter=args.delimiter,
                        ja_replace_num=args.ja_replace_num,
                        ja_dicdir=args.ja_dicdir,
                        ja_require_unidic=args.ja_require_unidic,
                        dict_no_below=args.dict_no_below,
                        dict_no_above=args.dict_no_above,
                        dict_exclude_tokens=_dict_exclude_tokens(args),
                        dict_exclude_single_alpha=args.dict_exclude_single_alpha,
                        dict_exclude_with_digit=args.dict_exclude_with_digit,
                        dict_exclude_hiragana_only=args.dict_exclude_hiragana_only,
                        exclude_labels=None,
                        split_csvs=split_csvs,
                        target_column=resolved_target_column,
                        **_reference_band_kwargs(args),
                    )
                    logger.info(
                        "wb %s corpus done data_run=%s model=%s category=%s "
                        "iteration=%s docs=%s vocab=%s sec=%.1f",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                        len(topic_word_texts),
                        len(topic_word_dictionary),
                        perf_counter() - stage_started,
                    )
                    stage_started = perf_counter()
                    logger.info(
                        "wb %s topic_words start data_run=%s model=%s category=%s "
                        "iteration=%s",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                    )
                    try:
                        (
                            topic_words_result,
                            topic_word_texts,
                            topic_word_dictionary,
                            topic_word_corpus_bow,
                        ) = _resolve_topic_words_result(
                            args=args,
                            cache=coherence_cache,
                            model=model,
                            data_run=data_run,
                            category=category,
                            iteration=iteration,
                            split_csvs=split_csvs,
                            target_column=resolved_target_column,
                            texts=topic_word_texts,
                            dictionary=topic_word_dictionary,
                            corpus_bow=topic_word_corpus_bow,
                        )
                    except ConditionEvaluationError as exc:
                        if not _isolates_condition_failures(args):
                            raise
                        failure_task = PendingWordBasedGroupTask(
                            sort_index=0,
                            data_run=data_run,
                            model=model,
                            num_topics=int(args.num_topics),
                            category=category,
                            progress_start=0,
                        )
                        failure = _condition_failure_payload(
                            task=failure_task,
                            args=args,
                            exc=exc,
                            stage="topic_words",
                            iteration=int(iteration),
                        )
                        condition_failures.append(failure)
                        save_failure_record(
                            checkpoint_root=_checkpoint_root(args),
                            identity=failure,
                            payload=failure,
                        )
                        logger.error("word_based iteration excluded: %s", failure)
                        continue

                    topic_words: TopicWords = topic_words_result.topic_words
                    topic_word_source = topic_words_result.topic_word_source
                    topic_word_score_mode = topic_words_result.score_mode or ""
                    topic_word_score_definition = (
                        topic_words_result.score_definition or ""
                    )
                    assert isinstance(
                        topic_words_result.runtime_payload, RuntimeTopicWords
                    )
                    runtime = topic_words_result.runtime_payload
                    # Runtimes are resolved with allow_empty_topics=True, so a
                    # degenerate topic set arrives as a report rather than an
                    # exception. Exclude it here exactly as the shared-counts
                    # path does, otherwise the same model publishes normal
                    # metrics over an empty topic whenever the reference corpus
                    # or the coherence set avoids the streaming path.
                    if runtime.empty_topic_ids and not _uses_mvtm_fixed_k_policy(
                        args,
                        model=model,
                    ):
                        empty_topic_task = PendingWordBasedGroupTask(
                            sort_index=0,
                            data_run=data_run,
                            model=model,
                            num_topics=int(args.num_topics),
                            category=category,
                            progress_start=0,
                        )
                        empty_topic_identity = _topic_word_checkpoint_identity(
                            args=args,
                            model=model,
                            data_run=data_run,
                            category=category,
                            iteration=int(iteration),
                            vocabulary_fingerprint=fingerprint_jsonable(
                                {
                                    "ordered_vocabulary": [
                                        str(topic_word_dictionary[index])
                                        for index in range(len(topic_word_dictionary))
                                    ]
                                }
                            ),
                        )
                        partial_pointer = _persist_partial_topic_words(
                            args=args,
                            task=empty_topic_task,
                            iteration=int(iteration),
                            runtime=runtime,
                            checkpoint_identity=empty_topic_identity,
                        )
                        empty_topic_failure = _condition_failure_payload(
                            task=empty_topic_task,
                            args=args,
                            exc=EmptyTopicError(
                                topic_ids=list(runtime.empty_topic_ids)
                            ),
                            stage="topic_words",
                            iteration=int(iteration),
                        )
                        empty_topic_failure["partial_artifact_current"] = str(
                            partial_pointer
                        )
                        condition_failures.append(empty_topic_failure)
                        save_failure_record(
                            checkpoint_root=_checkpoint_root(args),
                            identity={
                                key: value
                                for key, value in empty_topic_failure.items()
                                if key != "partial_artifact_current"
                            },
                            payload=empty_topic_failure,
                        )
                        logger.error(
                            "word_based iteration excluded: %s", empty_topic_failure
                        )
                        continue
                    if runtime.empty_topic_ids:
                        logger.warning(
                            "wb %s MvTM fixed-K evaluation includes empty topics "
                            "ids=%s",
                            condition_progress,
                            list(runtime.empty_topic_ids),
                        )
                    runtime_iterations.append((int(iteration), runtime))
                    logger.info(
                        "wb %s topic_words done data_run=%s model=%s category=%s "
                        "iteration=%s source=%s topics=%s sec=%.1f",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                        topic_word_source,
                        len(topic_words),
                        perf_counter() - stage_started,
                    )
                    stage_started = perf_counter()
                    logger.info(
                        "wb %s metrics start data_run=%s model=%s category=%s "
                        "iteration=%s reference=%s coherences=%s",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                        args.coherence_reference,
                        ",".join(coherences),
                    )
                    coherence_reference_streaming = False
                    fixed_k_reporting = _uses_fixed_k_reporting(args)
                    scoring_topic_words = (
                        [topic for topic in topic_words if topic]
                        if fixed_k_reporting
                        else topic_words
                    )
                    scoring_metric_names = (
                        _metric_names_for_coherences(coherences)
                        if fixed_k_reporting
                        else metric_names
                    )
                    if args.coherence_reference == "wikipedia":
                        assert args.coherence_reference_path is not None
                        if _uses_streaming_reference(args):
                            coherence_reference_streaming = True
                            metrics = {}
                            coherence_topic_words = (
                                truncate_topic_words(
                                    scoring_topic_words,
                                    args.coherence_topn,
                                )
                                if args.coherence_topn is not None
                                else scoring_topic_words
                            )
                            try:
                                streaming_result = (
                                    compute_streaming_reference_coherence_scores(
                                        topic_words=coherence_topic_words,
                                        reference_path=resolve_project_path(
                                            args.coherence_reference_path
                                        ),
                                        coherences=coherences,
                                        window_size=args.coherence_window_size,
                                        max_docs=args.coherence_reference_max_docs,
                                        min_doc_tokens=(
                                            args.coherence_reference_min_doc_tokens
                                        ),
                                        min_window_count=getattr(
                                            args,
                                            "coherence_min_window_count",
                                            None,
                                        ),
                                        progress_label=(
                                            f"wb {condition_progress} metrics"
                                        ),
                                    )
                                )
                            except ConditionEvaluationError as exc:
                                if not _isolates_condition_failures(args):
                                    raise
                                runtime_iterations.pop()
                                _record_condition_failure(
                                    args=args,
                                    task=PendingWordBasedGroupTask(
                                        sort_index=0,
                                        data_run=data_run,
                                        model=model,
                                        num_topics=int(args.num_topics),
                                        category=category,
                                        progress_start=0,
                                    ),
                                    exc=exc,
                                    stage="scoring",
                                    failures=condition_failures,
                                    iteration=int(iteration),
                                )
                                continue
                            for (
                                coherence_name,
                                score,
                            ) in streaming_result.scores.items():
                                metrics[
                                    coherence_metric_key(
                                        coherence_name,
                                        multiple=multiple_coherences,
                                    )
                                ] = score
                            coherence_reference_num_docs = streaming_result.num_docs
                            coherence_reference_vocab_size = streaming_result.vocab_size
                            if "diversity" in metric_names:
                                diversity_topic_words = (
                                    truncate_topic_words(
                                        scoring_topic_words,
                                        args.diversity_topn,
                                    )
                                    if args.diversity_topn is not None
                                    else scoring_topic_words
                                )
                                metrics["diversity"] = compute_topic_diversity(
                                    diversity_topic_words
                                )
                        else:
                            (
                                coherence_texts,
                                coherence_dictionary,
                                coherence_corpus_bow,
                            ) = _get_reference_corpus_bundle_cached(
                                cache=reference_corpus_cache,
                                path=args.coherence_reference_path,
                                max_docs=args.coherence_reference_max_docs,
                                min_doc_tokens=(
                                    args.coherence_reference_min_doc_tokens
                                ),
                                dict_no_below=args.dict_no_below,
                                dict_no_above=args.dict_no_above,
                                dict_exclude_tokens=_dict_exclude_tokens(args),
                                dict_exclude_single_alpha=(
                                    args.dict_exclude_single_alpha
                                ),
                                dict_exclude_with_digit=(args.dict_exclude_with_digit),
                                dict_exclude_hiragana_only=(
                                    args.dict_exclude_hiragana_only
                                ),
                            )
                            try:
                                metrics = evaluate_topic_words(
                                    topic_words=scoring_topic_words,
                                    metric_names=scoring_metric_names,
                                    texts=coherence_texts,
                                    dictionary=coherence_dictionary,
                                    corpus_bow=coherence_corpus_bow,
                                    coherence=coherences,
                                    coherence_topn=args.coherence_topn,
                                    diversity_topn=args.diversity_topn,
                                    coherence_window_size=args.coherence_window_size,
                                    coherence_min_window_count=getattr(
                                        args,
                                        "coherence_min_window_count",
                                        None,
                                    ),
                                    progress_label=f"wb {condition_progress} metrics",
                                )
                            except ConditionEvaluationError as exc:
                                if not _isolates_condition_failures(args):
                                    raise
                                runtime_iterations.pop()
                                _record_condition_failure(
                                    args=args,
                                    task=PendingWordBasedGroupTask(
                                        sort_index=0,
                                        data_run=data_run,
                                        model=model,
                                        num_topics=int(args.num_topics),
                                        category=category,
                                        progress_start=0,
                                    ),
                                    exc=exc,
                                    stage="scoring",
                                    failures=condition_failures,
                                    iteration=int(iteration),
                                )
                                continue
                            coherence_reference_num_docs = len(coherence_texts)
                            coherence_reference_vocab_size = len(coherence_dictionary)
                    else:
                        coherence_texts = topic_word_texts
                        coherence_dictionary = topic_word_dictionary
                        coherence_corpus_bow = topic_word_corpus_bow
                        try:
                            metrics = evaluate_topic_words(
                                topic_words=scoring_topic_words,
                                metric_names=scoring_metric_names,
                                texts=coherence_texts,
                                dictionary=coherence_dictionary,
                                corpus_bow=coherence_corpus_bow,
                                coherence=coherences,
                                coherence_topn=args.coherence_topn,
                                diversity_topn=args.diversity_topn,
                                coherence_window_size=args.coherence_window_size,
                                coherence_min_window_count=getattr(
                                    args,
                                    "coherence_min_window_count",
                                    None,
                                ),
                                progress_label=f"wb {condition_progress} metrics",
                            )
                        except ConditionEvaluationError as exc:
                            if not _isolates_condition_failures(args):
                                raise
                            runtime_iterations.pop()
                            _record_condition_failure(
                                args=args,
                                task=PendingWordBasedGroupTask(
                                    sort_index=0,
                                    data_run=data_run,
                                    model=model,
                                    num_topics=int(args.num_topics),
                                    category=category,
                                    progress_start=0,
                                ),
                                exc=exc,
                                stage="scoring",
                                failures=condition_failures,
                                iteration=int(iteration),
                            )
                            continue
                        coherence_reference_num_docs = len(coherence_texts)
                        coherence_reference_vocab_size = len(coherence_dictionary)
                    if fixed_k_reporting:
                        metrics = _apply_fixed_k_reporting(
                            metrics=metrics,
                            topic_words=topic_words,
                            coherences=coherences,
                            args=args,
                        )
                    metrics["num_topics"] = float(args.num_topics)
                    per_iter_metrics.append(metrics)
                    per_iter_topic_words.append(
                        {
                            "iteration": int(iteration),
                            "topics": serialize_topic_words(topic_words),
                        }
                    )
                    used_iterations.append(iteration)
                    logger.info(
                        "wb %s metrics done data_run=%s model=%s category=%s "
                        "iteration=%s coherence=%s diversity=%s ref_docs=%s "
                        "ref_vocab=%s sec=%.1f total_sec=%.1f",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                        metrics.get(
                            coherence_metric_key(
                                primary_coherence,
                                multiple=multiple_coherences,
                            )
                        ),
                        metrics.get("diversity"),
                        coherence_reference_num_docs,
                        coherence_reference_vocab_size,
                        perf_counter() - stage_started,
                        perf_counter() - iteration_started,
                    )

                if topic_word_source is None or not used_iterations:
                    logger.error(
                        "word_based group has no evaluable iterations "
                        "data_run=%s model=%s category=%s",
                        data_run,
                        model,
                        category,
                    )
                    continue
                write_started = perf_counter()
                logger.info(
                    "wb write start data_run=%s model=%s category=%s iterations=%s",
                    data_run,
                    model,
                    category,
                    list(used_iterations),
                )
                agg = aggregate_metrics(per_iter_metrics, metric_names=metric_names)
                provenance = resolve_model_provenance(
                    model=model,
                    dataset=args.dataset,
                    # Keyed on the requested iterations so the written condition
                    # id matches what _expected_output_condition_id probes.
                    iteration=int(min(args.iteration)),
                    num_topics=args.num_topics,
                    category=category,
                    data_run=data_run,
                    embedding_variant=_effective_embedding_variant_for_model(
                        model, args
                    ),
                    prior_scale=_effective_prior_scale_for_model(model, args),
                    covariance_type=_effective_covariance_type_for_model(model, args),
                    vmf_variant=_effective_vmf_variant_for_model(model, args),
                )
                condition_id, condition_fingerprint = _build_output_condition_id(
                    model=model,
                    dataset=args.dataset,
                    data_run=data_run,
                    category=category,
                    iterations=[int(value) for value in args.iteration],
                    num_topics=args.num_topics,
                    coherence=primary_coherence,
                    coherences=coherences if multiple_coherences else None,
                    coherence_topn=args.coherence_topn,
                    coherence_window_size=(
                        coherence_window_sizes
                        if multiple_coherences
                        else coherence_window_size
                    ),
                    coherence_implementation=(
                        coherence_implementations
                        if multiple_coherences
                        else coherence_implementation
                    ),
                    coherence_min_window_count=(
                        coherence_min_window_counts
                        if multiple_coherences
                        else coherence_min_window_count
                    ),
                    coherence_reference=args.coherence_reference,
                    coherence_reference_path=(
                        None
                        if args.coherence_reference_path is None
                        else str(resolve_project_path(args.coherence_reference_path))
                    ),
                    coherence_reference_format=(
                        args.coherence_reference_format
                        if args.coherence_reference == "wikipedia"
                        else None
                    ),
                    coherence_reference_max_docs=args.coherence_reference_max_docs,
                    coherence_reference_min_doc_tokens=(
                        args.coherence_reference_min_doc_tokens
                    ),
                    coherence_reference_streaming=coherence_reference_streaming,
                    diversity_topn=args.diversity_topn,
                    coherence_split=args.coherence_split,
                    topic_word_source=topic_word_source,
                    embedding_variant=_effective_embedding_variant_for_model(
                        model, args
                    ),
                    prior_scale=_effective_prior_scale_for_model(model, args),
                    covariance_type=_effective_covariance_type_for_model(model, args),
                    source_condition_id=(
                        None
                        if provenance.get("condition_id") is None
                        else str(provenance["condition_id"])
                    ),
                    source_condition_fingerprint=(
                        None
                        if provenance.get("condition_fingerprint") is None
                        else str(provenance["condition_fingerprint"])
                    ),
                    parameter_variant=(
                        None
                        if provenance.get("parameter_variant") is None
                        else str(provenance["parameter_variant"])
                    ),
                    metric_names=metric_names,
                    dict_exclude_tokens=_dict_exclude_tokens(args),
                    posterior_settings=_posterior_settings(args),
                    topic_word_score_mode=_topic_word_score_mode(args),
                    dict_no_above=float(args.dict_no_above),
                    reference_min_df=_reference_min_df(args),
                    reference_max_df_ratio=_reference_max_df_ratio(args),
                )
                display_key = condition_id
                started_at = datetime.now(UTC).isoformat()
                execution_id = build_execution_id(
                    prefix="exec",
                    started_at=started_at,
                )
                uses_default_output_layout = _uses_default_output_layout(args.out_root)
                if uses_default_output_layout:
                    archive_out_dir = build_archive_result_dir(
                        base_root=args.out_root,
                        dataset=args.dataset,
                        data_run=data_run,
                        category=category,
                        display_key=display_key,
                        started_at=started_at,
                        execution_id=execution_id,
                    )
                    latest_out_dir = build_latest_result_dir(
                        base_root=args.out_root,
                        dataset=args.dataset,
                        data_run=data_run,
                        category=category,
                        display_key=display_key,
                    )
                    out_dir = archive_out_dir
                else:
                    archive_out_dir = None
                    latest_out_dir = None
                    out_dir = (
                        args.out_root
                        / args.dataset
                        / data_run
                        / category
                        / condition_id
                    )
                ensure_directory(out_dir)
                coherence_reference_path = (
                    None
                    if args.coherence_reference_path is None
                    else str(resolve_project_path(args.coherence_reference_path))
                )
                coherence_reference_meta = {
                    "coherence_reference": args.coherence_reference,
                    "coherence_reference_path": coherence_reference_path,
                    "coherence_reference_format": (
                        args.coherence_reference_format
                        if args.coherence_reference == "wikipedia"
                        else None
                    ),
                    "coherence_reference_num_docs": int(coherence_reference_num_docs),
                    "coherence_reference_vocab_size": int(
                        coherence_reference_vocab_size
                    ),
                    "coherence_reference_max_docs": args.coherence_reference_max_docs,
                    "coherence_reference_min_doc_tokens": int(
                        args.coherence_reference_min_doc_tokens
                    ),
                    "coherence_reference_streaming": bool(
                        coherence_reference_streaming
                    ),
                    "coherence_reference_language": (
                        "en" if args.coherence_reference == "wikipedia" else None
                    ),
                }
                requested_embedding_variant = _requested_embedding_variant(args)
                effective_embedding_variant = (
                    model_inputs_module.effective_embedding_variant(
                        model,
                        requested_embedding_variant,
                    )
                )

                metrics_meta = build_evaluation_meta(
                    task="word_based_metrics",
                    model=model,
                    dataset=args.dataset,
                    data_run=data_run,
                    num_topics=args.num_topics,
                    category=category,
                    condition_id=condition_id,
                    display_key=display_key,
                    condition_fingerprint=condition_fingerprint,
                    embedding_variant=requested_embedding_variant,
                    effective_embedding_variant=effective_embedding_variant,
                    prior_scale=_effective_prior_scale_for_model(model, args),
                    covariance_type=_effective_covariance_type_for_model(model, args),
                    iterations=used_iterations,
                    started_at=started_at,
                    execution_id=execution_id,
                    archive_dir=str(out_dir),
                    latest_dir=None if latest_out_dir is None else str(latest_out_dir),
                    model_provenance=provenance,
                    source_condition_id=provenance.get("condition_id"),
                    source_condition_fingerprint=provenance.get(
                        "condition_fingerprint"
                    ),
                    parameter_variant=provenance.get("parameter_variant"),
                    metric_names=metric_names,
                    topic_words={
                        "topn": int(_requested_topic_word_topn(args)),
                        "coherence_topn": int(args.coherence_topn),
                        "diversity_topn": int(args.diversity_topn),
                        "source": topic_word_source,
                        "score_mode": topic_word_score_mode,
                        "score_definition": topic_word_score_definition,
                    },
                    coherence=_coherence_meta(
                        coherences=coherences,
                        args=args,
                        model=model,
                        topic_word_source=topic_word_source,
                        topic_word_score_mode=topic_word_score_mode,
                        topic_word_score_definition=topic_word_score_definition,
                        reference_meta=coherence_reference_meta,
                        window_sizes=coherence_window_sizes,
                        window_size_sources=coherence_window_size_sources,
                        min_window_counts=coherence_min_window_counts,
                    ),
                    diversity={
                        "topn": int(args.diversity_topn),
                        "topic_word_source": topic_word_source,
                    },
                )
                (
                    evaluation_topic_words_path,
                    display_topic_words_path,
                    iteration_artifacts,
                    runtime_meta,
                ) = _persist_runtime_topic_word_artifacts(
                    out_dir=out_dir,
                    model=model,
                    split=args.coherence_split,
                    runtimes_by_iteration=runtime_iterations,
                    common_meta={
                        "dataset": args.dataset,
                        "data_run": data_run,
                        "category": category,
                        "num_topics": args.num_topics,
                        "condition_id": condition_id,
                        "condition_fingerprint": condition_fingerprint,
                        "iterations": used_iterations,
                        "model_provenance": provenance,
                    },
                    evaluation_score_mode=_topic_word_score_mode(args),
                )
                metrics_meta.update(runtime_meta)
                metrics_meta["topic_word_score_mode"] = _topic_word_score_mode(args)
                metrics_meta["topic_word_ranking_schema_version"] = (
                    TOPIC_WORD_RANKING_SCHEMA_VERSION
                )
                metrics_meta["posterior_settings"] = _posterior_settings(args)
                metrics_meta["empty_topic_evaluation"] = _empty_topic_policy_meta(
                    args=args,
                    model=model,
                )
                metrics_meta["requested_iterations"] = [
                    int(value) for value in args.iteration
                ]
                metrics_meta["evaluated_iterations"] = [
                    int(value) for value in used_iterations
                ]
                metrics_meta["degenerate_iterations"] = sorted(
                    set(int(value) for value in args.iteration) - set(used_iterations)
                )
                metrics_results = {
                    "aggregate": agg,
                    "per_iteration": per_iter_metrics,
                    "topic_words_evaluation_topk": {
                        "topn": int(_requested_topic_word_topn(args)),
                        "coherence_topn": int(args.coherence_topn),
                        "diversity_topn": int(args.diversity_topn),
                        "per_iteration": per_iter_topic_words,
                    },
                    **runtime_meta,
                }
                out_path = out_dir / "metrics_agg.json"
                write_evaluation_json(
                    meta=metrics_meta,
                    results=metrics_results,
                    path=out_path,
                )
                logger.info(f"[{model}] aggregated metrics saved to {out_path}")

                logger.info(
                    "[%s] evaluation/display top words saved to %s and %s",
                    model,
                    evaluation_topic_words_path,
                    display_topic_words_path,
                )

                metadata_path = out_dir / "metadata.json"
                save_json(metrics_meta, metadata_path)
                logger.info(f"[{model}] metadata saved to {metadata_path}")

                if uses_default_output_layout and archive_out_dir is not None:
                    pointer_path = write_latest_result_pointer(
                        base_root=args.out_root,
                        task="word_based_metrics",
                        dataset=args.dataset,
                        data_run=data_run,
                        category=category,
                        display_key=display_key,
                        archive_dir=archive_out_dir,
                        started_at=started_at,
                        execution_id=execution_id,
                        condition_fingerprint=condition_fingerprint,
                        artifacts={
                            "metrics_agg": out_path.name,
                            "topic_words_evaluation_topk": (
                                evaluation_topic_words_path.name
                            ),
                            "topic_words_display_topk": display_topic_words_path.name,
                            "topic_words_probability_topk": runtime_meta[
                                "topic_words_probability_topk"
                            ],
                            "iteration_artifacts": iteration_artifacts,
                            "metadata": metadata_path.name,
                        },
                    )
                    logger.info(
                        "[%s] updated latest pointer at %s",
                        model,
                        pointer_path,
                    )

                logger.info(
                    "wb write done data_run=%s model=%s category=%s out_dir=%s "
                    "sec=%.1f",
                    data_run,
                    model,
                    category,
                    out_dir,
                    perf_counter() - write_started,
                )

                for metric_name, stats in agg.items():
                    row_coherence = _coherence_from_metric_name(
                        metric_name,
                        coherences=coherences,
                    )
                    row_coherence_details = (
                        describe_coherence_metric(row_coherence)
                        if row_coherence is not None
                        else None
                    )
                    summary_rows.append(
                        {
                            "dataset": args.dataset,
                            "data_run": data_run,
                            "num_topics": args.num_topics,
                            "category": category,
                            "model": model,
                            "metric": metric_name,
                            "mean": round_sigfigs(stats.get("mean", float("nan"))),
                            "std": round_sigfigs(stats.get("std", float("nan"))),
                            "coherence_metric": (
                                row_coherence if row_coherence is not None else ""
                            ),
                            "coherence_implementation": (
                                coherence_implementations[row_coherence]
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_definition": (
                                row_coherence_details["definition"]
                                if row_coherence_details is not None
                                else ""
                            ),
                            "coherence_cooccurrence_unit": (
                                row_coherence_details["cooccurrence_unit"]
                                if row_coherence_details is not None
                                else ""
                            ),
                            "coherence_zero_cooccurrence_policy": (
                                row_coherence_details["zero_cooccurrence_policy"]
                                if row_coherence_details is not None
                                else ""
                            ),
                            "coherence_split": (
                                args.coherence_split
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_topn": (
                                args.coherence_topn if row_coherence is not None else ""
                            ),
                            "coherence_window_size": (
                                coherence_window_sizes[row_coherence]
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_window_size_source": (
                                coherence_window_size_sources[row_coherence]
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_min_window_count": (
                                coherence_min_window_counts[row_coherence]
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_reference": (
                                args.coherence_reference
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_reference_path": (
                                coherence_reference_path
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_reference_format": (
                                (
                                    args.coherence_reference_format
                                    if args.coherence_reference == "wikipedia"
                                    else ""
                                )
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_reference_num_docs": (
                                coherence_reference_num_docs
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_reference_vocab_size": (
                                coherence_reference_vocab_size
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_reference_max_docs": (
                                args.coherence_reference_max_docs
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_reference_min_doc_tokens": (
                                args.coherence_reference_min_doc_tokens
                                if row_coherence is not None
                                else ""
                            ),
                            "coherence_reference_streaming": (
                                coherence_reference_streaming
                                if row_coherence is not None
                                else ""
                            ),
                            "diversity_topn": (
                                args.diversity_topn
                                if metric_name == "diversity"
                                else ""
                            ),
                            "topic_word_topn": _requested_topic_word_topn(args),
                            "dict_no_below": (
                                args.dict_no_below if row_coherence is not None else ""
                            ),
                            "dict_no_above": (
                                args.dict_no_above if row_coherence is not None else ""
                            ),
                            "dict_exclude_tokens": (
                                ",".join(sorted(_dict_exclude_tokens(args)))
                                if row_coherence is not None
                                else ""
                            ),
                            "dict_exclude_single_alpha": (
                                args.dict_exclude_single_alpha
                                if row_coherence is not None
                                else ""
                            ),
                            "dict_exclude_with_digit": (
                                args.dict_exclude_with_digit
                                if row_coherence is not None
                                else ""
                            ),
                            "dict_exclude_hiragana_only": (
                                args.dict_exclude_hiragana_only
                                if row_coherence is not None
                                else ""
                            ),
                            "language": (
                                args.language if row_coherence is not None else ""
                            ),
                            "embedding_variant": requested_embedding_variant,
                            "effective_embedding_variant": effective_embedding_variant,
                            "prior_scale": _effective_prior_scale_for_model(
                                model, args
                            ),
                            "topic_word_source": topic_word_source,
                            "topic_word_score_mode": topic_word_score_mode,
                        }
                    )
                summary_provenance.append(
                    {
                        "model": model,
                        "data_run": data_run,
                        "category": category,
                        "model_provenance": provenance,
                    }
                )

    output_root = reporting_module.write_summary_outputs(
        out_root=args.out_root,
        summary_rows=summary_rows,
        dataset=args.dataset,
        data_runs=list(args.data_run),
        num_topics=args.num_topics,
        iterations=list(args.iteration),
        coherence_metric=",".join(coherences),
        metric_names=metric_names,
        summary_provenance=summary_provenance,
        failure_records=condition_failures,
        failure_checkpoint_root=_checkpoint_root(args),
    )
    logger.info(
        "word_based_metrics done dataset=%s num_topics=%s total_conditions=%s",
        args.dataset,
        args.num_topics,
        total_conditions,
    )
    if condition_failures and _raises_after_isolated_failures(args):
        raise WordBasedConditionFailures(condition_failures)
    return output_root


def run_topic_coherence_analysis(
    *,
    models: list[str],
    dataset: str,
    data_runs: list[str] | tuple[str, ...] = ("default",),
    iterations: list[int],
    num_topics: int,
    categories: list[str],
    embedding_variant: str | None = DEFAULT_EMBEDDING_VARIANT,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
    out_root: Path = DEFAULT_OUT_ROOT,
    coherence: str | list[str] | tuple[str, ...] = "c_v",
    coherence_topn: int = 10,
    coherence_window_size: int | None = None,
    coherence_min_window_count: int | None = None,
    diversity_topn: int = 25,
    topic_word_score_mode: str = DEFAULT_TOPIC_WORD_SCORE_MODE,
    gaussian_word2vec: str = "word2vec-google-news-300",
    coherence_split: str = "train",
    coherence_min_token_len: int = 2,
    dict_no_below: int = 3,
    dict_no_above: float = 0.7,
    dict_exclude_tokens: frozenset[str] = frozenset(),
    dict_exclude_single_alpha: bool = False,
    dict_exclude_with_digit: bool = False,
    dict_exclude_hiragana_only: bool = False,
    reference_min_df: int = 0,
    reference_max_df_ratio: float = 1.0,
    posterior_num_chains: int = 1,
    posterior_burn_in_sweeps: int = 20,
    posterior_retained_samples: int = 20,
    posterior_thinning: int = 1,
    posterior_seed: int = 0,
    posterior_backend: str = "numba",
    etm_theta_samples: int = 100,
    etm_posterior_seed: int = 0,
    npmi_min_expected_count: float | None = None,
    coherence_reference: str = "dataset",
    coherence_reference_path: Path | None = None,
    coherence_reference_format: str = "tokenized_jsonl",
    coherence_reference_max_docs: int | None = None,
    coherence_reference_min_doc_tokens: int = 1,
    coherence_reference_streaming: bool = False,
    coherence_count_backend: str = "numba",
    coherence_count_workers: int = DEFAULT_REFERENCE_COUNT_WORKERS,
    coherence_count_chunk_size: int = DEFAULT_REFERENCE_COUNT_CHUNK_SIZE,
    coherence_topic_word_workers: int = 1,
    topic_word_encoder_device: str = "auto",
    topic_word_encode_batch_size: int | None = None,
    coherence_score_workers: int = 1,
    skip_existing: bool = False,
    checkpoint_mode: str = "auto",
    checkpoint_root: Path | None = None,
    reference_count_cache_mode: str = "auto",
    reference_index_mode: str = "off",
    reference_index_root: Path | None = None,
    reference_count_max_pending: int | None = None,
    condition_failure_policy: str = "exclude-condition",
    mvtm_empty_topic_policy: str = "exclude",
    language: str = "english",
    delimiter: str = " / ",
    ja_replace_num: bool = True,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
) -> Path:
    args = argparse.Namespace(
        model=models,
        dataset=dataset,
        data_run=list(data_runs),
        iteration=iterations,
        num_topics=(
            [int(value) for value in num_topics]
            if isinstance(num_topics, (list, tuple))
            else int(num_topics)
        ),
        category=categories,
        embedding_variant=embedding_variant,
        prior_scale=prior_scale,
        covariance_type=covariance_type,
        vmf_variant=vmf_variant,
        out_root=out_root,
        coherence=coherence,
        coherence_topn=int(coherence_topn),
        coherence_window_size=(
            None if coherence_window_size is None else int(coherence_window_size)
        ),
        coherence_min_window_count=(
            None
            if coherence_min_window_count is None
            else int(coherence_min_window_count)
        ),
        diversity_topn=int(diversity_topn),
        topic_word_score_mode=str(topic_word_score_mode),
        gaussian_word2vec=gaussian_word2vec,
        coherence_split=coherence_split,
        coherence_min_token_len=int(coherence_min_token_len),
        dict_no_below=int(dict_no_below),
        dict_no_above=float(dict_no_above),
        dict_exclude_tokens=frozenset(str(token) for token in dict_exclude_tokens),
        dict_exclude_single_alpha=bool(dict_exclude_single_alpha),
        dict_exclude_with_digit=bool(dict_exclude_with_digit),
        dict_exclude_hiragana_only=bool(dict_exclude_hiragana_only),
        reference_min_df=int(reference_min_df),
        reference_max_df_ratio=float(reference_max_df_ratio),
        posterior_num_chains=int(posterior_num_chains),
        posterior_burn_in_sweeps=int(posterior_burn_in_sweeps),
        posterior_retained_samples=int(posterior_retained_samples),
        posterior_thinning=int(posterior_thinning),
        posterior_seed=int(posterior_seed),
        posterior_backend=posterior_backend,
        etm_theta_samples=int(etm_theta_samples),
        etm_posterior_seed=int(etm_posterior_seed),
        npmi_min_expected_count=(
            None if npmi_min_expected_count is None else float(npmi_min_expected_count)
        ),
        coherence_reference=coherence_reference,
        coherence_reference_path=coherence_reference_path,
        coherence_reference_format=coherence_reference_format,
        coherence_reference_max_docs=(
            None
            if coherence_reference_max_docs is None
            else int(coherence_reference_max_docs)
        ),
        coherence_reference_min_doc_tokens=int(coherence_reference_min_doc_tokens),
        coherence_reference_streaming=bool(coherence_reference_streaming),
        coherence_count_backend=coherence_count_backend,
        coherence_count_workers=int(coherence_count_workers),
        coherence_count_chunk_size=int(coherence_count_chunk_size),
        coherence_topic_word_workers=int(coherence_topic_word_workers),
        topic_word_encoder_device=str(topic_word_encoder_device),
        topic_word_encode_batch_size=(
            None
            if topic_word_encode_batch_size is None
            else int(topic_word_encode_batch_size)
        ),
        coherence_score_workers=int(coherence_score_workers),
        skip_existing=bool(skip_existing),
        checkpoint_mode=str(checkpoint_mode),
        checkpoint_root=checkpoint_root,
        reference_count_cache_mode=str(reference_count_cache_mode),
        reference_index_mode=str(reference_index_mode),
        reference_index_root=reference_index_root,
        reference_count_max_pending=reference_count_max_pending,
        condition_failure_policy=str(condition_failure_policy),
        mvtm_empty_topic_policy=str(mvtm_empty_topic_policy),
        language=language,
        delimiter=delimiter,
        ja_replace_num=bool(ja_replace_num),
        ja_dicdir=ja_dicdir,
        ja_require_unidic=bool(ja_require_unidic),
    )
    return run_topic_coherence_analysis_from_args(args)


run_word_based_metrics = run_topic_coherence_analysis


def main() -> None:
    run_topic_coherence_analysis_from_args(parse_args())


if __name__ == "__main__":
    main()

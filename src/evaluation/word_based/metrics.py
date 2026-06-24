from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from gensim.corpora import Dictionary

from src.core.artifacts import load_artifact_pickle, save_json
from src.core.paths import (
    build_archive_result_dir,
    build_latest_result_dir,
    resolve_project_path,
    write_latest_result_pointer,
)
from src.core.result_identity import build_execution_id
from src.evaluation.reporting import write_evaluation_json
from src.evaluation.schema import build_evaluation_meta
from src.evaluation.word_based import cli as cli_module
from src.evaluation.word_based import corpus_bundle as corpus_bundle_module
from src.evaluation.word_based import model_inputs as model_inputs_module
from src.evaluation.word_based import reporting as reporting_module
from src.evaluation.word_based.reference_counts import (
    DEFAULT_REFERENCE_COUNT_CHUNK_SIZE,
    DEFAULT_REFERENCE_COUNT_WORKERS,
    ReferenceCountBackend,
    build_shared_reference_counts,
    collect_target_words,
    compute_shared_reference_coherence_scores,
    effective_window_sizes_for_coherences,
)
from src.evaluation.word_based.topic_word_metrics import (
    DEFAULT_PALMETTO_CV_MIN_WINDOW_COUNT,
    EPSILON_SMOOTHED_COHERENCES,
    MULTI_COHERENCE_CHOICES,
    PALMETTO_CV_IMPLEMENTATION,
    STREAMING_REFERENCE_COHERENCES,
    aggregate_metrics,
    coherence_metric_key,
    compute_streaming_reference_coherence_scores,
    compute_topic_diversity,
    describe_coherence_metric,
    evaluate_topic_words,
    normalize_coherences,
    truncate_topic_words,
)
from src.evaluation.word_based.topic_words import (
    TopicWords,
    TopicWordsResult,
    extract_topic_words_from_doc_topic_npmi,
    extract_topic_words_from_learned_model,
    extract_topic_words_from_sentence_topic_npmi,
    serialize_topic_words,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

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
]

ANALYSIS_ROOT = model_inputs_module.ANALYSIS_ROOT
DEFAULT_OUT_ROOT = model_inputs_module.DEFAULT_OUT_ROOT
DEFAULT_EMBEDDING_VARIANT = model_inputs_module.DEFAULT_EMBEDDING_VARIANT
MODEL_ALIASES = model_inputs_module.MODEL_ALIASES
MODEL_CHOICES = model_inputs_module.MODEL_CHOICES
LEARNED_WORD_TOPIC_MODELS: set[str] = set()
PROXY_WORD_TOPIC_MODELS = {
    "vmf",
    "sentence_gaussianlda",
    "sentlda",
}


@dataclass
class PendingWordBasedIteration:
    iteration: int
    topic_words: TopicWords


@dataclass
class PendingWordBasedGroup:
    data_run: str
    model: str
    num_topics: int
    category: str
    iterations: list[PendingWordBasedIteration]
    topic_word_source: str
    proxy_word_score_mode: str
    proxy_word_score_definition: str


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
) -> Path:
    return model_inputs_module.build_result_dir(
        model=model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
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
) -> dict[str, object]:
    return model_inputs_module.resolve_model_provenance(
        model=model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
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
    proxy_npmi_mode: str,
    proxy_word_score_mode: str,
    embedding_variant: str | None,
    metric_names: list[str],
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
        proxy_npmi_mode=proxy_npmi_mode,
        proxy_word_score_mode=proxy_word_score_mode,
        embedding_variant=embedding_variant,
        metric_names=metric_names,
    )


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
    dict_exclude_single_alpha: bool = False,
    dict_exclude_with_digit: bool = False,
    dict_exclude_hiragana_only: bool = False,
    exclude_labels: set[str] | None = None,
    split_csvs: tuple[str, ...] | None = None,
    target_column: str = "target_str",
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
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
        exclude_labels=exclude_labels,
        split_csvs=split_csvs,
        target_column=target_column,
    )


def build_reference_corpus_bundle(
    *,
    path: Path,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
    dict_no_below: int = 3,
    dict_no_above: float = 0.7,
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
    )


def aggregate_doc_topics_from_sentence_topics(
    sentence_topics_by_doc,
    num_topics: int,
):
    return model_inputs_module.aggregate_doc_topics_from_sentence_topics(
        sentence_topics_by_doc=sentence_topics_by_doc,
        num_topics=num_topics,
    )


def load_doc_topics_proxy_soft_preferred(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
):
    return model_inputs_module.load_doc_topics_proxy_soft_preferred(
        model=model,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        embedding_variant=embedding_variant,
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
    )


def load_sentence_topics(
    model: ModelType,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    embedding_variant: str | None = None,
):
    return model_inputs_module.load_sentence_topics(
        model=model,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        embedding_variant=embedding_variant,
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
    proxy_word_score_mode: str,
    proxy_word_score_definition: str,
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
        "dict_exclude_single_alpha": bool(args.dict_exclude_single_alpha),
        "dict_exclude_with_digit": bool(args.dict_exclude_with_digit),
        "dict_exclude_hiragana_only": bool(args.dict_exclude_hiragana_only),
        "language": args.language,
        "delimiter": args.delimiter,
        "ja_replace_num": bool(args.ja_replace_num),
        "ja_dicdir": args.ja_dicdir,
        "ja_require_unidic": bool(args.ja_require_unidic),
        "proxy_npmi_mode": (
            args.proxy_npmi_mode if model in PROXY_WORD_TOPIC_MODELS else ""
        ),
        "proxy_word_score_mode": (
            proxy_word_score_mode if model in PROXY_WORD_TOPIC_MODELS else ""
        ),
        "proxy_word_score_definition": (
            proxy_word_score_definition if model in PROXY_WORD_TOPIC_MODELS else ""
        ),
        "topic_word_source": topic_word_source,
        "gaussian_word2vec": args.gaussian_word2vec,
    }


def _coherence_meta(
    *,
    coherences: list[str],
    args: argparse.Namespace,
    model: str,
    topic_word_source: str,
    proxy_word_score_mode: str,
    proxy_word_score_definition: str,
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
            proxy_word_score_mode=proxy_word_score_mode,
            proxy_word_score_definition=proxy_word_score_definition,
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


def _uses_default_output_layout(out_root: Path) -> bool:
    return resolve_project_path(out_root) == DEFAULT_OUT_ROOT


def _expected_topic_word_identity(
    *,
    model: str,
    args: argparse.Namespace,
) -> tuple[str, str, str]:
    if model in LEARNED_WORD_TOPIC_MODELS:
        return "learned_topic_word_distribution", "", ""
    if model in PROXY_WORD_TOPIC_MODELS:
        topic_word_source = (
            "sentence_topic_proxy_npmi"
            if args.proxy_npmi_mode == "sentence"
            else "document_topic_proxy_npmi"
        )
        return topic_word_source, args.proxy_npmi_mode, args.proxy_word_score_mode
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
    topic_word_source, proxy_npmi_mode, proxy_word_score_mode = (
        _expected_topic_word_identity(model=model, args=args)
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
        proxy_npmi_mode=proxy_npmi_mode,
        proxy_word_score_mode=proxy_word_score_mode,
        embedding_variant=_effective_embedding_variant_for_model(model, args),
        metric_names=metric_names,
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
    if output_path.exists():
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
    if backend not in {"python", "numba"}:
        raise ValueError(
            "coherence_count_backend must be one of {'python', 'numba'}, "
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
    dict_exclude_single_alpha: bool,
    dict_exclude_with_digit: bool,
    dict_exclude_hiragana_only: bool,
    exclude_labels: set[str] | None = None,
    split_csvs: tuple[str, ...] | None = None,
    target_column: str = "target_str",
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
        dict_exclude_single_alpha,
        dict_exclude_with_digit,
        dict_exclude_hiragana_only,
        exclude_key,
        split_csvs,
        target_column,
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
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
        exclude_labels=exclude_labels,
        split_csvs=split_csvs,
        target_column=target_column,
    )
    cache[cache_key] = bundle
    return bundle


def _get_reference_corpus_bundle_cached(
    *,
    cache: dict[
        tuple[str, str, int | None, int, int, float, bool, bool, bool],
        tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]],
    ],
    path: Path,
    max_docs: int | None,
    min_doc_tokens: int,
    dict_no_below: int,
    dict_no_above: float,
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
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
    )
    cache[cache_key] = bundle
    return bundle


def _rebuild_twenty_newsgroup_all_bundle(
    *,
    args: argparse.Namespace,
    cache,
    data_run: str,
    category: str,
    split_csvs: tuple[str, ...] | None,
    target_column: str,
) -> tuple[
    list[list[str]],
    Dictionary,
    list[list[tuple[int, int]]],
    list[str],
    list[list[list[str]]],
]:
    exclude_labels = {"misc.forsale"}
    texts, dictionary, corpus_bow = _get_corpus_bundle_cached(
        cache=cache,
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
        dict_exclude_single_alpha=args.dict_exclude_single_alpha,
        dict_exclude_with_digit=args.dict_exclude_with_digit,
        dict_exclude_hiragana_only=args.dict_exclude_hiragana_only,
        exclude_labels=exclude_labels,
        split_csvs=split_csvs,
        target_column=target_column,
    )
    documents = load_documents(
        dataset=args.dataset,
        category=category,
        split=args.coherence_split,
        split_csvs=split_csvs,
        target_column=target_column,
        exclude_labels=exclude_labels,
    )
    sentence_tokens_by_doc = tokenize_sentence_documents(
        documents=documents,
        min_token_len=args.coherence_min_token_len,
        language=args.language,
        delimiter=args.delimiter,
        ja_replace_num=args.ja_replace_num,
        ja_dicdir=args.ja_dicdir,
        ja_require_unidic=args.ja_require_unidic,
    )
    return texts, dictionary, corpus_bow, documents, sentence_tokens_by_doc


def _load_sentlda_effective_corpus_bundle(
    *,
    args: argparse.Namespace,
    data_run: str,
    category: str,
    iteration: int,
) -> tuple[
    list[list[str]],
    Dictionary,
    list[list[tuple[int, int]]],
    list[list[list[str]]],
]:
    preprocessed_path = resolve_preprocessed_corpus_path(
        model="sentlda",
        dataset=args.dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=args.num_topics,
        category=category,
        split=args.coherence_split,
        embedding_variant=model_inputs_module.effective_embedding_variant(
            "sentlda",
            _requested_embedding_variant(args),
        ),
    )
    logger.info(
        "word_based_metrics stage load_sentlda_preprocessed data_run=%s category=%s "
        "iteration=%s path=%s",
        data_run,
        category,
        iteration,
        preprocessed_path,
    )
    raw_preprocessed = load_artifact_pickle(preprocessed_path)
    if not isinstance(raw_preprocessed, list):
        raise ValueError(
            "Expected sentlda preprocessed corpus to be a list, got "
            f"{type(raw_preprocessed)} at {preprocessed_path}"
        )
    texts: list[list[str]] = []
    sentence_tokens_by_doc: list[list[list[str]]] = []
    for doc_index, document in enumerate(raw_preprocessed):
        doc_tokens = list(getattr(document, "document_tokens", []))
        doc_sentences = [
            list(sentence_tokens)
            for sentence_tokens in getattr(document, "sentences_tokenized", [])
        ]
        if not doc_tokens:
            logger.warning(
                "word_based_metrics skipping empty sentlda preprocessed doc "
                "data_run=%s category=%s iteration=%s doc_index=%s path=%s",
                data_run,
                category,
                iteration,
                doc_index,
                preprocessed_path,
            )
            continue
        texts.append(doc_tokens)
        sentence_tokens_by_doc.append(doc_sentences)
    dictionary, corpus_bow = corpus_bundle_module.build_dictionary_and_corpus(
        texts,
        dict_no_below=args.dict_no_below,
        dict_no_above=args.dict_no_above,
        dict_exclude_single_alpha=args.dict_exclude_single_alpha,
        dict_exclude_with_digit=args.dict_exclude_with_digit,
        dict_exclude_hiragana_only=args.dict_exclude_hiragana_only,
    )
    logger.info(
        "word_based_metrics stage sentlda_aligned_corpus data_run=%s category=%s "
        "iteration=%s docs=%s",
        data_run,
        category,
        iteration,
        len(texts),
    )
    return texts, dictionary, corpus_bow, sentence_tokens_by_doc


def _resolve_proxy_topic_words(
    *,
    args: argparse.Namespace,
    cache,
    model: str,
    data_run: str,
    category: str,
    iteration: int,
    split_csvs: tuple[str, ...] | None,
    target_column: str,
    dictionary: Dictionary,
    corpus_bow: list[list[tuple[int, int]]],
) -> tuple[
    TopicWordsResult,
    list[list[str]],
    Dictionary,
    list[list[tuple[int, int]]],
]:
    aligned_texts: list[list[str]] | None = None
    if model == "sentlda":
        aligned_texts, dictionary, corpus_bow, sentence_tokens_by_doc = (
            _load_sentlda_effective_corpus_bundle(
                args=args,
                data_run=data_run,
                category=category,
                iteration=iteration,
            )
        )
    if args.proxy_npmi_mode == "sentence":
        logger.info(
            "wb topic_words load_sentence_topics data_run=%s model=%s category=%s "
            "iteration=%s",
            data_run,
            model,
            category,
            iteration,
        )
        sentence_topics_by_doc = load_sentence_topics(
            model=model,
            dataset=args.dataset,
            data_run=data_run,
            iteration=iteration,
            num_topics=args.num_topics,
            category=category,
            split=args.coherence_split,
            embedding_variant=_effective_embedding_variant_for_model(model, args),
        )
        if model != "sentlda":
            documents = load_documents(
                dataset=args.dataset,
                category=category,
                split=args.coherence_split,
                split_csvs=split_csvs,
                target_column=target_column,
                exclude_labels=None,
            )
            sentence_tokens_by_doc = tokenize_sentence_documents(
                documents=documents,
                min_token_len=args.coherence_min_token_len,
                language=args.language,
                delimiter=args.delimiter,
                ja_replace_num=args.ja_replace_num,
                ja_dicdir=args.ja_dicdir,
                ja_require_unidic=args.ja_require_unidic,
            )
        sentence_bow_by_doc = build_sentence_bow_by_document(
            sentence_tokens_by_doc=sentence_tokens_by_doc,
            dictionary=dictionary,
        )
        if len(sentence_topics_by_doc) != len(sentence_bow_by_doc):
            if args.dataset.startswith("20newsgroup") and category == "all":
                (
                    texts,
                    dictionary,
                    corpus_bow,
                    _documents,
                    sentence_tokens_by_doc,
                ) = _rebuild_twenty_newsgroup_all_bundle(
                    args=args,
                    cache=cache,
                    data_run=data_run,
                    category=category,
                    split_csvs=split_csvs,
                    target_column=target_column,
                )
                sentence_bow_by_doc = build_sentence_bow_by_document(
                    sentence_tokens_by_doc=sentence_tokens_by_doc,
                    dictionary=dictionary,
                )
            else:
                texts = None
            if len(sentence_topics_by_doc) != len(sentence_bow_by_doc):
                raise ValueError(
                    "sentence_topics rows "
                    f"{len(sentence_topics_by_doc)} do not match corpus size "
                    f"{len(sentence_bow_by_doc)}"
                )
            if texts is not None:
                return (
                    extract_topic_words_from_sentence_topic_npmi(
                        sentence_topics_by_doc=sentence_topics_by_doc,
                        sentence_bow_by_doc=sentence_bow_by_doc,
                        num_topics=args.num_topics,
                        dictionary=dictionary,
                        topn=_requested_topic_word_topn(args),
                        score_mode=args.proxy_word_score_mode,
                    ),
                    aligned_texts if aligned_texts is not None else texts,
                    dictionary,
                    corpus_bow,
                )
        return (
            extract_topic_words_from_sentence_topic_npmi(
                sentence_topics_by_doc=sentence_topics_by_doc,
                sentence_bow_by_doc=sentence_bow_by_doc,
                num_topics=args.num_topics,
                dictionary=dictionary,
                topn=_requested_topic_word_topn(args),
                score_mode=args.proxy_word_score_mode,
            ),
            aligned_texts,
            dictionary,
            corpus_bow,
        )

    logger.info(
        "wb topic_words load_doc_topics data_run=%s model=%s category=%s iteration=%s",
        data_run,
        model,
        category,
        iteration,
    )
    doc_topics = load_doc_topics_proxy_soft_preferred(
        model=model,
        dataset=args.dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=args.num_topics,
        category=category,
        split=args.coherence_split,
        embedding_variant=_effective_embedding_variant_for_model(model, args),
    )
    if doc_topics.shape[0] != len(corpus_bow):
        if args.dataset.startswith("20newsgroup") and category == "all":
            texts, dictionary, corpus_bow, _documents, _sentence_tokens_by_doc = (
                _rebuild_twenty_newsgroup_all_bundle(
                    args=args,
                    cache=cache,
                    data_run=data_run,
                    category=category,
                    split_csvs=split_csvs,
                    target_column=target_column,
                )
            )
        else:
            texts = None
        if doc_topics.shape[0] != len(corpus_bow):
            raise ValueError(
                f"doc_topics rows {doc_topics.shape[0]} do not match corpus size "
                f"{len(corpus_bow)}"
            )
        if texts is not None:
            return (
                extract_topic_words_from_doc_topic_npmi(
                    doc_topics=doc_topics,
                    corpus_bow=corpus_bow,
                    dictionary=dictionary,
                    topn=_requested_topic_word_topn(args),
                    score_mode=args.proxy_word_score_mode,
                ),
                aligned_texts if aligned_texts is not None else texts,
                dictionary,
                corpus_bow,
            )
    return (
        extract_topic_words_from_doc_topic_npmi(
            doc_topics=doc_topics,
            corpus_bow=corpus_bow,
            dictionary=dictionary,
            topn=_requested_topic_word_topn(args),
            score_mode=args.proxy_word_score_mode,
        ),
        aligned_texts,
        dictionary,
        corpus_bow,
    )


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
    if model in LEARNED_WORD_TOPIC_MODELS:
        return (
            extract_topic_words_from_learned_model(
                model=model,
                dataset=args.dataset,
                iteration=iteration,
                num_topics=args.num_topics,
                category=category,
                topn=_requested_topic_word_topn(args),
                gaussian_word2vec=args.gaussian_word2vec,
                dictionary=dictionary,
                data_run=data_run,
                embedding_variant=_effective_embedding_variant_for_model(model, args),
            ),
            texts,
            dictionary,
            corpus_bow,
        )
    if model in PROXY_WORD_TOPIC_MODELS:
        topic_words_result, maybe_texts, dictionary, corpus_bow = (
            _resolve_proxy_topic_words(
                args=args,
                cache=cache,
                model=model,
                data_run=data_run,
                category=category,
                iteration=iteration,
                split_csvs=split_csvs,
                target_column=target_column,
                dictionary=dictionary,
                corpus_bow=corpus_bow,
            )
        )
        return (
            topic_words_result,
            maybe_texts if maybe_texts is not None else texts,
            dictionary,
            corpus_bow,
        )
    raise ValueError(f"Unsupported model for coherence analysis: {model}")


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
    proxy_word_score_mode: str,
    proxy_word_score_definition: str,
    coherence_reference_num_docs: int,
    coherence_reference_vocab_size: int,
    coherence_reference_streaming: bool,
    summary_rows: list[dict[str, str | float]],
    summary_provenance: list[dict[str, object]],
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
        iteration=used_iterations[0],
        num_topics=args.num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=_effective_embedding_variant_for_model(model, args),
    )
    condition_id, condition_fingerprint = _build_output_condition_id(
        model=model,
        dataset=args.dataset,
        data_run=data_run,
        category=category,
        iterations=used_iterations,
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
        proxy_npmi_mode=(
            args.proxy_npmi_mode if model in PROXY_WORD_TOPIC_MODELS else ""
        ),
        proxy_word_score_mode=(
            proxy_word_score_mode if model in PROXY_WORD_TOPIC_MODELS else ""
        ),
        embedding_variant=_effective_embedding_variant_for_model(model, args),
        metric_names=metric_names,
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
        iterations=used_iterations,
        started_at=started_at,
        execution_id=execution_id,
        archive_dir=str(out_dir),
        latest_dir=None if latest_out_dir is None else str(latest_out_dir),
        model_provenance=provenance,
        metric_names=metric_names,
        topic_words={
            "topn": int(_requested_topic_word_topn(args)),
            "coherence_topn": int(args.coherence_topn),
            "diversity_topn": int(args.diversity_topn),
            "source": topic_word_source,
            "score_mode": (
                proxy_word_score_mode if model in PROXY_WORD_TOPIC_MODELS else ""
            ),
            "score_definition": (
                proxy_word_score_definition if model in PROXY_WORD_TOPIC_MODELS else ""
            ),
        },
        coherence=_coherence_meta(
            coherences=coherences,
            args=args,
            model=model,
            topic_word_source=topic_word_source,
            proxy_word_score_mode=proxy_word_score_mode,
            proxy_word_score_definition=proxy_word_score_definition,
            reference_meta=coherence_reference_meta,
            window_sizes=coherence_window_sizes,
            window_size_sources=coherence_window_size_sources,
            min_window_counts=coherence_min_window_counts,
        ),
        diversity={
            "topn": int(args.diversity_topn),
            "topic_word_source": topic_word_source,
            "proxy_word_score_mode": (
                proxy_word_score_mode if model in PROXY_WORD_TOPIC_MODELS else ""
            ),
        },
    )
    metrics_results = {
        "aggregate": agg,
        "per_iteration": per_iter_metrics,
        "topic_words_topk": {
            "topn": int(_requested_topic_word_topn(args)),
            "coherence_topn": int(args.coherence_topn),
            "diversity_topn": int(args.diversity_topn),
            "per_iteration": per_iter_topic_words,
        },
    }
    out_path = out_dir / "metrics_agg.json"
    write_evaluation_json(meta=metrics_meta, results=metrics_results, path=out_path)
    logger.info(f"[{model}] aggregated metrics saved to {out_path}")
    topic_words_path = out_dir / "topic_words_topk.json"
    topic_words_meta = build_evaluation_meta(
        task="word_based_topic_words",
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
        iterations=used_iterations,
        started_at=started_at,
        execution_id=execution_id,
        archive_dir=str(out_dir),
        latest_dir=None if latest_out_dir is None else str(latest_out_dir),
        model_provenance=provenance,
        metric_names=metric_names,
        topic_word_source=topic_word_source,
        proxy_npmi_mode=(
            args.proxy_npmi_mode if model in PROXY_WORD_TOPIC_MODELS else ""
        ),
        score_mode=proxy_word_score_mode if model in PROXY_WORD_TOPIC_MODELS else "",
        score_definition=(
            proxy_word_score_definition if model in PROXY_WORD_TOPIC_MODELS else ""
        ),
        coherence_reference=coherence_reference_meta,
        topn=int(_requested_topic_word_topn(args)),
        coherence_topn=int(args.coherence_topn),
        diversity_topn=int(args.diversity_topn),
    )
    write_evaluation_json(
        meta=topic_words_meta,
        results={"per_iteration": per_iter_topic_words},
        path=topic_words_path,
    )
    logger.info(f"[{model}] top words saved to {topic_words_path}")
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
                "metrics": out_path.name,
                "topic_words": topic_words_path.name,
                "metadata": metadata_path.name,
            },
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
                "topic_word_source": topic_word_source,
                "proxy_word_score_mode": (
                    proxy_word_score_mode if model in PROXY_WORD_TOPIC_MODELS else ""
                ),
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
    proxy_word_score_mode = ""
    proxy_word_score_definition = ""
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
        )
        if model == "sentlda":
            topic_word_texts = []
            topic_word_dictionary = Dictionary()
            topic_word_corpus_bow = []
        else:
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
                dict_exclude_single_alpha=local_args.dict_exclude_single_alpha,
                dict_exclude_with_digit=local_args.dict_exclude_with_digit,
                dict_exclude_hiragana_only=local_args.dict_exclude_hiragana_only,
                exclude_labels=None,
                split_csvs=split_csvs,
                target_column=resolved_target_column,
            )
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
        topic_word_source = topic_words_result.topic_word_source
        proxy_word_score_mode = topic_words_result.score_mode or ""
        proxy_word_score_definition = topic_words_result.score_definition or ""
        pending_iterations.append(
            PendingWordBasedIteration(
                iteration=int(iteration),
                topic_words=topic_words_result.topic_words,
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
        proxy_word_score_mode=proxy_word_score_mode,
        proxy_word_score_definition=proxy_word_score_definition,
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
        metrics = compute_shared_reference_coherence_scores(
            topic_words=pending_iteration.topic_words,
            metric_names=metric_names,
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


def run_topic_coherence_analysis_from_args(args: argparse.Namespace) -> Path:
    args.out_root = resolve_project_path(args.out_root)
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
    metric_names = _metric_names_for_coherences(coherences)
    summary_rows: list[dict[str, str | float]] = []
    summary_provenance: list[dict[str, object]] = []
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
                    ): task.sort_index
                    for task in group_tasks
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        grouped_results.append((futures[future], result))
            pending_groups = [group for _sort_index, group in sorted(grouped_results)]
            logger.info(
                "word_based_metrics topic_words parallel done workers=%s groups=%s",
                topic_word_workers,
                len(pending_groups),
            )
        else:
            for task in group_tasks:
                result = _collect_pending_word_based_group(
                    args=args,
                    task=task,
                    total_conditions=total_conditions,
                )
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
            shared_counts = build_shared_reference_counts(
                reference_path=resolve_project_path(args.coherence_reference_path),
                target_words=collect_target_words(coherence_topic_words),
                window_sizes=effective_window_sizes_for_coherences(
                    coherences,
                    window_size=args.coherence_window_size,
                ),
                max_docs=args.coherence_reference_max_docs,
                min_doc_tokens=args.coherence_reference_min_doc_tokens,
                backend=_coherence_count_backend(args),
                workers=_coherence_count_workers(args),
                chunk_size=_coherence_count_chunk_size(args),
                progress_label="wb reference_counts",
            )
            score_workers = min(_coherence_score_workers(args), len(pending_groups))
            if score_workers > 1:
                logger.info(
                    "word_based_metrics scoring parallel start workers=%s groups=%s",
                    score_workers,
                    len(pending_groups),
                )
                scored_by_index: list[tuple[int, ScoredWordBasedGroup]] = []
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
                        scored_by_index.append((futures[future], future.result()))
                scored_groups = [
                    scored_group for _index, scored_group in sorted(scored_by_index)
                ]
                logger.info(
                    "word_based_metrics scoring parallel done workers=%s groups=%s",
                    score_workers,
                    len(scored_groups),
                )
            else:
                scored_groups = [
                    _score_pending_word_based_group(
                        group=group,
                        args=args,
                        metric_names=metric_names,
                        coherences=coherences,
                        shared_counts=shared_counts,
                    )
                    for group in pending_groups
                ]
            for scored_group in scored_groups:
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
                    proxy_word_score_mode=group.proxy_word_score_mode,
                    proxy_word_score_definition=group.proxy_word_score_definition,
                    coherence_reference_num_docs=shared_counts.num_docs,
                    coherence_reference_vocab_size=shared_counts.vocab_size,
                    coherence_reference_streaming=True,
                    summary_rows=summary_rows,
                    summary_provenance=summary_provenance,
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
        )
        logger.info(
            "word_based_metrics done dataset=%s num_topics=%s total_conditions=%s",
            args.dataset,
            display_num_topics,
            total_conditions,
        )
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
                used_iterations: list[int] = []
                topic_word_source: str | None = None
                proxy_word_score_mode: str = ""
                proxy_word_score_definition: str = ""
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
                    if model == "sentlda":
                        topic_word_texts = []
                        topic_word_dictionary = Dictionary()
                        topic_word_corpus_bow = []
                    else:
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
                            dict_exclude_single_alpha=args.dict_exclude_single_alpha,
                            dict_exclude_with_digit=args.dict_exclude_with_digit,
                            dict_exclude_hiragana_only=args.dict_exclude_hiragana_only,
                            exclude_labels=None,
                            split_csvs=split_csvs,
                            target_column=resolved_target_column,
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
                        "iteration=%s mode=%s",
                        condition_progress,
                        data_run,
                        model,
                        category,
                        iteration,
                        args.proxy_npmi_mode,
                    )
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

                    topic_words: TopicWords = topic_words_result.topic_words
                    topic_word_source = topic_words_result.topic_word_source
                    proxy_word_score_mode = topic_words_result.score_mode or ""
                    proxy_word_score_definition = (
                        topic_words_result.score_definition or ""
                    )
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
                    if args.coherence_reference == "wikipedia":
                        assert args.coherence_reference_path is not None
                        if _uses_streaming_reference(args):
                            coherence_reference_streaming = True
                            metrics = {}
                            coherence_topic_words = (
                                truncate_topic_words(
                                    topic_words,
                                    args.coherence_topn,
                                )
                                if args.coherence_topn is not None
                                else topic_words
                            )
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
                                    progress_label=f"wb {condition_progress} metrics",
                                )
                            )
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
                                        topic_words,
                                        args.diversity_topn,
                                    )
                                    if args.diversity_topn is not None
                                    else topic_words
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
                                dict_exclude_single_alpha=(
                                    args.dict_exclude_single_alpha
                                ),
                                dict_exclude_with_digit=(args.dict_exclude_with_digit),
                                dict_exclude_hiragana_only=(
                                    args.dict_exclude_hiragana_only
                                ),
                            )
                            metrics = evaluate_topic_words(
                                topic_words=topic_words,
                                metric_names=metric_names,
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
                            coherence_reference_num_docs = len(coherence_texts)
                            coherence_reference_vocab_size = len(coherence_dictionary)
                    else:
                        coherence_texts = topic_word_texts
                        coherence_dictionary = topic_word_dictionary
                        coherence_corpus_bow = topic_word_corpus_bow
                        metrics = evaluate_topic_words(
                            topic_words=topic_words,
                            metric_names=metric_names,
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
                        coherence_reference_num_docs = len(coherence_texts)
                        coherence_reference_vocab_size = len(coherence_dictionary)
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

                assert topic_word_source is not None
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
                    iteration=used_iterations[0],
                    num_topics=args.num_topics,
                    category=category,
                    data_run=data_run,
                    embedding_variant=_effective_embedding_variant_for_model(
                        model, args
                    ),
                )
                condition_id, condition_fingerprint = _build_output_condition_id(
                    model=model,
                    dataset=args.dataset,
                    data_run=data_run,
                    category=category,
                    iterations=used_iterations,
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
                    proxy_npmi_mode=(
                        args.proxy_npmi_mode if model in PROXY_WORD_TOPIC_MODELS else ""
                    ),
                    proxy_word_score_mode=(
                        proxy_word_score_mode
                        if model in PROXY_WORD_TOPIC_MODELS
                        else ""
                    ),
                    embedding_variant=_effective_embedding_variant_for_model(
                        model, args
                    ),
                    metric_names=metric_names,
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
                    iterations=used_iterations,
                    started_at=started_at,
                    execution_id=execution_id,
                    archive_dir=str(out_dir),
                    latest_dir=None if latest_out_dir is None else str(latest_out_dir),
                    model_provenance=provenance,
                    metric_names=metric_names,
                    topic_words={
                        "topn": int(_requested_topic_word_topn(args)),
                        "coherence_topn": int(args.coherence_topn),
                        "diversity_topn": int(args.diversity_topn),
                        "source": topic_word_source,
                        "score_mode": (
                            proxy_word_score_mode
                            if model in PROXY_WORD_TOPIC_MODELS
                            else ""
                        ),
                        "score_definition": (
                            proxy_word_score_definition
                            if model in PROXY_WORD_TOPIC_MODELS
                            else ""
                        ),
                    },
                    coherence=_coherence_meta(
                        coherences=coherences,
                        args=args,
                        model=model,
                        topic_word_source=topic_word_source,
                        proxy_word_score_mode=proxy_word_score_mode,
                        proxy_word_score_definition=proxy_word_score_definition,
                        reference_meta=coherence_reference_meta,
                        window_sizes=coherence_window_sizes,
                        window_size_sources=coherence_window_size_sources,
                        min_window_counts=coherence_min_window_counts,
                    ),
                    diversity={
                        "topn": int(args.diversity_topn),
                        "topic_word_source": topic_word_source,
                        "proxy_word_score_mode": (
                            proxy_word_score_mode
                            if model in PROXY_WORD_TOPIC_MODELS
                            else ""
                        ),
                    },
                )
                metrics_results = {
                    "aggregate": agg,
                    "per_iteration": per_iter_metrics,
                    "topic_words_topk": {
                        "topn": int(_requested_topic_word_topn(args)),
                        "coherence_topn": int(args.coherence_topn),
                        "diversity_topn": int(args.diversity_topn),
                        "per_iteration": per_iter_topic_words,
                    },
                }
                out_path = out_dir / "metrics_agg.json"
                write_evaluation_json(
                    meta=metrics_meta,
                    results=metrics_results,
                    path=out_path,
                )
                logger.info(f"[{model}] aggregated metrics saved to {out_path}")

                topic_words_path = out_dir / "topic_words_topk.json"
                topic_words_meta = build_evaluation_meta(
                    task="word_based_topic_words",
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
                    iterations=used_iterations,
                    started_at=started_at,
                    execution_id=execution_id,
                    archive_dir=str(out_dir),
                    latest_dir=(
                        None if latest_out_dir is None else str(latest_out_dir)
                    ),
                    model_provenance=provenance,
                    metric_names=metric_names,
                    topic_word_source=topic_word_source,
                    proxy_npmi_mode=(
                        args.proxy_npmi_mode if model in PROXY_WORD_TOPIC_MODELS else ""
                    ),
                    score_mode=(
                        proxy_word_score_mode
                        if model in PROXY_WORD_TOPIC_MODELS
                        else ""
                    ),
                    score_definition=(
                        proxy_word_score_definition
                        if model in PROXY_WORD_TOPIC_MODELS
                        else ""
                    ),
                    coherence_reference=coherence_reference_meta,
                    topn=int(_requested_topic_word_topn(args)),
                    coherence_topn=int(args.coherence_topn),
                    diversity_topn=int(args.diversity_topn),
                )
                write_evaluation_json(
                    meta=topic_words_meta,
                    results={"per_iteration": per_iter_topic_words},
                    path=topic_words_path,
                )
                logger.info(f"[{model}] top words saved to {topic_words_path}")

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
                            "metrics": out_path.name,
                            "topic_words": topic_words_path.name,
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
                            "topic_word_source": topic_word_source,
                            "proxy_word_score_mode": (
                                proxy_word_score_mode
                                if model in PROXY_WORD_TOPIC_MODELS
                                else ""
                            ),
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
    )
    logger.info(
        "word_based_metrics done dataset=%s num_topics=%s total_conditions=%s",
        args.dataset,
        args.num_topics,
        total_conditions,
    )
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
    out_root: Path = DEFAULT_OUT_ROOT,
    coherence: str | list[str] | tuple[str, ...] = "c_v",
    coherence_topn: int = 10,
    coherence_window_size: int | None = None,
    coherence_min_window_count: int | None = None,
    diversity_topn: int = 25,
    gaussian_word2vec: str = "glove-wiki-gigaword-100",
    coherence_split: str = "train",
    coherence_min_token_len: int = 2,
    dict_no_below: int = 3,
    dict_no_above: float = 0.7,
    dict_exclude_single_alpha: bool = False,
    dict_exclude_with_digit: bool = False,
    dict_exclude_hiragana_only: bool = False,
    proxy_npmi_mode: str = "sentence",
    proxy_word_score_mode: str = "word_npmi",
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
    coherence_score_workers: int = 1,
    skip_existing: bool = False,
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
        gaussian_word2vec=gaussian_word2vec,
        coherence_split=coherence_split,
        coherence_min_token_len=int(coherence_min_token_len),
        dict_no_below=int(dict_no_below),
        dict_no_above=float(dict_no_above),
        dict_exclude_single_alpha=bool(dict_exclude_single_alpha),
        dict_exclude_with_digit=bool(dict_exclude_with_digit),
        dict_exclude_hiragana_only=bool(dict_exclude_hiragana_only),
        proxy_npmi_mode=proxy_npmi_mode,
        proxy_word_score_mode=proxy_word_score_mode,
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
        coherence_score_workers=int(coherence_score_workers),
        skip_existing=bool(skip_existing),
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

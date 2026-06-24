from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .config import DEFAULT_ALIGNMENT_MODE, DEFAULT_FEATURE_RESOLVE_MODE, RESULT_ROOT
from .limited import (
    DEFAULT_SAMPLING_MAX_ATTEMPTS,
    DEFAULT_SAMPLING_RETRY_SEED_STRIDE,
    run_limited_classification_evaluation,
)
from .summary import write_summary
from .train import run_classification_evaluation


def run_classification_suite(
    *,
    iterations: Iterable[int],
    datasets: Iterable[str],
    data_runs: Sequence[str] = ("default",),
    categories: Sequence[str] | None = None,
    topics: Iterable[int],
    classifiers: Sequence[str],
    vmf_assignment: str,
    result_root: Path = RESULT_ROOT,
    target_column: str = "target_str",
    label_schema: str = "identity",
    seed: int | None = 42,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
) -> None:
    run_classification_evaluation(
        iterations=iterations,
        datasets=datasets,
        data_runs=data_runs,
        categories=categories,
        topics=topics,
        classifiers=classifiers,
        vmf_assignment=vmf_assignment,
        result_root=result_root,
        target_column=target_column,
        label_schema=label_schema,
        seed=seed,
        alignment_mode=alignment_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
    )


def run_limited_classification_suite(
    *,
    mode: str,
    value: float | int,
    iterations: Iterable[int],
    datasets: Iterable[str],
    data_runs: Sequence[str] = ("default",),
    categories: Sequence[str] | None = None,
    topics: Iterable[int],
    classifiers: Sequence[str],
    vmf_assignment: str,
    result_root: Path = RESULT_ROOT,
    target_column: str = "target_str",
    label_schema: str = "identity",
    stratified: bool = True,
    seed: int | None = 42,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    sampling_repeats: Sequence[int] | None = None,
    sampling_seed_stride: int = 1000,
    sampling_max_attempts: int = DEFAULT_SAMPLING_MAX_ATTEMPTS,
    sampling_retry_seed_stride: int = DEFAULT_SAMPLING_RETRY_SEED_STRIDE,
) -> None:
    run_limited_classification_evaluation(
        mode=mode,
        value=value,
        result_root=result_root,
        data_runs=data_runs,
        iterations=iterations,
        datasets=datasets,
        categories=categories,
        topics=topics,
        classifiers=classifiers,
        vmf_assignment=vmf_assignment,
        target_column=target_column,
        label_schema=label_schema,
        stratified=stratified,
        seed=seed,
        alignment_mode=alignment_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
        sampling_repeats=sampling_repeats,
        sampling_seed_stride=sampling_seed_stride,
        sampling_max_attempts=sampling_max_attempts,
        sampling_retry_seed_stride=sampling_retry_seed_stride,
    )


def write_classification_summary(
    *,
    metric: str,
    dataset: str,
    topics: int,
    iterations: list[int],
    data_run: str = "default",
    classifiers: Sequence[str] | None = None,
    vmf_assignment: str = "hard",
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    result_root: Path = RESULT_ROOT,
    target_column: str = "target_str",
    label_schema: str = "identity",
    resolve_mode: str = "latest",
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    excluded_categories: Sequence[str] | None = None,
    include_all_category: bool = False,
    output_path: Path | None = None,
) -> None:
    write_summary(
        metric=metric,
        dataset=dataset,
        topics=topics,
        iterations=iterations,
        data_run=data_run,
        classifiers=classifiers,
        vmf_assignment=vmf_assignment,
        alignment_mode=alignment_mode,
        result_root=result_root,
        target_column=target_column,
        label_schema=label_schema,
        resolve_mode=resolve_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
        excluded_categories=excluded_categories,
        include_all_category=include_all_category,
        output_path=output_path,
    )

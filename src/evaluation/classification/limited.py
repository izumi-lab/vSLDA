from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.logging import get_logger

from .config import DEFAULT_ALIGNMENT_MODE, DEFAULT_FEATURE_RESOLVE_MODE, RESULT_ROOT
from .pipeline import load_classification_labels
from .train import train
from .workflow import (
    ClassificationCondition,
    build_classification_write_spec,
    build_sampling_meta,
    run_classification_grid,
)

logger = get_logger(__name__)

DEFAULT_SAMPLING_MAX_ATTEMPTS = 1
DEFAULT_SAMPLING_RETRY_SEED_STRIDE = 100_000


def _sample_indices(
    labels: Sequence[str],
    *,
    train_ratio: Optional[float],
    train_count: Optional[int],
    stratified: bool,
    seed: Optional[int],
) -> Tuple[List[int], Dict[str, int]]:
    if train_ratio is None and train_count is None:
        raise ValueError("Either train_ratio or train_count must be provided.")

    total = len(labels)
    if total == 0:
        return [], {}

    if train_ratio is not None:
        if train_ratio >= 1.0:
            indices = list(range(total))
        elif train_ratio <= 0.0:
            indices = []
        else:
            indices = _sample_with_stratified_split(
                labels, train_ratio, stratified=stratified, seed=seed
            )
    else:
        if train_count is None:
            raise ValueError("train_count must be set when train_ratio is None")
        if train_count >= total:
            indices = list(range(total))
        elif train_count <= 0:
            indices = []
        else:
            indices = _sample_with_stratified_split(
                labels, train_count, stratified=stratified, seed=seed
            )

    counts: Dict[str, int] = {}
    for idx in indices:
        lbl = labels[idx]
        counts[lbl] = counts.get(lbl, 0) + 1
    return sorted(indices), counts


def _sample_with_stratified_split(
    labels: Sequence[str],
    train_size: float | int,
    *,
    stratified: bool,
    seed: Optional[int],
) -> List[int]:
    total = len(labels)
    if total == 0:
        return []

    if not stratified:
        rng = np.random.default_rng(seed)
        sample_size = (
            int(round(total * train_size))
            if isinstance(train_size, float)
            else int(train_size)
        )
        sample_size = max(0, min(sample_size, total))
        return rng.choice(total, size=sample_size, replace=False).tolist()

    num_classes = len(set(labels))
    if isinstance(train_size, float):
        min_ratio = num_classes / total
        if train_size < min_ratio:
            logger.warning(
                f"[warn] train_ratio {train_size:.4f} < min_ratio {min_ratio:.4f}; "
                f"adjusting to {min_ratio:.4f} to keep all classes."
            )
            train_size = min_ratio
        train_size = min(train_size, 1.0)
    else:
        if train_size < num_classes:
            logger.warning(
                f"[warn] train_count {train_size} < num_classes {num_classes}; "
                f"adjusting to {num_classes} to keep all classes."
            )
            train_size = num_classes
        train_size = min(train_size, total)

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, random_state=seed
    )
    x_dummy = np.zeros(total)
    train_idx, _ = next(splitter.split(x_dummy, labels))
    return train_idx.tolist()


def _sampling_seed(
    base_seed: Optional[int],
    it: int,
    *,
    sampling_repeat: int | None,
    sampling_seed_stride: int,
    retry_attempt: int = 0,
    sampling_retry_seed_stride: int = DEFAULT_SAMPLING_RETRY_SEED_STRIDE,
) -> Optional[int]:
    if sampling_seed_stride <= 0:
        raise ValueError("sampling_seed_stride must be a positive integer.")
    if sampling_retry_seed_stride <= 0:
        raise ValueError("sampling_retry_seed_stride must be a positive integer.")
    if sampling_repeat is not None and sampling_repeat < 0:
        raise ValueError("sampling_repeat must be a non-negative integer.")
    if retry_attempt < 0:
        raise ValueError("retry_attempt must be a non-negative integer.")
    if base_seed is None:
        return None
    repeat_offset = (
        0 if sampling_repeat is None else sampling_repeat * sampling_seed_stride
    )
    retry_offset = int(retry_attempt) * int(sampling_retry_seed_stride)
    return int(base_seed) + int(it) + int(repeat_offset) + retry_offset


def _resolve_sampling_repeats(
    sampling_repeats: Sequence[int] | None,
) -> list[int | None]:
    if not sampling_repeats:
        return [None]

    repeats: list[int | None] = []
    seen: set[int] = set()
    for repeat in sampling_repeats:
        repeat_int = int(repeat)
        if repeat_int < 0:
            raise ValueError("sampling_repeat must be a non-negative integer.")
        if repeat_int in seen:
            raise ValueError(f"duplicate sampling_repeat: {repeat_int}")
        seen.add(repeat_int)
        repeats.append(repeat_int)
    return repeats


def _run_setting(
    *,
    mode: str,
    value: float | int,
    data_run: str,
    result_root: Path,
    iterations: Iterable[int],
    datasets: Iterable[str],
    categories: Sequence[str] | None,
    topics: Iterable[int],
    classifiers: Sequence[str],
    vmf_assignment: str,
    target_column: str,
    label_schema: str,
    stratified: bool,
    seed: Optional[int],
    alignment_mode: str,
    embedding_variants: Sequence[str] | None,
    feature_resolve_mode: str,
    selected_models: Sequence[str] | None,
    sampling_repeat: int | None,
    sampling_seed_stride: int,
    sampling_max_attempts: int,
    sampling_retry_seed_stride: int,
) -> None:
    def resolve_train_indices(
        dataset: str,
        category: str,
        iteration: int,
        _num_topics: int,
        retry_attempt: int,
    ) -> tuple[list[int], dict[str, Any]]:
        return _resolve_train_indices(
            dataset=dataset,
            category=category,
            iteration=iteration,
            mode=mode,
            value=value,
            stratified=stratified,
            seed=seed,
            sampling_repeat=sampling_repeat,
            sampling_seed_stride=sampling_seed_stride,
            sampling_max_attempts=sampling_max_attempts,
            sampling_retry_seed_stride=sampling_retry_seed_stride,
            retry_attempt=retry_attempt,
            target_column=target_column,
            label_schema=label_schema,
        )

    run_classification_grid(
        iterations=iterations,
        datasets=datasets,
        data_run=data_run,
        categories=categories,
        topics=topics,
        classifiers=classifiers,
        vmf_assignment=vmf_assignment,
        target_column=target_column,
        label_schema=label_schema,
        alignment_mode=alignment_mode,
        write_spec_builder=lambda iteration, dataset, num_topics: build_classification_write_spec(
            result_root=result_root,
            condition=ClassificationCondition(
                dataset=dataset,
                data_run=data_run,
                topics=num_topics,
                iteration=iteration,
                classifiers=classifiers,
                vmf_assignment=vmf_assignment,
                target_column=target_column,
                label_schema=label_schema,
                alignment_mode=alignment_mode,
                mode=mode,
                value=value,
                stratified=stratified,
                sampling_repeat=sampling_repeat,
                embedding_variants=embedding_variants,
                feature_resolve_mode=feature_resolve_mode,
                selected_models=selected_models,
            ),
            acc_filename=f"acc_{dataset}_{num_topics}topic_{mode}{value}.json",
            f1_filename=f"f1_{dataset}_{num_topics}topic_{mode}{value}.json",
            feature_filename=f"feat_{dataset}_{num_topics}topic_{mode}{value}.json",
            seed=_sampling_seed(
                seed,
                iteration,
                sampling_repeat=sampling_repeat,
                sampling_seed_stride=sampling_seed_stride,
            ),
        ),
        train_runner=train,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
        train_index_resolver=resolve_train_indices,
    )


def _resolve_train_indices(
    *,
    dataset: str,
    category: str,
    iteration: int,
    mode: str,
    value: float | int,
    stratified: bool,
    seed: Optional[int],
    sampling_repeat: int | None,
    sampling_seed_stride: int,
    target_column: str,
    label_schema: str,
    sampling_max_attempts: int = DEFAULT_SAMPLING_MAX_ATTEMPTS,
    sampling_retry_seed_stride: int = DEFAULT_SAMPLING_RETRY_SEED_STRIDE,
    retry_attempt: int = 0,
) -> tuple[list[int], dict[str, Any]]:
    if sampling_max_attempts <= 0:
        raise ValueError("sampling_max_attempts must be a positive integer.")
    if retry_attempt >= sampling_max_attempts:
        return [], {
            "skip_reason": "sampling retry attempts exhausted",
            "sampling_retry_attempt": retry_attempt,
            "sampling_max_attempts": sampling_max_attempts,
        }
    labels = load_classification_labels(
        dataset,
        category,
        "train",
        target_column=target_column,
        label_schema=label_schema,
    )
    train_idx, label_counts = _sample_indices(
        labels,
        train_ratio=value if mode == "ratio" else None,
        train_count=value if mode == "count" else None,
        stratified=stratified,
        seed=_sampling_seed(
            seed,
            iteration,
            sampling_repeat=sampling_repeat,
            sampling_seed_stride=sampling_seed_stride,
            retry_attempt=retry_attempt,
            sampling_retry_seed_stride=sampling_retry_seed_stride,
        ),
    )
    effective_seed = _sampling_seed(
        seed,
        iteration,
        sampling_repeat=sampling_repeat,
        sampling_seed_stride=sampling_seed_stride,
        retry_attempt=retry_attempt,
        sampling_retry_seed_stride=sampling_retry_seed_stride,
    )
    meta = build_sampling_meta(
        dataset=dataset,
        category=category,
        target_column=target_column,
        label_schema=label_schema,
        train_indices=train_idx,
        label_counts=label_counts,
        load_labels=load_classification_labels,
    )
    meta.update(
        {
            "sampling_retry_attempt": retry_attempt,
            "sampling_max_attempts": sampling_max_attempts,
            "sampling_effective_seed": effective_seed,
        }
    )
    return train_idx, meta


def run_limited_classification_evaluation(
    *,
    mode: str,
    value: float | int,
    result_root: Path = RESULT_ROOT,
    data_runs: Iterable[str] = ("default",),
    iterations: Iterable[int],
    datasets: Iterable[str],
    categories: Sequence[str] | None = None,
    topics: Iterable[int],
    classifiers: Sequence[str],
    vmf_assignment: str,
    target_column: str,
    label_schema: str,
    stratified: bool,
    seed: Optional[int],
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    sampling_repeats: Sequence[int] | None = None,
    sampling_seed_stride: int = 1000,
    sampling_max_attempts: int = DEFAULT_SAMPLING_MAX_ATTEMPTS,
    sampling_retry_seed_stride: int = DEFAULT_SAMPLING_RETRY_SEED_STRIDE,
) -> None:
    if sampling_max_attempts <= 0:
        raise ValueError("sampling_max_attempts must be a positive integer.")
    if sampling_retry_seed_stride <= 0:
        raise ValueError("sampling_retry_seed_stride must be a positive integer.")
    resolved_sampling_repeats = _resolve_sampling_repeats(sampling_repeats)
    for data_run in data_runs:
        for sampling_repeat in resolved_sampling_repeats:
            _run_setting(
                mode=mode,
                value=value,
                data_run=data_run,
                result_root=result_root,
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
                sampling_repeat=sampling_repeat,
                sampling_seed_stride=sampling_seed_stride,
                sampling_max_attempts=sampling_max_attempts,
                sampling_retry_seed_stride=sampling_retry_seed_stride,
            )

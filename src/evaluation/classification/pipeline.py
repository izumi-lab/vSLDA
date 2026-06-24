from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.splits import load_filtered_split_labels

from .alignment import build_common_feature_alignment, build_label_space_indices
from .classifier_registry import fit_classifiers, get_classifier_specs
from .config import DEFAULT_ALIGNMENT_MODE, model_matches_selector
from .feature_registry import (
    iter_available_features,
    resolve_feature_catalog_entry,
    resolve_feature_display_name,
)
from .metrics import (
    build_coverage_payload,
    build_feature_importance_payload,
    build_metrics_payload,
)

DELIMITER = " / "
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationLabelBundle:
    category_labels: list[str]
    label_map: dict[str, int]
    train_y: list[int]
    test_y: list[int]
    train_indices: np.ndarray | None
    train_source_indices: np.ndarray
    test_source_indices: np.ndarray
    raw_train_count: int
    raw_test_count: int
    label_filtered_train_count: int
    label_filtered_test_count: int


@dataclass(frozen=True)
class FeatureSet:
    name: str
    train_x: np.ndarray
    test_x: np.ndarray
    catalog_entry: dict[str, Any] | None = None
    available_train_docs: int | None = None
    available_test_docs: int | None = None


def load_classification_labels(
    dataset: str,
    category: str,
    split: str,
    *,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> list[str]:
    """Load labels for a split, filtering rows with empty sentence segments."""
    return list(
        _load_classification_labels_cached(
            dataset=dataset,
            category=category,
            split=split,
            target_column=target_column,
            label_schema=label_schema,
        )
    )


@lru_cache(maxsize=None)
def _load_classification_labels_cached(
    *,
    dataset: str,
    category: str,
    split: str,
    target_column: str,
    label_schema: str,
) -> tuple[str, ...]:
    return tuple(
        load_filtered_split_labels(
            dataset,
            category,
            split,
            data_column="data",
            target_column=target_column,
            label_schema=label_schema,
            delimiter=DELIMITER,
        )
    )


def build_label_bundle(
    *,
    dataset: str,
    category: str,
    category_labels: Sequence[str],
    train_indices: Sequence[int] | None = None,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> ClassificationLabelBundle:
    train_alignment, test_alignment = build_label_space_indices(
        dataset,
        category,
        target_column=target_column,
        label_schema=label_schema,
        delimiter=DELIMITER,
    )
    train_labels = load_classification_labels(
        dataset,
        category,
        "train",
        target_column=target_column,
        label_schema=label_schema,
    )
    if len(train_labels) != len(train_alignment.available_indices):
        raise ValueError(
            "train label count does not match reconstructed label-space indices "
            f"({len(train_labels)} != {len(train_alignment.available_indices)})"
        )

    label_map = {label: idx for idx, label in enumerate(sorted(set(train_labels)))}
    train_y_full = [label_map[label] for label in train_labels]
    train_source_indices = np.asarray(train_alignment.available_indices, dtype=int)

    train_indices_arr = None
    if train_indices is not None:
        train_indices_arr = np.asarray(train_indices, dtype=int)
        train_y = [train_y_full[idx] for idx in train_indices_arr]
        train_source_indices = train_source_indices[train_indices_arr]
    else:
        train_y = train_y_full

    test_labels = load_classification_labels(
        dataset,
        category,
        "test",
        target_column=target_column,
        label_schema=label_schema,
    )
    if len(test_labels) != len(test_alignment.available_indices):
        raise ValueError(
            "test label count does not match reconstructed label-space indices "
            f"({len(test_labels)} != {len(test_alignment.available_indices)})"
        )
    test_y = [label_map[label] for label in test_labels]
    return ClassificationLabelBundle(
        category_labels=list(category_labels),
        label_map=label_map,
        train_y=train_y,
        test_y=test_y,
        train_indices=train_indices_arr,
        train_source_indices=train_source_indices,
        test_source_indices=np.asarray(test_alignment.available_indices, dtype=int),
        raw_train_count=len(train_alignment.raw_indices),
        raw_test_count=len(test_alignment.raw_indices),
        label_filtered_train_count=len(train_y_full),
        label_filtered_test_count=len(test_y),
    )


def collect_feature_sets(
    *,
    dataset: str,
    data_run: str = "default",
    iteration: int,
    num_topics: int,
    category: str,
    vmf_assignment: str,
    label_bundle: ClassificationLabelBundle,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = "all",
    selected_models: Sequence[str] | None = None,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> tuple[list[FeatureSet], list[int], list[int], dict[str, Any]]:
    selectors = [str(model) for model in (selected_models or []) if str(model).strip()]
    candidate_entries: list[tuple[FeatureSet, np.ndarray, np.ndarray]] = []
    for spec, train_path, test_path in iter_available_features(
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        vmf_assignment=vmf_assignment,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
    ):
        if selectors and not any(
            model_matches_selector(
                spec.display_name,
                selector,
                model_key=spec.model_key,
            )
            for selector in selectors
        ):
            continue
        if not train_path.exists() or not test_path.exists():
            LOGGER.warning(
                "[skip] %s: missing files (%s %s)",
                spec.display_name,
                train_path if not train_path.exists() else "",
                test_path if not test_path.exists() else "",
            )
            continue

        train_x = np.asarray(spec.train_loader(train_path))
        test_x = np.asarray(spec.test_loader(test_path))
        resolver = spec.available_index_resolver or _resolve_label_space_availability
        train_alignment, test_alignment = resolver(
            dataset,
            category,
            train_path,
            test_path,
            target_column,
            label_schema,
        )
        train_available_indices = np.asarray(
            train_alignment.available_indices, dtype=int
        )
        test_available_indices = np.asarray(test_alignment.available_indices, dtype=int)
        if train_x.shape[0] != len(train_available_indices) or test_x.shape[0] != len(
            test_available_indices
        ):
            LOGGER.warning(
                "[skip] %s: resolved availability mismatch (features %s/%s vs indices %s/%s)",
                spec.display_name,
                train_x.shape[0],
                test_x.shape[0],
                len(train_available_indices),
                len(test_available_indices),
            )
            continue

        feature_name = resolve_feature_display_name(spec, train_path)
        candidate_entries.append(
            (
                FeatureSet(
                    name=feature_name,
                    train_x=train_x,
                    test_x=test_x,
                    catalog_entry=resolve_feature_catalog_entry(spec, train_path),
                    available_train_docs=len(train_available_indices),
                    available_test_docs=len(test_available_indices),
                ),
                train_available_indices,
                test_available_indices,
            )
        )

    if not candidate_entries:
        return [], [], [], _build_empty_coverage(label_bundle=label_bundle)

    if alignment_mode == "strict_skip":
        return _collect_feature_sets_strict_skip(
            candidate_entries=candidate_entries,
            label_bundle=label_bundle,
        )
    if alignment_mode != "intersection":
        raise ValueError(f"Unknown alignment_mode: {alignment_mode}")

    feature_index_map = {
        feature.name: (train_available_indices, test_available_indices)
        for feature, train_available_indices, test_available_indices in candidate_entries
    }
    alignments, _common_train_raw, _common_test_raw = build_common_feature_alignment(
        train_label_source_indices=label_bundle.train_source_indices,
        test_label_source_indices=label_bundle.test_source_indices,
        feature_available_indices=feature_index_map,
    )
    first_alignment = next(iter(alignments.values()))
    aligned_train_y = [
        label_bundle.train_y[idx]
        for idx in first_alignment.train_indices_in_label_space.tolist()
    ]
    aligned_test_y = [
        label_bundle.test_y[idx]
        for idx in first_alignment.test_indices_in_label_space.tolist()
    ]

    feature_sets: list[FeatureSet] = []
    for feature, _train_available_indices, _test_available_indices in candidate_entries:
        alignment = alignments[feature.name]
        aligned_feature = FeatureSet(
            name=feature.name,
            train_x=feature.train_x[alignment.train_row_selector],
            test_x=feature.test_x[alignment.test_row_selector],
            catalog_entry=feature.catalog_entry,
            available_train_docs=feature.available_train_docs,
            available_test_docs=feature.available_test_docs,
        )
        if aligned_feature.train_x.shape[0] != len(
            aligned_train_y
        ) or aligned_feature.test_x.shape[0] != len(aligned_test_y):
            LOGGER.warning(
                "[skip] %s: aligned length mismatch (features %s/%s vs labels %s/%s)",
                feature.name,
                aligned_feature.train_x.shape[0],
                aligned_feature.test_x.shape[0],
                len(aligned_train_y),
                len(aligned_test_y),
            )
            continue
        feature_sets.append(aligned_feature)

    coverage = _build_coverage_payload(
        label_bundle=label_bundle,
        common_train_docs=len(aligned_train_y),
        common_test_docs=len(aligned_test_y),
        feature_sets=feature_sets,
    )
    return feature_sets, aligned_train_y, aligned_test_y, coverage


def run_classification_task(
    *,
    dataset: str,
    data_run: str = "default",
    category: str,
    num_topics: int,
    iteration: int,
    category_labels: Sequence[str],
    classifiers_to_use: Sequence[str],
    vmf_assignment: str,
    train_indices: Sequence[int] | None = None,
    target_column: str = "target_str",
    label_schema: str = "identity",
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = "all",
    selected_models: Sequence[str] | None = None,
) -> (
    tuple[
        dict[str, float],
        dict[str, dict[str, float]],
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any],
    ]
    | None
):
    label_bundle = build_label_bundle(
        dataset=dataset,
        category=category,
        category_labels=category_labels,
        train_indices=train_indices,
        target_column=target_column,
        label_schema=label_schema,
    )
    feature_sets, aligned_train_y, aligned_test_y, coverage = collect_feature_sets(
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        vmf_assignment=vmf_assignment,
        label_bundle=label_bundle,
        alignment_mode=alignment_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
        target_column=target_column,
        label_schema=label_schema,
    )
    if not feature_sets:
        LOGGER.warning("[skip] no feature sets found. Did you run any models?")
        return None
    if not aligned_train_y or not aligned_test_y:
        LOGGER.warning(
            "[skip] empty aligned label set after applying %s alignment",
            alignment_mode,
        )
        return None

    classifier_specs = get_classifier_specs(classifiers_to_use)
    if not classifier_specs:
        LOGGER.warning("[skip] no classifiers selected")
        return None

    classifiers, predictions = fit_classifiers(
        [(feature.name, feature.train_x, feature.test_x) for feature in feature_sets],
        train_y=aligned_train_y,
        classifier_specs=classifier_specs,
    )
    acc_result, f1_result = build_metrics_payload(
        test_y=aligned_test_y,
        predictions=predictions,
    )
    feature_importance = build_feature_importance_payload(
        classifiers=classifiers,
        label_map=label_bundle.label_map,
        category=category,
        category_labels=label_bundle.category_labels,
    )
    feature_catalog = [
        dict(feature.catalog_entry)
        for feature in feature_sets
        if feature.catalog_entry is not None
    ]
    return acc_result, f1_result, feature_importance, feature_catalog, coverage


def _collect_feature_sets_strict_skip(
    *,
    candidate_entries: list[tuple[FeatureSet, np.ndarray, np.ndarray]],
    label_bundle: ClassificationLabelBundle,
) -> tuple[list[FeatureSet], list[int], list[int], dict[str, Any]]:
    feature_sets: list[FeatureSet] = []
    for feature, _train_available_indices, _test_available_indices in candidate_entries:
        if feature.train_x.shape[0] != len(
            label_bundle.train_y
        ) or feature.test_x.shape[0] != len(label_bundle.test_y):
            LOGGER.warning(
                "[skip] %s: length mismatch (features %s/%s vs labels %s/%s)",
                feature.name,
                feature.train_x.shape[0],
                feature.test_x.shape[0],
                len(label_bundle.train_y),
                len(label_bundle.test_y),
            )
            continue
        feature_sets.append(feature)

    coverage = _build_coverage_payload(
        label_bundle=label_bundle,
        common_train_docs=len(label_bundle.train_y),
        common_test_docs=len(label_bundle.test_y),
        feature_sets=feature_sets,
    )
    return feature_sets, list(label_bundle.train_y), list(label_bundle.test_y), coverage


def _build_empty_coverage(
    *,
    label_bundle: ClassificationLabelBundle,
) -> dict[str, Any]:
    return _build_coverage_payload(
        label_bundle=label_bundle,
        common_train_docs=0,
        common_test_docs=0,
        feature_sets=[],
    )


def _build_coverage_payload(
    *,
    label_bundle: ClassificationLabelBundle,
    common_train_docs: int,
    common_test_docs: int,
    feature_sets: Sequence[FeatureSet],
) -> dict[str, Any]:
    return build_coverage_payload(
        raw_train_docs=label_bundle.raw_train_count,
        raw_test_docs=label_bundle.raw_test_count,
        label_filtered_train_docs=label_bundle.label_filtered_train_count,
        label_filtered_test_docs=label_bundle.label_filtered_test_count,
        selected_train_docs=len(label_bundle.train_y),
        selected_test_docs=len(label_bundle.test_y),
        common_train_docs=common_train_docs,
        common_test_docs=common_test_docs,
        available_train_docs={
            feature.name: (
                feature.available_train_docs
                if feature.available_train_docs is not None
                else int(feature.train_x.shape[0])
            )
            for feature in feature_sets
        },
        available_test_docs={
            feature.name: (
                feature.available_test_docs
                if feature.available_test_docs is not None
                else int(feature.test_x.shape[0])
            )
            for feature in feature_sets
        },
    )


def _resolve_label_space_availability(
    dataset: str,
    category: str,
    _train_path: Path,
    _test_path: Path,
    target_column: str,
    label_schema: str,
) -> tuple[Any, Any]:
    return build_label_space_indices(
        dataset,
        category,
        target_column=target_column,
        label_schema=label_schema,
        delimiter=DELIMITER,
    )

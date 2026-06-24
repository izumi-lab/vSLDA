from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.baselines.dataset_adapters import use_legacy_category_behavior
from src.data.catalog import get_dataset_targets
from src.data.preprocessing import PreprocessedDocument, preprocess_document
from src.data.splits import load_dataset_split

DEFAULT_DELIMITER = " / "

DocumentAvailabilityPredicate = Callable[[PreprocessedDocument], bool]


@dataclass(frozen=True)
class SplitAlignment:
    raw_indices: np.ndarray
    available_indices: np.ndarray


@dataclass(frozen=True)
class FeatureAlignment:
    train_indices_in_label_space: np.ndarray
    test_indices_in_label_space: np.ndarray
    train_row_selector: np.ndarray
    test_row_selector: np.ndarray


@dataclass(frozen=True)
class PreprocessingAlignmentOptions:
    text_column: str
    target_column: str | None
    targets: tuple[str, ...] | None
    language: str
    delimiter: str | None
    segmenter: str
    tokenizer: str
    ja_replace_num: bool
    ja_dicdir: str | None
    ja_require_unidic: bool
    use_legacy: bool


def build_label_space_indices(
    dataset: str,
    category: str,
    *,
    data_column: str = "data",
    target_column: str = "target_str",
    label_schema: str = "identity",
    delimiter: str = DEFAULT_DELIMITER,
) -> tuple[SplitAlignment, SplitAlignment]:
    return tuple(
        _build_label_space_split_alignment(
            dataset,
            category,
            split,
            data_column=data_column,
            target_column=target_column,
            label_schema=label_schema,
            delimiter=delimiter,
        )
        for split in ("train", "test")
    )


def build_baseline_available_indices(
    dataset: str,
    category: str,
    metadata: dict[str, Any] | None,
    *,
    target_column: str = "target_str",
    label_schema: str = "identity",
    require_document_tokens: bool = True,
    require_contextual_text: bool = False,
    require_sentences: bool = False,
) -> tuple[SplitAlignment, SplitAlignment]:
    options = resolve_preprocessing_alignment_options(
        dataset,
        category,
        metadata,
        target_column=target_column,
        label_schema=label_schema,
    )
    availability_key = _baseline_availability_key(
        require_document_tokens=require_document_tokens,
        require_contextual_text=require_contextual_text,
        require_sentences=require_sentences,
    )
    return tuple(
        _build_preprocessed_split_alignment_cached(
            dataset,
            split,
            options=options,
            label_schema=label_schema,
            availability_key=availability_key,
        )
        for split in ("train", "test")
    )


def build_preprocessed_available_indices(
    dataset: str,
    category: str,
    metadata: dict[str, Any] | None,
    *,
    availability_predicate: DocumentAvailabilityPredicate,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> tuple[SplitAlignment, SplitAlignment]:
    options = resolve_preprocessing_alignment_options(
        dataset,
        category,
        metadata,
        target_column=target_column,
        label_schema=label_schema,
    )
    availability_key = _predicate_cache_key(availability_predicate)
    if availability_key is not None:
        return tuple(
            _build_preprocessed_split_alignment_cached(
                dataset,
                split,
                options=options,
                label_schema=label_schema,
                availability_key=availability_key,
            )
            for split in ("train", "test")
        )
    return tuple(
        _build_preprocessed_split_alignment(
            dataset,
            split,
            options=options,
            label_schema=label_schema,
            availability_predicate=availability_predicate,
        )
        for split in ("train", "test")
    )


def resolve_preprocessing_alignment_options(
    dataset: str,
    category: str,
    metadata: dict[str, Any] | None,
    *,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> PreprocessingAlignmentOptions:
    payload = metadata or {}
    language = str(payload.get("language", "english"))
    legacy_preprocessing = payload.get("legacy_preprocessing")
    if legacy_preprocessing is None:
        use_legacy = use_legacy_category_behavior(dataset, language)
    else:
        use_legacy = bool(legacy_preprocessing)

    metadata_targets = payload.get("targets")
    if isinstance(metadata_targets, (list, tuple)):
        targets = tuple(str(item) for item in metadata_targets)
    else:
        dataset_targets = (
            get_dataset_targets(
                dataset,
                target_column=target_column,
                label_schema=label_schema,
            )
            or {}
        )
        allowed_labels = dataset_targets.get(category)
        targets = (
            None
            if allowed_labels is None
            else tuple(str(item) for item in allowed_labels)
        )

    return PreprocessingAlignmentOptions(
        text_column=str(payload.get("text_column", "data")),
        target_column=(
            None
            if "target_column" in payload and payload.get("target_column") is None
            else str(payload.get("target_column", target_column))
        ),
        targets=targets,
        language=language,
        delimiter=payload.get("delimiter", DEFAULT_DELIMITER),
        segmenter=str(payload.get("segmenter", "delimiter")),
        tokenizer=str(payload.get("tokenizer", "default")),
        ja_replace_num=bool(payload.get("ja_replace_num", True)),
        ja_dicdir=(
            None if payload.get("ja_dicdir") is None else str(payload.get("ja_dicdir"))
        ),
        ja_require_unidic=bool(payload.get("ja_require_unidic", True)),
        use_legacy=use_legacy,
    )


def build_common_feature_alignment(
    *,
    train_label_source_indices: Sequence[int],
    test_label_source_indices: Sequence[int],
    feature_available_indices: dict[str, tuple[Sequence[int], Sequence[int]]],
) -> tuple[dict[str, FeatureAlignment], np.ndarray, np.ndarray]:
    label_train = np.asarray(train_label_source_indices, dtype=int)
    label_test = np.asarray(test_label_source_indices, dtype=int)

    common_train = label_train.copy()
    common_test = label_test.copy()
    for available_train, available_test in feature_available_indices.values():
        common_train = np.intersect1d(
            common_train, np.asarray(available_train, dtype=int)
        )
        common_test = np.intersect1d(common_test, np.asarray(available_test, dtype=int))

    ordered_common_train = _ordered_intersection(label_train, common_train)
    ordered_common_test = _ordered_intersection(label_test, common_test)

    alignments = {
        name: FeatureAlignment(
            train_indices_in_label_space=_selector_from_raw_indices(
                label_train,
                ordered_common_train,
            ),
            test_indices_in_label_space=_selector_from_raw_indices(
                label_test,
                ordered_common_test,
            ),
            train_row_selector=_selector_from_raw_indices(
                np.asarray(available_train, dtype=int),
                ordered_common_train,
            ),
            test_row_selector=_selector_from_raw_indices(
                np.asarray(available_test, dtype=int),
                ordered_common_test,
            ),
        )
        for name, (available_train, available_test) in feature_available_indices.items()
    }
    return alignments, ordered_common_train, ordered_common_test


@lru_cache(maxsize=None)
def _build_label_space_split_alignment(
    dataset: str,
    category: str,
    split: str,
    *,
    data_column: str,
    target_column: str,
    label_schema: str,
    delimiter: str,
) -> SplitAlignment:
    split_path, frame = load_dataset_split(dataset, split)
    if data_column not in frame.columns:
        raise ValueError(f"data_column '{data_column}' not found in {split_path}")
    if target_column not in frame.columns:
        raise ValueError(f"target_column '{target_column}' not found in {split_path}")

    dataset_targets = (
        get_dataset_targets(
            dataset,
            target_column=target_column,
            label_schema=label_schema,
        )
        or {}
    )
    allowed_labels = dataset_targets.get(category)
    available_indices: list[int] = []
    for raw_index, row in frame.iterrows():
        label = _normalize_row_label(
            row=row,
            dataset_path=str(split_path),
            target_column=target_column,
            label_schema=label_schema,
        )
        if allowed_labels is not None and label not in allowed_labels:
            continue
        sentences = [
            sentence.strip()
            for sentence in str(row[data_column]).split(delimiter)
            if sentence.strip()
        ]
        if not sentences:
            continue
        available_indices.append(int(raw_index))

    return SplitAlignment(
        raw_indices=np.arange(len(frame), dtype=int),
        available_indices=np.asarray(available_indices, dtype=int),
    )


@lru_cache(maxsize=None)
def _build_preprocessed_split_alignment_cached(
    dataset: str,
    split: str,
    *,
    options: PreprocessingAlignmentOptions,
    label_schema: str,
    availability_key: str,
) -> SplitAlignment:
    return _build_preprocessed_split_alignment(
        dataset,
        split,
        options=options,
        label_schema=label_schema,
        availability_predicate=lambda document: _document_available_for_key(
            document,
            availability_key,
        ),
    )


def _build_preprocessed_split_alignment(
    dataset: str,
    split: str,
    *,
    options: PreprocessingAlignmentOptions,
    label_schema: str,
    availability_predicate: DocumentAvailabilityPredicate,
) -> SplitAlignment:
    split_path, frame = load_dataset_split(dataset, split)
    if options.text_column not in frame.columns:
        raise ValueError(
            f"text_column '{options.text_column}' not found in {split_path}"
        )
    if options.targets is not None:
        if options.target_column is None:
            raise ValueError("target filtering requires target_column.")
        if options.target_column not in frame.columns:
            raise ValueError(
                f"target_column '{options.target_column}' not found in {split_path}"
            )

    available_indices: list[int] = []
    for raw_index, row in frame.iterrows():
        if not _row_matches_targets(
            row=row,
            dataset_path=str(split_path),
            target_column=options.target_column,
            targets=options.targets,
            label_schema=label_schema,
        ):
            continue
        text = _row_text_value(row, options.text_column)
        if not text.strip():
            continue
        document = preprocess_document(
            text,
            language=options.language,
            delimiter=DEFAULT_DELIMITER if options.use_legacy else options.delimiter,
            segmenter="delimiter" if options.use_legacy else options.segmenter,
            tokenizer=options.tokenizer,
            ja_replace_num=options.ja_replace_num,
            ja_stopwords=None,
            ja_dicdir=options.ja_dicdir,
            ja_require_unidic=options.ja_require_unidic,
        )
        if availability_predicate(document):
            available_indices.append(int(raw_index))

    return SplitAlignment(
        raw_indices=np.arange(len(frame), dtype=int),
        available_indices=np.asarray(available_indices, dtype=int),
    )


def _baseline_availability_key(
    *,
    require_document_tokens: bool,
    require_contextual_text: bool,
    require_sentences: bool,
) -> str:
    return (
        "baseline:"
        f"{int(require_document_tokens)}:"
        f"{int(require_contextual_text)}:"
        f"{int(require_sentences)}"
    )


def _predicate_cache_key(
    availability_predicate: DocumentAvailabilityPredicate,
) -> str | None:
    predicate_name = (
        f"{availability_predicate.__module__}." f"{availability_predicate.__qualname__}"
    )
    if (
        predicate_name
        == "src.evaluation.classification.feature_registry._sentence_available"
    ):
        return "sentences"
    return None


def _document_available_for_key(
    document: PreprocessedDocument,
    availability_key: str,
) -> bool:
    if availability_key == "sentences":
        return _has_modelable_sentence(document)
    if availability_key.startswith("baseline:"):
        _prefix, require_document_tokens, require_contextual_text, require_sentences = (
            availability_key.split(":")
        )
        if bool(int(require_sentences)) and not _has_modelable_sentence(document):
            return False
        if bool(int(require_document_tokens)) and not document.document_tokens:
            return False
        if bool(int(require_contextual_text)) and not document.contextual_text:
            return False
        return True
    raise ValueError(f"Unsupported availability cache key: {availability_key}")


def _has_modelable_sentence(document: PreprocessedDocument) -> bool:
    raw_sentences = getattr(document, "sentences_raw", None)
    tokenized_sentences = getattr(document, "sentences_tokenized", None)
    if tokenized_sentences is None:
        return bool(raw_sentences)
    return any(
        str(raw_sentence).strip() and bool(sentence_tokens)
        for raw_sentence, sentence_tokens in zip(raw_sentences, tokenized_sentences)
    )


def _ordered_intersection(source: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    allowed_set = set(int(item) for item in allowed.tolist())
    return np.asarray(
        [int(item) for item in source.tolist() if int(item) in allowed_set],
        dtype=int,
    )


def _selector_from_raw_indices(
    source_indices: np.ndarray,
    selected_raw_indices: np.ndarray,
) -> np.ndarray:
    index_map = {
        int(raw_index): position
        for position, raw_index in enumerate(source_indices.tolist())
    }
    return np.asarray(
        [index_map[int(raw_index)] for raw_index in selected_raw_indices.tolist()],
        dtype=int,
    )


def _row_text_value(row: pd.Series, text_column: str) -> str:
    value = row[text_column]
    if pd.isna(value):
        return ""
    return str(value)


def _row_matches_targets(
    *,
    row: pd.Series,
    dataset_path: str,
    target_column: str | None,
    targets: tuple[str, ...] | None,
    label_schema: str,
) -> bool:
    if targets is None:
        return True
    if target_column is None:
        return False
    label = _normalize_row_label(
        row=row,
        dataset_path=dataset_path,
        target_column=target_column,
        label_schema=label_schema,
    )
    return label in set(targets)


def _normalize_row_label(
    *,
    row: pd.Series,
    dataset_path: str,
    target_column: str,
    label_schema: str,
) -> str:
    return str(row[target_column]).strip()

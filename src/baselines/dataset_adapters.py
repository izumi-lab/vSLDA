from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.core.paths import DATA_ROOT
from src.data.catalog import has_builtin_category_mapping, resolve_category_targets
from src.data.preprocessing import PreprocessedDocument, preprocess_documents


def normalize_baseline_language(language: str) -> str:
    return (language or "").strip().lower()


def use_legacy_category_behavior(dataset: str, language: str) -> bool:
    return has_builtin_category_mapping(dataset) and normalize_baseline_language(
        language
    ) in {"", "en", "english"}


def resolve_baseline_targets(
    dataset: str,
    category: str,
    targets: Sequence[str] | None,
    *,
    language: str,
) -> list[str] | None:
    return resolve_category_targets(
        dataset,
        category,
        targets,
        allow_all_unfiltered=not use_legacy_category_behavior(dataset, language),
    )


def resolve_split_csv_paths(
    dataset: str,
    split: str,
    csv_paths: Sequence[str] | None,
) -> list[str]:
    if csv_paths is not None:
        return list(csv_paths)
    return [str(DATA_ROOT / dataset / f"{split}.csv")]


def load_filtered_texts(
    *,
    csv_paths: Sequence[str],
    text_column: str,
    target_column: str | None,
    targets: Sequence[str] | None,
) -> list[str]:
    texts: list[str] = []
    allowed = set(targets) if targets is not None else None
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        if text_column not in frame.columns:
            raise ValueError(f"text_column '{text_column}' not found in CSV {csv_path}")
        if allowed is not None:
            if target_column is None:
                raise ValueError("target filtering requires target_column.")
            if target_column not in frame.columns:
                raise ValueError(
                    f"target_column '{target_column}' not found in CSV {csv_path}"
                )
            frame = frame.loc[frame[target_column].isin(allowed)]

        values = frame[text_column].fillna("").astype(str).tolist()
        texts.extend([value for value in values if value.strip()])
    return texts


def load_filtered_texts_with_indices(
    *,
    csv_paths: Sequence[str],
    text_column: str,
    target_column: str | None,
    targets: Sequence[str] | None,
) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    raw_indices: list[int] = []
    allowed = set(targets) if targets is not None else None
    raw_offset = 0
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        if text_column not in frame.columns:
            raise ValueError(f"text_column '{text_column}' not found in CSV {csv_path}")
        if allowed is not None:
            if target_column is None:
                raise ValueError("target filtering requires target_column.")
            if target_column not in frame.columns:
                raise ValueError(
                    f"target_column '{target_column}' not found in CSV {csv_path}"
                )

        for row_index, row in frame.iterrows():
            if allowed is not None and row[target_column] not in allowed:
                continue
            value = row[text_column]
            text = "" if pd.isna(value) else str(value)
            if not text.strip():
                continue
            texts.append(text)
            raw_indices.append(raw_offset + int(row_index))
        raw_offset += int(len(frame))
    return texts, raw_indices


def load_sentence_corpus(
    *,
    csv_paths: Sequence[str],
    text_column: str,
    target_column: str | None,
    targets: Sequence[str] | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
) -> list[list[str]]:
    documents = load_preprocessed_documents(
        csv_paths=csv_paths,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
        delimiter=delimiter,
        language=language,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )
    return [doc.sentences_raw for doc in documents if doc.sentences_raw]


def load_document_texts(
    *,
    csv_paths: Sequence[str],
    text_column: str,
    target_column: str | None,
    targets: Sequence[str] | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
) -> list[str]:
    documents = load_preprocessed_documents(
        csv_paths=csv_paths,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
        delimiter=delimiter,
        language=language,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )
    return [doc.lexical_text for doc in documents if doc.document_tokens]


def _baseline_japanese_stopwords(
    stopwords_path: str | None,
) -> None:
    _ = stopwords_path
    # Preserve the current baseline contract: the path is recorded in metadata
    # but stopword filtering is not applied during shared preprocessing.
    return None


def load_preprocessed_documents(
    *,
    csv_paths: Sequence[str],
    text_column: str,
    target_column: str | None,
    targets: Sequence[str] | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
) -> list[PreprocessedDocument]:
    texts = load_filtered_texts(
        csv_paths=csv_paths,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
    )
    return preprocess_documents(
        texts,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords=_baseline_japanese_stopwords(ja_stopwords_path),
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    ).documents


def load_preprocessed_documents_with_indices(
    *,
    csv_paths: Sequence[str],
    text_column: str,
    target_column: str | None,
    targets: Sequence[str] | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
) -> tuple[list[PreprocessedDocument], list[int]]:
    texts, raw_indices = load_filtered_texts_with_indices(
        csv_paths=csv_paths,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
    )
    documents = preprocess_documents(
        texts,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords=_baseline_japanese_stopwords(ja_stopwords_path),
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    ).documents
    if len(documents) != len(raw_indices):
        raise RuntimeError("Preprocessed document count lost raw-index alignment.")
    return documents, raw_indices

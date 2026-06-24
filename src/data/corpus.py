from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

from src.core.errors import require_dataset_path
from src.data.preprocessing import PreprocessedCorpus, preprocess_documents
from src.data.text_processing import split_sentences


def split_document(
    text: str,
    *,
    language: str = "english",
    delimiter: str | None = None,
    segmenter: str = "delimiter",
) -> list[str]:
    return split_sentences(
        text,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
    )


def load_corpus(
    path: str | Path,
    *,
    language: str = "english",
    text_column: str = "data",
    target_column: str | None = "target_str",
    target_filter: Iterable[str] | None = None,
    delimiter: str | None = None,
    segmenter: str = "delimiter",
) -> list[list[str]]:
    corpus = load_preprocessed_corpus(
        path,
        language=language,
        text_column=text_column,
        target_column=target_column,
        target_filter=target_filter,
        delimiter=delimiter,
        segmenter=segmenter,
    )
    return [doc.sentences_raw for doc in corpus.documents if doc.sentences_raw]


def _load_texts(
    path: str | Path,
    *,
    text_column: str,
    target_column: str | None,
    target_filter: Iterable[str] | None,
) -> list[str]:
    resolved_path = require_dataset_path(
        path,
        detail="load_corpus expected an existing dataset input file.",
    )
    texts: list[str] = []

    if resolved_path.suffix.lower() == ".csv":
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(2**31 - 1)

        allowed_targets = set(target_filter) if target_filter is not None else None
        with resolved_path.open("r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if text_column not in fieldnames:
                raise ValueError(
                    f"text_column '{text_column}' not found in CSV {resolved_path}"
                )
            if allowed_targets is not None and target_column not in fieldnames:
                raise ValueError(
                    f"target_column '{target_column}' not found in CSV {resolved_path}"
                )

            for row in reader:
                if allowed_targets is not None:
                    target_value = row.get(target_column or "", None)
                    if target_value not in allowed_targets:
                        continue

                paragraph = (row.get(text_column) or "").strip()
                if not paragraph:
                    continue
                texts.append(paragraph)
        return texts

    with resolved_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            texts.append(text)
    return texts


def load_preprocessed_corpus(
    path: str | Path,
    *,
    language: str = "english",
    text_column: str = "data",
    target_column: str | None = "target_str",
    target_filter: Iterable[str] | None = None,
    delimiter: str | None = None,
    segmenter: str = "delimiter",
    tokenizer: str = "default",
    ja_replace_num: bool = True,
    ja_stopwords: set[str] | None = None,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
) -> PreprocessedCorpus:
    return preprocess_documents(
        _load_texts(
            path,
            text_column=text_column,
            target_column=target_column,
            target_filter=target_filter,
        ),
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords=ja_stopwords,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )

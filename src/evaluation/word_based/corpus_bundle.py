from __future__ import annotations

import json
import re
from pathlib import Path
from sys import stderr
from time import perf_counter
from typing import Any

import pandas as pd
from gensim.corpora import Dictionary

from src.data.catalog import DATASET_TARGETS
from src.data.splits import load_filtered_split_texts
from src.data.text_processing import tokenize_documents as tokenize_documents_shared
from src.data.text_processing import (
    tokenize_sentence_documents as tokenize_sentence_documents_shared,
)
from src.utils.logging import get_logger, get_progress_bar

SINGLE_ASCII_ALPHA_RE = re.compile(r"^[A-Za-z]$")
HIRAGANA_ONLY_RE = re.compile(r"^[ぁ-ゟ]+$")
REFERENCE_PROGRESS_DOC_INTERVAL = 100_000

logger = get_logger(__name__)


def get_targets(dataset: str) -> dict[str, list[str]]:
    if dataset in DATASET_TARGETS:
        return DATASET_TARGETS[dataset]
    if dataset.endswith("_tiny") and dataset.replace("_tiny", "") in DATASET_TARGETS:
        return DATASET_TARGETS[dataset.replace("_tiny", "")]
    return {}


def load_filtered_split_texts_from_csvs(
    *,
    dataset: str,
    csv_paths: tuple[str, ...],
    category: str,
    data_column: str = "data",
    target_column: str = "target_str",
    exclude_labels: set[str] | None = None,
) -> list[str]:
    frames = [pd.read_csv(Path(csv_path)) for csv_path in csv_paths]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if data_column not in frame.columns:
        raise ValueError(f"data_column '{data_column}' not found in {csv_paths[0]}")

    targets = get_targets(dataset)
    if targets:
        if target_column not in frame.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in {csv_paths[0]}"
            )
        if category not in targets:
            raise ValueError(f"Unknown category '{category}' for dataset '{dataset}'")
        frame = frame.loc[frame[target_column].isin(targets[category])]
    if exclude_labels:
        if target_column not in frame.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in {csv_paths[0]}"
            )
        frame = frame.loc[~frame[target_column].isin(exclude_labels)]
    return [str(value) for value in frame[data_column].fillna("")]


def load_documents(
    dataset: str,
    category: str,
    split: str,
    exclude_labels: set[str] | None = None,
    split_csvs: tuple[str, ...] | None = None,
    target_column: str = "target_str",
) -> list[str]:
    if split_csvs:
        return load_filtered_split_texts_from_csvs(
            dataset=dataset,
            csv_paths=split_csvs,
            category=category,
            target_column=target_column,
            exclude_labels=exclude_labels,
        )
    return load_filtered_split_texts(
        dataset,
        category,
        split,
        data_column="data",
        target_column=target_column,
        exclude_labels=exclude_labels,
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
    return tokenize_documents_shared(
        documents,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        min_token_len=min_token_len,
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
    return tokenize_sentence_documents_shared(
        [text],
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        min_token_len=min_token_len,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )[0]


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
    return tokenize_sentence_documents_shared(
        documents,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
        tokenizer=tokenizer,
        min_token_len=min_token_len,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )


def build_sentence_bow_by_document(
    sentence_tokens_by_doc: list[list[list[str]]],
    dictionary: Dictionary,
) -> list[list[list[tuple[int, int]]]]:
    return [
        [dictionary.doc2bow(sentence_tokens) for sentence_tokens in sentence_tokens_doc]
        for sentence_tokens_doc in sentence_tokens_by_doc
    ]


def build_dictionary_and_corpus(
    texts: list[list[str]],
    *,
    dict_no_below: int = 3,
    dict_no_above: float = 0.7,
    dict_exclude_single_alpha: bool = False,
    dict_exclude_with_digit: bool = False,
    dict_exclude_hiragana_only: bool = False,
) -> tuple[Dictionary, list[list[tuple[int, int]]]]:
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=dict_no_below, no_above=dict_no_above)
    if (
        dict_exclude_single_alpha
        or dict_exclude_with_digit
        or dict_exclude_hiragana_only
    ):
        bad_ids: list[int] = []
        for token, token_id in dictionary.token2id.items():
            is_single_alpha = bool(SINGLE_ASCII_ALPHA_RE.fullmatch(token))
            has_digit = any(ch.isdigit() for ch in token)
            is_hiragana_only = bool(HIRAGANA_ONLY_RE.fullmatch(token))
            if (
                (dict_exclude_single_alpha and is_single_alpha)
                or (dict_exclude_with_digit and has_digit)
                or (dict_exclude_hiragana_only and is_hiragana_only)
            ):
                bad_ids.append(token_id)
        if bad_ids:
            dictionary.filter_tokens(bad_ids=bad_ids)
            dictionary.compactify()
    corpus_bow = [dictionary.doc2bow(doc) for doc in texts]
    return dictionary, corpus_bow


def _tokens_from_reference_row(payload: Any, *, line_number: int) -> list[str]:
    if isinstance(payload, dict):
        raw_tokens = payload.get("tokens")
    elif isinstance(payload, list):
        raw_tokens = payload
    else:
        raise ValueError(
            "Reference corpus JSONL rows must be objects with a 'tokens' field "
            f"or raw token lists (line {line_number})."
        )
    if not isinstance(raw_tokens, list):
        raise ValueError(
            f"Reference corpus row has no list-valued 'tokens' field (line {line_number})."
        )
    tokens: list[str] = []
    for token in raw_tokens:
        if not isinstance(token, str):
            raise ValueError(
                "Reference corpus tokens must be strings "
                f"(line {line_number}, token={token!r})."
            )
        normalized = token.strip()
        if normalized:
            tokens.append(normalized)
    return tokens


def load_tokenized_reference_corpus(
    path: Path,
    *,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
) -> list[list[str]]:
    texts = list(
        iter_tokenized_reference_corpus(
            path=path,
            max_docs=max_docs,
            min_doc_tokens=min_doc_tokens,
        )
    )
    if not texts:
        raise ValueError(f"Reference corpus is empty after filtering: {path}")
    return texts


def iter_tokenized_reference_corpus(
    path: Path,
    *,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
):
    if max_docs is not None and max_docs < 1:
        raise ValueError(f"max_docs must be >= 1 when provided, got {max_docs}")
    if min_doc_tokens < 1:
        raise ValueError(f"min_doc_tokens must be >= 1, got {min_doc_tokens}")
    if not path.exists():
        raise FileNotFoundError(f"Reference corpus not found: {path}")

    kept_docs = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in reference corpus at {path}:{line_number}: {exc}"
                ) from exc
            tokens = _tokens_from_reference_row(payload, line_number=line_number)
            if len(tokens) < min_doc_tokens:
                continue
            yield tokens
            kept_docs += 1
            if max_docs is not None and kept_docs >= max_docs:
                break


class CountingTokenizedReferenceCorpus:
    def __init__(
        self,
        path: Path,
        *,
        max_docs: int | None = None,
        min_doc_tokens: int = 1,
    ) -> None:
        self.path = path
        self.max_docs = max_docs
        self.min_doc_tokens = min_doc_tokens
        self.num_docs = 0

    def __iter__(self):
        self.num_docs = 0
        yielded = False
        for tokens in iter_tokenized_reference_corpus(
            path=self.path,
            max_docs=self.max_docs,
            min_doc_tokens=self.min_doc_tokens,
        ):
            yielded = True
            self.num_docs += 1
            yield tokens
        if not yielded:
            raise ValueError(f"Reference corpus is empty after filtering: {self.path}")


def build_reference_corpus_bundle(
    path: Path,
    *,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
    dict_no_below: int = 3,
    dict_no_above: float = 0.7,
    dict_exclude_single_alpha: bool = False,
    dict_exclude_with_digit: bool = False,
    dict_exclude_hiragana_only: bool = False,
) -> tuple[list[list[str]], Dictionary, list[list[tuple[int, int]]]]:
    total_started = perf_counter()
    load_started = perf_counter()
    texts: list[list[str]] = []
    logger.info(
        "reference corpus load start path=%s max_docs=%s min_doc_tokens=%s",
        path,
        max_docs,
        min_doc_tokens,
    )
    reference_iter = iter_tokenized_reference_corpus(
        path=path,
        max_docs=max_docs,
        min_doc_tokens=min_doc_tokens,
    )
    for tokens in get_progress_bar(
        reference_iter,
        total=max_docs,
        desc="reference load",
        unit="docs",
        mininterval=1.0,
        disable=not stderr.isatty(),
    ):
        texts.append(tokens)
        if len(texts) % REFERENCE_PROGRESS_DOC_INTERVAL == 0:
            logger.info(
                "reference corpus load progress docs=%s max_docs=%s sec=%.1f",
                len(texts),
                max_docs,
                perf_counter() - load_started,
            )
    if not texts:
        raise ValueError(f"Reference corpus is empty after filtering: {path}")
    logger.info(
        "reference corpus load done docs=%s sec=%.1f",
        len(texts),
        perf_counter() - load_started,
    )
    dictionary_started = perf_counter()
    logger.info("reference corpus dictionary start docs=%s", len(texts))
    dictionary, corpus_bow = build_dictionary_and_corpus(
        texts,
        dict_no_below=dict_no_below,
        dict_no_above=dict_no_above,
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
    )
    logger.info(
        "reference corpus dictionary done docs=%s vocab=%s sec=%.1f total_sec=%.1f",
        len(texts),
        len(dictionary),
        perf_counter() - dictionary_started,
        perf_counter() - total_started,
    )
    return texts, dictionary, corpus_bow


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
    documents = load_documents(
        dataset=dataset,
        category=category,
        split=split,
        split_csvs=split_csvs,
        target_column=target_column,
        exclude_labels=exclude_labels,
    )
    texts = tokenize_documents(
        documents=documents,
        min_token_len=min_token_len,
        language=language,
        delimiter=delimiter,
        ja_replace_num=ja_replace_num,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )
    dictionary, corpus_bow = build_dictionary_and_corpus(
        texts,
        dict_no_below=dict_no_below,
        dict_no_above=dict_no_above,
        dict_exclude_single_alpha=dict_exclude_single_alpha,
        dict_exclude_with_digit=dict_exclude_with_digit,
        dict_exclude_hiragana_only=dict_exclude_hiragana_only,
    )
    return texts, dictionary, corpus_bow

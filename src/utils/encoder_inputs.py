from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.data.preprocessing import PreprocessedDocument, select_modelable_documents


def flatten_sentence_tokens(
    documents: Sequence[PreprocessedDocument],
) -> list[list[str]]:
    selection = select_modelable_documents(documents)
    return [
        list(tokens)
        for document in selection.documents
        for tokens in document.sentences_tokenized
    ]


def fit_encoder_on_sentences(
    encoder: Any,
    documents: Sequence[PreprocessedDocument],
) -> None:
    if getattr(encoder, "requires_fit", False):
        encoder.fit_tokenized(flatten_sentence_tokens(documents))


def sentence_corpus_for_encoder(
    documents: Sequence[PreprocessedDocument],
    encoder: Any,
) -> list[list[str]]:
    use_tokenized = bool(getattr(encoder, "accepts_tokenized", False))
    selection = select_modelable_documents(documents)
    corpus: list[list[str]] = []
    for document in selection.documents:
        if use_tokenized:
            corpus.append([" ".join(tokens) for tokens in document.sentences_tokenized])
        else:
            corpus.append(list(document.sentences_raw))
    return corpus


def sentence_flat_inputs_for_encoder(
    documents: Sequence[PreprocessedDocument],
    encoder: Any,
) -> tuple[list[str], list[list[str]], np.ndarray, list[PreprocessedDocument]]:
    use_tokenized = bool(getattr(encoder, "accepts_tokenized", False))
    selection = select_modelable_documents(documents)
    filtered: list[PreprocessedDocument] = []
    raw_sentences: list[str] = []
    tokenized_sentences: list[list[str]] = []
    offsets = [0]
    for document in selection.documents:
        filtered.append(document)
        if use_tokenized:
            raw_sentences.extend(
                [" ".join(tokens) for tokens in document.sentences_tokenized]
            )
        else:
            raw_sentences.extend(document.sentences_raw)
        tokenized_sentences.extend(
            [list(tokens) for tokens in document.sentences_tokenized]
        )
        offsets.append(len(raw_sentences))
    return (
        raw_sentences,
        tokenized_sentences,
        np.asarray(offsets, dtype=np.int32),
        filtered,
    )


def encode_sentences(
    encoder: Any,
    raw_sentences: Sequence[str],
    tokenized_sentences: Sequence[Sequence[str]] | None = None,
    **encode_kwargs: Any,
) -> np.ndarray:
    if getattr(encoder, "accepts_tokenized", False):
        if tokenized_sentences is None:
            tokenized_sentences = [str(sentence).split() for sentence in raw_sentences]
        return encoder.encode_tokenized(tokenized_sentences, **encode_kwargs)
    return encoder.encode(raw_sentences, **encode_kwargs)


def fit_encoder_on_documents(
    encoder: Any,
    documents: Sequence[PreprocessedDocument],
) -> None:
    if getattr(encoder, "requires_fit", False):
        encoder.fit_tokenized(
            [list(document.document_tokens) for document in documents]
        )


def document_texts_for_encoder(
    documents: Sequence[PreprocessedDocument],
    encoder: Any,
) -> list[str]:
    if getattr(encoder, "accepts_tokenized", False):
        return [" ".join(document.document_tokens) for document in documents]
    return [document.contextual_text for document in documents]


def encode_documents(
    encoder: Any,
    documents: Sequence[PreprocessedDocument],
    **encode_kwargs: Any,
) -> np.ndarray:
    if getattr(encoder, "accepts_tokenized", False):
        return encoder.encode_tokenized(
            [list(document.document_tokens) for document in documents],
            **encode_kwargs,
        )
    return encoder.encode(
        document_texts_for_encoder(documents, encoder), **encode_kwargs
    )

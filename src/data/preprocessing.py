from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.data.text_processing import split_sentences, tokenize_text


@dataclass(frozen=True)
class PreprocessedDocument:
    raw_text: str
    sentences_raw: list[str]
    sentences_tokenized: list[list[str]]
    sentences_joined: list[str]
    document_tokens: list[str]

    @property
    def contextual_text(self) -> str:
        return " ".join(sentence for sentence in self.sentences_raw if sentence).strip()

    @property
    def lexical_text(self) -> str:
        return " ".join(self.document_tokens).strip()


@dataclass(frozen=True)
class PreprocessedCorpus:
    documents: list[PreprocessedDocument]


@dataclass(frozen=True)
class SelectedCorpus:
    documents: list[PreprocessedDocument]
    raw_doc_indices: list[int]
    sentence_indices_by_doc: list[list[int]]
    dropped_doc_indices: list[int]
    drop_reasons: dict[int, str]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "raw_doc_indices": [int(index) for index in self.raw_doc_indices],
            "sentence_indices_by_doc": [
                [int(index) for index in sentence_indices]
                for sentence_indices in self.sentence_indices_by_doc
            ],
            "dropped_doc_indices": [int(index) for index in self.dropped_doc_indices],
            "drop_reasons": {
                str(int(index)): str(reason)
                for index, reason in self.drop_reasons.items()
            },
        }


def _rebuild_document(
    document: PreprocessedDocument,
    *,
    sentence_indices: Sequence[int],
    token_filter: set[str] | None = None,
) -> PreprocessedDocument:
    kept_raw: list[str] = []
    kept_tokens: list[list[str]] = []
    for sentence_index in sentence_indices:
        raw_sentence = document.sentences_raw[sentence_index]
        sentence_tokens = list(document.sentences_tokenized[sentence_index])
        if token_filter is not None:
            sentence_tokens = [
                token for token in sentence_tokens if token in token_filter
            ]
        kept_raw.append(raw_sentence)
        kept_tokens.append(sentence_tokens)
    return PreprocessedDocument(
        raw_text=document.raw_text,
        sentences_raw=kept_raw,
        sentences_tokenized=kept_tokens,
        sentences_joined=[" ".join(tokens) for tokens in kept_tokens],
        document_tokens=[
            token for sentence_tokens in kept_tokens for token in sentence_tokens
        ],
    )


def select_modelable_documents(
    documents: Sequence[PreprocessedDocument],
    *,
    raw_doc_indices: Sequence[int] | None = None,
    require_sentence_tokens: bool = True,
    require_document_tokens: bool = True,
) -> SelectedCorpus:
    source_indices = (
        list(range(len(documents)))
        if raw_doc_indices is None
        else [int(index) for index in raw_doc_indices]
    )
    if len(source_indices) != len(documents):
        raise ValueError(
            "raw_doc_indices length must match the number of preprocessed documents."
        )

    selected: list[PreprocessedDocument] = []
    selected_doc_indices: list[int] = []
    sentence_indices_by_doc: list[list[int]] = []
    dropped_doc_indices: list[int] = []
    drop_reasons: dict[int, str] = {}

    for raw_doc_index, document in zip(source_indices, documents, strict=True):
        sentence_indices: list[int] = []
        pair_count = min(len(document.sentences_raw), len(document.sentences_tokenized))
        for sentence_index in range(pair_count):
            raw_sentence = str(document.sentences_raw[sentence_index]).strip()
            sentence_tokens = document.sentences_tokenized[sentence_index]
            if not raw_sentence:
                continue
            if require_sentence_tokens and not sentence_tokens:
                continue
            sentence_indices.append(sentence_index)

        if not sentence_indices:
            dropped_doc_indices.append(raw_doc_index)
            drop_reasons[raw_doc_index] = (
                "no_tokenized_sentences" if require_sentence_tokens else "no_sentences"
            )
            continue

        rebuilt = _rebuild_document(document, sentence_indices=sentence_indices)
        if require_document_tokens and not rebuilt.document_tokens:
            dropped_doc_indices.append(raw_doc_index)
            drop_reasons[raw_doc_index] = "no_document_tokens"
            continue

        selected.append(rebuilt)
        selected_doc_indices.append(raw_doc_index)
        sentence_indices_by_doc.append(sentence_indices)

    return SelectedCorpus(
        documents=selected,
        raw_doc_indices=selected_doc_indices,
        sentence_indices_by_doc=sentence_indices_by_doc,
        dropped_doc_indices=dropped_doc_indices,
        drop_reasons=drop_reasons,
    )


def filter_selected_corpus_by_vocabulary(
    selection: SelectedCorpus,
    vocabulary: set[str],
    *,
    drop_reason: str = "no_vocabulary_tokens",
) -> SelectedCorpus:
    selected: list[PreprocessedDocument] = []
    selected_doc_indices: list[int] = []
    sentence_indices_by_doc: list[list[int]] = []
    dropped_doc_indices = list(selection.dropped_doc_indices)
    drop_reasons = dict(selection.drop_reasons)

    for raw_doc_index, document, original_sentence_indices in zip(
        selection.raw_doc_indices,
        selection.documents,
        selection.sentence_indices_by_doc,
        strict=True,
    ):
        kept_positions: list[int] = []
        kept_original_indices: list[int] = []
        for position, original_sentence_index in enumerate(original_sentence_indices):
            tokens = [
                token
                for token in document.sentences_tokenized[position]
                if token in vocabulary
            ]
            if not tokens:
                continue
            kept_positions.append(position)
            kept_original_indices.append(original_sentence_index)

        if not kept_positions:
            dropped_doc_indices.append(raw_doc_index)
            drop_reasons[raw_doc_index] = drop_reason
            continue

        selected.append(
            _rebuild_document(
                document,
                sentence_indices=kept_positions,
                token_filter=vocabulary,
            )
        )
        selected_doc_indices.append(raw_doc_index)
        sentence_indices_by_doc.append(kept_original_indices)

    return SelectedCorpus(
        documents=selected,
        raw_doc_indices=selected_doc_indices,
        sentence_indices_by_doc=sentence_indices_by_doc,
        dropped_doc_indices=dropped_doc_indices,
        drop_reasons=drop_reasons,
    )


def preprocess_document(
    text: str,
    *,
    language: str = "english",
    delimiter: str | None = " / ",
    segmenter: str = "delimiter",
    tokenizer: str = "default",
    min_token_len: int = 2,
    ja_replace_num: bool = True,
    ja_stopwords: set[str] | None = None,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
) -> PreprocessedDocument:
    sentences_raw = split_sentences(
        text,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
    )
    sentences_tokenized = [
        tokenize_text(
            sentence,
            language=language,
            tokenizer=tokenizer,
            min_token_len=min_token_len,
            ja_replace_num=ja_replace_num,
            ja_stopwords=ja_stopwords,
            ja_dicdir=ja_dicdir,
            ja_require_unidic=ja_require_unidic,
        )
        for sentence in sentences_raw
    ]
    return PreprocessedDocument(
        raw_text=text,
        sentences_raw=list(sentences_raw),
        sentences_tokenized=[list(tokens) for tokens in sentences_tokenized],
        sentences_joined=[" ".join(tokens) for tokens in sentences_tokenized],
        document_tokens=[
            token
            for sentence_tokens in sentences_tokenized
            for token in sentence_tokens
        ],
    )


def preprocess_documents(
    documents: Sequence[str],
    *,
    language: str = "english",
    delimiter: str | None = " / ",
    segmenter: str = "delimiter",
    tokenizer: str = "default",
    min_token_len: int = 2,
    ja_replace_num: bool = True,
    ja_stopwords: set[str] | None = None,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
) -> PreprocessedCorpus:
    return PreprocessedCorpus(
        documents=[
            preprocess_document(
                text,
                language=language,
                delimiter=delimiter,
                segmenter=segmenter,
                tokenizer=tokenizer,
                min_token_len=min_token_len,
                ja_replace_num=ja_replace_num,
                ja_stopwords=ja_stopwords,
                ja_dicdir=ja_dicdir,
                ja_require_unidic=ja_require_unidic,
            )
            for text in documents
            if text and text.strip()
        ]
    )

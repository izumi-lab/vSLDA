from __future__ import annotations

import re
from functools import lru_cache

import pysbd

from src.utils.english_tokenizer import tokenize_english_text
from src.utils.japanese_tokenizer import (
    is_japanese_language,
    tokenize_japanese_document_tokens,
    tokenize_japanese_text,
)

SUPPORTED_SEGMENTERS: frozenset[str] = frozenset({"pysbd", "delimiter"})
SUPPORTED_TOKENIZERS: frozenset[str] = frozenset({"default", "simple", "mecab"})
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_language(language: str) -> str:
    """Map common language names to pySBD codes with a safe fallback."""
    normalized = language.strip().lower()
    aliases = {
        "english": "en",
        "en": "en",
        "japanese": "ja",
        "ja": "ja",
    }
    if normalized in aliases:
        return aliases[normalized]
    if len(normalized) == 2:
        return normalized
    return normalized[:2]


def normalize_segmenter_name(segmenter: str) -> str:
    normalized = segmenter.strip().lower()
    aliases = {
        "pysbd": "pysbd",
        "default": "delimiter",
        "delimiter": "delimiter",
        "split_on_delimiter": "delimiter",
    }
    if normalized not in aliases:
        raise ValueError(
            "segmenter must be one of "
            f"{sorted(SUPPORTED_SEGMENTERS)}, got: {segmenter}"
        )
    return aliases[normalized]


def normalize_tokenizer_name(language: str, tokenizer: str) -> str:
    normalized = tokenizer.strip().lower()
    if normalized == "default":
        return "mecab" if is_japanese_language(language) else "simple"

    aliases = {
        "simple": "simple",
        "gensim": "simple",
        "mecab": "mecab",
    }
    if normalized not in aliases:
        raise ValueError(
            "tokenizer must be one of "
            f"{sorted(SUPPORTED_TOKENIZERS)}, got: {tokenizer}"
        )
    resolved = aliases[normalized]
    if resolved == "mecab" and not is_japanese_language(language):
        raise ValueError("tokenizer='mecab' is only supported for Japanese text.")
    if resolved == "simple" and is_japanese_language(language):
        raise ValueError("tokenizer='simple' is not supported for Japanese text.")
    return resolved


@lru_cache(maxsize=None)
def get_segmenter(language: str) -> pysbd.Segmenter:
    return pysbd.Segmenter(language=normalize_language(language), clean=False)


def split_sentences(
    text: str,
    *,
    language: str = "english",
    delimiter: str | None = None,
    segmenter: str = "delimiter",
) -> list[str]:
    normalized = _WHITESPACE_RE.sub(" ", text).strip()
    if not normalized:
        return []

    resolved_segmenter = normalize_segmenter_name(segmenter)
    if resolved_segmenter == "delimiter":
        if delimiter is None:
            return [normalized]
        return [
            sentence.strip()
            for sentence in normalized.split(delimiter)
            if sentence.strip()
        ]

    return [
        sentence.strip()
        for sentence in get_segmenter(language).segment(normalized)
        if sentence.strip()
    ]


def tokenize_text(
    text: str,
    *,
    language: str = "english",
    tokenizer: str = "default",
    min_token_len: int = 2,
    ja_replace_num: bool = True,
    ja_stopwords: set[str] | None = None,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = True,
) -> list[str]:
    if not text or not text.strip():
        return []

    resolved_tokenizer = normalize_tokenizer_name(language, tokenizer)
    if resolved_tokenizer == "mecab":
        return tokenize_japanese_text(
            text,
            replace_num=ja_replace_num,
            stopwords=ja_stopwords,
            dicdir=ja_dicdir,
            require_unidic=ja_require_unidic,
        )

    return tokenize_english_text(
        text,
        min_token_len=min_token_len,
        # Keep the legacy parameter name for compatibility with current call sites.
        replace_num=ja_replace_num,
    )


def tokenize_document_tokens(
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
) -> list[str]:
    if is_japanese_language(language):
        resolved_tokenizer = normalize_tokenizer_name(language, tokenizer)
        if resolved_tokenizer != "mecab":
            raise ValueError(
                "Japanese document tokenization requires tokenizer='default' or 'mecab'."
            )
        segments = split_sentences(
            text,
            language=language,
            delimiter=delimiter,
            segmenter=segmenter,
        )
        return tokenize_japanese_document_tokens(
            delimiter=None,
            text=" / ".join(segments),
            replace_num=ja_replace_num,
            stopwords=ja_stopwords,
            dicdir=ja_dicdir,
            require_unidic=ja_require_unidic,
        )

    tokens: list[str] = []
    for sentence in split_sentences(
        text,
        language=language,
        delimiter=delimiter,
        segmenter=segmenter,
    ):
        tokens.extend(
            tokenize_text(
                sentence,
                language=language,
                tokenizer=tokenizer,
                min_token_len=min_token_len,
            )
        )
    return tokens


def tokenize_sentence_documents(
    documents: list[str],
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
) -> list[list[list[str]]]:
    tokenized: list[list[list[str]]] = []
    for text in documents:
        sentence_tokens: list[list[str]] = []
        for sentence in split_sentences(
            text,
            language=language,
            delimiter=delimiter,
            segmenter=segmenter,
        ):
            sentence_tokens.append(
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
            )
        tokenized.append(sentence_tokens)
    return tokenized


def tokenize_documents(
    documents: list[str],
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
) -> list[list[str]]:
    return [
        tokenize_document_tokens(
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
    ]

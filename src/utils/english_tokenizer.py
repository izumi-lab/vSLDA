from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

_NUM_RE = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")
_TOKEN_RE = re.compile(r"[A-Za-z]+|[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")
_IRREGULAR_VERB_FORMS = frozenset(
    {
        "am",
        "are",
        "arose",
        "ate",
        "be",
        "became",
        "been",
        "began",
        "begun",
        "bought",
        "came",
        "did",
        "does",
        "doing",
        "done",
        "drank",
        "driven",
        "drove",
        "fell",
        "felt",
        "flew",
        "forgot",
        "forgone",
        "forwent",
        "gave",
        "given",
        "gone",
        "grew",
        "grown",
        "had",
        "has",
        "having",
        "is",
        "kept",
        "knew",
        "known",
        "ran",
        "ridden",
        "rode",
        "rose",
        "said",
        "sang",
        "sat",
        "saw",
        "seen",
        "spoke",
        "spoken",
        "stood",
        "swam",
        "swum",
        "taken",
        "taught",
        "thought",
        "took",
        "went",
        "were",
        "won",
        "wrote",
        "written",
        "was",
    }
)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _strip_accents(normalized)
    return normalized.lower()


@lru_cache(maxsize=1)
def _get_wordnet_lemmatizer() -> WordNetLemmatizer:
    try:
        wordnet.ensure_loaded()
    except LookupError as exc:
        raise RuntimeError(
            "NLTK wordnet corpus is required for English lemmatization. "
            "Install it with: poetry run setup-nltk"
        ) from exc
    return WordNetLemmatizer()


def _candidate_pos_tags(token: str) -> tuple[str, ...]:
    if token in _IRREGULAR_VERB_FORMS or token.endswith(("ing", "ed")):
        return ("v", "n", "a", "r")
    if token.endswith("ly"):
        return ("r", "a", "v", "n")
    if token.endswith(("er", "est")):
        return ("a", "r", "n", "v")
    if token.endswith("s") and len(token) > 3:
        return ("n", "v", "a", "r")
    return ("n", "v", "a", "r")


def lemmatize_english_token(token: str) -> str:
    lemmatizer = _get_wordnet_lemmatizer()
    for pos in _candidate_pos_tags(token):
        lemma = lemmatizer.lemmatize(token, pos=pos)
        if lemma and lemma != token:
            return lemma
    return lemmatizer.lemmatize(token)


def tokenize_english_text(
    text: str,
    *,
    min_token_len: int = 2,
    replace_num: bool = True,
) -> list[str]:
    if not text or not text.strip():
        return []

    tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(_normalize_text(text)):
        token = raw_token.strip()
        if not token:
            continue
        if _NUM_RE.fullmatch(token):
            tokens.append("<NUM>" if replace_num else token)
            continue
        lemma = lemmatize_english_token(token)
        if not lemma or len(lemma) < min_token_len:
            continue
        tokens.append(lemma)
    return tokens

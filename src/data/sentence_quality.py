from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable

from src.data.text_processing import split_sentences

PREPROCESSING_VERSION = "english_sentence_quality_v6"

_DISALLOWED_CHARS_RE = re.compile(r"[^0-9A-Za-z.,!?;:'\"()\s-]+")
_BAD_SENTENCE_BOUNDARY_RE = re.compile(
    r"\b[a-z]{2,}[.!?][\"']?[a-z]{2,}\b",
    re.IGNORECASE,
)
_BAD_SENTENCE_BOUNDARY_SPLIT_RE = re.compile(
    r"\b([A-Za-z]{2,}[.!?][\"']?)(?=[A-Za-z]{2,}\b)"
)
_DOMAIN_ENDING_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9.-]*\.(?:com|edu|net|org)\.?$",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_AGE_LOCATION_APPOSITIVE_RE = re.compile(
    r"^[a-z][a-z' -]+,\s*\d{1,3},\s+[^.]+\bof\b[^.]+,\s*"
    r"(?:a[krlz]|c[aot]|d[ce]|fl|ga|hi|i[adln]|k[sy]|la|m[adeinost]|"
    r"n[cdehjmvy]|o[hkr]|pa|ri|s[cd]|t[nx]|ut|v[at]|w[aivy])\.$",
    re.IGNORECASE,
)
_LEADING_ARTIFACT_RE = re.compile(r"^(?:[:>|#*=_\"']\s*)+")
_LEADING_ENUMERATOR_RE = re.compile(
    r"^(?:\(?\d+[.)]|[ivxlcdm]{1,6}\))\s+",
    re.IGNORECASE,
)
_PARENTHETICAL_ONLY_RE = re.compile(r"^\([^()]+\)\.?$")
_PUNCT_RE = re.compile(r"[^A-Za-z0-9\s]")
_REPEATED_PUNCT_RE = re.compile(r"(?:[-_=*~.]){3,}")
_TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+")
_CHUNK_RE = re.compile(r"[A-Za-z0-9]+")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")
_TRAILING_EMOTICON_RE = re.compile(r"\s*[:;]-?[)D]\s*$")
_SHORT_DISCOURSE_FRAGMENT_KEYS = frozenset(
    {
        "hi terry",
        "no nagging",
        "non toxic",
        "not necessarily",
    }
)

_TRANSLATION_TABLE = str.maketrans(
    {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)


@dataclass(frozen=True)
class SentenceQualityConfig:
    version: str = PREPROCESSING_VERSION
    min_word_tokens: int = 4
    min_alpha_chars: int = 4
    max_word_tokens: int = 120
    min_bad_boundary_word_tokens: int = 50
    max_parenthetical_only_word_tokens: int = 8
    max_punctuation_ratio: float = 0.45
    max_upper_noise_ratio: float = 0.65
    strip_leading_quote_markers: bool = True
    strip_leading_enumerators: bool = True

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SentenceQualityDecision:
    keep: bool
    reason: str
    cleaned_sentence: str
    char_count: int
    word_token_count: int
    alpha_char_count: int
    punctuation_ratio: float
    alpha_ratio: float

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedSentence:
    raw_sentence: str
    decision: SentenceQualityDecision

    @property
    def cleaned_sentence(self) -> str:
        return self.decision.cleaned_sentence


@dataclass(frozen=True)
class PreparedDocumentText:
    text: str
    kept_sentences: list[str]
    sentence_decisions: list[PreparedSentence]

    @property
    def candidate_sentence_count(self) -> int:
        return len(self.sentence_decisions)

    @property
    def kept_sentence_count(self) -> int:
        return len(self.kept_sentences)

    @property
    def dropped_sentence_count(self) -> int:
        return self.candidate_sentence_count - self.kept_sentence_count


@dataclass
class SentencePreparationStats:
    documents_seen: int = 0
    documents_kept: int = 0
    documents_dropped: int = 0
    candidate_sentences: int = 0
    kept_sentences: int = 0
    dropped_sentences: int = 0
    drop_reasons: Counter[str] = field(default_factory=Counter)

    def add_document(self, prepared: PreparedDocumentText) -> None:
        self.documents_seen += 1
        if prepared.kept_sentences:
            self.documents_kept += 1
        else:
            self.documents_dropped += 1

        self.candidate_sentences += prepared.candidate_sentence_count
        self.kept_sentences += prepared.kept_sentence_count
        self.dropped_sentences += prepared.dropped_sentence_count
        for prepared_sentence in prepared.sentence_decisions:
            decision = prepared_sentence.decision
            if not decision.keep:
                self.drop_reasons[decision.reason] += 1

    def extend(self, prepared_documents: Iterable[PreparedDocumentText]) -> None:
        for prepared in prepared_documents:
            self.add_document(prepared)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "documents_seen": int(self.documents_seen),
            "documents_kept": int(self.documents_kept),
            "documents_dropped": int(self.documents_dropped),
            "candidate_sentences": int(self.candidate_sentences),
            "kept_sentences": int(self.kept_sentences),
            "dropped_sentences": int(self.dropped_sentences),
            "drop_reasons": {
                reason: int(count)
                for reason, count in sorted(self.drop_reasons.items())
            },
        }


DEFAULT_SENTENCE_QUALITY_CONFIG = SentenceQualityConfig()


def clean_english_sentence(
    text: str,
    *,
    strip_leading_quote_markers: bool = True,
    strip_leading_enumerators: bool = True,
) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(str(text)))
    normalized = normalized.translate(_TRANSLATION_TABLE)
    normalized = _URL_RE.sub(" ", normalized)
    normalized = _EMAIL_RE.sub(" ", normalized)
    normalized = _REPEATED_PUNCT_RE.sub(" ", normalized)
    normalized = _DISALLOWED_CHARS_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    if strip_leading_quote_markers:
        normalized = _LEADING_ARTIFACT_RE.sub("", normalized).strip()
    if strip_leading_enumerators:
        normalized = _LEADING_ENUMERATOR_RE.sub("", normalized).strip()
    normalized = _TRAILING_EMOTICON_RE.sub("", normalized).strip()
    normalized = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", normalized)
    return normalized.strip()


def repair_bad_sentence_boundaries(text: str) -> str:
    return _BAD_SENTENCE_BOUNDARY_SPLIT_RE.sub(r"\1 ", str(text))


def split_english_sentence_candidates(
    text: str,
    *,
    language: str = "english",
) -> list[str]:
    return split_sentences(
        repair_bad_sentence_boundaries(text),
        language=language,
        delimiter=None,
        segmenter="pysbd",
    )


def _is_upper_noise_sentence(cleaned: str, *, max_upper_noise_ratio: float) -> bool:
    chunks = _CHUNK_RE.findall(cleaned)
    if len(chunks) < 3:
        return False
    noisy_chunks = 0
    for chunk in chunks:
        has_digit = any(char.isdigit() for char in chunk)
        has_alpha = any(char.isalpha() for char in chunk)
        is_short_upper = has_alpha and chunk.upper() == chunk and len(chunk) <= 5
        if has_digit or is_short_upper:
            noisy_chunks += 1
    return noisy_chunks / len(chunks) >= max_upper_noise_ratio


def _short_fragment_key(cleaned: str) -> str:
    return " ".join(re.findall(r"[a-z]+", cleaned.lower()))


def _has_unbalanced_parentheses(cleaned: str) -> bool:
    return cleaned.count("(") > cleaned.count(")")


def assess_sentence_quality(
    sentence: str,
    *,
    config: SentenceQualityConfig = DEFAULT_SENTENCE_QUALITY_CONFIG,
) -> SentenceQualityDecision:
    cleaned = clean_english_sentence(
        sentence,
        strip_leading_quote_markers=config.strip_leading_quote_markers,
        strip_leading_enumerators=config.strip_leading_enumerators,
    )
    char_count = len(cleaned)
    word_token_count = len(_TOKEN_RE.findall(cleaned))
    alpha_char_count = sum(1 for char in cleaned if char.isalpha())
    punctuation_count = len(_PUNCT_RE.findall(cleaned))
    punctuation_ratio = punctuation_count / char_count if char_count else 0.0
    alpha_ratio = alpha_char_count / char_count if char_count else 0.0

    reason = "kept"
    keep = True
    if not cleaned:
        keep = False
        reason = "empty_after_cleaning"
    elif _PUNCT_ONLY_RE.fullmatch(cleaned):
        keep = False
        reason = "punctuation_only"
    elif (
        _PARENTHETICAL_ONLY_RE.fullmatch(cleaned)
        and word_token_count <= config.max_parenthetical_only_word_tokens
    ):
        keep = False
        reason = "parenthetical_only"
    elif _DOMAIN_ENDING_RE.search(cleaned):
        keep = False
        reason = "domain_ending_fragment"
    elif _has_unbalanced_parentheses(cleaned):
        keep = False
        reason = "unbalanced_parentheses"
    elif word_token_count > config.max_word_tokens:
        keep = False
        reason = "too_many_word_tokens"
    elif (
        word_token_count >= config.min_bad_boundary_word_tokens
        and _BAD_SENTENCE_BOUNDARY_RE.search(cleaned)
    ):
        keep = False
        reason = "bad_sentence_boundary"
    elif _AGE_LOCATION_APPOSITIVE_RE.fullmatch(cleaned):
        keep = False
        reason = "age_location_appositive_fragment"
    elif _is_upper_noise_sentence(
        cleaned,
        max_upper_noise_ratio=config.max_upper_noise_ratio,
    ):
        keep = False
        reason = "upper_noise"
    elif word_token_count <= 3 and not cleaned.endswith((".", "?", "!")):
        keep = False
        reason = "short_fragment_no_terminal_punctuation"
    elif (
        word_token_count <= 2
        and _short_fragment_key(cleaned) in _SHORT_DISCOURSE_FRAGMENT_KEYS
    ):
        keep = False
        reason = "short_discourse_fragment"
    elif word_token_count < config.min_word_tokens:
        keep = False
        reason = "too_few_word_tokens"
    elif alpha_char_count < config.min_alpha_chars:
        keep = False
        reason = "too_few_alpha_chars"
    elif punctuation_ratio > config.max_punctuation_ratio:
        keep = False
        reason = "punctuation_heavy"

    return SentenceQualityDecision(
        keep=keep,
        reason=reason,
        cleaned_sentence=cleaned,
        char_count=char_count,
        word_token_count=word_token_count,
        alpha_char_count=alpha_char_count,
        punctuation_ratio=punctuation_ratio,
        alpha_ratio=alpha_ratio,
    )


def prepare_english_document_text(
    text: str,
    *,
    delimiter: str = " / ",
    language: str = "english",
    config: SentenceQualityConfig = DEFAULT_SENTENCE_QUALITY_CONFIG,
) -> PreparedDocumentText:
    decisions: list[PreparedSentence] = []
    kept_sentences: list[str] = []
    for raw_sentence in split_english_sentence_candidates(text, language=language):
        decision = assess_sentence_quality(raw_sentence, config=config)
        decisions.append(PreparedSentence(raw_sentence=raw_sentence, decision=decision))
        if decision.keep:
            kept_sentences.append(decision.cleaned_sentence)

    return PreparedDocumentText(
        text=delimiter.join(kept_sentences),
        kept_sentences=kept_sentences,
        sentence_decisions=decisions,
    )

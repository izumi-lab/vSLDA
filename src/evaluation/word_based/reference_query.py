from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .topic_word_metrics import topic_words_to_word_lists
from .topic_words import TopicWords


def canonical_pair(word_i: str, word_j: str) -> tuple[str, str]:
    return (word_i, word_j) if word_i <= word_j else (word_j, word_i)


@dataclass(frozen=True)
class ReferenceCountQuery:
    """Exact reference counts required by a set of coherence evaluations."""

    target_words: tuple[str, ...]
    requested_pairs: tuple[tuple[str, str], ...]
    window_sizes: tuple[int, ...]
    need_document_counts: bool = False

    def __post_init__(self) -> None:
        target_words = tuple(sorted(set(self.target_words)))
        target_set = set(target_words)
        requested_pairs = tuple(
            sorted(
                {
                    canonical_pair(str(word_i), str(word_j))
                    for word_i, word_j in self.requested_pairs
                    if word_i != word_j
                }
            )
        )
        if any(
            word_i not in target_set or word_j not in target_set
            for word_i, word_j in requested_pairs
        ):
            raise ValueError("requested_pairs must only contain target words")
        window_sizes = tuple(sorted({int(value) for value in self.window_sizes}))
        if any(value < 1 for value in window_sizes):
            raise ValueError("window_sizes must be positive")
        object.__setattr__(self, "target_words", target_words)
        object.__setattr__(self, "requested_pairs", requested_pairs)
        object.__setattr__(self, "window_sizes", window_sizes)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "target_words": self.target_words,
                "requested_pairs": self.requested_pairs,
                "window_sizes": self.window_sizes,
                "need_document_counts": self.need_document_counts,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    def is_subset_of(self, other: ReferenceCountQuery) -> bool:
        return (
            set(self.target_words).issubset(other.target_words)
            and set(self.requested_pairs).issubset(other.requested_pairs)
            and set(self.window_sizes).issubset(other.window_sizes)
            and (not self.need_document_counts or other.need_document_counts)
        )


def requested_pairs_from_topic_words(
    topic_words_by_condition: Iterable[TopicWords],
) -> set[tuple[str, str]]:
    return {
        canonical_pair(word_i, word_j)
        for topic_words in topic_words_by_condition
        for words in topic_words_to_word_lists(topic_words)
        for word_i, word_j in combinations(words, 2)
        if word_i != word_j
    }


def build_reference_count_query(
    *,
    topic_words_by_condition: Iterable[TopicWords],
    window_sizes: Iterable[int],
    need_document_counts: bool = False,
) -> ReferenceCountQuery:
    topic_words = list(topic_words_by_condition)
    target_words = {
        word
        for condition in topic_words
        for words in topic_words_to_word_lists(condition)
        for word in words
    }
    return ReferenceCountQuery(
        target_words=tuple(target_words),
        requested_pairs=tuple(requested_pairs_from_topic_words(topic_words)),
        window_sizes=tuple(window_sizes),
        need_document_counts=need_document_counts,
    )


def full_pair_query(
    *,
    target_words: Iterable[str],
    window_sizes: Iterable[int],
    need_document_counts: bool = True,
) -> ReferenceCountQuery:
    words = tuple(sorted(set(target_words)))
    return ReferenceCountQuery(
        target_words=words,
        requested_pairs=tuple(combinations(words, 2)),
        window_sizes=tuple(window_sizes),
        need_document_counts=need_document_counts,
    )

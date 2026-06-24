from __future__ import annotations

from .doc_topic import get_doc_topic_features
from .sentence_topic import flatten_sentence_topic_features
from .topic_word_proxy import (
    build_topic_word_from_doc_topic,
    build_topic_word_from_sentence_topic,
)

__all__ = [
    "build_topic_word_from_doc_topic",
    "build_topic_word_from_sentence_topic",
    "flatten_sentence_topic_features",
    "get_doc_topic_features",
]

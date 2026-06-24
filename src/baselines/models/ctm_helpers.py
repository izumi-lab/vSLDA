from __future__ import annotations

import string
from collections.abc import Sequence

from gensim.utils import deaccent
from sklearn.feature_extraction.text import CountVectorizer


class WhiteSpacePreprocessingStopwords:
    """Compatibility helper for the historical CTM whitespace preprocessing."""

    def __init__(
        self,
        documents: Sequence[str],
        stopwords_list: Sequence[str] | None = None,
        vocabulary_size: int = 2000,
        max_df: float | int = 1.0,
        min_words: int = 1,
        remove_numbers: bool = True,
    ) -> None:
        self.documents = list(documents)
        self.stopwords = set(stopwords_list or [])
        self.vocabulary_size = vocabulary_size
        self.max_df = max_df
        self.min_words = min_words
        self.remove_numbers = remove_numbers

    def preprocess(self) -> tuple[list[str], list[str], list[str], list[int]]:
        normalized_docs = [deaccent(doc.lower()) for doc in self.documents]
        normalized_docs = [
            doc.translate(
                str.maketrans(string.punctuation, " " * len(string.punctuation))
            )
            for doc in normalized_docs
        ]
        if self.remove_numbers:
            normalized_docs = [
                doc.translate(str.maketrans("0123456789", " " * len("0123456789")))
                for doc in normalized_docs
            ]
        normalized_docs = [
            " ".join(
                token for token in doc.split() if token and token not in self.stopwords
            )
            for doc in normalized_docs
        ]

        vectorizer = CountVectorizer(
            max_features=self.vocabulary_size,
            max_df=self.max_df,
        )
        vectorizer.fit_transform(normalized_docs)
        vocabulary = set(vectorizer.get_feature_names_out())

        preprocessed_docs = [
            " ".join(token for token in doc.split() if token in vocabulary)
            for doc in normalized_docs
        ]
        unpreprocessed_docs = list(self.documents)
        retained_indices = list(range(len(self.documents)))

        return (
            preprocessed_docs,
            unpreprocessed_docs,
            list({token for doc in preprocessed_docs for token in doc.split()}),
            retained_indices,
        )

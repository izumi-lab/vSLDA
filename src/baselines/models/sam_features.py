"""Document featurization for SAM: L2-normalized tf-idf on the unit sphere.

SAM consumes one unit vector per document in vocabulary space.  This module
builds those vectors from the repository's shared preprocessing output, keeping
two contracts intact:

* Vocabulary semantics match :mod:`src.baselines.models.bleilda` -- a
  ``gensim.corpora.Dictionary`` with ``filter_extremes`` -- so coherence
  evaluation sees the same kind of vocabulary it sees for LDA.
* Tokenization stays inside the repository's shared pipeline.  ``TfidfVectorizer``
  is deliberately *not* used on raw text, because it would bypass the tokenizer
  contract that ``tests/baselines/test_shared_preprocessing_integration.py``
  guards.

The tf-idf weights reproduce ``sklearn.feature_extraction.text.TfidfTransformer``
with ``norm="l2", smooth_idf=True`` exactly, but are computed here so the fitted
idf is a plain ``ndarray`` that can be persisted and replayed at inference time
without pickling a versioned scikit-learn estimator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import gensim
import numpy as np
import scipy.sparse as sp

from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    filter_selected_corpus_by_vocabulary,
)

__all__ = [
    "SamCorpus",
    "SamVocabulary",
    "build_inference_corpus",
    "build_training_corpus",
]

# The vMF numerics require at least three dimensions.
MIN_VOCABULARY_SIZE = 3


@dataclass(frozen=True)
class SamVocabulary:
    """Training vocabulary plus the idf weights fitted alongside it."""

    words: tuple[str, ...]
    idf: np.ndarray
    feature_scheme: str
    sublinear_tf: bool

    def __post_init__(self) -> None:
        if len(self.words) != int(self.idf.shape[0]):
            raise ValueError("SAM vocabulary and idf vector have different lengths.")

    @property
    def size(self) -> int:
        return len(self.words)

    def index_map(self) -> dict[str, int]:
        return {word: index for index, word in enumerate(self.words)}


@dataclass(frozen=True)
class SamCorpus:
    """L2-normalized tf-idf matrix aligned with a :class:`SelectedCorpus`."""

    matrix: sp.csr_matrix  # (D, V)
    documents: list[PreprocessedDocument]
    selection: SelectedCorpus

    def __post_init__(self) -> None:
        if self.matrix.shape[0] != len(self.documents):
            raise ValueError("SAM corpus matrix and documents are misaligned.")
        if len(self.selection.documents) != len(self.documents):
            raise ValueError("SAM corpus selection and documents are misaligned.")


def _token_documents(documents: Sequence[PreprocessedDocument]) -> list[list[str]]:
    return [list(document.document_tokens) for document in documents]


def _count_matrix(
    token_documents: Sequence[Sequence[str]], index_map: dict[str, int]
) -> sp.csr_matrix:
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    for tokens in token_documents:
        counts = Counter(index_map[token] for token in tokens if token in index_map)
        for column, count in sorted(counts.items()):
            indices.append(int(column))
            values.append(float(count))
        indptr.append(len(indices))
    return sp.csr_matrix(
        (
            np.asarray(values, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(indptr) - 1, len(index_map)),
    )


def _fit_idf(counts: sp.csr_matrix, feature_scheme: str = "tfidf") -> np.ndarray:
    """``idf = ln((1 + n_docs) / (1 + df)) + 1``, i.e. sklearn's ``smooth_idf``.

    ``feature_scheme="tf"`` returns all ones, which makes the weighting step a
    no-op and leaves plain L2-normalized term frequencies.  The paper reports
    ``tf`` and ``tf-idf`` side by side ("SAM uses kappa = 1500, l2-normalized tf
    or tf-idf document representations"), and the gap between them is large --
    around 4.7 accuracy points on its news-20 tasks -- so the two are separate
    conditions rather than one default and one variant.
    """

    num_documents = counts.shape[0]
    if str(feature_scheme).strip().lower() == "tf":
        return np.ones(counts.shape[1], dtype=np.float64)
    document_frequency = np.asarray((counts > 0).sum(axis=0), dtype=np.float64).ravel()
    return np.log((1.0 + num_documents) / (1.0 + document_frequency)) + 1.0


def _apply_tfidf(counts: sp.csr_matrix, vocabulary: SamVocabulary) -> sp.csr_matrix:
    matrix = counts.astype(np.float64).tocsr(copy=True)
    if vocabulary.sublinear_tf:
        matrix.data = 1.0 + np.log(matrix.data)
    matrix = matrix @ sp.diags(vocabulary.idf)
    matrix = sp.csr_matrix(matrix)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if np.any(norms <= 0.0):
        empty = int(np.count_nonzero(norms <= 0.0))
        raise ValueError(
            f"{empty} SAM document(s) have zero feature norm; vocabulary filtering "
            "must run through filter_selected_corpus_by_vocabulary first."
        )
    matrix = sp.diags(1.0 / norms) @ matrix
    return sp.csr_matrix(matrix)


def _retained_words(*, selection: SelectedCorpus, params) -> tuple[str, ...]:
    dictionary = gensim.corpora.Dictionary(_token_documents(selection.documents))
    if len(dictionary) == 0:
        raise ValueError("Tokenization produced an empty dictionary for SAM.")
    dictionary.filter_extremes(
        no_below=int(params.min_df),
        no_above=float(params.max_df),
        keep_n=(
            None
            if params.max_vocabulary_size is None
            else int(params.max_vocabulary_size)
        ),
    )
    dictionary.compactify()
    if len(dictionary) < MIN_VOCABULARY_SIZE:
        raise ValueError(
            "SAM vocabulary collapsed to "
            f"{len(dictionary)} term(s) after filter_extremes(min_df={params.min_df}, "
            f"max_df={params.max_df}); relax the pruning thresholds."
        )
    return tuple(str(dictionary[index]) for index in range(len(dictionary)))


def build_training_corpus(
    *, selection: SelectedCorpus, params
) -> tuple[SamVocabulary, SamCorpus]:
    """Fit the vocabulary and idf on the training split and featurize it.

    Documents left with no in-vocabulary token are dropped through
    :func:`filter_selected_corpus_by_vocabulary`, which is the only place a SAM
    document drop may happen: routing it there is what keeps
    ``SelectedCorpus.raw_doc_indices`` and ``drop_reasons`` correct, and hence
    what lets ``validate_no_additional_document_drop`` pass in ``persist_sam_run``.

    Known one-pass caveat, shared with ETM: ``filter_extremes`` computes document
    frequencies before the empty documents are removed, so the df of a surviving
    term shifts very slightly afterwards.  Iterating to a fixed point is not
    worth the extra pass.
    """

    words = _retained_words(selection=selection, params=params)
    filtered = filter_selected_corpus_by_vocabulary(selection, set(words))
    if not filtered.documents:
        raise ValueError("SAM vocabulary filtering removed every training document.")

    index_map = {word: index for index, word in enumerate(words)}
    counts = _count_matrix(_token_documents(filtered.documents), index_map)
    vocabulary = SamVocabulary(
        words=words,
        idf=_fit_idf(counts, str(params.feature_scheme)),
        feature_scheme=str(params.feature_scheme),
        sublinear_tf=bool(params.sublinear_tf),
    )
    corpus = SamCorpus(
        matrix=_apply_tfidf(counts, vocabulary),
        documents=list(filtered.documents),
        selection=filtered,
    )
    return vocabulary, corpus


def build_inference_corpus(
    *, selection: SelectedCorpus, vocabulary: SamVocabulary, empty_error_message: str
) -> SamCorpus:
    """Featurize a held-out split with the *training* vocabulary and idf.

    Refitting idf on the evaluation split would make held-out features depend on
    the split's own composition and would not be reproducible from the persisted
    artifacts, so the fitted idf is replayed verbatim.
    """

    words = set(vocabulary.words)
    filtered = filter_selected_corpus_by_vocabulary(selection, words)
    if not filtered.documents:
        raise ValueError(empty_error_message)

    counts = _count_matrix(_token_documents(filtered.documents), vocabulary.index_map())
    return SamCorpus(
        matrix=_apply_tfidf(counts, vocabulary),
        documents=list(filtered.documents),
        selection=filtered,
    )

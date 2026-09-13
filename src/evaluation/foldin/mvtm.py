"""Collapsed fold-in document-topic distributions of one MvTM (vLDA) run.

MvTM applies the vMF Sentence LDA sampler to word vectors, one token per
assignment unit, and estimates the document-topic distribution of a training
document from the topic counts of its final sweep and that of a held-out
document from per-token argmax counts (``infer/<category>.pkl``). This module
re-estimates both splits with the estimator of the vMF Sentence LDA fold-in
(:mod:`.theta`): the frozen per-topic vMF mixtures give a ``(V, K)`` table of
token log likelihoods over the model's word vectors, and the collapsed fold-in
runs over each document's in-vocabulary tokens (out-of-vocabulary tokens are
dropped exactly as at training time), ``chunk_docs`` documents per sampler
call so that a whole-corpus run at large K stays within memory.

The artifacts follow the baseline layout: ``params/<category>_doc_topic_foldin.pkl``
and ``..._foldin_counts.pkl`` for the training split, the same names under
``infer/`` for the held-out split, and ``foldin_meta.json`` in the run directory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.core.artifacts import load_artifact_json, load_artifact_pickle
from src.evaluation.foldin.artifacts import (
    DEFAULT_CHUNK_DOCS,
    FoldInLayout,
    FoldInRunResult,
    compute_foldin_from_type_table,
    condition_fingerprint_of,
    foldin_fingerprint,
    token_corpus_fingerprint,
    word_vector_fingerprint,
)
from src.evaluation.word_based.topic_assignment import CollapsedFoldInConfig
from src.evaluation.word_based.topic_word_model_adapters import (
    vmf_mixture_log_likelihood,
)
from src.evaluation.word_based.topic_word_runtime import (
    _alpha_vector,
    _condition_metadata,
    _load_documents_and_ids,
)
from src.models.vmf_encoding import VMFDocumentEncoder

MODEL = "mvtm"
ASSIGNMENT_UNIT_TYPE = "token"
PARAMS_DIRNAME = "params"
INFER_DIRNAME = "infer"


@dataclass(frozen=True)
class TokenCorpus:
    """The preprocessed documents of one split of an MvTM run."""

    condition_dir: Path
    split: str
    documents: list[Any]
    raw_doc_indices: list[int]
    token_sha1: str

    @property
    def num_documents(self) -> int:
        return len(self.documents)


def load_token_corpus(condition_dir: Path, *, split: str) -> TokenCorpus:
    """``params/`` (train) or ``infer/`` (test) ``preprocessed_corpus.pkl`` with its selection."""

    condition_dir = Path(condition_dir)
    documents, raw_ids = _load_documents_and_ids(
        model=MODEL, condition_dir=condition_dir, split=split
    )
    return TokenCorpus(
        condition_dir=condition_dir,
        split=split,
        documents=list(documents),
        raw_doc_indices=[int(value) for value in raw_ids],
        token_sha1=token_corpus_fingerprint(documents),
    )


def word_vector_source(condition_dir: Path) -> tuple[str, str | None]:
    """``(word2vec, wikientvec_cache_dir)`` the run was trained with, from its metadata."""

    metadata = _condition_metadata(Path(condition_dir))
    baseline_params = metadata.get("baseline_params")
    if not isinstance(baseline_params, dict) or not baseline_params.get("word2vec"):
        raise ValueError(f"Saved word-vector source is missing from {condition_dir}")
    cache_dir = baseline_params.get("wikientvec_cache_dir")
    return str(baseline_params["word2vec"]), (
        None if cache_dir in {None, ""} else str(cache_dir)
    )


def mvtm_encoder_fingerprint(condition_dir: Path) -> str:
    word2vec, cache_dir = word_vector_source(condition_dir)
    return word_vector_fingerprint(word2vec, wikientvec_cache_dir=cache_dir)


def load_mvtm_word_vectors(condition_dir: Path):
    """The run's word vectors (``params/local_word2vec.kv`` when it trained its own)."""

    from src.baselines.models.gaussian_helpers import load_gaussian_word_vectors

    word2vec, cache_dir = word_vector_source(condition_dir)
    return load_gaussian_word_vectors(
        word2vec,
        param_dir=Path(condition_dir) / PARAMS_DIRNAME,
        wikientvec_cache_dir=cache_dir,
    )


def normalized_word_rows(vectors, words: Sequence[str]) -> np.ndarray:
    """The unit vectors MvTM observes for ``words``: ``WordVectorEncoder.encode``
    followed by ``VMFDocumentEncoder.encode_and_normalize`` (no transform), in the
    trainer's storage precision, so that the likelihoods equal the training-time ones.
    """

    if not words:
        return np.zeros((0, int(vectors.vector_size)), dtype=np.float64)
    rows = np.vstack(
        [np.asarray(vectors[str(word)], dtype=np.float64) for word in words]
    ).astype(np.float64, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12
    stored = np.asarray(rows / norms, dtype=VMFDocumentEncoder.STORAGE_DTYPE)
    return np.asarray(stored, dtype=np.float64)


def token_type_table(
    documents: Sequence[Any], *, supported: Mapping[str, int]
) -> tuple[list[str], list[np.ndarray]]:
    """The in-vocabulary word types of the corpus (first-occurrence order) and each
    document's tokens as row indices of that list; out-of-vocabulary tokens are
    dropped like ``WordVectorEncoder.encode`` drops them at training time."""

    type_index: dict[str, int] = {}
    covered_words: list[str] = []
    type_ids_by_doc: list[np.ndarray] = []
    for document in documents:
        ids: list[int] = []
        for token in getattr(document, "document_tokens", document):
            word = str(token)
            if word not in supported:
                continue
            index = type_index.get(word)
            if index is None:
                index = len(covered_words)
                type_index[word] = index
                covered_words.append(word)
            ids.append(index)
        type_ids_by_doc.append(np.asarray(ids, dtype=np.int64))
    return covered_words, type_ids_by_doc


def mvtm_token_log_likelihoods(
    *,
    documents: Sequence[Any],
    vectors,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    kappa_per_topic: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """``(table (V_cov, K), type_ids_by_doc, covered_words)`` under the frozen topics."""

    covered_words, type_ids_by_doc = token_type_table(
        documents, supported=vectors.key_to_index
    )
    rows = normalized_word_rows(vectors, covered_words)
    means = np.asarray(component_means, dtype=np.float64)
    if rows.shape[0]:
        table = vmf_mixture_log_likelihood(
            rows,
            mixture_weights=np.asarray(mixture_weights, dtype=np.float64),
            component_means=means,
            kappa_per_topic=np.asarray(kappa_per_topic, dtype=np.float64),
        )
    else:
        table = np.zeros((0, means.shape[0]), dtype=np.float64)
    return table, type_ids_by_doc, covered_words


def mvtm_frozen_parameters(
    condition_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(mixture_weights, component_means, kappa_per_topic, alpha)`` from ``params/``."""

    params_dir = Path(condition_dir) / PARAMS_DIRNAME
    params = load_artifact_json(params_dir / "params.json")
    weights = np.asarray(load_artifact_pickle(params_dir / "mixture_weights.pkl"))
    means = np.asarray(load_artifact_pickle(params_dir / "component_means.pkl"))
    kappa = np.asarray(load_artifact_pickle(params_dir / "kappa_per_topic.pkl"))
    alpha = _alpha_vector(params["alpha"], int(means.shape[0]))
    return weights, means, kappa, alpha


def _existing_doc_topic_rows(
    condition_dir: Path, *, split: str, category: str
) -> int | None:
    path = (
        Path(condition_dir) / PARAMS_DIRNAME / "table_counts_per_doc.pkl"
        if split == "train"
        else Path(condition_dir) / INFER_DIRNAME / f"{category}.pkl"
    )
    if not path.exists():
        return None
    values = np.asarray(load_artifact_pickle(path))
    return int(values.shape[0]) if values.ndim == 2 else None


def compute_mvtm_foldin(
    *,
    documents: Sequence[Any],
    vectors,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    kappa_per_topic: np.ndarray,
    alpha: np.ndarray,
    split: str,
    dataset: str,
    data_run: str,
    category: str,
    encoder_fp: str,
    condition_fp: str,
    word2vec_name: str | None,
    corpus_sha1: str | None = None,
    foldin_config: CollapsedFoldInConfig | None = None,
    chunk_docs: int = DEFAULT_CHUNK_DOCS,
    extra_metadata: Mapping[str, Any] | None = None,
    timing: Mapping[str, float] | None = None,
) -> FoldInRunResult:
    """The fold-in of one split from in-memory parameters (shared by the post-hoc
    command and the training runner, which therefore agree bit for bit)."""

    timings: dict[str, float] = dict(timing or {})
    started = time.perf_counter()
    table, type_ids_by_doc, covered_words = mvtm_token_log_likelihoods(
        documents=documents,
        vectors=vectors,
        mixture_weights=mixture_weights,
        component_means=component_means,
        kappa_per_topic=kappa_per_topic,
    )
    timings["log_likelihood_sec"] = time.perf_counter() - started
    return compute_foldin_from_type_table(
        log_likelihood_by_type=table,
        type_ids_by_doc=type_ids_by_doc,
        alpha=alpha,
        split=split,
        config=foldin_config,
        encoder_fp=encoder_fp,
        corpus_sha1=(
            corpus_sha1
            if corpus_sha1 is not None
            else token_corpus_fingerprint(documents)
        ),
        condition_fp=condition_fp,
        dataset=dataset,
        data_run=data_run,
        category=category,
        encoder_model_name=word2vec_name,
        extra_metadata={
            "num_covered_word_types": len(covered_words),
            **dict(extra_metadata or {}),
        },
        timing=timings,
        model=MODEL,
        assignment_unit_type=ASSIGNMENT_UNIT_TYPE,
        chunk_docs=chunk_docs,
        layout=FoldInLayout.for_model(MODEL, category=category),
    )


def mvtm_foldin_fingerprint(
    condition_dir: Path,
    *,
    split: str,
    config: CollapsedFoldInConfig,
    chunk_docs: int = DEFAULT_CHUNK_DOCS,
    corpus: TokenCorpus | None = None,
) -> str:
    """The fingerprint the split's artifacts must carry to be current."""

    corpus = corpus or load_token_corpus(condition_dir, split=split)
    return foldin_fingerprint(
        config=config,
        encoder_fp=mvtm_encoder_fingerprint(condition_dir),
        corpus_sha1=corpus.token_sha1,
        condition_fp=condition_fingerprint_of(condition_dir),
        assignment_unit_type=ASSIGNMENT_UNIT_TYPE,
        chunk_docs=chunk_docs,
    )


def compute_mvtm_foldin_for_run(
    condition_dir: Path,
    *,
    split: str,
    dataset: str,
    data_run: str,
    category: str,
    foldin_config: CollapsedFoldInConfig | None = None,
    chunk_docs: int = DEFAULT_CHUNK_DOCS,
    corpus: TokenCorpus | None = None,
    vectors=None,
) -> FoldInRunResult:
    """Fold-in θ of one MvTM run on one split from the artifacts on disk.

    ``corpus`` (the run's own documents of the split) and ``vectors`` (the word
    vectors named in its metadata) are loaded when not given; several runs of
    one word-vector source share one ``vectors`` object.
    """

    condition_dir = Path(condition_dir)
    timing: dict[str, float] = {}
    started = time.perf_counter()
    if corpus is None:
        corpus = load_token_corpus(condition_dir, split=split)
    elif corpus.split != split or corpus.condition_dir != condition_dir:
        raise ValueError(
            f"corpus describes {corpus.condition_dir} ({corpus.split}), "
            f"not {condition_dir} ({split})"
        )
    timing["load_corpus_sec"] = time.perf_counter() - started

    started = time.perf_counter()
    if vectors is None:
        vectors = load_mvtm_word_vectors(condition_dir)
    timing["encode_sec"] = time.perf_counter() - started
    word2vec_name, _cache_dir = word_vector_source(condition_dir)
    weights, means, kappa, alpha = mvtm_frozen_parameters(condition_dir)
    result = compute_mvtm_foldin(
        documents=corpus.documents,
        vectors=vectors,
        mixture_weights=weights,
        component_means=means,
        kappa_per_topic=kappa,
        alpha=alpha,
        split=split,
        dataset=dataset,
        data_run=data_run,
        category=category,
        encoder_fp=mvtm_encoder_fingerprint(condition_dir),
        condition_fp=condition_fingerprint_of(condition_dir),
        word2vec_name=word2vec_name,
        corpus_sha1=corpus.token_sha1,
        foldin_config=foldin_config,
        chunk_docs=chunk_docs,
        extra_metadata={"written_by": "vmf-foldin-theta"},
        timing=timing,
    )
    existing_rows = _existing_doc_topic_rows(
        condition_dir, split=split, category=category
    )
    if existing_rows is not None and existing_rows != result.theta.shape[0]:
        raise ValueError(
            f"the {split} doc-topic artifact of {condition_dir} has {existing_rows} rows "
            f"but the {split} corpus has {result.theta.shape[0]} documents"
        )
    return result


__all__ = [
    "ASSIGNMENT_UNIT_TYPE",
    "MODEL",
    "TokenCorpus",
    "compute_mvtm_foldin",
    "compute_mvtm_foldin_for_run",
    "load_mvtm_word_vectors",
    "load_token_corpus",
    "mvtm_encoder_fingerprint",
    "mvtm_foldin_fingerprint",
    "mvtm_frozen_parameters",
    "mvtm_token_log_likelihoods",
    "normalized_word_rows",
    "token_type_table",
    "word_vector_source",
]

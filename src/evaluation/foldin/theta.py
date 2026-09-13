"""Collapsed fold-in document-topic distributions of one vMF Sentence LDA run.

The training run estimates the document-topic distribution of a training
document from the topic counts of its final Gibbs sweep, and that of a held-out
document from the mean of independent per-sentence posteriors; the latter omits
the document-level term ``n_dk^{-i} + alpha_k`` of the collapsed conditional.
This module re-estimates both splits with the same estimator: the collapsed
fold-in of the representative-word protocol (frozen topic parameters, Rao-
Blackwellized conditionals averaged over the retained sweeps) gives
``E[n_dk]`` as the column sums of the per-sentence posterior means, and

    theta_dk = (E[n_dk] + alpha_k) / (N_d + sum_j alpha_j)

is the posterior mean of the document-topic distribution under the fitted
model; an empty document falls back to the prior mean ``alpha / sum(alpha)``.

The result is written next to the run's other artifacts as
``doc_topic_{split}_foldin.pkl``, together with the expected counts
``E[n_dk]`` in ``doc_topic_{split}_foldin_counts.pkl`` (and, on request, the
per-sentence posterior means in ``sentence_topic_{split}_foldin.pkl``); the
sampler settings, the fingerprints of the inputs and the timing go to
``foldin_meta.json``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.artifacts import load_artifact_pickle
from src.evaluation.foldin.artifacts import (
    DOC_TOPIC_FILENAME,
    EXPECTED_COUNTS_FILENAME,
    EXPECTED_COUNTS_POINTER_ARTIFACT_KEYS,
    FOLDIN_ASSIGNMENT,
    META_FILENAME,
    META_SCHEMA,
    META_SCHEMA_VERSION,
    POINTER_ARTIFACT_KEYS,
    SENTENCE_POINTER_ARTIFACT_KEYS,
    SENTENCE_TOPIC_FILENAME,
    SPLITS,
    THETA_DEFINITION,
    FoldInRunResult,
    _atomic_write_json,
    _validate_split,
    compute_foldin_from_likelihoods,
    condition_fingerprint_of,
    doc_topic_filename,
    expected_counts_filename,
    foldin_fingerprint,
    foldin_is_current,
    foldin_theta_from_posterior,
    read_foldin_meta,
    sentence_topic_filename,
    write_foldin_artifacts,
)
from src.evaluation.topic_pairs.inputs import (
    CachedEmbeddings,
    SentenceAlignmentError,
    TrainCorpus,
    assert_same_sentences,
    encode_train_corpus,
    encoder_config_of,
    encoder_fingerprint,
    load_train_corpus,
)
from src.evaluation.word_based.topic_assignment import CollapsedFoldInConfig
from src.evaluation.word_based.topic_word_runtime import vmf_sentence_log_likelihoods


def _existing_doc_topic_rows(condition_dir: Path, split: str) -> int | None:
    """Rows of the run's own ``doc_topic_{split}.pkl``, when it exists."""

    path = Path(condition_dir) / f"doc_topic_{split}.pkl"
    if not path.exists():
        return None
    values = np.asarray(load_artifact_pickle(path))
    return int(values.shape[0]) if values.ndim == 2 else None


def compute_foldin_for_run(
    condition_dir: Path,
    *,
    split: str,
    dataset: str,
    data_run: str,
    category: str,
    cache_root: Path,
    encoder_device: str,
    encode_batch_size: int | None,
    foldin_config: CollapsedFoldInConfig | None = None,
    corpus: TrainCorpus | None = None,
    reference_corpus: TrainCorpus | None = None,
    embeddings: CachedEmbeddings | None = None,
) -> FoldInRunResult:
    """Fold-in θ of one run on one split.

    ``corpus`` (the run's own sentences of the split) is loaded when not given.
    ``embeddings`` are the raw encoder outputs of ``reference_corpus``; when
    given, the run's corpus must describe exactly those sentences, so several
    runs of one unit share one encoding.
    """

    condition_dir = Path(condition_dir)
    split = _validate_split(split)
    config = foldin_config or CollapsedFoldInConfig()
    config.validate()
    timing: dict[str, float] = {}

    started = time.perf_counter()
    if corpus is None:
        corpus = load_train_corpus(condition_dir, model="vmf", split=split)
    timing["load_corpus_sec"] = time.perf_counter() - started

    encoder_config = encoder_config_of(condition_dir)
    encoder_fp = encoder_fingerprint(encoder_config)
    started = time.perf_counter()
    if embeddings is None:
        embeddings = encode_train_corpus(
            corpus,
            encoder_config=encoder_config,
            cache_root=Path(cache_root),
            dataset=dataset,
            data_run=data_run,
            category=category,
            split=split,
            device=encoder_device,
            encode_batch_size=encode_batch_size,
        )
    else:
        if reference_corpus is not None:
            assert_same_sentences(reference_corpus, corpus)
        manifest_fp = embeddings.manifest.get("encoder_fingerprint")
        if manifest_fp not in {None, ""} and str(manifest_fp) != encoder_fp:
            raise SentenceAlignmentError(
                f"embeddings were produced by encoder {manifest_fp}, "
                f"but {condition_dir} was trained with {encoder_fp}"
            )
        manifest_sha1 = embeddings.manifest.get("sentence_sha1")
        if (
            manifest_sha1 not in {None, ""}
            and str(manifest_sha1) != corpus.sentence_sha1
        ):
            raise SentenceAlignmentError(
                f"embeddings describe sentences {str(manifest_sha1)[:12]} but "
                f"{condition_dir} ({split}) holds {corpus.sentence_sha1[:12]}"
            )
        if int(embeddings.embeddings.shape[0]) != corpus.num_sentences:
            raise SentenceAlignmentError(
                f"{embeddings.embeddings.shape[0]} embeddings for "
                f"{corpus.num_sentences} sentences at {condition_dir}"
            )
    timing["encode_sec"] = time.perf_counter() - started

    started = time.perf_counter()
    likelihoods, alpha = vmf_sentence_log_likelihoods(
        condition_dir=condition_dir, encoded_documents=embeddings.iter_documents()
    )
    timing["log_likelihood_sec"] = time.perf_counter() - started

    condition_fp = condition_fingerprint_of(condition_dir)
    result = compute_foldin_from_likelihoods(
        likelihoods,
        alpha,
        split=split,
        config=config,
        encoder_fp=encoder_fp,
        corpus_sha1=corpus.sentence_sha1,
        condition_fp=condition_fp,
        dataset=dataset,
        data_run=data_run,
        category=category,
        encoder_model_name=encoder_config.get("model_name"),
        extra_metadata={
            "embedding_cache_dir": str(embeddings.cache_dir),
            "embedding_cache_hit": bool(embeddings.cache_hit),
        },
        timing=timing,
    )
    if result.theta.shape[0] != corpus.num_documents:
        raise SentenceAlignmentError(
            f"fold-in produced {result.theta.shape[0]} documents for "
            f"{corpus.num_documents} at {condition_dir}"
        )
    existing_rows = _existing_doc_topic_rows(condition_dir, split)
    if existing_rows is not None and existing_rows != result.theta.shape[0]:
        raise SentenceAlignmentError(
            f"doc_topic_{split}.pkl of {condition_dir} has {existing_rows} rows "
            f"but the {split} corpus has {result.theta.shape[0]} documents"
        )
    return result


__all__ = [
    "DOC_TOPIC_FILENAME",
    "EXPECTED_COUNTS_FILENAME",
    "EXPECTED_COUNTS_POINTER_ARTIFACT_KEYS",
    "FOLDIN_ASSIGNMENT",
    "META_FILENAME",
    "META_SCHEMA",
    "META_SCHEMA_VERSION",
    "POINTER_ARTIFACT_KEYS",
    "SENTENCE_POINTER_ARTIFACT_KEYS",
    "SENTENCE_TOPIC_FILENAME",
    "SPLITS",
    "THETA_DEFINITION",
    "FoldInRunResult",
    "compute_foldin_for_run",
    "condition_fingerprint_of",
    "doc_topic_filename",
    "expected_counts_filename",
    "foldin_fingerprint",
    "foldin_is_current",
    "foldin_theta_from_posterior",
    "read_foldin_meta",
    "sentence_topic_filename",
    "write_foldin_artifacts",
]

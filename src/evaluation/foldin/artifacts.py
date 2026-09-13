"""Fold-in artifacts of a vMF-family run: file names, theta, fingerprints, I/O.

Two models share the collapsed fold-in estimator: vMF Sentence LDA (sentence
units, artifacts flat in the run directory) and MvTM / vLDA (word-token units,
artifacts under the baseline layout ``params/<category>_...`` for the training
split and ``infer/<category>_...`` for the held-out split). :class:`FoldInLayout`
names the files of either model; the pointer artifact keys are shared.

Kept free of the evaluation runtime (encoders, gensim) so that the training
runners (:mod:`src.models.registry`, :mod:`src.baselines.models.mvtm`) can write
the same artifacts that ``evaluation vmf-foldin-theta`` (:mod:`.theta`,
:mod:`.mvtm`) writes after the fact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from src.core.artifacts import load_artifact_json, save_pickle
from src.evaluation.word_based.topic_assignment import (
    CollapsedFoldInConfig,
    fingerprint_jsonable,
    run_collapsed_fold_in,
)

AssignmentUnitType = Literal["sentence", "token"]
ASSIGNMENT_UNIT_TYPES: tuple[str, ...] = ("sentence", "token")
FOLDIN_MODELS: tuple[str, ...] = ("vmf_sentence_lda", "mvtm")
# Documents per call of the collapsed sampler when the likelihoods are given as a
# type table (MvTM): the sampler materializes every ``(N_d, K)`` block of a call,
# which for a whole-corpus run at large K would be several gigabytes at once.
DEFAULT_CHUNK_DOCS = 256

FOLDIN_ASSIGNMENT = "foldin"
THETA_DEFINITION = "collapsed_fold_in_posterior_mean_dirichlet_smoothed"
DOC_TOPIC_FILENAME = "doc_topic_{split}_foldin.pkl"
# The expected topic counts E[n_dk] the theta was smoothed from; kept so that an
# unsmoothed estimate E[n_dk] / N_d can be formed without re-running the sampler.
EXPECTED_COUNTS_FILENAME = "doc_topic_{split}_foldin_counts.pkl"
SENTENCE_TOPIC_FILENAME = "sentence_topic_{split}_foldin.pkl"
META_FILENAME = "foldin_meta.json"
META_SCHEMA = "vmf_foldin_theta"
META_SCHEMA_VERSION = 1
SPLITS: tuple[str, ...] = ("train", "test")
POINTER_ARTIFACT_KEYS: dict[str, str] = {
    "train": "train_doc_topic_foldin",
    "test": "test_doc_topic_foldin",
}
EXPECTED_COUNTS_POINTER_ARTIFACT_KEYS: dict[str, str] = {
    "train": "train_doc_topic_foldin_counts",
    "test": "test_doc_topic_foldin_counts",
}
SENTENCE_POINTER_ARTIFACT_KEYS: dict[str, str] = {
    "train": "train_sentence_topic_foldin",
    "test": "test_sentence_topic_foldin",
}


def _validate_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    return split


def _validate_unit_type(value: str) -> str:
    if value not in ASSIGNMENT_UNIT_TYPES:
        raise ValueError(
            f"assignment_unit_type must be one of {ASSIGNMENT_UNIT_TYPES}, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class FoldInLayout:
    """Where one model keeps its fold-in files, relative to the run directory.

    ``vmf_sentence_lda`` writes ``doc_topic_<split>_foldin*.pkl`` flat into the
    run directory. ``mvtm`` follows the baseline layout: the training split goes
    to ``params/<category>_doc_topic_foldin*.pkl`` and the held-out split to
    ``infer/<category>_doc_topic_foldin*.pkl`` (next to ``<category>.pkl``, the
    argmax counts; the entropy task already looks for these names).
    ``foldin_meta.json`` always sits in the run directory.
    """

    model: str = "vmf_sentence_lda"
    train_dir: str = ""
    test_dir: str = ""
    prefix: str = ""
    # vSLDA names the split in the file (both splits share one directory); the
    # baseline layout names it by the directory, like ``infer/<category>.pkl``.
    split_in_name: bool = True

    @classmethod
    def for_model(cls, model: str, *, category: str | None = None) -> "FoldInLayout":
        if model == "vmf_sentence_lda":
            return cls()
        if model == "mvtm":
            if not category:
                raise ValueError("the mvtm fold-in layout needs the run's category")
            return cls(
                model="mvtm",
                train_dir="params",
                test_dir="infer",
                prefix=f"{category}_",
                split_in_name=False,
            )
        raise ValueError(
            f"no fold-in layout for model {model!r}; use one of {FOLDIN_MODELS}"
        )

    def split_dir(self, split: str) -> str:
        return self.train_dir if _validate_split(split) == "train" else self.test_dir

    def _relpath(self, split: str, template: str) -> str:
        split = _validate_split(split)
        name = self.prefix + (
            template.format(split=split)
            if self.split_in_name
            else template.replace("{split}_", "")
        )
        directory = self.split_dir(split)
        return f"{directory}/{name}" if directory else name

    def doc_topic_relpath(self, split: str) -> str:
        return self._relpath(split, DOC_TOPIC_FILENAME)

    def expected_counts_relpath(self, split: str) -> str:
        return self._relpath(split, EXPECTED_COUNTS_FILENAME)

    def sentence_topic_relpath(self, split: str) -> str:
        return self._relpath(split, SENTENCE_TOPIC_FILENAME)


DEFAULT_LAYOUT = FoldInLayout()


def doc_topic_filename(split: str, layout: FoldInLayout = DEFAULT_LAYOUT) -> str:
    """Run-relative path of the split's fold-in θ (a bare name for vSLDA)."""

    return layout.doc_topic_relpath(split)


def expected_counts_filename(split: str, layout: FoldInLayout = DEFAULT_LAYOUT) -> str:
    return layout.expected_counts_relpath(split)


def sentence_topic_filename(split: str, layout: FoldInLayout = DEFAULT_LAYOUT) -> str:
    return layout.sentence_topic_relpath(split)


def token_corpus_fingerprint(documents: Iterable[Any]) -> str:
    """sha1 of the word tokens of a preprocessed corpus (document order matters).

    The token counterpart of ``sentence_fingerprint``: MvTM observes
    ``document_tokens``, so its fold-in inputs are identified by them.
    """

    digest = hashlib.sha1()
    for document in documents:
        tokens = getattr(document, "document_tokens", document)
        digest.update("\x1f".join(str(token) for token in tokens).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def word_vector_fingerprint(
    word2vec: str, *, wikientvec_cache_dir: str | None = None
) -> str:
    """The MvTM counterpart of the sentence-encoder fingerprint: the word-vector source."""

    return fingerprint_jsonable(
        {
            "word_vectors": str(word2vec),
            "wikientvec_cache_dir": (
                None
                if wikientvec_cache_dir in {None, ""}
                else str(wikientvec_cache_dir)
            ),
        }
    )


def foldin_theta_from_posterior(
    posterior_mean_by_doc: Sequence[np.ndarray], alpha: np.ndarray
) -> np.ndarray:
    """``(E[n_dk] + alpha_k) / (N_d + sum alpha)`` from per-sentence posterior means.

    ``posterior_mean_by_doc[d]`` is the ``(S_d, K)`` block of the collapsed
    fold-in, whose rows sum to one, so its column sums are the expected topic
    counts of document ``d`` under the retained sweeps. An empty block gives
    the prior mean.
    """

    alpha_values = np.asarray(alpha, dtype=np.float64)
    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("alpha must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(alpha_values)) or np.any(alpha_values <= 0.0):
        raise ValueError("alpha must contain only finite positive values")
    num_topics = int(alpha_values.size)
    alpha_total = float(alpha_values.sum())
    theta = np.empty((len(posterior_mean_by_doc), num_topics), dtype=np.float64)
    for doc_index, block in enumerate(posterior_mean_by_doc):
        values = np.asarray(block, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != num_topics:
            raise ValueError(
                f"posterior for document {doc_index} must have width {num_topics}"
            )
        expected_counts = (
            values.sum(axis=0) if values.shape[0] else np.zeros(num_topics)
        )
        theta[doc_index] = (expected_counts + alpha_values) / (
            float(values.shape[0]) + alpha_total
        )
    return theta


def condition_fingerprint_of(condition_dir: Path) -> str:
    """The training run's condition fingerprint from its ``metadata.json``."""

    metadata = load_artifact_json(Path(condition_dir) / "metadata.json")
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid condition metadata: {condition_dir}")
    value = metadata.get("condition_fingerprint")
    if value in {None, ""} and isinstance(metadata.get("axes"), dict):
        value = metadata.get("condition_id")
    if value in {None, ""}:
        raise ValueError(f"Condition fingerprint is missing: {condition_dir}")
    return str(value)


def foldin_fingerprint(
    *,
    config: CollapsedFoldInConfig,
    encoder_fp: str,
    corpus_sha1: str,
    condition_fp: str,
    assignment_unit_type: str = "sentence",
    chunk_docs: int | None = None,
) -> str:
    """Identity of one fold-in result: sampler settings and every input.

    The sentence-unit fingerprint of vMF Sentence LDA is unchanged by the two
    optional fields (they are only added for token units / chunked runs), so
    artifacts written before they existed stay current.
    """

    payload: dict[str, Any] = {
        "schema": META_SCHEMA,
        "schema_version": META_SCHEMA_VERSION,
        "theta_definition": THETA_DEFINITION,
        "config": asdict(config),
        "encoder_fingerprint": str(encoder_fp),
        "corpus_sha1": str(corpus_sha1),
        "condition_fingerprint": str(condition_fp),
    }
    unit_type = _validate_unit_type(assignment_unit_type)
    if unit_type != "sentence":
        payload["assignment_unit_type"] = unit_type
    if chunk_docs is not None:
        payload["chunk_docs"] = int(chunk_docs)
    return fingerprint_jsonable(payload)


def run_collapsed_fold_in_chunked(
    *,
    alpha: np.ndarray,
    assignment_unit_type: str,
    config: CollapsedFoldInConfig,
    log_likelihood_by_type: np.ndarray,
    type_ids_by_doc: Sequence[np.ndarray],
    chunk_docs: int = DEFAULT_CHUNK_DOCS,
    source_condition_fingerprint: str | None = None,
    corpus_fingerprint: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collapsed fold-in over a type table, ``chunk_docs`` documents per sampler call.

    Returns ``(expected_counts (D, K), num_units (D,), metadata)``. Only the
    column sums of each document's posterior means are kept, so the memory of a
    call is bounded by the chunk's units. Chunk ``c`` is sampled with
    ``random_seed + c``, which makes the result independent of the machine but
    dependent on ``chunk_docs`` (recorded in the fingerprint).
    """

    if int(chunk_docs) <= 0:
        raise ValueError("chunk_docs must be positive")
    unit_type = _validate_unit_type(assignment_unit_type)
    alpha_values = np.asarray(alpha, dtype=np.float64)
    num_topics = int(alpha_values.size)
    num_documents = len(type_ids_by_doc)
    expected = np.zeros((num_documents, num_topics), dtype=np.float64)
    num_units = np.zeros(num_documents, dtype=np.int64)
    metadata: dict[str, Any] = {}
    for chunk_index, start in enumerate(range(0, num_documents, int(chunk_docs))):
        stop = min(start + int(chunk_docs), num_documents)
        chunk_ids = [
            np.asarray(item, dtype=np.int64) for item in type_ids_by_doc[start:stop]
        ]
        chunk_config = replace(
            config, random_seed=int(config.random_seed) + chunk_index
        )
        posterior = run_collapsed_fold_in(
            alpha=alpha_values,
            assignment_unit_type=unit_type,  # type: ignore[arg-type]
            config=chunk_config,
            log_likelihood_by_type=log_likelihood_by_type,
            type_ids_by_doc=chunk_ids,
            source_condition_fingerprint=source_condition_fingerprint,
            corpus_fingerprint=corpus_fingerprint,
        )
        if not metadata:
            metadata = dict(posterior.metadata)
            metadata["random_seed"] = int(config.random_seed)
            metadata["chunk_docs"] = int(chunk_docs)
            metadata["chunk_seed_rule"] = "random_seed + chunk_index"
        for offset, block in enumerate(posterior.posterior_mean_by_doc):
            values = np.asarray(block, dtype=np.float64)
            if values.shape[0]:
                expected[start + offset] = values.sum(axis=0)
            num_units[start + offset] = int(values.shape[0])
    if not metadata:
        metadata = {
            "assignment_unit_type": unit_type,
            "global_topic_parameters": "frozen",
            **asdict(config),
            "chunk_docs": int(chunk_docs),
            "chunk_seed_rule": "random_seed + chunk_index",
            "source_condition_fingerprint": source_condition_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
        }
    return expected, num_units, metadata


@dataclass(frozen=True)
class FoldInRunResult:
    split: str
    theta: np.ndarray  # (D, K)
    sentence_posteriors: list[np.ndarray]  # (S_d, K) blocks; empty for chunked runs
    alpha: np.ndarray
    fingerprint: str
    metadata: dict[str, Any]
    timing: dict[str, float]
    # Set by the chunked (token-unit) path, which keeps no per-unit posteriors.
    expected_counts_override: np.ndarray | None = None
    num_units_override: np.ndarray | None = None
    layout: FoldInLayout = DEFAULT_LAYOUT

    @property
    def expected_counts(self) -> np.ndarray:
        """``E[n_dk]``: column sums of the per-unit posterior means, ``(D, K)``."""

        if self.expected_counts_override is not None:
            return np.asarray(self.expected_counts_override, dtype=np.float64)
        counts = np.zeros(self.theta.shape, dtype=np.float64)
        for doc_index, block in enumerate(self.sentence_posteriors):
            if block.shape[0]:
                counts[doc_index] = np.asarray(block, dtype=np.float64).sum(axis=0)
        return counts

    @property
    def num_units(self) -> np.ndarray:
        """Assignment units (sentences or tokens) per document, ``(D,)``."""

        if self.num_units_override is not None:
            return np.asarray(self.num_units_override, dtype=np.int64)
        return np.asarray(
            [int(block.shape[0]) for block in self.sentence_posteriors], dtype=np.int64
        )

    @property
    def num_documents(self) -> int:
        return int(self.theta.shape[0])

    @property
    def num_topics(self) -> int:
        return int(self.theta.shape[1])

    @property
    def total_sentences(self) -> int:
        return int(self.num_units.sum())

    @property
    def num_empty_documents(self) -> int:
        return int(np.count_nonzero(self.num_units == 0))


def compute_foldin_from_likelihoods(
    likelihoods: Sequence[np.ndarray],
    alpha: np.ndarray,
    *,
    split: str,
    config: CollapsedFoldInConfig | None = None,
    encoder_fp: str,
    corpus_sha1: str,
    condition_fp: str,
    dataset: str,
    data_run: str,
    category: str,
    encoder_model_name: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    timing: Mapping[str, float] | None = None,
    model: str = "vmf_sentence_lda",
    layout: FoldInLayout | None = None,
) -> FoldInRunResult:
    """Run the collapsed fold-in on per-document ``(S_d, K)`` sentence log likelihoods
    and build the result (theta, expected counts, fingerprint, metadata) the
    artifacts are written from.

    Shared by the post-hoc command (:mod:`.theta`, likelihoods from the frozen
    parameters on disk) and the training runner (likelihoods from the trainer).
    Token-unit models use :func:`compute_foldin_from_type_table`.
    """

    split = _validate_split(split)
    config = config or CollapsedFoldInConfig()
    config.validate()
    timings: dict[str, float] = dict(timing or {})
    started = time.perf_counter()
    posterior = run_collapsed_fold_in(
        alpha=alpha,
        assignment_unit_type="sentence",
        config=config,
        log_likelihood_by_doc=list(likelihoods),
        source_condition_fingerprint=condition_fp,
        corpus_fingerprint=corpus_sha1,
    )
    timings["collapsed_foldin_sec"] = time.perf_counter() - started
    blocks = [
        np.asarray(block, dtype=np.float64) for block in posterior.posterior_mean_by_doc
    ]
    theta = foldin_theta_from_posterior(blocks, alpha)
    fingerprint = foldin_fingerprint(
        config=config,
        encoder_fp=encoder_fp,
        corpus_sha1=corpus_sha1,
        condition_fp=condition_fp,
    )
    return _build_result(
        split=split,
        theta=theta,
        alpha=alpha,
        blocks=blocks,
        expected_counts=None,
        num_units=None,
        fingerprint=fingerprint,
        config=config,
        posterior_metadata=dict(posterior.metadata),
        assignment_unit_type="sentence",
        model=model,
        layout=layout or FoldInLayout.for_model(model, category=category),
        encoder_fp=encoder_fp,
        encoder_model_name=encoder_model_name,
        corpus_sha1=corpus_sha1,
        condition_fp=condition_fp,
        dataset=dataset,
        data_run=data_run,
        category=category,
        extra_metadata=extra_metadata,
        timings=timings,
    )


def compute_foldin_from_type_table(
    *,
    log_likelihood_by_type: np.ndarray,
    type_ids_by_doc: Sequence[np.ndarray],
    alpha: np.ndarray,
    split: str,
    config: CollapsedFoldInConfig | None = None,
    encoder_fp: str,
    corpus_sha1: str,
    condition_fp: str,
    dataset: str,
    data_run: str,
    category: str,
    encoder_model_name: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    timing: Mapping[str, float] | None = None,
    model: str = "mvtm",
    assignment_unit_type: str = "token",
    chunk_docs: int = DEFAULT_CHUNK_DOCS,
    layout: FoldInLayout | None = None,
) -> FoldInRunResult:
    """The token-unit counterpart of :func:`compute_foldin_from_likelihoods`.

    ``log_likelihood_by_type`` is the ``(V, K)`` table of the model's supported
    word types and ``type_ids_by_doc[d]`` the (in-vocabulary) tokens of document
    ``d`` as row indices; the sampler runs ``chunk_docs`` documents at a time
    (:func:`run_collapsed_fold_in_chunked`) and only the expected counts are kept.
    """

    split = _validate_split(split)
    config = config or CollapsedFoldInConfig()
    config.validate()
    unit_type = _validate_unit_type(assignment_unit_type)
    timings: dict[str, float] = dict(timing or {})
    started = time.perf_counter()
    expected, num_units, posterior_metadata = run_collapsed_fold_in_chunked(
        alpha=alpha,
        assignment_unit_type=unit_type,
        config=config,
        log_likelihood_by_type=log_likelihood_by_type,
        type_ids_by_doc=type_ids_by_doc,
        chunk_docs=chunk_docs,
        source_condition_fingerprint=condition_fp,
        corpus_fingerprint=corpus_sha1,
    )
    timings["collapsed_foldin_sec"] = time.perf_counter() - started
    alpha_values = np.asarray(alpha, dtype=np.float64)
    theta = (expected + alpha_values[None, :]) / (
        num_units[:, None].astype(np.float64) + float(alpha_values.sum())
    )
    fingerprint = foldin_fingerprint(
        config=config,
        encoder_fp=encoder_fp,
        corpus_sha1=corpus_sha1,
        condition_fp=condition_fp,
        assignment_unit_type=unit_type,
        chunk_docs=chunk_docs,
    )
    return _build_result(
        split=split,
        theta=theta,
        alpha=alpha,
        blocks=[],
        expected_counts=expected,
        num_units=num_units,
        fingerprint=fingerprint,
        config=config,
        posterior_metadata=posterior_metadata,
        assignment_unit_type=unit_type,
        model=model,
        layout=layout or FoldInLayout.for_model(model, category=category),
        encoder_fp=encoder_fp,
        encoder_model_name=encoder_model_name,
        corpus_sha1=corpus_sha1,
        condition_fp=condition_fp,
        dataset=dataset,
        data_run=data_run,
        category=category,
        extra_metadata=extra_metadata,
        timings=timings,
    )


def _build_result(
    *,
    split: str,
    theta: np.ndarray,
    alpha: np.ndarray,
    blocks: list[np.ndarray],
    expected_counts: np.ndarray | None,
    num_units: np.ndarray | None,
    fingerprint: str,
    config: CollapsedFoldInConfig,
    posterior_metadata: Mapping[str, Any],
    assignment_unit_type: str,
    model: str,
    layout: FoldInLayout,
    encoder_fp: str,
    encoder_model_name: str | None,
    corpus_sha1: str,
    condition_fp: str,
    dataset: str,
    data_run: str,
    category: str,
    extra_metadata: Mapping[str, Any] | None,
    timings: dict[str, float],
) -> FoldInRunResult:
    if model not in FOLDIN_MODELS:
        raise ValueError(
            f"unsupported fold-in model {model!r}; use one of {FOLDIN_MODELS}"
        )
    if not np.all(np.isfinite(theta)) or not np.allclose(
        theta.sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError(f"fold-in theta rows do not sum to one ({dataset}/{category})")
    result = FoldInRunResult(
        split=split,
        theta=theta,
        sentence_posteriors=blocks,
        alpha=np.asarray(alpha, dtype=np.float64),
        fingerprint=fingerprint,
        metadata={},
        timing=timings,
        expected_counts_override=expected_counts,
        num_units_override=num_units,
        layout=layout,
    )
    metadata: dict[str, Any] = {
        "schema": META_SCHEMA,
        "schema_version": META_SCHEMA_VERSION,
        "theta_definition": THETA_DEFINITION,
        "assignment": FOLDIN_ASSIGNMENT,
        "model": str(model),
        "assignment_unit_type": str(assignment_unit_type),
        "split": split,
        "dataset": str(dataset),
        "data_run": str(data_run),
        "category": str(category),
        "config": asdict(config),
        "posterior_metadata": dict(posterior_metadata),
        "encoder_fingerprint": str(encoder_fp),
        "encoder_model_name": encoder_model_name,
        "corpus_sha1": str(corpus_sha1),
        "condition_fingerprint": str(condition_fp),
        "num_documents": int(theta.shape[0]),
        "num_topics": int(theta.shape[1]),
        "total_sentences": result.total_sentences,
        "total_units": result.total_sentences,
        "num_empty_documents": result.num_empty_documents,
        "alpha": [float(value) for value in np.asarray(alpha, dtype=np.float64)],
        **dict(extra_metadata or {}),
    }
    result.metadata.update(metadata)
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_foldin_meta(condition_dir: Path) -> dict[str, Any]:
    """The run's ``foldin_meta.json`` (``{}`` when absent or unreadable)."""

    path = Path(condition_dir) / META_FILENAME
    if not path.exists():
        return {}
    try:
        payload = load_artifact_json(path)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def foldin_is_current(
    condition_dir: Path,
    *,
    split: str,
    fingerprint: str,
    layout: FoldInLayout = DEFAULT_LAYOUT,
) -> bool:
    """True when the split's artifacts exist and were written for ``fingerprint``."""

    condition_dir = Path(condition_dir)
    split = _validate_split(split)
    if not (condition_dir / doc_topic_filename(split, layout)).exists():
        return False
    if not (condition_dir / expected_counts_filename(split, layout)).exists():
        return False
    entry = read_foldin_meta(condition_dir).get("splits", {}).get(split)
    return isinstance(entry, dict) and str(entry.get("fingerprint")) == str(fingerprint)


def write_foldin_artifacts(
    condition_dir: Path,
    *,
    result: FoldInRunResult,
    write_sentence_posteriors: bool = False,
) -> dict[str, str]:
    """Write the split's θ (and optionally the sentence posteriors) and update the meta.

    Returns the pointer artifact keys that now name a file, as paths relative to
    ``condition_dir`` (bare names for vSLDA, ``params/...`` / ``infer/...`` for MvTM).
    """

    condition_dir = Path(condition_dir)
    split = _validate_split(result.split)
    layout = result.layout
    artifacts: dict[str, str] = {}
    theta_name = doc_topic_filename(split, layout)
    save_pickle(
        np.ascontiguousarray(result.theta, dtype=np.float64), condition_dir / theta_name
    )
    artifacts[POINTER_ARTIFACT_KEYS[split]] = theta_name
    counts_name = expected_counts_filename(split, layout)
    save_pickle(
        np.ascontiguousarray(result.expected_counts, dtype=np.float64),
        condition_dir / counts_name,
    )
    artifacts[EXPECTED_COUNTS_POINTER_ARTIFACT_KEYS[split]] = counts_name
    if write_sentence_posteriors:
        if not result.sentence_posteriors and result.total_sentences:
            raise ValueError(
                "per-unit posteriors were not kept (chunked token-unit fold-in)"
            )
        sentence_name = sentence_topic_filename(split, layout)
        save_pickle(
            [
                np.ascontiguousarray(block, dtype=np.float32)
                for block in result.sentence_posteriors
            ],
            condition_dir / sentence_name,
        )
        artifacts[SENTENCE_POINTER_ARTIFACT_KEYS[split]] = sentence_name

    meta = read_foldin_meta(condition_dir)
    meta.setdefault("schema", META_SCHEMA)
    meta.setdefault("schema_version", META_SCHEMA_VERSION)
    meta["theta_definition"] = THETA_DEFINITION
    meta["model"] = layout.model
    splits = meta.setdefault("splits", {})
    if not isinstance(splits, dict):
        splits = {}
        meta["splits"] = splits
    splits[split] = {
        **result.metadata,
        "fingerprint": result.fingerprint,
        "artifacts": dict(artifacts),
        "timing": {key: float(value) for key, value in result.timing.items()},
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(condition_dir / META_FILENAME, meta)
    return artifacts


__all__ = [
    "ASSIGNMENT_UNIT_TYPES",
    "DEFAULT_CHUNK_DOCS",
    "DEFAULT_LAYOUT",
    "FOLDIN_MODELS",
    "FoldInLayout",
    "compute_foldin_from_type_table",
    "run_collapsed_fold_in_chunked",
    "token_corpus_fingerprint",
    "word_vector_fingerprint",
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
    "compute_foldin_from_likelihoods",
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

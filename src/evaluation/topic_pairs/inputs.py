"""Inputs of the topic-pair analysis: corpora, embeddings, labels, model state.

One training corpus per (dataset, data run, category) is the reference: the
three sentence-level models were trained on the same selected documents, and
every model's own corpus is checked against the reference (document count,
sentences per document, hash of the raw sentences) before its posteriors are
used. The raw encoder output of the reference sentences is cached on disk so
the encoder runs once per category, not once per run.

Fine labels come from the dataset split file through the reference
condition's ``raw_doc_indices`` (row positions of ``<split>.csv``, the same
convention as the classification pipeline). SentLDA runs trained before the
raw-index fix store positions within the category frame instead, so labels
are never read through a SentLDA selection file when another model is present.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.core.artifacts import load_artifact_json, load_artifact_pickle, load_json
from src.data.catalog import get_dataset_targets
from src.data.preprocessing import PreprocessedDocument
from src.data.splits import load_dataset_split
from src.evaluation.entropy_based.inputs import (
    effective_embedding_variant,
    parameter_variant_for,
    provenance_for,
)
from src.evaluation.entropy_based.inputs import (
    resolve_condition_dir as _resolve_entropy_condition_dir,
)
from src.evaluation.word_based.sentence_encoding import (
    encode_sentence_corpus,
    flatten_encoder_sentences,
)
from src.evaluation.word_based.topic_word_runtime import (
    _condition_metadata,
    _encoder_from_metadata,
    _load_documents_and_ids,
)

TASK_MODELS: tuple[str, ...] = ("vmf", "sentlda", "sentence_gaussianlda")
MODEL_ALIASES: dict[str, str] = {
    "vmf_sentence_lda": "vmf",
    "gaussian": "sentence_gaussianlda",
}
SENTENCE_ENCODER_MODELS = frozenset({"vmf", "sentence_gaussianlda"})
# Preference order of the model whose corpus and selection file serve as the
# category reference (labels are read through its raw document indices).
REFERENCE_MODEL_ORDER: tuple[str, ...] = ("vmf", "sentence_gaussianlda", "sentlda")
EMBEDDING_CACHE_SCHEMA_VERSION = 1
EMBEDDING_SPACE = "l2_normalized_raw_encoder_output"
ENCODER_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "model_name",
    "backend",
    "pooling",
    "encode_prefix",
    "encode_prompt",
    "encode_prompt_name",
    "truncate_dim",
    "strip_terminal_normalize",
    "normalize_embeddings",
    "model_kwargs",
    "tokenizer_kwargs",
)


class SentenceAlignmentError(ValueError):
    """A model's training corpus differs from the category reference."""


def normalize_model_name(model: str) -> str:
    key = str(model).strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key not in TASK_MODELS:
        raise ValueError(
            f"Unsupported model for topic-pair metrics: '{model}'. "
            f"Use one of {list(TASK_MODELS)}."
        )
    return key


def normalize_model_names(models: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for model in models:
        key = normalize_model_name(model)
        if key not in resolved:
            resolved.append(key)
    return resolved


def reference_model(models: Sequence[str]) -> str:
    """The model whose run defines the category corpus and labels."""

    resolved = normalize_model_names(models)
    for candidate in REFERENCE_MODEL_ORDER:
        if candidate in resolved:
            return candidate
    raise ValueError("no supported model selected")


def model_embedding_variant(model: str, embedding_variant: str | None) -> str | None:
    """The result-path suffix a model's runs are stored under."""

    return effective_embedding_variant(
        normalize_model_name(model), embedding_variant, word_embedding_variant=None
    )


def resolve_condition_dir(
    *,
    model: str,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    embedding_variant: str | None,
    prior_scale: float | None = None,
) -> Path:
    key = normalize_model_name(model)
    return _resolve_entropy_condition_dir(
        model=key,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        embedding_variant=model_embedding_variant(key, embedding_variant),
        prior_scale=prior_scale,
    )


def sentence_fingerprint(sentences: Sequence[str]) -> str:
    digest = hashlib.sha1()
    for sentence in sentences:
        digest.update(str(sentence).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainCorpus:
    model: str
    condition_dir: Path
    documents: list[PreprocessedDocument]
    raw_doc_indices: list[int]
    sentences: list[str]
    doc_offsets: np.ndarray
    sentence_sha1: str

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    def sentences_per_document(self) -> np.ndarray:
        return np.diff(self.doc_offsets)


def load_train_corpus(
    condition_dir: Path, *, model: str, split: str = "train"
) -> TrainCorpus:
    key = normalize_model_name(model)
    documents, raw_ids = _load_documents_and_ids(
        model=key, condition_dir=Path(condition_dir), split=split
    )
    sentences, offsets = flatten_encoder_sentences(documents, use_tokenized=False)
    return TrainCorpus(
        model=key,
        condition_dir=Path(condition_dir),
        documents=list(documents),
        raw_doc_indices=[int(value) for value in raw_ids],
        sentences=sentences,
        doc_offsets=np.asarray(offsets, dtype=np.int64),
        sentence_sha1=sentence_fingerprint(sentences),
    )


def assert_same_sentences(reference: TrainCorpus, other: TrainCorpus) -> None:
    """Require ``other`` to describe exactly the reference sentences."""

    if other.num_documents != reference.num_documents:
        raise SentenceAlignmentError(
            f"{other.model} ({other.condition_dir}) has {other.num_documents} "
            f"documents but the {reference.model} reference has "
            f"{reference.num_documents}"
        )
    reference_counts = reference.sentences_per_document()
    other_counts = other.sentences_per_document()
    mismatch = np.flatnonzero(reference_counts != other_counts)
    if mismatch.size:
        index = int(mismatch[0])
        raise SentenceAlignmentError(
            f"{other.model} ({other.condition_dir}) has {int(other_counts[index])} "
            f"sentences in document {index} but the {reference.model} reference "
            f"has {int(reference_counts[index])}"
        )
    if other.sentence_sha1 != reference.sentence_sha1:
        raise SentenceAlignmentError(
            f"{other.model} ({other.condition_dir}) sentences differ from the "
            f"{reference.model} reference (sha1 {other.sentence_sha1[:12]} vs "
            f"{reference.sentence_sha1[:12]})"
        )


# ---------------------------------------------------------------------------
# Sentence embeddings and their cache
# ---------------------------------------------------------------------------


def encoder_config_of(condition_dir: Path) -> dict[str, Any]:
    metadata = _condition_metadata(Path(condition_dir))
    config = metadata.get("encoder_config")
    if not isinstance(config, dict) or not config.get("model_name"):
        raise ValueError(f"encoder_config.model_name is missing: {condition_dir}")
    return dict(config)


def encoder_fingerprint(encoder_config: Mapping[str, Any]) -> str:
    subset = {
        key: encoder_config.get(key)
        for key in ENCODER_FINGERPRINT_FIELDS
        if encoder_config.get(key) not in (None, {}, "")
    }
    encoded = json.dumps(subset, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def embedding_cache_dir(
    cache_root: Path,
    *,
    dataset: str,
    data_run: str,
    category: str,
    split: str,
    encoder_fp: str,
    sentence_sha1: str,
) -> Path:
    payload = {
        "dataset": str(dataset),
        "data_run": str(data_run),
        "category": str(category),
        "split": str(split),
        "encoder_fingerprint": str(encoder_fp),
        "sentence_sha1": str(sentence_sha1),
        "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
    }
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[
        :20
    ]
    return Path(cache_root) / f"v{EMBEDDING_CACHE_SCHEMA_VERSION}" / key


def _atomic_write(path: Path, writer: Callable[[Any], None], *, binary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            fd, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})
        ) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_cached_embeddings(
    cache_dir: Path,
    *,
    embeddings: np.ndarray,
    doc_offsets: np.ndarray,
    manifest: Mapping[str, Any],
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    completion = cache_dir / "COMPLETE.json"
    if completion.exists():
        completion.unlink()
    values = np.ascontiguousarray(embeddings, dtype=np.float32)
    _atomic_write(
        cache_dir / "embeddings.npy",
        lambda handle: np.save(handle, values, allow_pickle=False),
        binary=True,
    )
    _atomic_write(
        cache_dir / "doc_offsets.npy",
        lambda handle: np.save(
            handle, np.asarray(doc_offsets, dtype=np.int64), allow_pickle=False
        ),
        binary=True,
    )
    payload = dict(manifest)
    payload.setdefault("schema", "topic_pairs_sentence_embedding_cache")
    payload.setdefault("schema_version", EMBEDDING_CACHE_SCHEMA_VERSION)
    _atomic_write(
        cache_dir / "manifest.json",
        lambda handle: json.dump(payload, handle, ensure_ascii=False, indent=2),
        binary=False,
    )
    _atomic_write(
        completion,
        lambda handle: json.dump(
            {
                "schema": "topic_pairs_sentence_embedding_cache_completion",
                "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
                "sentence_sha1": payload.get("sentence_sha1"),
            },
            handle,
        ),
        binary=False,
    )


def load_cached_embeddings(
    cache_dir: Path, *, expected_sentences: int, sentence_sha1: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    cache_dir = Path(cache_dir)
    completion = cache_dir / "COMPLETE.json"
    manifest_path = cache_dir / "manifest.json"
    if not completion.exists() or not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("sentence_sha1") != sentence_sha1:
        return None
    embeddings = np.load(cache_dir / "embeddings.npy", allow_pickle=False)
    doc_offsets = np.load(cache_dir / "doc_offsets.npy", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != int(expected_sentences):
        return None
    return embeddings, doc_offsets, manifest


@dataclass(frozen=True)
class CachedEmbeddings:
    embeddings: np.ndarray  # raw encoder output, float32, (S, D)
    doc_offsets: np.ndarray
    manifest: dict[str, Any]
    cache_dir: Path
    cache_hit: bool

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def document(self, doc_index: int) -> np.ndarray:
        start = int(self.doc_offsets[doc_index])
        stop = int(self.doc_offsets[doc_index + 1])
        return self.embeddings[start:stop]

    def iter_documents(self):
        for doc_index in range(len(self.doc_offsets) - 1):
            yield self.document(doc_index)


EncoderFactory = Callable[[Mapping[str, Any], str, int | None], tuple[Any, Any]]


def default_encoder_factory(
    encoder_config: Mapping[str, Any], device: str, batch_size_override: int | None
) -> tuple[Any, Any]:
    """Rebuild the training encoder exactly as the representative-word protocol does."""

    return _encoder_from_metadata(
        {"encoder_config": dict(encoder_config)},
        device=device,
        encode_batch_size_override=batch_size_override,
    )


def encode_train_corpus(
    corpus: TrainCorpus,
    *,
    encoder_config: Mapping[str, Any],
    cache_root: Path,
    dataset: str,
    data_run: str,
    category: str,
    split: str,
    device: str,
    encode_batch_size: int | None = None,
    encoder_factory: EncoderFactory = default_encoder_factory,
) -> CachedEmbeddings:
    """Raw encoder output of the reference sentences, from the cache when present."""

    encoder_fp = encoder_fingerprint(encoder_config)
    cache_dir = embedding_cache_dir(
        cache_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        split=split,
        encoder_fp=encoder_fp,
        sentence_sha1=corpus.sentence_sha1,
    )
    cached = load_cached_embeddings(
        cache_dir,
        expected_sentences=corpus.num_sentences,
        sentence_sha1=corpus.sentence_sha1,
    )
    if cached is not None:
        embeddings, doc_offsets, manifest = cached
        if not np.array_equal(doc_offsets, corpus.doc_offsets):
            raise SentenceAlignmentError(
                f"cached document offsets differ from the corpus: {cache_dir}"
            )
        return CachedEmbeddings(
            embeddings=embeddings,
            doc_offsets=doc_offsets,
            manifest=manifest,
            cache_dir=cache_dir,
            cache_hit=True,
        )

    encoder, batch_size = encoder_factory(encoder_config, device, encode_batch_size)
    encoded = encode_sentence_corpus(
        encoder=encoder,
        documents=corpus.documents,
        batch_size=int(getattr(batch_size, "value", batch_size)),
        show_progress_bar=False,
    )
    if not np.array_equal(encoded.doc_offsets, corpus.doc_offsets):
        raise SentenceAlignmentError("encoder document offsets differ from the corpus")
    manifest = {
        "dataset": str(dataset),
        "data_run": str(data_run),
        "category": str(category),
        "split": str(split),
        "encoder_config": {
            key: encoder_config.get(key) for key in ENCODER_FINGERPRINT_FIELDS
        },
        "encoder_fingerprint": encoder_fp,
        "sentence_sha1": corpus.sentence_sha1,
        "num_documents": int(corpus.num_documents),
        "total_sentences": int(corpus.num_sentences),
        "embedding_dim": int(encoded.embedding_dim),
        "embedding_storage": "raw_encoder_output_float32",
        "source_condition_dir": str(corpus.condition_dir),
        "encode_batch_size": int(getattr(batch_size, "value", batch_size)),
        "device": str(device),
        "created_at": datetime.now(UTC).isoformat(),
    }
    save_cached_embeddings(
        cache_dir,
        embeddings=encoded.embeddings,
        doc_offsets=encoded.doc_offsets,
        manifest=manifest,
    )
    return CachedEmbeddings(
        embeddings=np.ascontiguousarray(encoded.embeddings, dtype=np.float32),
        doc_offsets=np.asarray(encoded.doc_offsets, dtype=np.int64),
        manifest=manifest,
        cache_dir=cache_dir,
        cache_hit=False,
    )


def unit_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalised float64 rows; a zero row is an error, not a NaN."""

    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("encoder produced a zero sentence vector")
    return values / norms


# ---------------------------------------------------------------------------
# Fine labels and model state
# ---------------------------------------------------------------------------


def category_label_names(
    dataset: str,
    category: str,
    *,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> list[str]:
    targets = get_dataset_targets(
        dataset, target_column=target_column, label_schema=label_schema
    )
    if not targets:
        raise ValueError(f"no fine-label catalogue for dataset {dataset!r}")
    if category in targets:
        return [str(name) for name in targets[category]]
    if str(category) == "all":
        return sorted({str(name) for names in targets.values() for name in names})
    raise ValueError(
        f"category {category!r} is not in the fine-label catalogue of {dataset!r}: "
        f"{sorted(targets)}"
    )


def load_fine_labels(
    dataset: str,
    *,
    split: str,
    raw_doc_indices: Sequence[int],
    category: str,
    target_column: str = "target_str",
    label_schema: str = "identity",
) -> tuple[np.ndarray, list[str]]:
    """Fine-label index of every selected document, in selection order.

    ``raw_doc_indices`` are row positions of ``data/<dataset>/<split>.csv``,
    the convention of the classification pipeline; the label order is the
    dataset catalogue's, so per-topic label vectors are comparable across runs.
    """

    label_names = category_label_names(
        dataset, category, target_column=target_column, label_schema=label_schema
    )
    _, frame = load_dataset_split(dataset, split)
    if target_column not in frame.columns:
        raise ValueError(f"column {target_column!r} is missing from {dataset}/{split}")
    column = frame[target_column].astype(str).str.strip().to_numpy()
    indices = np.asarray(list(raw_doc_indices), dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= column.shape[0]):
        raise ValueError(
            f"raw document index out of range for {dataset}/{split} "
            f"({column.shape[0]} rows): min={indices.min()} max={indices.max()}"
        )
    labels = column[indices]
    index = {name: position for position, name in enumerate(label_names)}
    unknown = sorted({str(value) for value in labels if str(value) not in index})
    if unknown:
        raise ValueError(
            f"documents of {dataset}/{category} carry labels outside the category "
            f"catalogue {label_names}: {unknown[:5]}"
        )
    return (
        np.asarray([index[str(value)] for value in labels], dtype=np.int64),
        label_names,
    )


def sentence_labels(doc_labels: np.ndarray, doc_offsets: np.ndarray) -> np.ndarray:
    """Every sentence inherits the fine label of its document."""

    counts = np.diff(np.asarray(doc_offsets, dtype=np.int64))
    values = np.asarray(doc_labels, dtype=np.int64)
    if values.shape[0] != counts.shape[0]:
        raise ValueError("document labels and offsets are not aligned")
    return np.repeat(values, counts)


def load_vmf_model_reference(condition_dir: Path) -> dict[str, Any]:
    """The vMF Sentence LDA run's own centres and concentrations."""

    condition_dir = Path(condition_dir)
    params = load_artifact_json(condition_dir / "params.json")
    return {
        "topic_means": np.asarray(
            load_artifact_pickle(condition_dir / "topic_means.pkl"), dtype=np.float64
        ),
        "kappa_model": np.asarray(
            load_artifact_pickle(condition_dir / "kappa_per_topic.pkl"),
            dtype=np.float64,
        ),
        "topic_counts": np.asarray(
            load_artifact_pickle(condition_dir / "topic_counts.pkl"), dtype=np.float64
        ),
        "kappa_default": params.get("kappa_default"),
        "max_kappa": params.get("max_kappa"),
    }


__all__ = [
    "EMBEDDING_CACHE_SCHEMA_VERSION",
    "EMBEDDING_SPACE",
    "MODEL_ALIASES",
    "REFERENCE_MODEL_ORDER",
    "SENTENCE_ENCODER_MODELS",
    "TASK_MODELS",
    "CachedEmbeddings",
    "SentenceAlignmentError",
    "TrainCorpus",
    "assert_same_sentences",
    "category_label_names",
    "default_encoder_factory",
    "embedding_cache_dir",
    "encode_train_corpus",
    "encoder_config_of",
    "encoder_fingerprint",
    "load_cached_embeddings",
    "load_fine_labels",
    "load_train_corpus",
    "load_vmf_model_reference",
    "model_embedding_variant",
    "normalize_model_name",
    "normalize_model_names",
    "parameter_variant_for",
    "provenance_for",
    "reference_model",
    "resolve_condition_dir",
    "save_cached_embeddings",
    "sentence_fingerprint",
    "sentence_labels",
    "unit_normalize",
]

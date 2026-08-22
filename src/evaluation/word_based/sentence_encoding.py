from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.preprocessing import PreprocessedDocument
from src.utils.encoder import SentenceEncoder
from src.utils.encoder_profiles import normalize_encoder_backend

BACKEND_DEFAULT_ENCODE_BATCH_SIZES: dict[str, int] = {
    "sentence_transformers": 32,
    "simcse": 32,
    "usif": 128,
}
_CUDA_DEVICE_RE = re.compile(r"cuda:(\d+)")


@dataclass(frozen=True)
class ResolvedEncodeBatchSize:
    value: int
    source: str
    training_value: int | None


@dataclass(frozen=True)
class EncodedSentenceCorpus:
    embeddings: np.ndarray
    doc_offsets: np.ndarray
    num_documents: int
    total_sentences: int
    embedding_dim: int

    def document(self, doc_index: int) -> np.ndarray:
        if doc_index < 0 or doc_index >= self.num_documents:
            raise IndexError(
                f"document index out of range: {doc_index} "
                f"for {self.num_documents} documents"
            )
        start = int(self.doc_offsets[doc_index])
        stop = int(self.doc_offsets[doc_index + 1])
        return self.embeddings[start:stop]

    def iter_documents(self) -> Iterator[np.ndarray]:
        for doc_index in range(self.num_documents):
            yield self.document(doc_index)


def flatten_encoder_sentences(
    documents: Sequence[PreprocessedDocument],
    *,
    use_tokenized: bool,
) -> tuple[list[str], np.ndarray]:
    """Flatten sentences exactly as the fitted encoder saw them during training.

    ``sentence_corpus_for_encoder`` feeds ``sentences_raw`` to every backend
    except uSIF, which consumes tokenized-and-rejoined text.  Post-hoc encoding
    has to use the same text domain, otherwise the sentence log likelihoods land
    far from every fitted topic parameter and the fold-in assigns topics almost
    arbitrarily.

    Documents are not filtered here: callers align the encoded corpus with
    ``documents`` positionally.  ``sentences_raw``, ``sentences_tokenized`` and
    ``sentences_joined`` are index-aligned within a document, so the offsets are
    the same either way.
    """

    sentences: list[str] = []
    offsets = np.empty(len(documents) + 1, dtype=np.int64)
    offsets[0] = 0
    for doc_index, document in enumerate(documents):
        if use_tokenized:
            sentences.extend(document.sentences_joined)
        else:
            sentences.extend(document.sentences_raw)
        offsets[doc_index + 1] = len(sentences)
    return sentences, offsets


def encode_sentence_corpus(
    *,
    encoder: SentenceEncoder,
    documents: Sequence[PreprocessedDocument],
    batch_size: int,
    show_progress_bar: bool = False,
) -> EncodedSentenceCorpus:
    resolved_batch_size = int(batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    sentences, doc_offsets = flatten_encoder_sentences(
        documents,
        use_tokenized=bool(getattr(encoder, "accepts_tokenized", False)),
    )
    expected_dim = int(encoder.get_sentence_embedding_dimension())
    if not sentences:
        embeddings = np.empty((0, expected_dim), dtype=np.float32)
    else:
        try:
            encoded = encoder.encode(
                sentences,
                batch_size=resolved_batch_size,
                show_progress_bar=show_progress_bar,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "topic-word sentence encoding ran out of memory with "
                    f"batch_size={resolved_batch_size}; retry with "
                    "TOPIC_WORD_ENCODE_BATCH_SIZE=64 or 32"
                ) from exc
            raise
        embeddings = np.asarray(encoded)
        if embeddings.ndim == 1 and len(sentences) == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2:
            raise ValueError(
                "sentence encoder returned a non-matrix result: "
                f"shape={embeddings.shape} sentences={len(sentences)}"
            )
        if embeddings.shape[0] != len(sentences):
            raise ValueError(
                "sentence encoder row count mismatch: "
                f"rows={embeddings.shape[0]} sentences={len(sentences)}"
            )
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                "sentence encoder dimension mismatch: "
                f"actual={embeddings.shape[1]} expected={expected_dim}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "sentence encoder returned non-finite values: "
                f"shape={embeddings.shape} sentences={len(sentences)}"
            )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    return EncodedSentenceCorpus(
        embeddings=embeddings,
        doc_offsets=doc_offsets,
        num_documents=len(documents),
        total_sentences=len(sentences),
        embedding_dim=expected_dim,
    )


def _optional_torch():
    """Return torch if importable, else None (CPU-only installs lack torch)."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_topic_word_encoder_device(requested: str) -> str:
    normalized = str(requested).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "auto":
        torch = _optional_torch()
        if torch is None:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        torch = _optional_torch()
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError(
                "topic-word encoder device 'cuda' was requested, "
                "but CUDA is unavailable"
            )
        return "cuda"
    match = _CUDA_DEVICE_RE.fullmatch(normalized)
    if match is not None:
        torch = _optional_torch()
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError(
                f"topic-word encoder device {normalized!r} was requested, "
                "but CUDA is unavailable"
            )
        device_index = int(match.group(1))
        device_count = int(torch.cuda.device_count())
        if device_index >= device_count:
            raise RuntimeError(
                f"topic-word encoder device {normalized!r} is out of range; "
                f"CUDA device count is {device_count}"
            )
        return normalized
    raise ValueError(
        "topic_word_encoder_device must be auto, cpu, cuda, or cuda:N; "
        f"got {requested!r}"
    )


def resolve_topic_word_encode_batch_size(
    *,
    encoder_config: Mapping[str, Any],
    override: int | None,
) -> ResolvedEncodeBatchSize:
    raw_training_value = encoder_config.get("encode_batch_size")
    training_value = None if raw_training_value is None else int(raw_training_value)
    if training_value is not None and training_value <= 0:
        raise ValueError(
            "saved encoder_config.encode_batch_size must be > 0; "
            f"got {training_value}"
        )

    if override is not None:
        resolved_override = int(override)
        if resolved_override <= 0:
            raise ValueError("topic_word_encode_batch_size must be > 0")
        return ResolvedEncodeBatchSize(
            value=resolved_override,
            source="override",
            training_value=training_value,
        )

    if training_value is not None:
        return ResolvedEncodeBatchSize(
            value=training_value,
            source="training_metadata",
            training_value=training_value,
        )

    model_name = encoder_config.get("model_name")
    if model_name in {None, ""}:
        raise ValueError("encoder_config.model_name is required to resolve batch size")
    backend = normalize_encoder_backend(
        str(model_name),
        (
            None
            if encoder_config.get("backend") is None
            else str(encoder_config.get("backend"))
        ),
    )
    backend_default = BACKEND_DEFAULT_ENCODE_BATCH_SIZES.get(backend)
    if backend_default is None:
        raise ValueError(
            "saved encoder batch size is null and no stable backend default is "
            f"defined for backend={backend!r}; provide an explicit override"
        )
    return ResolvedEncodeBatchSize(
        value=backend_default,
        source="backend_default",
        training_value=None,
    )

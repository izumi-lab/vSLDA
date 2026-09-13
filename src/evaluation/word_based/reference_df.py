"""Document frequencies of every token in a coherence reference corpus.

The evaluation dictionary can be restricted to a reference-frequency band
(see ``build_dictionary_and_corpus``). Building that band needs one document
frequency table per reference corpus, which is expensive enough (one full pass)
to be worth caching but small enough to keep as JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.utils.logging import get_logger

from .corpus_bundle import iter_tokenized_reference_corpus
from .reference_index import reference_source_identity

REFERENCE_DF_SCHEMA_VERSION = 1
_PROGRESS_DOC_INTERVAL = 250_000

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReferenceDocumentFrequencies:
    """Per-token document frequency over a reference corpus."""

    document_frequencies: dict[str, int]
    num_docs: int
    source: dict[str, Any]

    def ratio(self, token: str) -> float:
        if self.num_docs <= 0:
            return 0.0
        return self.document_frequencies.get(token, 0) / self.num_docs


def reference_df_fingerprint(
    *, path: Path, max_docs: int | None, min_doc_tokens: int
) -> str:
    payload = {
        "schema_version": REFERENCE_DF_SCHEMA_VERSION,
        "source": reference_source_identity(Path(path)),
        "max_docs": None if max_docs is None else int(max_docs),
        "min_doc_tokens": int(min_doc_tokens),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_reference_document_frequencies(
    path: Path,
    *,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
) -> ReferenceDocumentFrequencies:
    """Count, in one pass, how many reference documents contain each token."""

    resolved = Path(path)
    started = perf_counter()
    logger.info(
        "reference df scan start path=%s max_docs=%s min_doc_tokens=%s",
        resolved,
        max_docs,
        min_doc_tokens,
    )
    counts: Counter[str] = Counter()
    num_docs = 0
    for tokens in iter_tokenized_reference_corpus(
        path=resolved, max_docs=max_docs, min_doc_tokens=min_doc_tokens
    ):
        num_docs += 1
        counts.update(set(tokens))
        if num_docs % _PROGRESS_DOC_INTERVAL == 0:
            logger.info(
                "reference df scan progress docs=%s vocab=%s sec=%.1f",
                num_docs,
                len(counts),
                perf_counter() - started,
            )
    if num_docs == 0:
        raise ValueError(f"Reference corpus is empty after filtering: {resolved}")
    logger.info(
        "reference df scan done docs=%s vocab=%s sec=%.1f",
        num_docs,
        len(counts),
        perf_counter() - started,
    )
    return ReferenceDocumentFrequencies(
        document_frequencies=dict(counts),
        num_docs=num_docs,
        source=reference_source_identity(resolved),
    )


def load_reference_document_frequencies(
    path: Path,
    *,
    max_docs: int | None = None,
    min_doc_tokens: int = 1,
    cache_root: Path | None = None,
) -> ReferenceDocumentFrequencies:
    """Return the document frequency table, reusing the on-disk cache."""

    resolved = Path(path)
    cache_path: Path | None = None
    if cache_root is not None:
        fingerprint = reference_df_fingerprint(
            path=resolved, max_docs=max_docs, min_doc_tokens=min_doc_tokens
        )
        cache_path = Path(cache_root) / "reference_df" / f"{fingerprint}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "reference df cache unreadable, rebuilding path=%s", cache_path
                )
            else:
                if int(payload.get("schema_version", 0)) == REFERENCE_DF_SCHEMA_VERSION:
                    logger.info("reference df cache hit path=%s", cache_path)
                    return ReferenceDocumentFrequencies(
                        document_frequencies={
                            str(word): int(count)
                            for word, count in payload["document_frequencies"].items()
                        },
                        num_docs=int(payload["num_docs"]),
                        source=dict(payload.get("source", {})),
                    )

    table = build_reference_document_frequencies(
        resolved, max_docs=max_docs, min_doc_tokens=min_doc_tokens
    )
    if cache_path is not None:
        _atomic_json(
            cache_path,
            {
                "schema_version": REFERENCE_DF_SCHEMA_VERSION,
                "source": table.source,
                "max_docs": None if max_docs is None else int(max_docs),
                "min_doc_tokens": int(min_doc_tokens),
                "num_docs": table.num_docs,
                "document_frequencies": table.document_frequencies,
            },
        )
        logger.info("reference df cache write path=%s", cache_path)
    return table

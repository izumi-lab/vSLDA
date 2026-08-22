"""Shared schema handling for preprocessing_selection.json artifacts.

Two on-disk layouts exist for the selection payload:

- Flat form (baseline models, one file per split)::

    {"raw_doc_indices": [...], "sentence_indices_by_doc": [...], ...}

- Combined form (vMF, one file for both splits)::

    {"train": {"raw_doc_indices": [...], ...}, "test": {...}}

Both the topic-word evaluation and the classification feature registry must
interpret these layouts with the same rules, so the normalization lives here
next to the ``SelectedCorpus`` serialization that produces the payloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

VALID_SELECTION_SPLITS = ("train", "test")

_RAW_DOC_INDICES_KEY = "raw_doc_indices"
_SENTENCE_INDICES_KEY = "sentence_indices_by_doc"


def resolve_preprocessing_selection(
    payload: object,
    *,
    split: str,
    selection_path: Path,
) -> Mapping[str, object]:
    """Return the per-split selection mapping for ``split``.

    Detection is structural, never model-based: a top-level
    ``raw_doc_indices`` marks the flat form, top-level ``train``/``test``
    keys mark the combined form. Ambiguous or unrecognized payloads fail
    instead of falling back to an empty selection.
    """
    if split not in VALID_SELECTION_SPLITS:
        raise ValueError(
            f"Unsupported preprocessing selection split {split!r} "
            f"(expected one of {list(VALID_SELECTION_SPLITS)}): {selection_path}"
        )
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Invalid preprocessing selection payload "
            f"(expected a JSON object, got {type(payload).__name__}): "
            f"{selection_path}"
        )
    top_level_keys = sorted(str(key) for key in payload)
    has_flat_marker = _RAW_DOC_INDICES_KEY in payload
    has_combined_marker = any(key in payload for key in VALID_SELECTION_SPLITS)
    if has_flat_marker and has_combined_marker:
        raise ValueError(
            "Ambiguous preprocessing selection payload: both top-level "
            f"{_RAW_DOC_INDICES_KEY!r} and split keys are present "
            f"(top-level keys: {top_level_keys}): {selection_path}"
        )
    if has_combined_marker:
        if split not in payload:
            raise ValueError(
                f"Preprocessing selection has no {split!r} split "
                f"(top-level keys: {top_level_keys}): {selection_path}"
            )
        section = payload[split]
        if not isinstance(section, Mapping):
            raise ValueError(
                f"Preprocessing selection split {split!r} is not a JSON object "
                f"(got {type(section).__name__}): {selection_path}"
            )
        if _RAW_DOC_INDICES_KEY not in section:
            raise ValueError(
                f"Preprocessing selection split {split!r} is missing "
                f"{_RAW_DOC_INDICES_KEY!r} "
                f"(split keys: {sorted(str(key) for key in section)}): "
                f"{selection_path}"
            )
        return section
    if has_flat_marker:
        return payload
    raise ValueError(
        "Unrecognized preprocessing selection payload: neither "
        f"{_RAW_DOC_INDICES_KEY!r} nor train/test split keys are present "
        f"(top-level keys: {top_level_keys}): {selection_path}"
    )


def parse_raw_doc_indices(
    selection: Mapping[str, object],
    *,
    selection_path: Path,
    split: str,
) -> list[int]:
    """Validate and return the ordered raw document IDs for one split.

    The order is preserved as stored: it defines both the document alignment
    and the corpus fingerprint, so it must never be sorted or deduplicated.
    """
    values = selection.get(_RAW_DOC_INDICES_KEY)
    if not isinstance(values, list):
        raise ValueError(
            f"Preprocessing selection split {split!r} field "
            f"{_RAW_DOC_INDICES_KEY!r} must be a list "
            f"(got {type(values).__name__}): {selection_path}"
        )
    raw_ids = [
        _coerce_raw_doc_index(
            value, position=position, selection_path=selection_path, split=split
        )
        for position, value in enumerate(values)
    ]
    seen: set[int] = set()
    for position, raw_id in enumerate(raw_ids):
        if raw_id in seen:
            raise ValueError(
                f"Duplicate raw document ID {raw_id} at position {position} "
                f"in split {split!r}: {selection_path}"
            )
        seen.add(raw_id)
    sentence_indices = selection.get(_SENTENCE_INDICES_KEY)
    if sentence_indices is not None:
        if not isinstance(sentence_indices, list):
            raise ValueError(
                f"Preprocessing selection split {split!r} field "
                f"{_SENTENCE_INDICES_KEY!r} must be a list "
                f"(got {type(sentence_indices).__name__}): {selection_path}"
            )
        if len(sentence_indices) != len(raw_ids):
            raise ValueError(
                f"Preprocessing selection split {split!r} has "
                f"{len(sentence_indices)} {_SENTENCE_INDICES_KEY!r} entries for "
                f"{len(raw_ids)} raw document IDs: {selection_path}"
            )
    return raw_ids


def _coerce_raw_doc_index(
    value: object,
    *,
    position: int,
    selection_path: Path,
    split: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid raw document ID {value!r} at position {position} "
            f"in split {split!r} (booleans are not IDs): {selection_path}"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(
            f"Invalid raw document ID {value!r} at position {position} "
            f"in split {split!r} (non-integral number): {selection_path}"
        )
    if isinstance(value, str):
        try:
            return int(value, 10)
        except ValueError:
            pass
    raise ValueError(
        f"Invalid raw document ID {value!r} at position {position} "
        f"in split {split!r} (not convertible to an integer): {selection_path}"
    )

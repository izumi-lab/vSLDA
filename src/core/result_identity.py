from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_identity_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_identity_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, set):
        return [_normalize_identity_value(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_identity_value(item) for item in value]
    return value


def build_condition_fingerprint(payload: Mapping[str, Any]) -> str:
    normalized = _normalize_identity_value(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:10]


def slugify_label(value: Any, *, max_length: int = 24) -> str:
    text = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    text = _NON_ALNUM_RE.sub("-", text.strip().lower()).strip("-")
    if not text:
        return "na"
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip("-") or "na"


def build_condition_id(
    *,
    iteration: int,
    num_topics: int,
    fingerprint_payload: Mapping[str, Any],
    category: str | None = None,
    extra_labels: Sequence[Any] = (),
    include_fingerprint: bool = True,
) -> tuple[str, str]:
    fingerprint = build_condition_fingerprint(fingerprint_payload)
    parts = [f"it{int(iteration)}", f"k{int(num_topics)}"]
    if category is not None:
        parts.append(slugify_label(category, max_length=18))
    for label in extra_labels:
        slug = slugify_label(label, max_length=18)
        if slug != "na":
            parts.append(slug)
    if include_fingerprint:
        parts.append(fingerprint[:8])
    return "__".join(parts), fingerprint


def _coerce_utc_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        dt = value
    else:
        normalized = str(value).strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def build_display_key(
    *,
    num_topics: int,
    iteration: int,
    extra_labels: Sequence[Any] = (),
) -> str:
    parts: list[str] = []
    for label in extra_labels:
        slug = slugify_label(label, max_length=24)
        if slug != "na":
            parts.append(slug)
    parts.extend([f"k{int(num_topics)}", f"it{int(iteration)}"])
    return "_".join(parts)


def build_execution_date(*, started_at: datetime | str | None = None) -> str:
    return _coerce_utc_datetime(started_at).strftime("%Y-%m-%d")


def build_execution_id(
    *,
    prefix: str = "exec",
    started_at: datetime | str | None = None,
) -> str:
    timestamp = _coerce_utc_datetime(started_at).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"

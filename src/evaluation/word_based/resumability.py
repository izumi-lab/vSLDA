from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.word_based.topic_word_runtime import RuntimeTopicWords
from src.evaluation.word_based.topic_words import TopicWords

CHECKPOINT_SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 1


def fingerprint_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_json(payload: Any, path: Path) -> None:
    _atomic_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def atomic_save_pickle(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def deserialize_topic_words(payload: Any) -> TopicWords:
    if not isinstance(payload, list):
        raise ValueError("serialized topic words must be a list")
    topics: TopicWords = []
    for expected_topic_id, raw_topic in enumerate(payload):
        if not isinstance(raw_topic, dict):
            raise ValueError("serialized topic must be an object")
        if int(raw_topic.get("topic_id", -1)) != expected_topic_id:
            raise ValueError("serialized topic IDs are not contiguous")
        raw_words = raw_topic.get("words")
        if not isinstance(raw_words, list):
            raise ValueError("serialized topic words must be a list")
        topics.append(
            [
                (str(item["word"]), float(item["score"]))
                for item in raw_words
                if isinstance(item, dict) and "word" in item and "score" in item
            ]
        )
    return topics


def topic_word_checkpoint_dir(
    *,
    checkpoint_root: Path,
    identity: Mapping[str, Any],
) -> Path:
    return (
        checkpoint_root
        / "word_based_topic_words"
        / f"v{CHECKPOINT_SCHEMA_VERSION}"
        / fingerprint_payload(identity)
    )


def save_topic_word_checkpoint(
    *,
    checkpoint_dir: Path,
    identity: Mapping[str, Any],
    runtime: RuntimeTopicWords,
    serialized_evaluation_words: list[dict[str, object]],
    serialized_display_words: list[dict[str, object]],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completion_path = checkpoint_dir / "COMPLETE.json"
    if completion_path.exists():
        completion_path.unlink()
    manifest = {
        "schema": "word_based_topic_word_checkpoint",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": _jsonable(dict(identity)),
        "runtime": {
            "evaluation_source": runtime.evaluation.topic_word_source,
            "evaluation_score_mode": runtime.evaluation.score_mode,
            "evaluation_score_definition": runtime.evaluation.score_definition,
            "display_source": runtime.display_source,
            "display_score_mode": runtime.display_score_mode,
            "protocol": runtime.protocol,
            "condition_dir": str(runtime.condition_dir),
            "source_condition_fingerprint": runtime.source_condition_fingerprint,
            "vocabulary_fingerprint": runtime.vocabulary_fingerprint,
            "corpus_fingerprint": runtime.corpus_fingerprint,
            "coverage": runtime.coverage,
            "posterior_metadata": runtime.posterior_metadata,
            "execution_metadata": runtime.execution_metadata,
            "empty_topic_ids": list(runtime.empty_topic_ids),
        },
        "evaluation_topic_words": serialized_evaluation_words,
        "display_topic_words": serialized_display_words,
        "artifacts": {
            "posterior_mean": (
                "posterior_mean.pkl"
                if runtime.posterior_mean_by_doc is not None
                else None
            ),
            "expected_counts": (
                "expected_counts.pkl" if runtime.expected_counts is not None else None
            ),
        },
    }
    if runtime.posterior_mean_by_doc is not None:
        atomic_save_pickle(
            runtime.posterior_mean_by_doc,
            checkpoint_dir / "posterior_mean.pkl",
        )
    if runtime.expected_counts is not None:
        atomic_save_pickle(
            runtime.expected_counts,
            checkpoint_dir / "expected_counts.pkl",
        )
    atomic_save_json(manifest, checkpoint_dir / "manifest.json")
    atomic_save_json(
        {
            "schema": "word_based_topic_word_checkpoint_completion",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity_fingerprint": fingerprint_payload(identity),
        },
        completion_path,
    )


def load_topic_word_checkpoint(
    *,
    checkpoint_dir: Path,
    expected_identity: Mapping[str, Any],
) -> RuntimeTopicWords | None:
    completion_path = checkpoint_dir / "COMPLETE.json"
    manifest_path = checkpoint_dir / "manifest.json"
    if not completion_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return None
        if manifest.get("identity") != _jsonable(dict(expected_identity)):
            return None
        meta = manifest["runtime"]
        artifacts = manifest["artifacts"]
        posterior = None
        expected_counts = None
        if artifacts.get("posterior_mean"):
            with (checkpoint_dir / artifacts["posterior_mean"]).open("rb") as handle:
                posterior = pickle.load(handle)
        if artifacts.get("expected_counts"):
            with (checkpoint_dir / artifacts["expected_counts"]).open("rb") as handle:
                expected_counts = pickle.load(handle)
        from src.evaluation.word_based.topic_words import TopicWordsResult

        evaluation_words = deserialize_topic_words(manifest["evaluation_topic_words"])
        return RuntimeTopicWords(
            evaluation=TopicWordsResult(
                topic_words=evaluation_words,
                topic_word_source=str(meta["evaluation_source"]),
                score_mode=meta.get("evaluation_score_mode"),
                score_definition=meta.get("evaluation_score_definition"),
            ),
            display_topic_words=deserialize_topic_words(
                manifest["display_topic_words"]
            ),
            display_source=str(meta["display_source"]),
            display_score_mode=str(meta["display_score_mode"]),
            protocol=str(meta["protocol"]),
            condition_dir=Path(meta["condition_dir"]),
            source_condition_fingerprint=str(meta["source_condition_fingerprint"]),
            vocabulary_fingerprint=str(meta["vocabulary_fingerprint"]),
            corpus_fingerprint=str(meta["corpus_fingerprint"]),
            coverage=dict(meta["coverage"]),
            posterior_mean_by_doc=posterior,
            posterior_metadata=meta.get("posterior_metadata"),
            expected_counts=expected_counts,
            execution_metadata=meta.get("execution_metadata"),
            empty_topic_ids=tuple(
                int(topic_id) for topic_id in meta.get("empty_topic_ids", [])
            ),
        )
    except (KeyError, TypeError, ValueError, OSError, pickle.UnpicklingError, EOFError):
        return None


def reference_corpus_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def save_failure_record(
    *,
    checkpoint_root: Path,
    identity: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Path:
    path = checkpoint_root / "failures" / "v1" / f"{fingerprint_payload(identity)}.json"
    atomic_save_json(
        {
            "schema": "word_based_condition_failure",
            "schema_version": 1,
            "identity": _jsonable(dict(identity)),
            **dict(payload),
        },
        path,
    )
    return path


def write_completion_marker(
    *,
    output_dir: Path,
    condition_fingerprint: str,
    artifacts: Mapping[str, Any],
) -> Path:
    path = output_dir / "COMPLETE.json"
    atomic_save_json(
        {
            "schema": "word_based_metrics_completion",
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "condition_fingerprint": condition_fingerprint,
            "artifacts": dict(artifacts),
        },
        path,
    )
    return path

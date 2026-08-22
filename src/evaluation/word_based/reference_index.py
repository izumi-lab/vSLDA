from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .reference_interval_counts import (
    _build_pair_adjacency,
    _get_interval_kernel,
    _to_public_counts,
    _tokens_from_line,
)
from .reference_query import ReferenceCountQuery
from .topic_word_metrics import SlidingWindowCounts

REFERENCE_INDEX_SCHEMA_VERSION = 1
REFERENCE_INDEX_BUILDER_VERSION = 1
DEFAULT_INDEX_COMMIT_DOCS = 25_000


def reference_source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS vocabulary (
            word_id INTEGER PRIMARY KEY,
            word TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY,
            length INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS postings (
            word_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            position INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS build_state (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            byte_offset INTEGER NOT NULL,
            next_doc_id INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            source_identity TEXT
        );
        INSERT OR IGNORE INTO build_state
            (singleton, byte_offset, next_doc_id, token_count, source_identity)
            VALUES (1, 0, 0, 0, NULL);
        """
    )
    # Databases written before the resume guard existed lack the column.
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(build_state)")
    }
    if "source_identity" not in columns:
        connection.execute("ALTER TABLE build_state ADD COLUMN source_identity TEXT")
    connection.commit()


def build_reference_index(
    *,
    reference_path: Path,
    index_root: Path,
    min_doc_tokens: int = 1,
    commit_docs: int = DEFAULT_INDEX_COMMIT_DOCS,
) -> Path:
    """Build or resume a collision-free SQLite positional index.

    SQLite is the v1 prototype storage. It keeps build checkpoints transactionally
    and permits measuring the corpus-specific size before adding compression.
    """
    if min_doc_tokens < 1:
        raise ValueError("min_doc_tokens must be >= 1")
    if commit_docs < 1:
        raise ValueError("commit_docs must be >= 1")
    reference_path = reference_path.resolve()
    source = reference_source_identity(reference_path)
    source_json = json.dumps(source, sort_keys=True)
    index_root.mkdir(parents=True, exist_ok=True)
    complete_path = index_root / "COMPLETE.json"
    manifest_path = index_root / "manifest.json"
    database_path = index_root / "index.sqlite3"
    if complete_path.exists() and is_reference_index_valid(
        reference_path=reference_path,
        index_root=index_root,
        min_doc_tokens=min_doc_tokens,
    ):
        return index_root
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source") != source or int(
            existing.get("min_doc_tokens", -1)
        ) != int(min_doc_tokens):
            raise ValueError(
                "Existing reference index is stale; choose a new index root or "
                "explicitly remove/refresh it."
            )
    else:
        _atomic_json(
            manifest_path,
            {
                "schema": "word_based_reference_positional_index",
                "schema_version": REFERENCE_INDEX_SCHEMA_VERSION,
                "builder_version": REFERENCE_INDEX_BUILDER_VERSION,
                "storage": "sqlite_uncompressed",
                "source": source,
                "min_doc_tokens": int(min_doc_tokens),
                "status": "building",
            },
        )
    if complete_path.exists():
        complete_path.unlink()

    connection = sqlite3.connect(database_path)
    try:
        _initialize_database(connection)
        byte_offset, next_doc_id, token_count, stored_source = connection.execute(
            "SELECT byte_offset, next_doc_id, token_count, source_identity "
            "FROM build_state WHERE singleton = 1"
        ).fetchone()
        # The resume offset lives in the database while the corpus identity used
        # to live only in manifest.json. If the manifest is missing the manifest
        # branch above rewrites it for the *current* corpus, so without this
        # check a stale offset would be replayed against a different file and
        # committed as COMPLETE. Validate the identity where the offset is kept.
        if int(byte_offset) > 0 or int(next_doc_id) > 0:
            resumable = (
                stored_source is not None and json.loads(stored_source) == source
            )
            if not resumable:
                raise ValueError(
                    "Existing reference index has build progress for a different "
                    "reference corpus; choose a new index root or explicitly "
                    "remove/refresh it."
                )
        vocabulary = {
            str(word): int(word_id)
            for word_id, word in connection.execute(
                "SELECT word_id, word FROM vocabulary"
            )
        }
        next_word_id = len(vocabulary)
        docs_since_commit = 0
        with reference_path.open("rb") as handle:
            handle.seek(int(byte_offset))
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                byte_offset = handle.tell()
                if not line.strip():
                    continue
                tokens = _tokens_from_line(line, path=reference_path, offset=line_start)
                if len(tokens) < min_doc_tokens:
                    continue
                doc_id = int(next_doc_id)
                connection.execute(
                    "INSERT INTO documents(doc_id, length) VALUES (?, ?)",
                    (doc_id, len(tokens)),
                )
                postings: list[tuple[int, int, int]] = []
                for position, word in enumerate(tokens):
                    word_id = vocabulary.get(word)
                    if word_id is None:
                        word_id = next_word_id
                        next_word_id += 1
                        vocabulary[word] = word_id
                        connection.execute(
                            "INSERT INTO vocabulary(word_id, word) VALUES (?, ?)",
                            (word_id, word),
                        )
                    postings.append((word_id, doc_id, position))
                connection.executemany(
                    "INSERT INTO postings(word_id, doc_id, position) "
                    "VALUES (?, ?, ?)",
                    postings,
                )
                next_doc_id += 1
                token_count += len(tokens)
                docs_since_commit += 1
                if docs_since_commit >= commit_docs:
                    connection.execute(
                        "UPDATE build_state SET byte_offset = ?, next_doc_id = ?, "
                        "token_count = ?, source_identity = ? WHERE singleton = 1",
                        (byte_offset, next_doc_id, token_count, source_json),
                    )
                    connection.commit()
                    docs_since_commit = 0
        connection.execute(
            "UPDATE build_state SET byte_offset = ?, next_doc_id = ?, "
            "token_count = ?, source_identity = ? WHERE singleton = 1",
            (byte_offset, next_doc_id, token_count, source_json),
        )
        connection.commit()
        connection.execute(
            "CREATE INDEX IF NOT EXISTS postings_word_doc_position "
            "ON postings(word_id, doc_id, position)"
        )
        connection.commit()
        database_size = database_path.stat().st_size
        _atomic_json(
            manifest_path,
            {
                "schema": "word_based_reference_positional_index",
                "schema_version": REFERENCE_INDEX_SCHEMA_VERSION,
                "builder_version": REFERENCE_INDEX_BUILDER_VERSION,
                "storage": "sqlite_uncompressed",
                "source": source,
                "min_doc_tokens": int(min_doc_tokens),
                "document_count": int(next_doc_id),
                "vocabulary_size": int(next_word_id),
                "token_count": int(token_count),
                "database_size": int(database_size),
                "status": "complete",
            },
        )
        _atomic_json(
            complete_path,
            {
                "schema": "word_based_reference_positional_index_completion",
                "schema_version": REFERENCE_INDEX_SCHEMA_VERSION,
                "source": source,
            },
        )
    finally:
        connection.close()
    return index_root


def refresh_reference_index(
    *,
    reference_path: Path,
    index_root: Path,
    min_doc_tokens: int = 1,
) -> Path:
    """Archive an existing index and build a fresh one.

    Refresh is explicit and recoverable: the previous directory is renamed rather
    than deleted.
    """
    if index_root.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archived = index_root.with_name(f"{index_root.name}.stale-{timestamp}")
        os.replace(index_root, archived)
    return build_reference_index(
        reference_path=reference_path,
        index_root=index_root,
        min_doc_tokens=min_doc_tokens,
    )


def is_reference_index_valid(
    *,
    reference_path: Path,
    index_root: Path,
    min_doc_tokens: int = 1,
) -> bool:
    try:
        if not (index_root / "COMPLETE.json").exists():
            return False
        manifest = json.loads(
            (index_root / "manifest.json").read_text(encoding="utf-8")
        )
        return (
            manifest.get("schema_version") == REFERENCE_INDEX_SCHEMA_VERSION
            and manifest.get("status") == "complete"
            and manifest.get("source")
            == reference_source_identity(reference_path.resolve())
            and int(manifest.get("min_doc_tokens", -1)) == int(min_doc_tokens)
            and (index_root / "index.sqlite3").exists()
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def query_reference_index(
    *,
    reference_path: Path,
    index_root: Path,
    query: ReferenceCountQuery,
    min_doc_tokens: int = 1,
) -> tuple[
    dict[int, SlidingWindowCounts],
    dict[str, int],
    dict[tuple[str, str], int],
    int,
]:
    if not is_reference_index_valid(
        reference_path=reference_path,
        index_root=index_root,
        min_doc_tokens=min_doc_tokens,
    ):
        raise ValueError("Reference index is incomplete, stale, or incompatible")
    connection = sqlite3.connect(index_root / "index.sqlite3")
    try:
        lengths = np.asarray(
            [
                int(length)
                for (length,) in connection.execute(
                    "SELECT length FROM documents ORDER BY doc_id"
                )
            ],
            dtype=np.int64,
        )
        connection.execute(
            "CREATE TEMP TABLE requested_words "
            "(word TEXT PRIMARY KEY, query_word_id INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO requested_words(word, query_word_id) VALUES (?, ?)",
            [(word, idx) for idx, word in enumerate(query.target_words)],
        )
        # Size the buffers up front: a wikipedia-scale reference corpus with a
        # five-figure target vocabulary yields O(10^8) postings, and accumulating
        # those in Python lists costs several GB before the numpy copy.
        (posting_count,) = connection.execute(
            """
            SELECT COUNT(*)
            FROM requested_words AS r
            JOIN vocabulary AS v ON v.word = r.word
            JOIN postings AS p ON p.word_id = v.word_id
            """
        ).fetchone()
        rows = connection.execute(
            """
            SELECT p.doc_id, p.position, r.query_word_id
            FROM requested_words AS r
            JOIN vocabulary AS v ON v.word = r.word
            JOIN postings AS p ON p.word_id = v.word_id
            ORDER BY p.doc_id, p.position
            """
        )
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        positions = np.empty(int(posting_count), dtype=np.int64)
        word_ids = np.empty(int(posting_count), dtype=np.int64)
        cursor = 0
        current_doc = 0
        for doc_id, position, query_word_id in rows:
            doc_id = int(doc_id)
            while current_doc < doc_id:
                offsets[current_doc + 1] = cursor
                current_doc += 1
            positions[cursor] = position
            word_ids[cursor] = query_word_id
            cursor += 1
        while current_doc < len(lengths):
            offsets[current_doc + 1] = cursor
            current_doc += 1
        if cursor != len(positions):
            raise RuntimeError(
                "reference index posting count changed while querying: "
                f"expected {len(positions)}, read {cursor}"
            )
        indptr, other_ids, pair_ids = _build_pair_adjacency(query)
        kernel = _get_interval_kernel()
        result = kernel(
            lengths,
            offsets,
            positions,
            word_ids,
            np.asarray(query.window_sizes, dtype=np.int64),
            len(query.target_words),
            indptr,
            other_ids,
            pair_ids,
            query.need_document_counts,
        )
        public = _to_public_counts(
            query=query,
            word_counts=result[0],
            pair_counts=result[1],
            num_windows=result[2],
            doc_word_counts=result[3],
            doc_pair_counts=result[4],
        )
        return (*public, len(lengths))
    finally:
        connection.close()

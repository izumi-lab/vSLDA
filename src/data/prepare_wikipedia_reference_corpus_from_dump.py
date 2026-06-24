from __future__ import annotations

import argparse
import bz2
import json
import random
import shutil
import sqlite3
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from gensim.corpora.wikicorpus import extract_pages, filter_wiki

from src.core.artifacts import save_json
from src.evaluation.word_based.corpus_bundle import tokenize_documents

WORD_BASED_LANGUAGE = "english"
WORD_BASED_SEGMENTER = "delimiter"
WIKIPEDIA_DELIMITER = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a tokenized Wikipedia JSONL reference corpus from an official "
            "Wikimedia pages-articles XML dump."
        )
    )
    parser.add_argument("--language", choices=["en"], default="en")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--dump-url", required=True)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/reference/wikipedia/en/raw"),
    )
    parser.add_argument("--dump-path", type=Path, default=None)
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Defaults to data/reference/wikipedia/en/processed/enwiki-<snapshot>.tokenized.jsonl.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Defaults to data/reference/wikipedia/en/metadata/enwiki-<snapshot>.metadata.json.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--sample-mode",
        choices=["random", "first"],
        default="random",
        help=(
            "When --max-docs is set, choose a deterministic random reservoir "
            "sample from the full dump or the first kept documents."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--min-doc-tokens", type=int, default=20)
    parser.add_argument("--tokenizer", default="default")
    parser.add_argument("--delimiter", default=WIKIPEDIA_DELIMITER)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes used for tokenization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of extracted Wikipedia articles tokenized per worker task.",
    )
    return parser.parse_args()


def _resolve_default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dump_name = f"enwiki-{args.snapshot}-pages-articles-multistream.xml.bz2"
    dump_path = args.dump_path or (args.raw_dir / dump_name)
    out_jsonl = args.out_jsonl or (
        Path("data/reference/wikipedia/en/processed")
        / f"enwiki-{args.snapshot}.tokenized.jsonl"
    )
    metadata = args.metadata or (
        Path("data/reference/wikipedia/en/metadata")
        / f"enwiki-{args.snapshot}.metadata.json"
    )
    return dump_path, out_jsonl, metadata


def _download_dump(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _iter_wikipedia_articles(dump_path: Path):
    with bz2.open(dump_path, "rb") as handle:
        for title, raw_text, page_id in extract_pages(handle, filter_namespaces=("0",)):
            text = filter_wiki(raw_text)
            if not text or not text.strip():
                continue
            yield str(page_id), str(title), text


def _write_jsonl_row(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def _init_reservoir_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE reservoir (
            slot INTEGER PRIMARY KEY,
            eligible_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            row_json TEXT NOT NULL
        )
        """
    )
    return conn


def _upsert_reservoir_row(
    conn: sqlite3.Connection,
    *,
    slot: int,
    eligible_index: int,
    token_count: int,
    row: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO reservoir(slot, eligible_index, token_count, row_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(slot) DO UPDATE SET
            eligible_index=excluded.eligible_index,
            token_count=excluded.token_count,
            row_json=excluded.row_json
        """,
        (
            int(slot),
            int(eligible_index),
            int(token_count),
            json.dumps(row, ensure_ascii=False, sort_keys=True),
        ),
    )


def _reference_row(page_id: str, title: str, tokens: list[str]) -> dict[str, Any]:
    return {
        "id": page_id,
        "title": title,
        "tokens": tokens,
    }


BatchItem = tuple[str, str, str]
BatchResult = tuple[list[dict[str, Any]], int]


def _tokenize_batch_like_word_based_metrics(
    batch: list[BatchItem],
    *,
    min_token_len: int,
    min_doc_tokens: int,
    delimiter: str | None,
    tokenizer: str,
) -> BatchResult:
    if not batch:
        return [], 0
    tokenized_docs = tokenize_documents(
        [text for _page_id, _title, text in batch],
        min_token_len=min_token_len,
        language=WORD_BASED_LANGUAGE,
        delimiter=delimiter,
        segmenter=WORD_BASED_SEGMENTER,
        tokenizer=tokenizer,
        ja_replace_num=True,
    )
    rows: list[dict[str, Any]] = []
    total_tokens = 0
    for (page_id, title, _text), tokens in zip(batch, tokenized_docs):
        if len(tokens) < min_doc_tokens:
            continue
        rows.append(_reference_row(page_id, title, tokens))
        total_tokens += len(tokens)
    return rows, total_tokens


def _iter_article_batches(
    dump_path: Path,
    batch_size: int,
) -> Iterable[tuple[int, list[BatchItem]]]:
    batch: list[BatchItem] = []
    total_seen = 0
    for page_id, title, text in _iter_wikipedia_articles(dump_path):
        total_seen += 1
        batch.append((page_id, title, text))
        if len(batch) >= batch_size:
            yield total_seen, batch
            batch = []
    if batch:
        yield total_seen, batch


def _process_batch(batch: list[BatchItem], args: argparse.Namespace) -> BatchResult:
    return _tokenize_batch_like_word_based_metrics(
        batch,
        min_token_len=args.min_token_len,
        min_doc_tokens=args.min_doc_tokens,
        delimiter=args.delimiter,
        tokenizer=args.tokenizer,
    )


def _process_batch_worker(
    batch: list[BatchItem],
    min_token_len: int,
    min_doc_tokens: int,
    delimiter: str | None,
    tokenizer: str,
) -> BatchResult:
    return _tokenize_batch_like_word_based_metrics(
        batch,
        min_token_len=min_token_len,
        min_doc_tokens=min_doc_tokens,
        delimiter=delimiter,
        tokenizer=tokenizer,
    )


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _shorten_title(title: str, max_chars: int = 80) -> str:
    normalized = " ".join(title.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def _log_progress(
    *,
    label: str,
    started_monotonic: float,
    total_seen: int,
    eligible_docs: int,
    kept_docs: int,
    total_tokens: int,
    last_title: str,
    target_docs: int | None,
    sample_mode: str,
) -> None:
    elapsed = max(time.monotonic() - started_monotonic, 1e-9)
    docs_per_sec = eligible_docs / elapsed
    tokens_per_sec = total_tokens / elapsed
    target_label = "all" if target_docs is None else str(target_docs)
    if sample_mode == "random" and target_docs is not None:
        selected_docs = min(eligible_docs, target_docs, kept_docs)
    else:
        selected_docs = kept_docs
    print(
        (
            f"[{label}] elapsed={_format_elapsed(elapsed)} "
            f"pages_seen={total_seen} eligible_docs_seen={eligible_docs} "
            f"selected={selected_docs}/{target_label} sample_mode={sample_mode} "
            f"tokens={total_tokens} docs/s={docs_per_sec:.2f} "
            f"tokens/s={tokens_per_sec:.0f} last_title={_shorten_title(last_title)!r}"
        ),
        file=sys.stderr,
        flush=True,
    )


def prepare_reference_corpus(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_docs is not None and args.max_docs < 1:
        raise ValueError(f"max-docs must be >= 1 when provided, got {args.max_docs}")
    if args.min_token_len < 1:
        raise ValueError(f"min-token-len must be >= 1, got {args.min_token_len}")
    if args.min_doc_tokens < 1:
        raise ValueError(f"min-doc-tokens must be >= 1, got {args.min_doc_tokens}")
    if args.log_every < 1:
        raise ValueError(f"log-every must be >= 1, got {args.log_every}")
    if args.workers < 1:
        raise ValueError(f"workers must be >= 1, got {args.workers}")
    if args.batch_size < 1:
        raise ValueError(f"batch-size must be >= 1, got {args.batch_size}")

    dump_path, out_jsonl, metadata_path = _resolve_default_paths(args)
    if not dump_path.exists():
        if not args.download:
            raise FileNotFoundError(
                f"Wikipedia dump not found: {dump_path}. Pass --download to fetch it."
            )
        _download_dump(args.dump_url, dump_path)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC).isoformat()
    total_pages = 0
    eligible_docs = 0
    kept_docs = 0
    total_tokens = 0
    eligible_total_tokens = 0
    sampled_total_tokens = 0
    rng = random.Random(args.random_seed)
    random_sampling = args.max_docs is not None and args.sample_mode == "random"
    started_monotonic = time.monotonic()
    print(
        (
            "[dump-wikipedia] start "
            f"snapshot={args.snapshot} dump_path={dump_path} "
            f"max_docs={args.max_docs} sample_mode={args.sample_mode} "
            f"random_seed={args.random_seed} workers={args.workers} "
            f"batch_size={args.batch_size} out={out_jsonl}"
        ),
        file=sys.stderr,
        flush=True,
    )

    next_log_at = args.log_every

    def consume_result(rows_seen: int, result: BatchResult) -> bool:
        nonlocal total_pages
        nonlocal eligible_docs
        nonlocal kept_docs
        nonlocal total_tokens
        nonlocal eligible_total_tokens
        nonlocal sampled_total_tokens
        nonlocal next_log_at

        total_pages = max(total_pages, rows_seen)
        output_rows, _batch_tokens = result
        last_title = ""
        for row in output_rows:
            if (
                args.max_docs is not None
                and args.sample_mode == "first"
                and kept_docs >= args.max_docs
            ):
                break

            row_tokens = row["tokens"]
            row_token_count = len(row_tokens)
            eligible_docs += 1
            eligible_total_tokens += row_token_count

            if random_sampling:
                assert args.max_docs is not None
                assert reservoir_conn is not None
                if kept_docs < args.max_docs:
                    _upsert_reservoir_row(
                        reservoir_conn,
                        slot=kept_docs,
                        eligible_index=eligible_docs,
                        token_count=row_token_count,
                        row=row,
                    )
                    sampled_total_tokens += row_token_count
                else:
                    replace_index = rng.randrange(eligible_docs)
                    if replace_index < args.max_docs:
                        previous = reservoir_conn.execute(
                            "SELECT token_count FROM reservoir WHERE slot = ?",
                            (int(replace_index),),
                        ).fetchone()
                        if previous is not None:
                            sampled_total_tokens -= int(previous[0])
                        _upsert_reservoir_row(
                            reservoir_conn,
                            slot=replace_index,
                            eligible_index=eligible_docs,
                            token_count=row_token_count,
                            row=row,
                        )
                        sampled_total_tokens += row_token_count
            else:
                assert out_handle is not None
                _write_jsonl_row(out_handle, row)

            kept_docs += 1
            total_tokens += row_token_count
            last_title = str(row.get("title") or "")

        if kept_docs >= next_log_at:
            _log_progress(
                label="dump-wikipedia",
                started_monotonic=started_monotonic,
                total_seen=total_pages,
                eligible_docs=eligible_docs,
                kept_docs=kept_docs,
                total_tokens=total_tokens,
                last_title=last_title,
                target_docs=args.max_docs,
                sample_mode=args.sample_mode,
            )
            while kept_docs >= next_log_at:
                next_log_at += args.log_every

        return (
            args.max_docs is not None
            and args.sample_mode == "first"
            and kept_docs >= args.max_docs
        )

    reservoir_db_path: Path | None = None
    reservoir_conn: sqlite3.Connection | None = None
    out_handle = None if random_sampling else out_jsonl.open("w", encoding="utf-8")
    if random_sampling:
        reservoir_db_path = out_jsonl.parent / f".{out_jsonl.name}.reservoir.sqlite3"
        reservoir_db_path.unlink(missing_ok=True)
        reservoir_conn = _init_reservoir_db(reservoir_db_path)
    try:
        if args.workers == 1:
            for rows_seen, batch in _iter_article_batches(dump_path, args.batch_size):
                if consume_result(rows_seen, _process_batch(batch, args)):
                    break
        else:
            max_pending = max(1, args.workers * 2)
            pending = deque()
            batch_iter = iter(_iter_article_batches(dump_path, args.batch_size))
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                while True:
                    while len(pending) < max_pending:
                        try:
                            rows_seen, batch = next(batch_iter)
                        except StopIteration:
                            break
                        pending.append(
                            (
                                rows_seen,
                                executor.submit(
                                    _process_batch_worker,
                                    batch,
                                    args.min_token_len,
                                    args.min_doc_tokens,
                                    args.delimiter,
                                    args.tokenizer,
                                ),
                            )
                        )

                    if not pending:
                        break

                    rows_seen, future = pending.popleft()
                    if consume_result(rows_seen, future.result()):
                        break
    finally:
        if out_handle is not None:
            out_handle.close()

    if random_sampling:
        assert reservoir_conn is not None
        reservoir_conn.commit()
        kept_docs = int(
            reservoir_conn.execute("SELECT COUNT(*) FROM reservoir").fetchone()[0]
        )
        with out_jsonl.open("w", encoding="utf-8") as out_handle_random:
            for (row_json,) in reservoir_conn.execute(
                "SELECT row_json FROM reservoir ORDER BY eligible_index"
            ):
                out_handle_random.write(row_json)
                out_handle_random.write("\n")
        total_tokens = sampled_total_tokens
        reservoir_conn.close()
        reservoir_conn = None
        if reservoir_db_path is not None:
            reservoir_db_path.unlink(missing_ok=True)

    _log_progress(
        label="dump-wikipedia done",
        started_monotonic=started_monotonic,
        total_seen=total_pages,
        eligible_docs=eligible_docs,
        kept_docs=kept_docs,
        total_tokens=total_tokens,
        last_title="",
        target_docs=args.max_docs,
        sample_mode=args.sample_mode,
    )

    finished_at = datetime.now(UTC).isoformat()
    metadata = {
        "task": "prepare_wikipedia_reference_corpus_from_dump",
        "language": args.language,
        "snapshot": args.snapshot,
        "dump_url": args.dump_url,
        "dump_path": str(dump_path),
        "out_jsonl": str(out_jsonl),
        "started_at": started_at,
        "finished_at": finished_at,
        "max_docs": args.max_docs,
        "sample_mode": args.sample_mode,
        "random_seed": int(args.random_seed),
        "workers": int(args.workers),
        "batch_size": int(args.batch_size),
        "min_token_len": int(args.min_token_len),
        "min_doc_tokens": int(args.min_doc_tokens),
        "word_based_tokenization": True,
        "tokenization_module": "src.evaluation.word_based.corpus_bundle",
        "tokenization_function": "tokenize_documents",
        "tokenization_language": WORD_BASED_LANGUAGE,
        "segmenter": WORD_BASED_SEGMENTER,
        "tokenizer": args.tokenizer,
        "delimiter": args.delimiter,
        "total_pages_seen": int(total_pages),
        "eligible_docs": int(eligible_docs),
        "eligible_total_tokens": int(eligible_total_tokens),
        "num_docs": int(kept_docs),
        "total_tokens": int(total_tokens),
        "reservoir_storage": "sqlite" if random_sampling else None,
    }
    save_json(metadata, metadata_path)
    return metadata


def main() -> None:
    try:
        metadata = prepare_reference_corpus(_parse_args())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

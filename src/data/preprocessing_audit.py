from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.sentence_quality import (
    DEFAULT_SENTENCE_QUALITY_CONFIG,
    SentenceQualityConfig,
    assess_sentence_quality,
    repair_bad_sentence_boundaries,
    split_english_sentence_candidates,
)


@dataclass(frozen=True)
class AuditSummary:
    input_path: str
    output_path: str
    summary_path: str
    rows_seen: int
    sentences_seen: int
    review_rows: int
    decision_counts: dict[str, int]
    reason_counts: dict[str, int]
    sample_group_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary_path": self.summary_path,
            "rows_seen": int(self.rows_seen),
            "sentences_seen": int(self.sentences_seen),
            "review_rows": int(self.review_rows),
            "decision_counts": dict(self.decision_counts),
            "reason_counts": dict(self.reason_counts),
            "sample_group_counts": dict(self.sample_group_counts),
        }


def _split_saved_sentences(text: str, *, delimiter: str) -> list[str]:
    sentences: list[str] = []
    for sentence in str(text).split(delimiter):
        stripped = sentence.strip()
        if not stripped:
            continue
        repaired = repair_bad_sentence_boundaries(stripped)
        if repaired == stripped:
            sentences.append(stripped)
        else:
            sentences.extend(split_english_sentence_candidates(stripped))
    return sentences


def _sample_group(row: dict[str, object], *, short_token_threshold: int) -> str:
    if row["decision"] == "drop":
        return "drop_candidate"
    if int(row["word_token_count"]) <= short_token_threshold:
        return "short_sentence"
    if float(row["punctuation_ratio"]) >= 0.25:
        return "punctuation_heavy"
    return "random"


def _sample_review_rows(
    rows: list[dict[str, object]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return list(rows)

    rng = random.Random(seed)
    priority_rows = [
        row
        for row in rows
        if row["sample_group"] in {"drop_candidate", "short_sentence"}
    ]
    other_rows = [
        row
        for row in rows
        if row["sample_group"] not in {"drop_candidate", "short_sentence"}
    ]

    priority_limit = min(
        len(priority_rows),
        max(sample_size // 2, sample_size - len(other_rows)),
    )
    selected = rng.sample(priority_rows, priority_limit) if priority_limit else []
    remaining_size = sample_size - len(selected)
    if remaining_size > 0:
        selected_ids = {id(row) for row in selected}
        remaining_pool = [row for row in rows if id(row) not in selected_ids]
        selected.extend(
            rng.sample(remaining_pool, min(remaining_size, len(remaining_pool)))
        )

    return sorted(
        selected,
        key=lambda row: (int(row["doc_index"]), int(row["sentence_index"])),
    )


def audit_preprocessed_csv(
    input_path: Path,
    output_path: Path,
    *,
    summary_path: Path | None = None,
    text_column: str = "data",
    target_column: str | None = "target_str",
    delimiter: str = " / ",
    sample_size: int = 100,
    seed: int = 42,
    short_token_threshold: int = 3,
    config: SentenceQualityConfig = DEFAULT_SENTENCE_QUALITY_CONFIG,
) -> AuditSummary:
    frame = pd.read_csv(input_path)
    if text_column not in frame.columns:
        raise ValueError(f"text_column '{text_column}' not found in CSV {input_path}")

    resolved_summary_path = (
        summary_path
        if summary_path is not None
        else output_path.with_suffix(".summary.json")
    )
    rows: list[dict[str, object]] = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    has_target = target_column is not None and target_column in frame.columns
    for doc_index, row in frame.reset_index(drop=True).iterrows():
        target_value = row[target_column] if has_target else ""
        for sentence_index, raw_sentence in enumerate(
            _split_saved_sentences(row[text_column], delimiter=delimiter)
        ):
            decision = assess_sentence_quality(str(raw_sentence), config=config)
            decision_label = "keep" if decision.keep else "drop"
            decision_counts[decision_label] += 1
            reason_counts[decision.reason] += 1
            rows.append(
                {
                    "doc_index": int(doc_index),
                    "sentence_index": int(sentence_index),
                    "target": target_value,
                    "raw_sentence": raw_sentence,
                    "cleaned_sentence": decision.cleaned_sentence,
                    "decision": decision_label,
                    "reason": decision.reason,
                    "char_count": decision.char_count,
                    "word_token_count": decision.word_token_count,
                    "alpha_char_count": decision.alpha_char_count,
                    "punctuation_ratio": round(decision.punctuation_ratio, 6),
                    "alpha_ratio": round(decision.alpha_ratio, 6),
                    "review_decision": "",
                    "review_comment": "",
                }
            )

    for row in rows:
        row["sample_group"] = _sample_group(
            row,
            short_token_threshold=short_token_threshold,
        )

    sampled_rows = _sample_review_rows(rows, sample_size=sample_size, seed=seed)
    sample_group_counts = Counter(str(row["sample_group"]) for row in sampled_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sampled_rows).to_csv(output_path, index=False, encoding="utf-8")

    summary = AuditSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        summary_path=str(resolved_summary_path),
        rows_seen=int(len(frame)),
        sentences_seen=int(len(rows)),
        review_rows=int(len(sampled_rows)),
        decision_counts={
            key: int(value) for key, value in sorted(decision_counts.items())
        },
        reason_counts={key: int(value) for key, value in sorted(reason_counts.items())},
        sample_group_counts={
            key: int(value) for key, value in sorted(sample_group_counts.items())
        },
    )
    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_json_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary

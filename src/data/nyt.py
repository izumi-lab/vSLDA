from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.data.sentence_quality import (
    DEFAULT_SENTENCE_QUALITY_CONFIG,
    SentencePreparationStats,
    SentenceQualityConfig,
    prepare_english_document_text,
)


def _prepare_text(text: str) -> str:
    return prepare_english_document_text(text).text


def prepare_nyt(
    raw_path: Path,
    output_dir: Path,
    *,
    test_size: float = 0.4,
    random_state: int = 42,
    quality_config: SentenceQualityConfig = DEFAULT_SENTENCE_QUALITY_CONFIG,
) -> None:
    raw = pd.read_pickle(raw_path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"NYT raw pickle is missing required columns: {missing}")

    cleaned_texts: list[str] = []
    cleaned_labels: list[str] = []
    quality_stats = SentencePreparationStats()
    rows = raw.loc[:, ["text", "label"]].dropna()
    for row in tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc="Cleaning & segmenting NYT",
    ):
        prepared = prepare_english_document_text(str(row.text), config=quality_config)
        quality_stats.add_document(prepared)
        text = prepared.text
        label = str(row.label).strip()
        if not text or not label:
            continue
        cleaned_texts.append(text)
        cleaned_labels.append(label)

    if not cleaned_texts:
        raise ValueError(f"No usable NYT documents found in {raw_path}")

    x_train, x_test, train_labels, test_labels = train_test_split(
        cleaned_texts,
        cleaned_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=cleaned_labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    pd.DataFrame({"data": x_train, "target_str": train_labels}).to_csv(
        train_path,
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame({"data": x_test, "target_str": test_labels}).to_csv(
        test_path,
        index=False,
        encoding="utf-8",
    )
    manifest = {
        "dataset": "nyt",
        "layout_version": 1,
        "raw_path": str(raw_path),
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "stratify": "target_str",
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "preprocessing_version": quality_config.version,
        "sentence_quality": {
            "config": quality_config.to_json_dict(),
            "stats": quality_stats.to_json_dict(),
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

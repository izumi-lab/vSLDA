from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.data.sentence_quality import (
    DEFAULT_SENTENCE_QUALITY_CONFIG,
    SentencePreparationStats,
    SentenceQualityConfig,
    clean_english_sentence,
    prepare_english_document_text,
)


def clean_text(text: str) -> str:
    return clean_english_sentence(text)


def prepare_20newsgroups(
    output_dir: Path,
    *,
    quality_config: SentenceQualityConfig = DEFAULT_SENTENCE_QUALITY_CONFIG,
) -> None:
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )
    quality_stats = SentencePreparationStats()

    cleaned_texts: list[str] = []
    cleaned_targets: list[int] = []
    cleaned_target_labels: list[str] = []

    for text, target in tqdm(
        zip(dataset.data, dataset.target),
        total=len(dataset.data),
        desc="Cleaning & segmenting",
    ):
        prepared = prepare_english_document_text(text, config=quality_config)
        quality_stats.add_document(prepared)
        if not prepared.text:
            continue
        cleaned_texts.append(prepared.text)
        cleaned_targets.append(int(target))
        cleaned_target_labels.append(dataset.target_names[int(target)])

    x_train, x_test, y_train, y_test, train_labels, test_labels = train_test_split(
        cleaned_texts,
        cleaned_targets,
        cleaned_target_labels,
        test_size=0.4,
        random_state=42,
        stratify=cleaned_targets,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "data": x_train,
            "target": y_train,
            "target_str": train_labels,
        }
    ).to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        {
            "data": x_test,
            "target": y_test,
            "target_str": test_labels,
        }
    ).to_csv(output_dir / "test.csv", index=False, encoding="utf-8")
    manifest = {
        "dataset": "20newsgroup",
        "layout_version": 1,
        "train_csv": str(output_dir / "train.csv"),
        "test_csv": str(output_dir / "test.csv"),
        "test_size": 0.4,
        "random_state": 42,
        "stratify": "target",
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

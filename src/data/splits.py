from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.catalog import get_dataset_targets, resolve_dataset_dir
from src.data.text_processing import split_sentences


def load_dataset_split(
    dataset: str,
    split: str,
) -> tuple[Path, pd.DataFrame]:
    dataset_dir = resolve_dataset_dir(dataset)
    if dataset_dir is None:
        raise ValueError(
            f"Could not resolve dataset directory for '{dataset}' under data/."
        )
    split_path = dataset_dir / f"{split}.csv"
    return split_path, pd.read_csv(split_path)


def _load_split_frame(
    dataset: str,
    split: str,
    *,
    split_csvs: tuple[str | Path, ...] | None = None,
) -> tuple[str, pd.DataFrame]:
    if split_csvs:
        csv_paths = tuple(Path(csv_path) for csv_path in split_csvs)
        frame = pd.concat(
            [pd.read_csv(csv_path) for csv_path in csv_paths],
            ignore_index=True,
        )
        return ", ".join(str(csv_path) for csv_path in csv_paths), frame
    split_path, frame = load_dataset_split(dataset, split)
    return str(split_path), frame


def load_filtered_split_labels(
    dataset: str,
    category: str,
    split: str,
    *,
    data_column: str = "data",
    target_column: str = "target_str",
    label_schema: str = "identity",
    delimiter: str = " / ",
    split_csvs: tuple[str | Path, ...] | None = None,
) -> list[str]:
    split_path, frame = _load_split_frame(dataset, split, split_csvs=split_csvs)
    if data_column not in frame.columns:
        raise ValueError(f"data_column '{data_column}' not found in {split_path}")
    if target_column not in frame.columns:
        raise ValueError(f"target_column '{target_column}' not found in {split_path}")

    dataset_targets = (
        get_dataset_targets(
            dataset,
            target_column=target_column,
            label_schema=label_schema,
        )
        or {}
    )
    allowed_labels = dataset_targets.get(category)

    labels: list[str] = []
    for _, row in frame.iterrows():
        sentences = [
            sentence.strip()
            for sentence in str(row[data_column]).split(delimiter)
            if sentence.strip()
        ]
        if not sentences:
            continue
        label = str(row[target_column]).strip()
        if allowed_labels is not None and label not in allowed_labels:
            continue
        labels.append(label)
    return labels


def load_filtered_split_texts(
    dataset: str,
    category: str,
    split: str,
    *,
    data_column: str = "data",
    target_column: str = "target_str",
    exclude_labels: set[str] | None = None,
    split_csvs: tuple[str | Path, ...] | None = None,
) -> list[str]:
    split_path, frame = _load_split_frame(dataset, split, split_csvs=split_csvs)
    if data_column not in frame.columns:
        raise ValueError(f"data_column '{data_column}' not found in {split_path}")

    targets = get_dataset_targets(dataset) or {}
    if targets:
        if target_column not in frame.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in {split_path}"
            )
        if category not in targets:
            raise ValueError(f"Unknown category '{category}' for dataset '{dataset}'")
        frame = frame.loc[frame[target_column].isin(targets[category])]
    if exclude_labels:
        if target_column not in frame.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in {split_path}"
            )
        frame = frame.loc[~frame[target_column].isin(exclude_labels)]
    return [str(value) for value in frame[data_column].fillna("")]


def load_filtered_split_sentences(
    dataset: str,
    category: str,
    split: str,
    *,
    data_column: str = "data",
    target_column: str = "target_str",
    exclude_labels: set[str] | None = None,
    language: str = "english",
    delimiter: str | None = " / ",
    segmenter: str = "delimiter",
    split_csvs: tuple[str | Path, ...] | None = None,
) -> list[str]:
    texts = load_filtered_split_texts(
        dataset,
        category,
        split,
        data_column=data_column,
        target_column=target_column,
        exclude_labels=exclude_labels,
        split_csvs=split_csvs,
    )
    sentences: list[str] = []
    for text in texts:
        sentences.extend(
            split_sentences(
                text,
                language=language,
                delimiter=delimiter,
                segmenter=segmenter,
            )
        )
    return [sentence for sentence in sentences if sentence.strip()]

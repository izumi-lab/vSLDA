"""
Create a small subset of 20 Newsgroups CSVs for quick smoke tests.

Reads train.csv / test.csv under --src-dir, samples up to --per-label rows
per target_str (stratified) with a fixed random seed, and writes them to
--dst-dir preserving the same columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def subset_file(csv_path: Path, out_path: Path, per_label: int, seed: int) -> None:
    df = pd.read_csv(csv_path)
    # Expect target_str column to be present
    grouped = df.groupby("target_str", group_keys=False)
    sampled = grouped.apply(
        lambda g: g.sample(n=min(per_label, len(g)), random_state=seed)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False, encoding="utf-8")
    print(f"{csv_path.name}: {len(sampled)} rows -> {out_path}")


def create_20newsgroups_subset(
    *,
    src_dir: Path,
    dst_dir: Path,
    per_label: int,
    seed: int,
) -> None:
    for name in ["train", "test"]:
        subset_file(
            csv_path=src_dir / f"{name}.csv",
            out_path=dst_dir / f"{name}.csv",
            per_label=per_label,
            seed=seed,
        )

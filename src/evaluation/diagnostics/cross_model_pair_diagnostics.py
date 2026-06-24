from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.core.artifacts import load_artifact_pickle
from src.core.paths import (
    RESULTS_ROOT,
    build_archive_result_dir,
    build_baseline_doc_topic_path,
    build_latest_result_dir,
    build_vmf_doc_topic_path,
    write_latest_result_pointer,
)
from src.core.result_identity import build_condition_id, build_execution_id
from src.data.catalog import DATASET_TARGETS, resolve_dataset_dir
from src.evaluation.model_provenance import load_model_provenance_for_artifact
from src.evaluation.reporting import write_csv_rows, write_evaluation_json
from src.evaluation.schema import build_evaluation_meta
from src.evaluation.source_data import resolve_artifact_split_config
from src.utils.random import DEFAULT_RANDOM_SEED

DEFAULT_OUT_ROOT = RESULTS_ROOT / "analysis" / "vmf_vs_baseline"
BASELINE_CHOICES = [
    "bleilda",
    "ctm",
    "gaussianlda",
    "etm",
    "mvtm",
    "sentence_gaussianlda",
    "sentlda",
    "senclu",
]


def _start_execution() -> tuple[str, str]:
    started_at = datetime.now(UTC).isoformat()
    return started_at, build_execution_id(prefix="exec", started_at=started_at)


def _get_targets(dataset: str) -> dict[str, list[str]]:
    if dataset in DATASET_TARGETS:
        return DATASET_TARGETS[dataset]
    if dataset.endswith("_tiny") and dataset.replace("_tiny", "") in DATASET_TARGETS:
        return DATASET_TARGETS[dataset.replace("_tiny", "")]
    return {}


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def _row_normalize(arr: np.ndarray) -> np.ndarray:
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return arr / row_sums


def _load_docs(
    dataset: str,
    category: str,
    split: str,
    *,
    split_csvs: tuple[str, ...] | None = None,
    target_column: str = "target_str",
    text_column: str = "data",
) -> pd.DataFrame:
    if split_csvs:
        csv_paths = [Path(csv_path) for csv_path in split_csvs]
        csv_path = csv_paths[0]
        df = pd.concat(
            [pd.read_csv(csv_path) for csv_path in csv_paths], ignore_index=True
        )
    else:
        dataset_dir = resolve_dataset_dir(dataset)
        if dataset_dir is None:
            raise ValueError(
                f"Could not resolve dataset directory for '{dataset}' under data/."
            )
        csv_path = dataset_dir / f"{split}.csv"
        df = pd.read_csv(csv_path)
    targets = _get_targets(dataset)
    if category != "all" and targets:
        if target_column not in df.columns:
            raise ValueError(
                f"target_column '{target_column}' missing in {csv_path}; cannot filter category '{category}'"
            )
        if category not in targets:
            raise ValueError(f"Unknown category '{category}' for dataset '{dataset}'")
        df = df.loc[df[target_column].isin(targets[category])]
    for column in [target_column, text_column]:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' missing in {csv_path}")
    return df.reset_index(drop=True)


def _resolve_vmf_doc_topic_path(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    vmf_assignment: str,
    results_root: Path,
    data_run: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
) -> Path:
    return build_vmf_doc_topic_path(
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        assignment=vmf_assignment,
        run_name=data_run,
        condition_id=condition_id,
        num_components=num_components,
        embedding_variant=embedding_variant,
        dataset_root=results_root / "experiments" / dataset,
    )


def _resolve_baseline_doc_topic_path(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    baseline: str,
    results_root: Path,
    data_run: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
) -> Path | None:
    return build_baseline_doc_topic_path(
        model=baseline,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        data_run=data_run,
        condition_id=condition_id,
        num_components=num_components,
        embedding_variant=embedding_variant,
        baseline_root=results_root / "baselines",
    )


def _build_output_dir(
    *,
    out_root: Path,
    dataset: str,
    data_run: str,
    condition_id: str,
    condition_fingerprint: str,
    iteration: int,
    num_topics: int,
    category: str,
) -> Path:
    _ = (condition_fingerprint, iteration, num_topics)
    return out_root / dataset / data_run / category / condition_id


def _build_output_condition_id(
    *,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    split: str,
    baseline: str,
    k_neighbors: int,
    baseline_max: float,
    vmf_min: float,
    topn: int,
    unique_docs: bool,
    row_normalize: bool,
    dump_vectors: bool,
    vmf_assignment: str,
    vmf_condition_id: str | None,
    vmf_num_components: int | None,
    vmf_embedding_variant: str | None,
    baseline_condition_id: str | None,
    baseline_num_components: int | None,
    baseline_embedding_variant: str | None,
) -> tuple[str, str]:
    return build_condition_id(
        iteration=int(iteration),
        num_topics=int(num_topics),
        fingerprint_payload={
            "task": "cross_model_pair_diagnostics",
            "dataset": dataset,
            "data_run": data_run,
            "category": category,
            "iteration": int(iteration),
            "num_topics": int(num_topics),
            "split": split,
            "baseline": baseline,
            "k_neighbors": int(k_neighbors),
            "baseline_max": float(baseline_max),
            "vmf_min": float(vmf_min),
            "topn": int(topn),
            "unique_docs": bool(unique_docs),
            "row_normalize": bool(row_normalize),
            "dump_vectors": bool(dump_vectors),
            "vmf_assignment": vmf_assignment,
            "vmf_condition_id": vmf_condition_id,
            "vmf_num_components": (
                None if vmf_num_components is None else int(vmf_num_components)
            ),
            "vmf_embedding_variant": vmf_embedding_variant,
            "baseline_condition_id": baseline_condition_id,
            "baseline_num_components": (
                None
                if baseline_num_components is None
                else int(baseline_num_components)
            ),
            "baseline_embedding_variant": baseline_embedding_variant,
        },
        extra_labels=[baseline, split],
    )


def _stringify_csv_records(
    records: list[dict[str, object]],
    *,
    dump_vectors: bool,
) -> list[dict[str, object]]:
    if not dump_vectors:
        return [dict(record) for record in records]
    csv_records: list[dict[str, object]] = []
    for record in records:
        row = dict(record)
        row["vmf_vec_i"] = json.dumps(row["vmf_vec_i"], ensure_ascii=False)
        row["vmf_vec_j"] = json.dumps(row["vmf_vec_j"], ensure_ascii=False)
        row["baseline_vec_i"] = json.dumps(row["baseline_vec_i"], ensure_ascii=False)
        row["baseline_vec_j"] = json.dumps(row["baseline_vec_j"], ensure_ascii=False)
        csv_records.append(row)
    return csv_records


def _align_legacy_bleilda_rows(
    *,
    baseline: str,
    baseline_path: Path,
    docs: pd.DataFrame,
    vmf: np.ndarray,
    baseline_arr: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, bool]:
    doc_indices = np.arange(len(docs), dtype=int)
    if (
        baseline != "bleilda"
        or len(docs) == baseline_arr.shape[0]
        or len(docs) != vmf.shape[0]
    ):
        return docs, vmf, doc_indices, False

    preprocessed_path = baseline_path.with_name("preprocessed_corpus.pkl")
    if not preprocessed_path.exists():
        return docs, vmf, doc_indices, False

    preprocessed = load_artifact_pickle(preprocessed_path)
    if len(preprocessed) != len(docs):
        return docs, vmf, doc_indices, False

    nonempty_indices = np.asarray(
        [
            row_index
            for row_index, doc in enumerate(preprocessed)
            if getattr(doc, "document_tokens", None)
        ],
        dtype=int,
    )
    if nonempty_indices.shape[0] != baseline_arr.shape[0]:
        return docs, vmf, doc_indices, False

    return (
        docs.iloc[nonempty_indices].reset_index(drop=True),
        vmf[nonempty_indices],
        nonempty_indices,
        True,
    )


def run_vmf_vs_baseline_pair_analysis(
    *,
    dataset: str,
    category: str,
    iteration: int,
    num_topics: int,
    split: str = "train",
    baseline: str = "bleilda",
    k_neighbors: int = 30,
    baseline_max: float = 0.05,
    vmf_min: float = 0.6,
    topn: int = 10,
    unique_docs: bool = False,
    row_normalize: bool = True,
    dump_vectors: bool = False,
    seed: int | None = DEFAULT_RANDOM_SEED,
    vmf_assignment: str = "hard",
    vmf_condition_id: str | None = None,
    vmf_num_components: int | None = None,
    vmf_embedding_variant: str | None = None,
    baseline_condition_id: str | None = None,
    baseline_num_components: int | None = None,
    baseline_embedding_variant: str | None = None,
    data_run: str = "default",
    results_root: Path = RESULTS_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
) -> Path:
    vmf_path = _resolve_vmf_doc_topic_path(
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        vmf_assignment=vmf_assignment,
        results_root=results_root,
        data_run=data_run,
        condition_id=vmf_condition_id,
        num_components=vmf_num_components,
        embedding_variant=vmf_embedding_variant,
    )
    baseline_path = _resolve_baseline_doc_topic_path(
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        baseline=baseline,
        results_root=results_root,
        data_run=data_run,
        condition_id=baseline_condition_id,
        num_components=baseline_num_components,
        embedding_variant=baseline_embedding_variant,
    )
    if baseline_path is None:
        raise ValueError(
            f"Baseline '{baseline}' does not have '{split}' outputs. "
            "Use --split test for ctm/gaussianlda/etm/mvtm/sentence_gaussianlda/sentlda."
        )
    if not vmf_path.exists():
        raise FileNotFoundError(f"vMF doc-topic not found: {vmf_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline doc-topic not found: {baseline_path}")

    vmf = np.asarray(load_artifact_pickle(vmf_path), dtype=float)
    baseline_arr = np.asarray(load_artifact_pickle(baseline_path), dtype=float)

    if row_normalize:
        vmf = _row_normalize(vmf)
        baseline_arr = _row_normalize(baseline_arr)

    artifact_split = resolve_artifact_split_config(vmf_path, split=split)
    docs = _load_docs(
        dataset,
        category,
        split,
        split_csvs=artifact_split.split_csvs,
        target_column=artifact_split.target_column,
        text_column=artifact_split.text_column,
    )
    docs, vmf, doc_indices, legacy_bleilda_row_alignment = _align_legacy_bleilda_rows(
        baseline=baseline,
        baseline_path=baseline_path,
        docs=docs,
        vmf=vmf,
        baseline_arr=baseline_arr,
    )
    if len(docs) != vmf.shape[0] or len(docs) != baseline_arr.shape[0]:
        raise ValueError(
            "doc-topic rows do not match filtered CSV length: "
            f"docs={len(docs)} vmf={vmf.shape[0]} baseline={baseline_arr.shape[0]}"
        )

    vmf_norm = _normalize_rows(vmf)
    baseline_norm = _normalize_rows(baseline_arr)

    neighbor_count = min(max(2, int(k_neighbors) + 1), baseline_norm.shape[0])
    nbrs = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine").fit(
        baseline_norm
    )
    baseline_dist, baseline_idx = nbrs.kneighbors(baseline_norm, return_distance=True)

    pairs: list[tuple[int, int, float, float]] = []
    for i in range(baseline_norm.shape[0]):
        for n in range(1, baseline_idx.shape[1]):
            j = int(baseline_idx[i, n])
            if i >= j:
                continue
            d_baseline = float(baseline_dist[i, n])
            if d_baseline > baseline_max:
                continue
            d_vmf = float(1.0 - float(np.dot(vmf_norm[i], vmf_norm[j])))
            if d_vmf < vmf_min:
                continue
            pairs.append((i, j, d_vmf, d_baseline))

    pairs.sort(key=lambda x: (-(x[2] - x[3]), -x[2], x[3]))

    selected: list[tuple[int, int, float, float]] = []
    used_docs: set[int] = set()
    for item in pairs:
        if unique_docs and (item[0] in used_docs or item[1] in used_docs):
            continue
        selected.append(item)
        used_docs.add(item[0])
        used_docs.add(item[1])
        if len(selected) >= topn:
            break

    records: list[dict[str, object]] = []
    for rank, (i, j, d_vmf, d_baseline) in enumerate(selected, start=1):
        record: dict[str, object] = {
            "rank": rank,
            "i": int(doc_indices[i]),
            "j": int(doc_indices[j]),
            "d_vmf": d_vmf,
            "d_baseline": d_baseline,
            "score": d_vmf - d_baseline,
            "baseline": baseline,
            "label_i": docs.loc[i].get(artifact_split.target_column, None),
            "label_j": docs.loc[j].get(artifact_split.target_column, None),
            "text_i": docs.loc[i].get(artifact_split.text_column, ""),
            "text_j": docs.loc[j].get(artifact_split.text_column, ""),
        }
        if dump_vectors:
            record["vmf_vec_i"] = vmf[i].tolist()
            record["vmf_vec_j"] = vmf[j].tolist()
            record["baseline_vec_i"] = baseline_arr[i].tolist()
            record["baseline_vec_j"] = baseline_arr[j].tolist()
        records.append(record)

    condition_id, condition_fingerprint = _build_output_condition_id(
        dataset=dataset,
        data_run=artifact_split.data_run or data_run,
        category=category,
        iteration=iteration,
        num_topics=num_topics,
        split=split,
        baseline=baseline,
        k_neighbors=k_neighbors,
        baseline_max=baseline_max,
        vmf_min=vmf_min,
        topn=topn,
        unique_docs=unique_docs,
        row_normalize=row_normalize,
        dump_vectors=dump_vectors,
        vmf_assignment=vmf_assignment,
        vmf_condition_id=vmf_condition_id,
        vmf_num_components=vmf_num_components,
        vmf_embedding_variant=vmf_embedding_variant,
        baseline_condition_id=baseline_condition_id,
        baseline_num_components=baseline_num_components,
        baseline_embedding_variant=baseline_embedding_variant,
    )
    resolved_data_run = artifact_split.data_run or data_run
    display_key = condition_id
    started_at, execution_id = _start_execution()
    out_dir = build_archive_result_dir(
        base_root=out_root,
        dataset=dataset,
        data_run=resolved_data_run,
        category=category,
        display_key=display_key,
        started_at=started_at,
        execution_id=execution_id,
    )
    latest_dir = build_latest_result_dir(
        base_root=out_root,
        dataset=dataset,
        data_run=resolved_data_run,
        category=category,
        display_key=display_key,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"pairs_{baseline}_{split}.json"
    csv_path = out_dir / f"pairs_{baseline}_{split}.csv"
    meta = build_evaluation_meta(
        task="cross_model_pair_diagnostics",
        output_kind="payload",
        dataset=dataset,
        data_run=resolved_data_run,
        category=category,
        condition_id=condition_id,
        display_key=display_key,
        condition_fingerprint=condition_fingerprint,
        started_at=started_at,
        execution_id=execution_id,
        archive_dir=str(out_dir),
        latest_dir=str(latest_dir),
        split=split,
        iteration=int(iteration),
        num_topics=int(num_topics),
        baseline=baseline,
        vmf_assignment=vmf_assignment,
        vmf_condition_id=vmf_condition_id,
        vmf_num_components=vmf_num_components,
        vmf_embedding_variant=vmf_embedding_variant,
        baseline_condition_id=baseline_condition_id,
        baseline_num_components=baseline_num_components,
        baseline_embedding_variant=baseline_embedding_variant,
        k_neighbors=int(k_neighbors),
        baseline_max=float(baseline_max),
        vmf_min=float(vmf_min),
        topn=int(topn),
        unique_docs=bool(unique_docs),
        row_normalize=bool(row_normalize),
        dump_vectors=bool(dump_vectors),
        seed=seed,
        vmf_doc_topic_path=str(vmf_path),
        baseline_doc_topic_path=str(baseline_path),
        pair_count=len(records),
        legacy_bleilda_row_alignment=legacy_bleilda_row_alignment,
        model_provenance={
            "vmf_sentence_lda": load_model_provenance_for_artifact(
                vmf_path,
                model_key="vmf_sentence_lda",
            ),
            baseline: load_model_provenance_for_artifact(
                baseline_path,
                model_key=baseline,
            ),
        },
    )
    write_evaluation_json(
        meta=meta,
        results={"pairs": records},
        path=json_path,
    )

    fieldnames = [
        "rank",
        "i",
        "j",
        "d_vmf",
        "d_baseline",
        "score",
        "baseline",
        "label_i",
        "label_j",
        "text_i",
        "text_j",
    ]
    if dump_vectors:
        fieldnames.extend(
            [
                "vmf_vec_i",
                "vmf_vec_j",
                "baseline_vec_i",
                "baseline_vec_j",
            ]
        )
    write_csv_rows(
        fieldnames=fieldnames,
        rows=_stringify_csv_records(records, dump_vectors=dump_vectors),
        path=csv_path,
    )
    write_latest_result_pointer(
        base_root=out_root,
        task="cross_model_pair_diagnostics",
        dataset=dataset,
        data_run=resolved_data_run,
        category=category,
        display_key=display_key,
        archive_dir=out_dir,
        started_at=started_at,
        execution_id=execution_id,
        condition_fingerprint=condition_fingerprint,
        artifacts={
            "json": json_path.name,
            "csv": csv_path.name,
        },
    )
    return json_path


run_cross_model_pair_diagnostics = run_vmf_vs_baseline_pair_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find doc pairs that are far in vMF space but close in a baseline space."
    )
    parser.add_argument("--dataset", default="20newsgroup")
    parser.add_argument("--category", default="computer")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--num-topics", type=int, default=20)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--data-run", default="default")
    parser.add_argument("--vmf-assignment", choices=["hard", "soft"], default="hard")
    parser.add_argument("--vmf-condition-id", default=None)
    parser.add_argument("--vmf-num-components", type=int, default=None)
    parser.add_argument("--vmf-embedding-variant", default=None)
    parser.add_argument("--baseline-condition-id", default=None)
    parser.add_argument("--baseline-num-components", type=int, default=None)
    parser.add_argument("--baseline-embedding-variant", default=None)
    parser.add_argument(
        "--baseline",
        dest="baseline",
        default="bleilda",
        choices=BASELINE_CHOICES,
    )
    parser.add_argument(
        "--lda-baseline",
        dest="baseline",
        choices=BASELINE_CHOICES,
        help="Deprecated alias for --baseline.",
    )
    parser.add_argument("--k-neighbors", type=int, default=30)
    parser.add_argument("--baseline-max", type=float, default=0.05)
    parser.add_argument(
        "--lda-max",
        dest="baseline_max",
        type=float,
        help="Deprecated alias for --baseline-max.",
    )
    parser.add_argument("--vmf-min", type=float, default=0.6)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--unique-docs", action="store_true")
    parser.add_argument("--no-row-normalize", action="store_true")
    parser.add_argument(
        "--dump-vectors",
        action="store_true",
        help="If set, include raw doc-topic vectors in outputs (CSV will store JSON strings).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = run_vmf_vs_baseline_pair_analysis(
        dataset=args.dataset,
        category=args.category,
        iteration=args.iteration,
        num_topics=args.num_topics,
        split=args.split,
        baseline=args.baseline,
        k_neighbors=args.k_neighbors,
        baseline_max=args.baseline_max,
        vmf_min=args.vmf_min,
        topn=args.topn,
        unique_docs=args.unique_docs,
        row_normalize=not args.no_row_normalize,
        dump_vectors=args.dump_vectors,
        seed=args.seed,
        vmf_assignment=args.vmf_assignment,
        vmf_condition_id=args.vmf_condition_id,
        vmf_num_components=args.vmf_num_components,
        vmf_embedding_variant=args.vmf_embedding_variant,
        baseline_condition_id=args.baseline_condition_id,
        baseline_num_components=args.baseline_num_components,
        baseline_embedding_variant=args.baseline_embedding_variant,
        data_run=args.data_run,
        results_root=args.results_root,
        out_root=args.out_root,
    )
    print(f"[ok] {out_path}")
    print(f"[ok] {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from src.core.artifacts import load_artifact_pickle, save_json
from src.core.paths import (
    RESULTS_ROOT,
    build_archive_result_dir,
    build_baseline_doc_topic_path,
    build_latest_result_dir,
    build_result_display_key,
    build_vmf_doc_topic_path,
    write_latest_result_pointer,
)
from src.core.result_identity import build_condition_id, build_execution_id
from src.data.splits import load_filtered_split_labels
from src.evaluation.model_provenance import load_model_provenance_for_artifact
from src.evaluation.reporting import write_csv_rows, write_evaluation_json
from src.evaluation.schema import build_evaluation_meta, build_evaluation_payload
from src.evaluation.source_data import resolve_artifact_split_config
from src.utils.logging import get_logger

ModelName = Literal["vmf_sentence_lda", "bleilda"]
SplitName = Literal["train", "test"]
VmfAssignment = Literal["soft", "hard"]

LOGGER = get_logger(__name__)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return arr / row_sums


def _load_doc_topics(path: Path) -> np.ndarray:
    arr = np.asarray(load_artifact_pickle(path), dtype=float)
    return _normalize_rows(arr)


def _resolve_doc_topic_path(
    *,
    model: ModelName,
    results_root: Path,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: SplitName,
    vmf_assignment: VmfAssignment,
    data_run: str = "default",
) -> Path:
    if model == "vmf_sentence_lda":
        return build_vmf_doc_topic_path(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            assignment=vmf_assignment,
            run_name=data_run,
            dataset_root=results_root / "experiments" / dataset,
        )

    path = build_baseline_doc_topic_path(
        model="bleilda",
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        data_run=data_run,
        baseline_root=results_root / "baselines",
    )
    if path is None:
        raise ValueError(f"Unsupported split '{split}' for model '{model}'.")
    return path


def _load_labels(
    dataset: str,
    category: str,
    split: SplitName,
    *,
    data_column: str = "data",
    target_column: str = "target_str",
    label_schema: str = "identity",
    delimiter: str = " / ",
    split_csvs: tuple[str, ...] | None = None,
) -> list[str]:
    return load_filtered_split_labels(
        dataset,
        category,
        split,
        data_column=data_column,
        target_column=target_column,
        label_schema=label_schema,
        delimiter=delimiter,
        split_csvs=split_csvs,
    )


def _build_output_identity(
    *,
    model: ModelName,
    dataset: str,
    data_run: str,
    category: str,
    split: SplitName,
    iteration: int,
    num_topics: int,
    top_n: int,
    sort_by: str,
    pmi_eps: float,
    min_docs_per_label: int,
    vmf_assignment: VmfAssignment,
) -> tuple[str, str]:
    display_key = build_result_display_key(
        num_topics=int(num_topics),
        iteration=int(iteration),
        extra_labels=[model, split],
    )
    _, condition_fingerprint = build_condition_id(
        iteration=int(iteration),
        num_topics=int(num_topics),
        fingerprint_payload={
            "task": "word_based_label_profile",
            "model": model,
            "dataset": dataset,
            "data_run": data_run,
            "category": category,
            "split": split,
            "iteration": int(iteration),
            "num_topics": int(num_topics),
            "top_n": int(top_n),
            "sort_by": sort_by,
            "pmi_eps": float(pmi_eps),
            "min_docs_per_label": int(min_docs_per_label),
            "vmf_assignment": vmf_assignment,
        },
        extra_labels=[model, split],
    )
    return display_key, condition_fingerprint


def run_label_topic_profile(
    *,
    model: ModelName,
    dataset: str,
    category: str,
    split: SplitName,
    iteration: int,
    num_topics: int,
    top_n: int = 5,
    sort_by: Literal["ratio", "pmi"] = "ratio",
    pmi_eps: float = 1e-12,
    min_docs_per_label: int = 1,
    vmf_assignment: VmfAssignment = "soft",
    data_run: str = "default",
    results_root: Path = RESULTS_ROOT,
    data_column: str = "data",
    target_column: str = "target_str",
    label_schema: str = "identity",
    delimiter: str = " / ",
    out_json: Path | None = None,
    out_csv: Path | None = None,
) -> dict[str, object]:
    doc_topic_path = _resolve_doc_topic_path(
        model=model,
        results_root=results_root,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        vmf_assignment=vmf_assignment,
        data_run=data_run,
    )
    if not doc_topic_path.exists():
        raise FileNotFoundError(
            f"Doc-topic file not found: {doc_topic_path}. "
            f"Check --results-root and path settings."
        )
    model_provenance = load_model_provenance_for_artifact(
        doc_topic_path,
        model_key=model,
    )
    artifact_split = resolve_artifact_split_config(
        doc_topic_path,
        split=split,
        default_text_column=data_column,
        default_target_column=target_column,
    )

    doc_topics = _load_doc_topics(doc_topic_path)
    labels = _load_labels(
        dataset,
        category,
        split,
        data_column=artifact_split.text_column,
        target_column=artifact_split.target_column,
        label_schema=label_schema,
        delimiter=delimiter,
        split_csvs=artifact_split.split_csvs,
    )

    if doc_topics.shape[0] != len(labels):
        raise ValueError(
            f"Length mismatch: doc_topics has {doc_topics.shape[0]} rows but labels has {len(labels)} rows. "
            "Ensure doc-topic output and CSV filtering settings match."
        )

    label_counts = Counter(labels)
    label_order = sorted(label_counts.keys(), key=lambda x: (-label_counts[x], x))
    global_mean = doc_topics.mean(axis=0)

    LOGGER.info(
        "label_topic_profile model=%s dataset=%s category=%s split=%s docs=%s topics=%s",
        model,
        dataset,
        category,
        split,
        doc_topics.shape[0],
        doc_topics.shape[1],
    )
    LOGGER.info("label_topic_profile labels=%s", len(label_order))

    global_top_ids = np.argsort(-global_mean)[: min(top_n, doc_topics.shape[1])]
    LOGGER.info("label_topic_profile global top topics")
    for rank, topic_id in enumerate(global_top_ids, start=1):
        LOGGER.info(
            "global rank=%s topic=%s mean=%.8f",
            rank,
            int(topic_id),
            float(global_mean[topic_id]),
        )

    per_label_rows: list[dict[str, object]] = []
    per_label_result: list[dict[str, object]] = []

    for label in label_order:
        count = label_counts[label]
        if count < min_docs_per_label:
            continue
        idx = np.where(np.asarray(labels) == label)[0]
        mean_weights = doc_topics[idx].mean(axis=0)
        if sort_by == "ratio":
            score = mean_weights
            metric_name = "mean"
        else:
            score = np.log((mean_weights + pmi_eps) / (global_mean + pmi_eps))
            metric_name = "pmi"

        top_ids = np.argsort(-score)[: min(top_n, mean_weights.shape[0])]

        LOGGER.info("label=%s docs=%s", label, count)
        top_topics_payload: list[dict[str, object]] = []
        for rank, topic_id in enumerate(top_ids, start=1):
            mean_val = float(mean_weights[topic_id])
            score_val = float(score[topic_id])
            global_val = float(global_mean[topic_id])
            LOGGER.info(
                "label=%s rank=%s topic=%s %s=%.8f mean=%.8f global=%.8f",
                label,
                rank,
                int(topic_id),
                metric_name,
                score_val,
                mean_val,
                global_val,
            )
            row = {
                "label": label,
                "docs": int(count),
                "rank": rank,
                "topic_id": int(topic_id),
                "mean_weight": mean_val,
                "global_mean_weight": global_val,
                "score_type": sort_by,
                "score_value": score_val,
            }
            per_label_rows.append(row)
            top_topics_payload.append(
                {
                    "rank": rank,
                    "topic_id": int(topic_id),
                    "mean_weight": mean_val,
                    "global_mean_weight": global_val,
                    "score_type": sort_by,
                    "score_value": score_val,
                }
            )

        per_label_result.append(
            {
                "label": label,
                "docs": int(count),
                "top_topics": top_topics_payload,
            }
        )

    resolved_data_run = artifact_split.data_run or data_run
    display_key, condition_fingerprint = _build_output_identity(
        model=model,
        dataset=dataset,
        data_run=resolved_data_run,
        category=category,
        split=split,
        iteration=iteration,
        num_topics=num_topics,
        top_n=top_n,
        sort_by=sort_by,
        pmi_eps=pmi_eps,
        min_docs_per_label=min_docs_per_label,
        vmf_assignment=vmf_assignment,
    )
    analysis_root = results_root / "topic_analysis" / "label_profile"
    started_at = datetime.now(UTC).isoformat()
    execution_id = build_execution_id(prefix="exec", started_at=started_at)
    uses_default_output_layout = out_json is None and out_csv is None
    if uses_default_output_layout:
        archive_out_dir = build_archive_result_dir(
            base_root=analysis_root,
            dataset=dataset,
            data_run=resolved_data_run,
            category=category,
            display_key=display_key,
            started_at=started_at,
            execution_id=execution_id,
        )
        resolved_out_json = archive_out_dir / "label_topic_profile.json"
        resolved_out_csv = archive_out_dir / "label_topic_profile.csv"
        latest_out_dir = build_latest_result_dir(
            base_root=analysis_root,
            dataset=dataset,
            data_run=resolved_data_run,
            category=category,
            display_key=display_key,
        )
    else:
        archive_out_dir = None
        latest_out_dir = None
        custom_root = (
            out_json.parent
            if out_json is not None
            else out_csv.parent if out_csv is not None else analysis_root
        )
        resolved_out_json = out_json or (custom_root / "label_topic_profile.json")
        resolved_out_csv = out_csv or (custom_root / "label_topic_profile.csv")
    resolved_metadata_path = resolved_out_json.parent / "metadata.json"

    meta: dict[str, object] = build_evaluation_meta(
        task="word_based_label_profile",
        output_kind="payload",
        model=model,
        dataset=dataset,
        category=category,
        split=split,
        data_run=resolved_data_run,
        iteration=int(iteration),
        num_topics=int(num_topics),
        condition_id=display_key,
        condition_fingerprint=condition_fingerprint,
        display_key=display_key,
        started_at=started_at,
        execution_id=execution_id,
        results_root=str(results_root),
        doc_topic_path=str(doc_topic_path),
        vmf_assignment=vmf_assignment if model == "vmf_sentence_lda" else None,
        target_column=artifact_split.target_column,
        label_schema=label_schema,
        sort_by=sort_by,
        pmi_eps=float(pmi_eps),
        num_docs=int(doc_topics.shape[0]),
        model_provenance=model_provenance,
        latest_dir=None if latest_out_dir is None else str(latest_out_dir),
    )
    results: dict[str, object] = {
        "global_top_topics": [
            {
                "rank": rank,
                "topic_id": int(topic_id),
                "mean_weight": float(global_mean[topic_id]),
            }
            for rank, topic_id in enumerate(global_top_ids, start=1)
        ],
        "labels": per_label_result,
    }
    report: dict[str, object] = build_evaluation_payload(meta=meta, results=results)

    write_evaluation_json(meta=meta, results=results, path=resolved_out_json)
    LOGGER.info("Wrote JSON report to %s", resolved_out_json)
    save_json(meta, resolved_metadata_path)
    LOGGER.info("Wrote metadata to %s", resolved_metadata_path)

    csv_rows = [
        {
            "label": row["label"],
            "docs": row["docs"],
            "rank": row["rank"],
            "topic_id": row["topic_id"],
            "mean_weight": f"{row['mean_weight']:.12f}",
            "global_mean_weight": f"{row['global_mean_weight']:.12f}",
            "score_type": row["score_type"],
            "score_value": f"{row['score_value']:.12f}",
        }
        for row in per_label_rows
    ]
    write_csv_rows(
        fieldnames=[
            "label",
            "docs",
            "rank",
            "topic_id",
            "mean_weight",
            "global_mean_weight",
            "score_type",
            "score_value",
        ],
        rows=csv_rows,
        path=resolved_out_csv,
    )
    LOGGER.info("Wrote CSV report to %s", resolved_out_csv)

    if uses_default_output_layout and archive_out_dir is not None:
        pointer_path = write_latest_result_pointer(
            base_root=analysis_root,
            task="word_based_label_profile",
            dataset=dataset,
            data_run=resolved_data_run,
            category=category,
            display_key=display_key,
            archive_dir=archive_out_dir,
            started_at=started_at,
            execution_id=execution_id,
            condition_fingerprint=condition_fingerprint,
            artifacts={
                "json": resolved_out_json.name,
                "csv": resolved_out_csv.name,
                "metadata": resolved_metadata_path.name,
            },
        )
        LOGGER.info(
            "Updated latest pointer at %s",
            pointer_path,
        )
        report["_meta"] = meta

    return report


run_word_based_label_profile = run_label_topic_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze topic proportions per class label used in classification."
    )
    parser.add_argument(
        "--model", choices=["vmf_sentence_lda", "bleilda"], required=True
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--category", default="all")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--num-topics", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--sort-by", choices=["ratio", "pmi"], default="ratio")
    parser.add_argument("--pmi-eps", type=float, default=1e-12)
    parser.add_argument("--min-docs-per-label", type=int, default=1)
    parser.add_argument("--vmf-assignment", choices=["soft", "hard"], default="soft")
    parser.add_argument("--data-run", default="default")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--data-column", default="data")
    parser.add_argument("--target-column", default="target_str")
    parser.add_argument("--label-schema", default="identity")
    parser.add_argument("--delimiter", default=" / ")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_label_topic_profile(
        model=args.model,
        dataset=args.dataset,
        category=args.category,
        split=args.split,
        iteration=args.iteration,
        num_topics=args.num_topics,
        top_n=args.top_n,
        sort_by=args.sort_by,
        pmi_eps=args.pmi_eps,
        min_docs_per_label=args.min_docs_per_label,
        vmf_assignment=args.vmf_assignment,
        data_run=args.data_run,
        results_root=args.results_root,
        data_column=args.data_column,
        target_column=args.target_column,
        label_schema=args.label_schema,
        delimiter=args.delimiter,
        out_json=args.out_json,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()

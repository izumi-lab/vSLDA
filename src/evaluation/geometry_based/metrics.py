from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.core.artifacts import load_artifact_pickle, save_json
from src.core.paths import (
    RESULTS_ROOT,
    build_archive_result_dir,
    build_latest_result_dir,
    resolve_baseline_condition_dir,
    resolve_project_path,
    resolve_vmf_experiment_dir,
    write_latest_result_pointer,
)
from src.core.result_identity import build_condition_id, build_execution_id
from src.evaluation.model_provenance import load_model_provenance
from src.evaluation.reporting import (
    write_csv_rows,
    write_evaluation_json,
    write_tabular_report_json,
)
from src.evaluation.schema import build_evaluation_meta
from src.utils.encoder_profiles import embedding_variant_base, encoder_model_alias
from src.utils.logging import get_logger

ModelType = Literal["vmf", "gaussian"]

ANALYSIS_ROOT = RESULTS_ROOT / "topic_analysis"
DEFAULT_OUT_ROOT = ANALYSIS_ROOT / "geometry_based"
logger = get_logger(__name__)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _uses_default_output_layout(out_root: Path) -> bool:
    return resolve_project_path(out_root) == DEFAULT_OUT_ROOT


def build_result_dir(
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> Path:
    if model == "vmf":
        return resolve_vmf_experiment_dir(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            run_name=data_run,
            embedding_variant=embedding_variant,
        )
    return (
        resolve_baseline_condition_dir(
            model="sentence_gaussianlda",
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
        / "params"
    )


def _build_output_condition_id(
    *,
    model: ModelType,
    dataset: str,
    data_run: str,
    category: str,
    iterations: Sequence[int],
    num_topics: int,
    dup_threshold: float,
    embedding_variant: str | None = None,
) -> tuple[str, str]:
    extra_labels = [model]
    if embedding_variant not in {None, ""}:
        extra_labels.append(str(embedding_variant))
    return build_condition_id(
        iteration=int(min(iterations)),
        num_topics=int(num_topics),
        fingerprint_payload={
            "task": "geometry_based_metrics",
            "model": model,
            "dataset": dataset,
            "data_run": data_run,
            "category": category,
            "iterations": [int(value) for value in iterations],
            "num_topics": int(num_topics),
            "dup_threshold": float(dup_threshold),
            "embedding_variant": embedding_variant,
        },
        extra_labels=extra_labels,
    )


def _resolve_requested_embedding_variant(
    *,
    embedding_variant: str | None,
    encoder_model: str | None,
) -> str | None:
    requested_variant = (
        None if embedding_variant in {None, ""} else str(embedding_variant)
    )
    if encoder_model in {None, ""}:
        return requested_variant

    encoder_variant = encoder_model_alias(str(encoder_model))
    if requested_variant is not None:
        requested_base = embedding_variant_base(requested_variant)
        encoder_base = embedding_variant_base(encoder_variant)
        if requested_base != encoder_base:
            raise ValueError(
                "encoder_model and embedding_variant mismatch: "
                f"encoder_model='{encoder_model}' resolves to '{encoder_base}', "
                f"but embedding_variant='{requested_variant}' resolves to "
                f"'{requested_base}'."
            )
        return requested_variant
    return encoder_variant


def _effective_embedding_variant(
    model: ModelType,
    embedding_variant: str | None,
) -> str | None:
    if embedding_variant in {None, ""}:
        return None
    variant = str(embedding_variant)
    if model == "gaussian" and not variant.endswith(("_raw", "_norm")):
        return f"{variant}_raw"
    return variant


def load_topic_vectors(
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> np.ndarray:
    result_dir = build_result_dir(
        model,
        dataset,
        iteration,
        num_topics,
        category,
        data_run=data_run,
        embedding_variant=embedding_variant,
    )
    path = result_dir / ("topic_means.pkl" if model == "vmf" else "table_means.pkl")
    arr = np.asarray(load_artifact_pickle(path), dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for topic vectors, got shape {arr.shape}")
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norm


def compute_cosine_similarity(topic_vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(topic_vectors, axis=1, keepdims=True) + 1e-12
    normalized = topic_vectors / norm
    return normalized @ normalized.T


def _to_unit_interval(
    matrix: np.ndarray, min_val: float = -1.0, max_val: float = 1.0
) -> np.ndarray:
    scaled = (matrix - min_val) / (max_val - min_val + 1e-12)
    return np.clip(scaled, 0.0, 1.0)


def round_sigfigs(value: float, sig: int = 4) -> float:
    if np.isnan(value) or np.isinf(value):
        return float(value)
    return float(f"{value:.{sig}g}")


def compute_overlap_metrics(
    cosine: np.ndarray, dup_threshold: float
) -> dict[str, float | str]:
    k = cosine.shape[0]
    if k < 2:
        return {
            "num_topics": float(k),
            "mean_pairwise_cosine": float("nan"),
            "diversity_score": float("nan"),
            "max_pairwise_cosine": float("nan"),
            "dup_threshold": float(dup_threshold),
            "num_pairs_above_threshold": float("nan"),
            "note": "Need at least two topics for pairwise metrics.",
        }

    tri_idx = np.triu_indices(k, k=1)
    off_cosine = cosine[tri_idx]
    mean_cos = float(off_cosine.mean())
    max_cos = float(off_cosine.max())
    num_pairs_above = int((off_cosine >= dup_threshold).sum())
    return {
        "num_topics": float(k),
        "mean_pairwise_cosine": mean_cos,
        "diversity_score": float(1.0 - mean_cos),
        "max_pairwise_cosine": max_cos,
        "dup_threshold": float(dup_threshold),
        "num_pairs_above_threshold": float(num_pairs_above),
    }


def aggregate_metrics(
    per_iter_metrics: list[dict[str, float | str]],
) -> dict[str, dict[str, float]]:
    keys_float = ["mean_pairwise_cosine", "diversity_score", "max_pairwise_cosine"]
    keys_int = ["num_pairs_above_threshold"]

    def _mean_std(values: np.ndarray) -> dict[str, float]:
        values = values.astype(float)
        if values.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        if values.size == 1:
            return {"mean": float(values[0]), "std": 0.0}
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}

    agg: dict[str, dict[str, float]] = {}
    for key in keys_float:
        agg[key] = _mean_std(np.array([m[key] for m in per_iter_metrics], dtype=float))
    for key in keys_int:
        agg[key] = _mean_std(np.array([m[key] for m in per_iter_metrics], dtype=float))
    return agg


def plot_matrix(
    matrix: np.ndarray,
    out_path: Path,
    vmin: float | None,
    vmax: float | None,
) -> None:
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, cmap="GnBu", vmin=vmin, vmax=vmax)
    ticks = np.arange(matrix.shape[0])
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    ensure_directory(out_path.parent)
    plt.savefig(out_path)
    plt.close()


def run_topic_overlap_analysis(
    *,
    models: Sequence[ModelType],
    dataset: str,
    iterations: Sequence[int],
    num_topics: int,
    categories: Sequence[str],
    data_runs: Sequence[str] = ("default",),
    dup_threshold: float = 0.90,
    out_root: Path = DEFAULT_OUT_ROOT,
    save_per_iter_artifacts: bool = False,
    embedding_variant: str | None = None,
    encoder_model: str | None = None,
) -> Path:
    summary_rows: list[dict[str, str | float | int]] = []
    summary_provenance: list[dict[str, object]] = []
    uses_default_output_layout = _uses_default_output_layout(out_root)
    requested_embedding_variant = _resolve_requested_embedding_variant(
        embedding_variant=embedding_variant,
        encoder_model=encoder_model,
    )

    for data_run in data_runs:
        for model in models:
            for category in categories:
                per_iter_metrics: list[dict[str, float | str]] = []
                used_iterations: list[int] = []
                per_iter_cosines: list[tuple[int, np.ndarray]] = []

                for iteration in iterations:
                    topic_vectors = load_topic_vectors(
                        model=model,
                        dataset=dataset,
                        iteration=iteration,
                        num_topics=num_topics,
                        category=category,
                        data_run=data_run,
                        embedding_variant=_effective_embedding_variant(
                            model,
                            requested_embedding_variant,
                        ),
                    )
                    cosine = compute_cosine_similarity(topic_vectors)
                    metrics = compute_overlap_metrics(
                        cosine, dup_threshold=dup_threshold
                    )
                    per_iter_metrics.append(metrics)
                    used_iterations.append(iteration)
                    if save_per_iter_artifacts:
                        per_iter_cosines.append((iteration, cosine))

                agg = aggregate_metrics(per_iter_metrics)
                condition_id, condition_fingerprint = _build_output_condition_id(
                    model=model,
                    dataset=dataset,
                    data_run=data_run,
                    category=category,
                    iterations=used_iterations,
                    num_topics=num_topics,
                    dup_threshold=dup_threshold,
                    embedding_variant=requested_embedding_variant,
                )
                display_key = condition_id
                started_at = datetime.now(UTC).isoformat()
                execution_id = build_execution_id(prefix="exec", started_at=started_at)

                if uses_default_output_layout:
                    archive_out_dir = build_archive_result_dir(
                        base_root=out_root,
                        dataset=dataset,
                        data_run=data_run,
                        category=category,
                        display_key=display_key,
                        started_at=started_at,
                        execution_id=execution_id,
                    )
                    latest_out_dir = build_latest_result_dir(
                        base_root=out_root,
                        dataset=dataset,
                        data_run=data_run,
                        category=category,
                        display_key=display_key,
                    )
                    out_dir = archive_out_dir
                else:
                    archive_out_dir = None
                    latest_out_dir = None
                    out_dir = out_root / dataset / data_run / category / condition_id
                ensure_directory(out_dir)

                if save_per_iter_artifacts:
                    for iteration, cosine in per_iter_cosines:
                        iter_out_dir = out_dir / f"iter{iteration}"
                        ensure_directory(iter_out_dir)
                        cosine_plot = _to_unit_interval(
                            cosine,
                            min_val=-1.0,
                            max_val=1.0,
                        )
                        plot_matrix(
                            cosine_plot,
                            out_path=iter_out_dir / "topic_cosine_similarity.png",
                            vmin=0.0,
                            vmax=1.0,
                        )
                        np.savetxt(
                            iter_out_dir / "topic_cosine_similarity.csv",
                            cosine,
                            delimiter=",",
                            fmt="%.6f",
                        )

                provenance = load_model_provenance(
                    build_result_dir(
                        model=model,
                        dataset=dataset,
                        iteration=used_iterations[0],
                        num_topics=num_topics,
                        category=category,
                        data_run=data_run,
                        embedding_variant=_effective_embedding_variant(
                            model,
                            requested_embedding_variant,
                        ),
                    ),
                    model_key=(
                        "vmf_sentence_lda" if model == "vmf" else "sentence_gaussianlda"
                    ),
                )

                meta = build_evaluation_meta(
                    task="geometry_based_metrics",
                    model=model,
                    dataset=dataset,
                    data_run=data_run,
                    num_topics=num_topics,
                    category=category,
                    condition_id=condition_id,
                    display_key=display_key,
                    condition_fingerprint=condition_fingerprint,
                    iterations=used_iterations,
                    started_at=started_at,
                    execution_id=execution_id,
                    archive_dir=str(out_dir),
                    latest_dir=None if latest_out_dir is None else str(latest_out_dir),
                    dup_threshold=float(dup_threshold),
                    embedding_variant=requested_embedding_variant,
                    encoder_model=encoder_model,
                    model_provenance=provenance,
                )
                results = {
                    "aggregate": agg,
                    "per_iteration": per_iter_metrics,
                }
                out_path = out_dir / "overlap_metrics_agg.json"
                write_evaluation_json(meta=meta, results=results, path=out_path)
                logger.info(f"[{model}] aggregated metrics saved to {out_path}")

                metadata_path = out_dir / "metadata.json"
                save_json(meta, metadata_path)
                logger.info(f"[{model}] metadata saved to {metadata_path}")

                if uses_default_output_layout and archive_out_dir is not None:
                    artifacts = {
                        "metrics": out_path.name,
                        "metadata": metadata_path.name,
                    }
                    if save_per_iter_artifacts:
                        for iteration in used_iterations:
                            artifacts[
                                f"topic_cosine_similarity_csv_iter{iteration}"
                            ] = f"iter{iteration}/topic_cosine_similarity.csv"
                            artifacts[
                                f"topic_cosine_similarity_png_iter{iteration}"
                            ] = f"iter{iteration}/topic_cosine_similarity.png"
                    pointer_path = write_latest_result_pointer(
                        base_root=out_root,
                        task="geometry_based_metrics",
                        dataset=dataset,
                        data_run=data_run,
                        category=category,
                        display_key=display_key,
                        archive_dir=archive_out_dir,
                        started_at=started_at,
                        execution_id=execution_id,
                        condition_fingerprint=condition_fingerprint,
                        artifacts=artifacts,
                    )
                    logger.info(
                        "[%s] updated latest pointer at %s",
                        model,
                        pointer_path,
                    )

                agg_div = agg["diversity_score"]
                agg_max = agg["max_pairwise_cosine"]
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "data_run": data_run,
                        "num_topics": num_topics,
                        "category": category,
                        "model": model,
                        "diversity_mean": round_sigfigs(agg_div["mean"]),
                        "diversity_std": round_sigfigs(agg_div["std"]),
                        "max_cosine_mean": round_sigfigs(agg_max["mean"]),
                        "max_cosine_std": round_sigfigs(agg_max["std"]),
                    }
                )
                summary_provenance.append(
                    {
                        "model": model,
                        "data_run": data_run,
                        "category": category,
                        "embedding_variant": requested_embedding_variant,
                        "model_provenance": provenance,
                    }
                )

    summary_path = out_root / "summary.csv"
    if summary_rows:
        fieldnames = [
            "dataset",
            "data_run",
            "num_topics",
            "category",
            "model",
            "diversity_mean",
            "diversity_std",
            "max_cosine_mean",
            "max_cosine_std",
        ]
        write_csv_rows(fieldnames=fieldnames, rows=summary_rows, path=summary_path)
        summary_json_path = out_root / "summary.json"
        write_tabular_report_json(
            meta={
                "task": "geometry_based_metrics_summary",
                "dataset": dataset,
                "data_runs": list(data_runs),
                "dup_threshold": float(dup_threshold),
                "embedding_variant": requested_embedding_variant,
                "encoder_model": encoder_model,
                "out_root": str(out_root),
                "model_provenance": summary_provenance,
            },
            columns=fieldnames,
            rows=summary_rows,
            path=summary_json_path,
        )
        logger.info(f"summary CSV written to {summary_path}")
    return summary_path


run_geometry_based_metrics = run_topic_overlap_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Sentence LDA topic vectors (Gaussian / vMF) "
            "with iteration-aggregated metrics."
        )
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=["vmf", "gaussian"],
        required=True,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_run", nargs="+", default=["default"])
    parser.add_argument("--iteration", type=int, nargs="+", required=True)
    parser.add_argument("--num_topics", type=int, required=True)
    parser.add_argument("--category", nargs="+", default=["all"])
    parser.add_argument("--dup_threshold", type=float, default=0.90)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--save_per_iter_artifacts", action="store_true")
    parser.add_argument("--embedding_variant", default=None)
    parser.add_argument("--encoder_model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_topic_overlap_analysis(
        models=args.model,
        dataset=args.dataset,
        data_runs=args.data_run,
        iterations=args.iteration,
        num_topics=args.num_topics,
        categories=args.category,
        dup_threshold=args.dup_threshold,
        out_root=args.out_root,
        save_per_iter_artifacts=args.save_per_iter_artifacts,
        embedding_variant=args.embedding_variant,
        encoder_model=args.encoder_model,
    )


if __name__ == "__main__":
    main()

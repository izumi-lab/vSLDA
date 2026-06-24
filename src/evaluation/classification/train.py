from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from src.utils.logging import get_logger
from src.utils.random import set_global_seed

from .config import (
    DEFAULT_ALIGNMENT_MODE,
    DEFAULT_FEATURE_RESOLVE_MODE,
    RESULT_ROOT,
    get_dataset_targets,
)
from .pipeline import run_classification_task
from .workflow import (
    ClassificationCondition,
    build_classification_write_spec,
    run_classification_grid,
)

logger = get_logger(__name__)


def train(
    category: str,
    dataset: str,
    num_topics: int,
    it: int,
    classifiers_to_use: Sequence[str],
    vmf_assignment: str,
    data_run: str = "default",
    train_indices: Sequence[int] | None = None,
    target_column: str = "target_str",
    label_schema: str = "identity",
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
) -> (
    tuple[
        dict[str, float],
        dict[str, dict[str, float]],
        dict[str, object],
        list[dict[str, object]],
        dict[str, object],
    ]
    | None
):
    """Train classifiers on doc-topic features across models."""
    dataset_targets = get_dataset_targets(
        dataset,
        target_column=target_column,
        label_schema=label_schema,
    )
    if dataset_targets is None or category not in dataset_targets:
        logger.warning(f"[skip] unknown category '{category}' for dataset '{dataset}'")
        return None
    return run_classification_task(
        dataset=dataset,
        category=category,
        num_topics=num_topics,
        iteration=it,
        data_run=data_run,
        category_labels=dataset_targets[category],
        classifiers_to_use=classifiers_to_use,
        vmf_assignment=vmf_assignment,
        train_indices=train_indices,
        target_column=target_column,
        label_schema=label_schema,
        alignment_mode=alignment_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
    )


def run_classification_evaluation(
    *,
    iterations: Sequence[int],
    datasets: Iterable[str],
    data_runs: Sequence[str] = ("default",),
    categories: Sequence[str] | None = None,
    topics: Sequence[int],
    classifiers: Sequence[str],
    vmf_assignment: str,
    result_root: Path = RESULT_ROOT,
    target_column: str = "target_str",
    label_schema: str = "identity",
    seed: int | None = 42,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
) -> None:
    if seed is not None:
        set_global_seed(seed)

    for data_run in data_runs:
        run_classification_grid(
            iterations=iterations,
            datasets=datasets,
            data_run=data_run,
            categories=categories,
            topics=topics,
            classifiers=classifiers,
            vmf_assignment=vmf_assignment,
            target_column=target_column,
            label_schema=label_schema,
            alignment_mode=alignment_mode,
            write_spec_builder=lambda iteration, dataset, num_topics, *, _data_run=data_run: build_classification_write_spec(
                result_root=result_root,
                condition=ClassificationCondition(
                    dataset=dataset,
                    data_run=_data_run,
                    topics=num_topics,
                    iteration=iteration,
                    classifiers=classifiers,
                    vmf_assignment=vmf_assignment,
                    target_column=target_column,
                    label_schema=label_schema,
                    alignment_mode=alignment_mode,
                    embedding_variants=embedding_variants,
                    feature_resolve_mode=feature_resolve_mode,
                    selected_models=selected_models,
                ),
                acc_filename=f"acc_{dataset}_{num_topics}topic.json",
                f1_filename=f"f1_{dataset}_{num_topics}topic.json",
                feature_filename=f"feat_{dataset}_{num_topics}topic.json",
                seed=seed,
            ),
            train_runner=train,
            embedding_variants=embedding_variants,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=selected_models,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classifier evaluation over doc-topic features."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="*",
        default=list(range(5)),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["20newsgroup", "nyt"],
    )
    parser.add_argument(
        "--topics",
        type=int,
        nargs="*",
        default=[10, 20],
    )
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="*",
        default=["svm"],
        choices=["logreg", "svm"],
    )
    parser.add_argument(
        "--vmf_assignment",
        type=str,
        default="hard",
        choices=["soft", "hard"],
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--target-column", type=str, default="target_str")
    parser.add_argument(
        "--label-schema",
        type=str,
        default="identity",
    )
    parser.add_argument("--embedding-variant", type=str, nargs="*", default=None)
    parser.add_argument(
        "--feature-resolve-mode",
        type=str,
        default=DEFAULT_FEATURE_RESOLVE_MODE,
        choices=["all", "strict"],
    )
    args = parser.parse_args()

    run_classification_evaluation(
        iterations=args.iterations,
        datasets=args.datasets,
        topics=args.topics,
        data_runs=("default",),
        classifiers=args.classifiers,
        vmf_assignment=args.vmf_assignment,
        result_root=args.result_root,
        target_column=args.target_column,
        label_schema=args.label_schema,
        seed=args.seed,
        alignment_mode=DEFAULT_ALIGNMENT_MODE,
        embedding_variants=args.embedding_variant,
        feature_resolve_mode=args.feature_resolve_mode,
    )

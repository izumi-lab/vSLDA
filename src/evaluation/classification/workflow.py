from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.core.path_builders import (
    build_archive_result_dir,
    build_latest_result_dir,
    build_result_display_key,
)
from src.core.path_pointers import write_latest_result_pointer
from src.core.result_identity import build_condition_id, build_execution_id
from src.evaluation.reporting import write_evaluation_json
from src.evaluation.schema import build_evaluation_meta
from src.utils.random import DEFAULT_RANDOM_SEED

from .config import (
    DEFAULT_ALIGNMENT_MODE,
    DEFAULT_FEATURE_RESOLVE_MODE,
    get_dataset_targets,
    resolve_dataset_name,
)

TrainIndexResolver = Callable[
    [str, str, int, int, int], tuple[Sequence[int] | None, dict[str, Any]]
]
TrainRunner = Callable[
    ...,
    tuple[
        dict[str, float],
        dict[str, dict[str, float]],
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any],
    ]
    | None,
]
LabelLoader = Callable[..., list[str]]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationCondition:
    dataset: str
    topics: int
    iteration: int
    classifiers: Sequence[str]
    vmf_assignment: str
    target_column: str
    label_schema: str
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE
    embedding_variants: Sequence[str] | None = None
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE
    selected_models: Sequence[str] | None = None
    data_run: str = "default"
    mode: str | None = None
    value: float | int | None = None
    stratified: bool | None = None
    sampling_repeat: int | None = None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "data_run": self.data_run,
            "topics": int(self.topics),
            "iteration": int(self.iteration),
            "classifiers": sorted(str(item) for item in self.classifiers),
            "vmf_assignment": self.vmf_assignment,
            "target_column": self.target_column,
            "label_schema": self.label_schema,
            "alignment_mode": self.alignment_mode,
            "embedding_variants": (
                None
                if self.embedding_variants is None
                else sorted(str(item) for item in self.embedding_variants)
            ),
            "feature_resolve_mode": self.feature_resolve_mode,
            "selected_models": (
                None
                if self.selected_models is None
                else sorted(str(item) for item in self.selected_models)
            ),
        }
        if self.mode is not None:
            payload["mode"] = self.mode
        if self.value is not None:
            payload["value"] = self.value
        if self.stratified is not None:
            payload["stratified"] = bool(self.stratified)
        if self.sampling_repeat is not None:
            payload["sampling_repeat"] = int(self.sampling_repeat)
        return payload

    def condition_id(self) -> tuple[str, str]:
        extra_labels: list[Any] = [self.vmf_assignment]
        if self.mode is not None:
            extra_labels.append(self.mode)
        if self.value is not None:
            extra_labels.append(self.value)
        if self.sampling_repeat is not None:
            extra_labels.append(f"sample-r{int(self.sampling_repeat)}")
        return build_condition_id(
            iteration=self.iteration,
            num_topics=self.topics,
            fingerprint_payload=self.payload(),
            extra_labels=extra_labels,
        )

    def display_key(self) -> str:
        extra_labels: list[Any] = []
        classifier_label = _classifier_display_key_label(self.classifiers)
        if classifier_label is not None:
            extra_labels.append(classifier_label)
        extra_labels.append(self.vmf_assignment)
        if self.alignment_mode != DEFAULT_ALIGNMENT_MODE:
            extra_labels.append(self.alignment_mode)
        if self.embedding_variants is not None:
            extra_labels.append(
                "emb-" + "-".join(sorted(str(item) for item in self.embedding_variants))
            )
        if self.feature_resolve_mode != DEFAULT_FEATURE_RESOLVE_MODE:
            extra_labels.append(self.feature_resolve_mode)
        if self.selected_models is not None:
            extra_labels.append(
                "models-" + "-".join(sorted(str(item) for item in self.selected_models))
            )
        if self.mode is not None:
            extra_labels.append(self.mode)
        if self.value is not None:
            extra_labels.append(self.value)
        if self.stratified is not None:
            extra_labels.append("stratified" if self.stratified else "unstratified")
        if self.sampling_repeat is not None:
            extra_labels.append(f"sample-r{int(self.sampling_repeat)}")
        return build_result_display_key(
            num_topics=self.topics,
            iteration=self.iteration,
            extra_labels=extra_labels,
        )

    def meta(self, *, seed: int | None = DEFAULT_RANDOM_SEED) -> dict[str, Any]:
        condition_id, condition_fingerprint = self.condition_id()
        meta: dict[str, Any] = build_evaluation_meta(
            task="classification",
            output_kind="payload",
            dataset=self.dataset,
            data_run=self.data_run,
            condition_id=condition_id,
            condition_fingerprint=condition_fingerprint,
            topics=self.topics,
            iteration=self.iteration,
            classifiers=list(self.classifiers),
            vmf_assignment=self.vmf_assignment,
            target_column=self.target_column,
            label_schema=self.label_schema,
            categories={},
        )
        meta["alignment_mode"] = self.alignment_mode
        meta["embedding_variants"] = (
            None
            if self.embedding_variants is None
            else sorted(str(item) for item in self.embedding_variants)
        )
        meta["feature_resolve_mode"] = self.feature_resolve_mode
        meta["selected_models"] = (
            None
            if self.selected_models is None
            else sorted(str(item) for item in self.selected_models)
        )
        meta["display_key"] = self.display_key()
        if self.mode is not None:
            meta["mode"] = self.mode
        if self.value is not None:
            meta["value"] = self.value
        if self.stratified is not None:
            meta["stratified"] = self.stratified
        if self.sampling_repeat is not None:
            meta["sampling_repeat"] = int(self.sampling_repeat)
        if seed is not None:
            meta["seed"] = seed
            if self.sampling_repeat is not None:
                meta["sampling_seed"] = seed
        return meta


def build_classification_condition_payload(
    *,
    dataset: str,
    data_run: str = "default",
    topics: int,
    iteration: int,
    classifiers: Sequence[str],
    vmf_assignment: str,
    target_column: str,
    label_schema: str,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    mode: str | None = None,
    value: float | int | None = None,
    stratified: bool | None = None,
    sampling_repeat: int | None = None,
) -> dict[str, Any]:
    return ClassificationCondition(
        dataset=dataset,
        data_run=data_run,
        topics=topics,
        iteration=iteration,
        classifiers=classifiers,
        vmf_assignment=vmf_assignment,
        target_column=target_column,
        label_schema=label_schema,
        alignment_mode=alignment_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
        mode=mode,
        value=value,
        stratified=stratified,
        sampling_repeat=sampling_repeat,
    ).payload()


def build_classification_condition_id(
    *,
    dataset: str,
    data_run: str = "default",
    topics: int,
    iteration: int,
    classifiers: Sequence[str],
    vmf_assignment: str,
    target_column: str,
    label_schema: str,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    mode: str | None = None,
    value: float | int | None = None,
    stratified: bool | None = None,
    sampling_repeat: int | None = None,
) -> tuple[str, str]:
    return ClassificationCondition(
        dataset=dataset,
        data_run=data_run,
        topics=topics,
        iteration=iteration,
        classifiers=classifiers,
        vmf_assignment=vmf_assignment,
        target_column=target_column,
        label_schema=label_schema,
        alignment_mode=alignment_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
        mode=mode,
        value=value,
        stratified=stratified,
        sampling_repeat=sampling_repeat,
    ).condition_id()


def build_classification_output_dir(
    *,
    result_root: Path,
    dataset: str,
    data_run: str,
    condition_id: str,
    category: str = "all",
) -> Path:
    category_first_dir = result_root / dataset / data_run / category / condition_id
    legacy_dir = result_root / dataset / data_run / condition_id
    if legacy_dir.exists():
        return legacy_dir
    return category_first_dir


def build_classification_latest_dir(
    *,
    result_root: Path,
    dataset: str,
    data_run: str,
    display_key: str,
    category: str = "all",
) -> Path:
    return build_latest_result_dir(
        base_root=result_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        display_key=display_key,
    )


def build_classification_archive_dir(
    *,
    result_root: Path,
    dataset: str,
    data_run: str,
    display_key: str,
    started_at: str,
    execution_id: str,
    category: str = "all",
) -> Path:
    return build_archive_result_dir(
        base_root=result_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        display_key=display_key,
        started_at=started_at,
        execution_id=execution_id,
    )


def build_classification_output_dir_from_condition(
    *,
    result_root: Path,
    condition: ClassificationCondition,
    category: str = "all",
) -> Path:
    condition_id, _ = condition.condition_id()
    return build_classification_output_dir(
        result_root=result_root,
        dataset=condition.dataset,
        data_run=condition.data_run,
        condition_id=condition_id,
        category=category,
    )


def build_classification_archive_dir_from_condition(
    *,
    result_root: Path,
    condition: ClassificationCondition,
    started_at: str,
    execution_id: str,
    category: str = "all",
) -> Path:
    return build_classification_archive_dir(
        result_root=result_root,
        dataset=condition.dataset,
        data_run=condition.data_run,
        display_key=condition.display_key(),
        started_at=started_at,
        execution_id=execution_id,
        category=category,
    )


def build_classification_meta(
    *,
    dataset: str,
    data_run: str = "default",
    topics: int,
    iteration: int,
    classifiers: Sequence[str],
    vmf_assignment: str,
    target_column: str,
    label_schema: str,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    mode: str | None = None,
    value: float | int | None = None,
    stratified: bool | None = None,
    sampling_repeat: int | None = None,
    seed: int | None = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    return ClassificationCondition(
        dataset=dataset,
        data_run=data_run,
        topics=topics,
        iteration=iteration,
        classifiers=classifiers,
        vmf_assignment=vmf_assignment,
        target_column=target_column,
        label_schema=label_schema,
        alignment_mode=alignment_mode,
        embedding_variants=embedding_variants,
        feature_resolve_mode=feature_resolve_mode,
        selected_models=selected_models,
        mode=mode,
        value=value,
        stratified=stratified,
        sampling_repeat=sampling_repeat,
    ).meta(seed=seed)


@dataclass(frozen=True)
class EvaluationWriteSpec:
    output_dir: Path
    acc_filename: str
    f1_filename: str
    feature_filename: str
    meta: dict[str, Any] = field(default_factory=dict)
    latest_base_root: Path | None = None
    latest_category: str = "all"
    display_key: str | None = None
    started_at: str | None = None
    execution_id: str | None = None


def build_classification_write_spec(
    *,
    result_root: Path,
    condition: ClassificationCondition,
    acc_filename: str,
    f1_filename: str,
    feature_filename: str,
    category: str = "all",
    seed: int | None = DEFAULT_RANDOM_SEED,
) -> EvaluationWriteSpec:
    started_at = datetime.now(UTC).isoformat()
    execution_id = build_execution_id(prefix="exec", started_at=started_at)
    display_key = condition.display_key()
    meta = condition.meta(seed=seed)
    meta["started_at"] = started_at
    meta["execution_id"] = execution_id
    meta["latest_dir"] = str(
        build_classification_latest_dir(
            result_root=result_root,
            dataset=condition.dataset,
            data_run=condition.data_run,
            display_key=display_key,
            category=category,
        )
    )
    return EvaluationWriteSpec(
        output_dir=build_classification_archive_dir_from_condition(
            result_root=result_root,
            condition=condition,
            started_at=started_at,
            execution_id=execution_id,
            category=category,
        ),
        acc_filename=acc_filename,
        f1_filename=f1_filename,
        feature_filename=feature_filename,
        meta=meta,
        latest_base_root=result_root,
        latest_category=category,
        display_key=display_key,
        started_at=started_at,
        execution_id=execution_id,
    )


@dataclass
class EvaluationArtifacts:
    acc_result: dict[str, Any] = field(default_factory=dict)
    f1_result: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def run_classification_grid(
    *,
    iterations: Iterable[int],
    datasets: Iterable[str],
    data_run: str = "default",
    categories: Sequence[str] | None = None,
    topics: Iterable[int],
    classifiers: Sequence[str],
    vmf_assignment: str,
    target_column: str,
    label_schema: str,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    embedding_variants: Sequence[str] | None = None,
    feature_resolve_mode: str = DEFAULT_FEATURE_RESOLVE_MODE,
    selected_models: Sequence[str] | None = None,
    write_spec_builder: Callable[[int, str, int], EvaluationWriteSpec],
    train_runner: TrainRunner,
    train_index_resolver: TrainIndexResolver | None = None,
) -> None:
    for iteration in iterations:
        for dataset in datasets:
            resolved_dataset = resolve_dataset_name(dataset)
            if resolved_dataset is None:
                LOGGER.warning("[skip] unknown dataset %s", dataset)
                continue

            dataset_targets = get_dataset_targets(
                resolved_dataset,
                target_column=target_column,
                label_schema=label_schema,
            )
            if dataset_targets is None:
                LOGGER.warning(
                    "[skip] no category definitions found for %s",
                    resolved_dataset,
                )
                continue
            if categories is None:
                categories_to_run = list(dataset_targets.keys())
            else:
                categories_to_run = [
                    str(category)
                    for category in categories
                    if str(category) in dataset_targets
                ]
                missing_categories = [
                    str(category)
                    for category in categories
                    if str(category) not in dataset_targets
                ]
                for category in missing_categories:
                    LOGGER.warning(
                        "[skip] unknown category '%s' for dataset '%s'",
                        category,
                        resolved_dataset,
                    )
                if not categories_to_run:
                    LOGGER.warning(
                        "[skip] no configured categories available for %s",
                        resolved_dataset,
                    )
                    continue

            for num_topics in topics:
                write_spec = write_spec_builder(
                    iteration,
                    resolved_dataset,
                    num_topics,
                )
                artifacts = EvaluationArtifacts(meta=dict(write_spec.meta))
                artifacts.meta.setdefault("categories", {})
                artifacts.meta.setdefault("data_run", data_run)

                for category in categories_to_run:
                    retry_attempt = 0
                    result = None
                    category_meta: dict[str, Any] = {}
                    skip_reason = ""
                    while True:
                        train_indices = None
                        category_meta = {}
                        if train_index_resolver is not None:
                            train_indices, category_meta = train_index_resolver(
                                resolved_dataset,
                                category,
                                iteration,
                                num_topics,
                                retry_attempt,
                            )
                            if train_indices is not None:
                                train_indices = list(train_indices)
                            if train_indices is not None and not train_indices:
                                skip_reason = str(
                                    category_meta.get(
                                        "skip_reason",
                                        "no training data after sampling",
                                    )
                                )
                                LOGGER.warning(
                                    "[skip] %s %s %stopic %s",
                                    skip_reason,
                                    resolved_dataset,
                                    num_topics,
                                    category,
                                )
                                break

                        try:
                            result = train_runner(
                                category,
                                resolved_dataset,
                                num_topics,
                                iteration,
                                classifiers,
                                vmf_assignment,
                                data_run=data_run,
                                train_indices=train_indices,
                                target_column=target_column,
                                label_schema=label_schema,
                                alignment_mode=alignment_mode,
                                embedding_variants=embedding_variants,
                                feature_resolve_mode=feature_resolve_mode,
                                selected_models=selected_models,
                            )
                            break
                        except ValueError as exc:
                            if (
                                train_index_resolver is not None
                                and _is_single_class_training_error(exc)
                            ):
                                LOGGER.warning(
                                    "[retry] one class after alignment %s %stopic %s attempt=%s",
                                    resolved_dataset,
                                    num_topics,
                                    category,
                                    retry_attempt,
                                )
                                retry_attempt += 1
                                continue
                            raise

                    if result is None:
                        if not skip_reason:
                            LOGGER.warning(
                                "[skip] no features -> skip %s %stopic %s",
                                resolved_dataset,
                                num_topics,
                                category,
                            )
                        continue

                    (
                        acc_result,
                        f1_result,
                        feature_importance,
                        feature_catalog,
                        coverage,
                    ) = result
                    artifacts.acc_result[category] = acc_result
                    artifacts.f1_result[category] = f1_result
                    artifacts.feature_importance[category] = feature_importance
                    category_payload = dict(category_meta)
                    category_payload["feature_catalog"] = list(feature_catalog)
                    category_payload["coverage"] = dict(coverage)
                    artifacts.meta["categories"][category] = category_payload

                if not artifacts.acc_result:
                    LOGGER.warning(
                        "[skip] nothing to write for iter%s %s %stopic",
                        iteration,
                        resolved_dataset,
                        num_topics,
                    )
                    continue

                _write_artifacts(write_spec, artifacts)


def build_sampling_meta(
    *,
    dataset: str,
    category: str,
    target_column: str,
    label_schema: str,
    train_indices: Sequence[int],
    label_counts: dict[str, int],
    load_labels: LabelLoader,
) -> dict[str, Any]:
    labels = load_labels(
        dataset,
        category,
        "train",
        target_column=target_column,
        label_schema=label_schema,
    )
    return {
        "train_count": len(train_indices),
        "label_counts": label_counts,
        "total_train": len(labels),
    }


def _is_single_class_training_error(exc: ValueError) -> bool:
    message = str(exc)
    return (
        (
            "Training data must contain at least 2 classes after alignment" in message
            and "got 1 class" in message
        )
        or (
            "The number of classes has to be greater than one" in message
            and "got 1 class" in message
        )
        or (
            "This solver needs samples of at least 2 classes" in message
            and "contains only one class" in message
        )
    )


def _write_artifacts(spec: EvaluationWriteSpec, artifacts: EvaluationArtifacts) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = spec.output_dir / "feature"
    feature_dir.mkdir(parents=True, exist_ok=True)

    write_evaluation_json(
        meta=artifacts.meta,
        results=artifacts.acc_result,
        path=spec.output_dir / spec.acc_filename,
    )
    write_evaluation_json(
        meta=artifacts.meta,
        results=artifacts.f1_result,
        path=spec.output_dir / spec.f1_filename,
    )
    write_evaluation_json(
        meta=artifacts.meta,
        results=artifacts.feature_importance,
        path=feature_dir / spec.feature_filename,
    )
    if (
        spec.latest_base_root is not None
        and spec.display_key is not None
        and spec.started_at is not None
        and spec.execution_id is not None
    ):
        write_latest_result_pointer(
            base_root=spec.latest_base_root,
            task="classification",
            dataset=str(artifacts.meta.get("dataset", "")),
            data_run=str(artifacts.meta.get("data_run", "default")),
            category=spec.latest_category,
            display_key=spec.display_key,
            archive_dir=spec.output_dir,
            started_at=spec.started_at,
            execution_id=spec.execution_id,
            condition_fingerprint=artifacts.meta.get("condition_fingerprint"),
            artifacts={
                "acc": spec.acc_filename,
                "f1": spec.f1_filename,
                "feature": f"feature/{spec.feature_filename}",
            },
        )


def _classifier_display_key_label(classifiers: Sequence[str]) -> str | None:
    normalized = sorted(
        str(item).strip().lower() for item in classifiers if str(item).strip()
    )
    if not normalized:
        return None
    return "-".join(normalized)

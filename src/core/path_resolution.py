from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from src.core.artifacts import (
    CURRENT_POINTER_FILENAME,
    METADATA_FILENAME,
    load_artifact_json,
)
from src.core.errors import MissingArtifactError

from .path_builders import (
    build_baseline_dir,
    build_baseline_display_key,
    build_baseline_latest_dir,
    build_latest_result_dir,
    build_vmf_display_key,
    build_vmf_experiment_dir,
    build_vmf_latest_dir,
    legacy_baseline_condition_dir,
    legacy_vmf_experiment_dir,
)
from .paths_roots import RESULTS_ROOT, VISUALIZATION_RESULTS_ROOT, resolve_project_path


def _current_experiment_results_root() -> Path:
    from . import paths as public_paths

    return public_paths.EXPERIMENT_RESULTS_ROOT


def _current_baseline_results_root() -> Path:
    from . import paths as public_paths

    return public_paths.BASELINE_RESULTS_ROOT


def _embedding_latest_dirs(
    *,
    exact_dir: Path,
    base_display_key: str,
    embedding_variant: str | None,
) -> list[Path]:
    if embedding_variant:
        return [exact_dir]
    dirs = [exact_dir]
    parent = exact_dir.parent
    if parent.exists():
        prefix = f"{base_display_key}_"
        dirs.extend(
            path
            for path in sorted(parent.iterdir())
            if path.is_dir() and path.name.startswith(prefix)
        )
    return dirs


def _archive_from_latest_pointers(
    latest_pointer_dirs: list[Path],
    *,
    require_condition_id: str | None = None,
) -> Path | None:
    matches: list[tuple[Path, Path]] = []
    for latest_pointer_dir in latest_pointer_dirs:
        pointer_path = latest_pointer_dir / CURRENT_POINTER_FILENAME
        if not pointer_path.exists():
            continue
        payload = load_artifact_json(pointer_path)
        if not isinstance(payload, dict) or not payload.get("archive_dir"):
            continue
        archive_dir = resolve_project_path(str(payload["archive_dir"]))
        if not archive_dir.exists():
            continue
        if require_condition_id is not None:
            metadata_path = archive_dir / METADATA_FILENAME
            if not metadata_path.exists():
                continue
            metadata = load_artifact_json(metadata_path)
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("condition_id")) != str(require_condition_id):
                continue
        matches.append((latest_pointer_dir, archive_dir))
    unique: list[Path] = []
    unique_display_keys: list[str] = []
    seen: set[Path] = set()
    for latest_pointer_dir, archive_dir in matches:
        if archive_dir in seen:
            continue
        seen.add(archive_dir)
        unique.append(archive_dir)
        unique_display_keys.append(latest_pointer_dir.name)
    if not unique:
        return None
    if len(unique) > 1:
        candidates = ", ".join(unique_display_keys)
        raise MissingArtifactError(
            latest_pointer_dirs[0].parent,
            detail=(
                "Multiple latest pointers match this request. Specify "
                "embedding_variant or condition_id to disambiguate. "
                f"Candidates: {candidates}"
            ),
        )
    return unique[0]


def _is_ambiguous_artifact_error(exc: MissingArtifactError) -> bool:
    return "Multiple" in str(exc)


def _vmf_dataset_root_candidates(
    *,
    dataset: str,
    dataset_root: Path | None = None,
) -> list[Path]:
    resolved_dataset_root = dataset_root or (
        _current_experiment_results_root() / dataset
    )
    candidates = [resolved_dataset_root]
    nested_dataset_root = resolved_dataset_root / dataset
    if nested_dataset_root != resolved_dataset_root:
        candidates.append(nested_dataset_root)
    return candidates


def _candidate_component_counts(
    *,
    requested: int | None,
    default: int | None,
) -> list[int | None]:
    candidates: list[int | None] = []
    if requested is not None:
        candidates.append(int(requested))
    elif default is not None:
        candidates.append(int(default))
    candidates.append(None)
    return candidates


def _algorithm_variant_num_components(value: object) -> int | None:
    text = str(value or "")
    prefix = "components_"
    if not text.startswith(prefix):
        return None
    raw_value = text[len(prefix) :].split("__", 1)[0]
    try:
        return int(raw_value)
    except ValueError:
        return None


def _vmf_payload_matches_component(
    payload: Mapping[str, Any],
    num_components: int | None,
) -> bool:
    if num_components is None:
        return True
    if payload.get("num_components") is not None:
        return int(payload["num_components"]) == int(num_components)
    axes = payload.get("axes")
    if isinstance(axes, Mapping):
        if axes.get("num_components") is not None:
            return int(axes["num_components"]) == int(num_components)
        variant_num_components = _algorithm_variant_num_components(
            axes.get("algorithm_variant")
        )
        if variant_num_components is not None:
            return variant_num_components == int(num_components)
    return int(num_components) == 1


def _baseline_payload_matches_component(
    payload: Mapping[str, Any],
    *,
    model: str,
    num_components: int | None,
) -> bool:
    if model != "mvtm":
        return True
    if num_components is None:
        return True
    baseline_params = payload.get("baseline_params")
    if (
        isinstance(baseline_params, Mapping)
        and baseline_params.get("num_components") is not None
    ):
        return int(baseline_params["num_components"]) == int(num_components)
    parameter_variant = str(payload.get("parameter_variant", ""))
    if f"num_components={int(num_components)}" in parameter_variant:
        return True
    return int(num_components) == 1


def build_vmf_doc_topic_path(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    assignment: str = "hard",
    run_name: str = "default",
    condition_id: str | None = None,
    condition_payload: Mapping[str, Any] | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    dataset_root: Path | None = None,
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if assignment not in {"hard", "soft"}:
        raise ValueError(f"Unsupported assignment: {assignment}")

    if condition_id is None:
        try:
            result_dir = resolve_vmf_experiment_dir(
                dataset=dataset,
                iteration=iteration,
                num_topics=num_topics,
                category=category,
                run_name=run_name,
                num_components=num_components,
                embedding_variant=embedding_variant,
                dataset_root=dataset_root,
            )
        except MissingArtifactError as exc:
            if _is_ambiguous_artifact_error(exc):
                raise
            result_dir = build_vmf_experiment_dir(
                dataset=dataset,
                iteration=iteration,
                num_topics=num_topics,
                category=category,
                run_name=run_name,
                condition_payload=condition_payload,
                dataset_root=dataset_root,
            )
    else:
        result_dir = build_vmf_experiment_dir(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            run_name=run_name,
            condition_id=condition_id,
            condition_payload=condition_payload,
            dataset_root=dataset_root,
        )
        if not result_dir.exists():
            try:
                result_dir = resolve_vmf_experiment_dir(
                    dataset=dataset,
                    iteration=iteration,
                    num_topics=num_topics,
                    category=category,
                    run_name=run_name,
                    condition_id=condition_id,
                    num_components=num_components,
                    embedding_variant=embedding_variant,
                    dataset_root=dataset_root,
                )
            except MissingArtifactError as exc:
                if _is_ambiguous_artifact_error(exc):
                    raise
                pass
    suffix = "_soft" if assignment == "soft" else ""
    return result_dir / f"doc_topic_{split}{suffix}.pkl"


def build_baseline_doc_topic_path(
    *,
    model: str,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str,
    prefer_soft: bool = False,
    data_run: str = "default",
    condition_id: str | None = None,
    condition_payload: Mapping[str, Any] | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    baseline_root: Path | None = None,
) -> Path | None:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    resolved_condition_dir: Path | None = None
    if condition_id is None:
        try:
            resolved_condition_dir = resolve_baseline_condition_dir(
                model=model,
                dataset=dataset,
                iteration=iteration,
                num_topics=num_topics,
                category=category,
                data_run=data_run,
                num_components=num_components,
                embedding_variant=embedding_variant,
                baseline_root=baseline_root,
            )
        except MissingArtifactError as exc:
            if _is_ambiguous_artifact_error(exc):
                raise
            pass

    if model == "bleilda":
        if split == "train":
            if resolved_condition_dir is not None:
                return resolved_condition_dir / "params" / "lda_comp.pkl"
            path = (
                build_baseline_dir(
                    model=model,
                    split_root="params",
                    dataset=dataset,
                    iteration=iteration,
                    num_topics=num_topics,
                    category=category,
                    data_run=data_run,
                    condition_id=condition_id,
                    condition_payload=condition_payload,
                    baseline_root=baseline_root,
                )
                / "lda_comp.pkl"
            )
            if not path.exists():
                try:
                    resolved_condition_dir = resolve_baseline_condition_dir(
                        model=model,
                        dataset=dataset,
                        iteration=iteration,
                        num_topics=num_topics,
                        category=category,
                        data_run=data_run,
                        condition_id=condition_id,
                        num_components=num_components,
                        embedding_variant=embedding_variant,
                        baseline_root=baseline_root,
                    )
                    return resolved_condition_dir / "params" / "lda_comp.pkl"
                except MissingArtifactError as exc:
                    if _is_ambiguous_artifact_error(exc):
                        raise
                    pass
            return path
        if resolved_condition_dir is not None:
            return resolved_condition_dir / "infer" / f"{category}.pkl"
        path = (
            build_baseline_dir(
                model=model,
                split_root="infer",
                dataset=dataset,
                iteration=iteration,
                num_topics=num_topics,
                category=category,
                data_run=data_run,
                condition_id=condition_id,
                condition_payload=condition_payload,
                baseline_root=baseline_root,
            )
            / f"{category}.pkl"
        )
        if not path.exists():
            try:
                resolved_condition_dir = resolve_baseline_condition_dir(
                    model=model,
                    dataset=dataset,
                    iteration=iteration,
                    num_topics=num_topics,
                    category=category,
                    data_run=data_run,
                    condition_id=condition_id,
                    num_components=num_components,
                    embedding_variant=embedding_variant,
                    baseline_root=baseline_root,
                )
                return resolved_condition_dir / "infer" / f"{category}.pkl"
            except MissingArtifactError as exc:
                if _is_ambiguous_artifact_error(exc):
                    raise
                pass
        return path

    if model in {
        "ctm",
        "etm",
        "gaussianlda",
        "mvtm",
        "sentence_gaussianlda",
        "sentlda",
        "bertopic_kmeans",
        "spherical_kmeans",
        "gaussian_kmeans",
        "movmf",
        "gaussian_mixture",
    }:
        if split == "train":
            filename = {
                "ctm": "ctm.pkl",
                "etm": "etm.pkl",
                "gaussianlda": "table_counts_per_doc.pkl",
                "mvtm": "table_counts_per_doc.pkl",
                "sentence_gaussianlda": "table_counts_per_doc.pkl",
                "sentlda": "table_counts_per_doc.pkl",
                "bertopic_kmeans": "bertopic_kmeans.pkl",
                "spherical_kmeans": f"{category}.pkl",
                "gaussian_kmeans": f"{category}.pkl",
                "movmf": f"{category}.pkl",
                "gaussian_mixture": f"{category}.pkl",
            }[model]
            if resolved_condition_dir is not None:
                return resolved_condition_dir / "params" / filename
            path = (
                build_baseline_dir(
                    model=model,
                    split_root="params",
                    dataset=dataset,
                    iteration=iteration,
                    num_topics=num_topics,
                    category=category,
                    data_run=data_run,
                    condition_id=condition_id,
                    condition_payload=condition_payload,
                    baseline_root=baseline_root,
                )
                / filename
            )
            if not path.exists():
                try:
                    resolved_condition_dir = resolve_baseline_condition_dir(
                        model=model,
                        dataset=dataset,
                        iteration=iteration,
                        num_topics=num_topics,
                        category=category,
                        data_run=data_run,
                        condition_id=condition_id,
                        num_components=num_components,
                        embedding_variant=embedding_variant,
                        baseline_root=baseline_root,
                    )
                    return resolved_condition_dir / "params" / filename
                except MissingArtifactError as exc:
                    if _is_ambiguous_artifact_error(exc):
                        raise
                    pass
            return path
        if resolved_condition_dir is not None:
            infer_dir = resolved_condition_dir / "infer"
            if prefer_soft:
                return infer_dir / f"{category}_doc_topic_soft.pkl"
            return infer_dir / f"{category}.pkl"
        infer_dir = build_baseline_dir(
            model=model,
            split_root="infer",
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            condition_id=condition_id,
            condition_payload=condition_payload,
            baseline_root=baseline_root,
        )
        if not infer_dir.exists():
            try:
                infer_dir = (
                    resolve_baseline_condition_dir(
                        model=model,
                        dataset=dataset,
                        iteration=iteration,
                        num_topics=num_topics,
                        category=category,
                        data_run=data_run,
                        condition_id=condition_id,
                        num_components=num_components,
                        embedding_variant=embedding_variant,
                        baseline_root=baseline_root,
                    )
                    / "infer"
                )
            except MissingArtifactError as exc:
                if _is_ambiguous_artifact_error(exc):
                    raise
                pass
        if prefer_soft:
            return infer_dir / f"{category}_doc_topic_soft.pkl"
        return infer_dir / f"{category}.pkl"

    if model == "senclu":
        split_root = "params" if split == "train" else "infer"
        if resolved_condition_dir is not None:
            return resolved_condition_dir / split_root / f"{category}.pkl"
        path = (
            build_baseline_dir(
                model=model,
                split_root=split_root,
                dataset=dataset,
                iteration=iteration,
                num_topics=num_topics,
                category=category,
                data_run=data_run,
                condition_id=condition_id,
                condition_payload=condition_payload,
                baseline_root=baseline_root,
            )
            / f"{category}.pkl"
        )
        if not path.exists():
            try:
                resolved_condition_dir = resolve_baseline_condition_dir(
                    model=model,
                    dataset=dataset,
                    iteration=iteration,
                    num_topics=num_topics,
                    category=category,
                    data_run=data_run,
                    condition_id=condition_id,
                    num_components=num_components,
                    embedding_variant=embedding_variant,
                    baseline_root=baseline_root,
                )
                return resolved_condition_dir / split_root / f"{category}.pkl"
            except MissingArtifactError as exc:
                if _is_ambiguous_artifact_error(exc):
                    raise
                pass
        return path

    raise ValueError(f"Unsupported baseline model: {model}")


def resolve_unique_condition_dir(
    *,
    root: Path,
    metadata_path_builder: Callable[[Path], Path],
    matches_payload: Callable[[dict[str, Any]], bool],
    missing_detail: str,
    ambiguous_detail: str,
) -> Path:
    matches: list[Path] = []
    if root.exists():
        for condition_dir in sorted(root.iterdir()):
            if not condition_dir.is_dir():
                continue
            metadata_path = metadata_path_builder(condition_dir)
            if not metadata_path.exists():
                continue
            payload = load_artifact_json(metadata_path)
            if isinstance(payload, dict) and matches_payload(payload):
                matches.append(condition_dir)
    if not matches:
        raise MissingArtifactError(root, detail=missing_detail)
    if len(matches) > 1:
        raise ValueError(
            f"{ambiguous_detail} Matches: {[str(path.name) for path in matches]}"
        )
    return matches[0]


def baseline_metadata_path(condition_dir: Path) -> Path:
    direct = condition_dir / METADATA_FILENAME
    if direct.exists():
        return direct
    nested = condition_dir / "params" / METADATA_FILENAME
    if nested.exists():
        return nested
    return direct


def resolve_analysis_result_dir(
    *,
    base_root: Path,
    dataset: str,
    data_run: str,
    category: str,
    display_key: str,
    fallback_dir: Path | None = None,
) -> Path:
    latest_pointer_dir = build_latest_result_dir(
        base_root=base_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        display_key=display_key,
    )
    pointer_path = latest_pointer_dir / CURRENT_POINTER_FILENAME
    if pointer_path.exists():
        payload = load_artifact_json(pointer_path)
        if isinstance(payload, dict) and payload.get("archive_dir"):
            archive_dir = resolve_project_path(str(payload["archive_dir"]))
            if archive_dir.exists():
                return archive_dir
    if fallback_dir is not None and fallback_dir.exists():
        return fallback_dir
    if fallback_dir is not None:
        return fallback_dir
    return latest_pointer_dir


def resolve_topic_count_analysis_dir(
    *,
    dataset: str,
    data_run: str,
    category: str,
    condition_id: str,
    base_root: Path | None = None,
) -> Path:
    resolved_base_root = base_root or (RESULTS_ROOT / "topic_count_analysis")
    legacy_dir = resolved_base_root / dataset / data_run / category / condition_id
    return resolve_analysis_result_dir(
        base_root=resolved_base_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        display_key=condition_id,
        fallback_dir=legacy_dir,
    )


def resolve_cross_model_pair_diagnostics_dir(
    *,
    dataset: str,
    data_run: str,
    category: str,
    condition_id: str,
    base_root: Path | None = None,
) -> Path:
    resolved_base_root = base_root or (RESULTS_ROOT / "analysis" / "vmf_vs_baseline")
    legacy_dir = resolved_base_root / dataset / data_run / category / condition_id
    return resolve_analysis_result_dir(
        base_root=resolved_base_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        display_key=condition_id,
        fallback_dir=legacy_dir,
    )


def resolve_sentence_topic_inspection_dir(
    *,
    dataset: str,
    data_run: str,
    category: str,
    condition_id: str,
    base_root: Path | None = None,
) -> Path:
    resolved_base_root = base_root or VISUALIZATION_RESULTS_ROOT
    legacy_dir = resolved_base_root / dataset / data_run / category / condition_id
    return resolve_analysis_result_dir(
        base_root=resolved_base_root,
        dataset=dataset,
        data_run=data_run,
        category=category,
        display_key=condition_id,
        fallback_dir=legacy_dir,
    )


def resolve_vmf_experiment_dir(
    *,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    run_name: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    dataset_root: Path | None = None,
) -> Path:
    dataset_root_candidates = _vmf_dataset_root_candidates(
        dataset=dataset,
        dataset_root=dataset_root,
    )
    fallback_dataset_root = dataset_root_candidates[0]
    missing_errors: list[MissingArtifactError] = []
    metadata_num_components = 1 if num_components is None else int(num_components)

    for resolved_dataset_root in dataset_root_candidates:
        model_root = (
            resolved_dataset_root / str(run_name or "default") / "vmf_sentence_lda"
        )
        category_root = model_root / str(category)
        latest_pointer_dirs: list[Path] = []
        for component_count in _candidate_component_counts(
            requested=num_components,
            default=1,
        ):
            base_display_key = build_vmf_display_key(
                iteration=iteration,
                num_topics=num_topics,
                num_components=component_count,
            )
            latest_pointer_dirs.extend(
                _embedding_latest_dirs(
                    exact_dir=build_vmf_latest_dir(
                        category=category,
                        iteration=iteration,
                        num_topics=num_topics,
                        num_components=component_count,
                        embedding_variant=embedding_variant,
                        run_name=run_name,
                        dataset_root=resolved_dataset_root,
                    ),
                    base_display_key=base_display_key,
                    embedding_variant=embedding_variant,
                )
            )
        if condition_id is not None:
            category_first_dir = category_root / condition_id
            if category_first_dir.exists():
                return category_first_dir
            legacy_dir = legacy_vmf_experiment_dir(
                dataset_root=resolved_dataset_root,
                run_name=str(run_name or "default"),
                condition_id=condition_id,
            )
            if legacy_dir.exists():
                return legacy_dir
            archive_dir = _archive_from_latest_pointers(
                latest_pointer_dirs,
                require_condition_id=condition_id,
            )
            if archive_dir is not None:
                return archive_dir
            continue
        archive_dir = _archive_from_latest_pointers(latest_pointer_dirs)
        if archive_dir is not None:
            return archive_dir
        if not category_root.exists():
            if model_root.exists():
                try:
                    return resolve_unique_condition_dir(
                        root=model_root,
                        metadata_path_builder=lambda condition_dir: condition_dir
                        / METADATA_FILENAME,
                        matches_payload=lambda payload: (
                            isinstance(payload.get("axes"), dict)
                            and int(payload["axes"].get("iteration", -1))
                            == int(iteration)
                            and int(payload["axes"].get("num_topics", -1))
                            == int(num_topics)
                            and str(payload["axes"].get("category")) == str(category)
                            and str(payload["axes"].get("data_run", "default"))
                            == str(run_name or "default")
                            and _vmf_payload_matches_component(
                                payload,
                                metadata_num_components,
                            )
                        ),
                        missing_detail=(
                            "Expected a unique vMF condition directory matching the requested axes."
                        ),
                        ambiguous_detail=(
                            "Multiple vMF condition directories match the requested axes. "
                            "Specify condition_id to disambiguate."
                        ),
                    )
                except MissingArtifactError as exc:
                    missing_errors.append(exc)
                    continue
            continue

        try:
            return resolve_unique_condition_dir(
                root=category_root,
                metadata_path_builder=lambda condition_dir: condition_dir
                / METADATA_FILENAME,
                matches_payload=lambda payload: (
                    isinstance(payload.get("axes"), dict)
                    and int(payload["axes"].get("iteration", -1)) == int(iteration)
                    and int(payload["axes"].get("num_topics", -1)) == int(num_topics)
                    and str(payload["axes"].get("category")) == str(category)
                    and str(payload["axes"].get("data_run", "default"))
                    == str(run_name or "default")
                    and _vmf_payload_matches_component(
                        payload,
                        metadata_num_components,
                    )
                ),
                missing_detail=(
                    "Expected a unique vMF condition directory matching the requested axes."
                ),
                ambiguous_detail=(
                    "Multiple vMF condition directories match the requested axes. "
                    "Specify condition_id to disambiguate."
                ),
            )
        except MissingArtifactError:
            try:
                return resolve_unique_condition_dir(
                    root=model_root,
                    metadata_path_builder=lambda condition_dir: condition_dir
                    / METADATA_FILENAME,
                    matches_payload=lambda payload: (
                        isinstance(payload.get("axes"), dict)
                        and int(payload["axes"].get("iteration", -1)) == int(iteration)
                        and int(payload["axes"].get("num_topics", -1))
                        == int(num_topics)
                        and str(payload["axes"].get("category")) == str(category)
                        and str(payload["axes"].get("data_run", "default"))
                        == str(run_name or "default")
                        and _vmf_payload_matches_component(
                            payload,
                            metadata_num_components,
                        )
                    ),
                    missing_detail=(
                        "Expected a unique vMF condition directory matching the requested axes."
                    ),
                    ambiguous_detail=(
                        "Multiple vMF condition directories match the requested axes. "
                        "Specify condition_id to disambiguate."
                    ),
                )
            except MissingArtifactError as exc:
                missing_errors.append(exc)

    if condition_id is not None:
        return build_vmf_experiment_dir(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            run_name=run_name,
            condition_id=condition_id,
            dataset_root=fallback_dataset_root,
        )
    if missing_errors:
        raise missing_errors[-1]
    return build_vmf_experiment_dir(
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        run_name=run_name,
        dataset_root=fallback_dataset_root,
    )


def resolve_baseline_condition_dir(
    *,
    model: str,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    condition_id: str | None = None,
    num_components: int | None = None,
    embedding_variant: str | None = None,
    baseline_root: Path | None = None,
) -> Path:
    resolved_baseline_root = baseline_root or _current_baseline_results_root()
    model_root = resolved_baseline_root / dataset / data_run / model
    category_root = model_root / str(category)
    default_num_components = 1 if model == "mvtm" else None
    metadata_num_components = (
        int(num_components) if num_components is not None else default_num_components
    )
    latest_pointer_dirs: list[Path] = []
    for component_count in _candidate_component_counts(
        requested=num_components if model == "mvtm" else None,
        default=default_num_components,
    ):
        display_component_count = component_count if model == "mvtm" else None
        base_display_key = build_baseline_display_key(
            iteration=iteration,
            num_topics=num_topics,
            num_components=display_component_count,
        )
        latest_pointer_dirs.extend(
            _embedding_latest_dirs(
                exact_dir=build_baseline_latest_dir(
                    model=model,
                    dataset=dataset,
                    data_run=data_run,
                    category=category,
                    iteration=iteration,
                    num_topics=num_topics,
                    num_components=display_component_count,
                    embedding_variant=embedding_variant,
                    baseline_root=resolved_baseline_root,
                ),
                base_display_key=base_display_key,
                embedding_variant=embedding_variant,
            )
        )
    if condition_id is not None:
        category_first_dir = category_root / condition_id
        if category_first_dir.exists():
            return category_first_dir
        legacy_dir = legacy_baseline_condition_dir(
            baseline_root=resolved_baseline_root,
            dataset=dataset,
            data_run=data_run,
            model=model,
            condition_id=condition_id,
        )
        if legacy_dir.exists():
            return legacy_dir
        archive_dir = _archive_from_latest_pointers(
            latest_pointer_dirs,
            require_condition_id=condition_id,
        )
        if archive_dir is not None:
            return archive_dir
        return category_first_dir
    archive_dir = _archive_from_latest_pointers(latest_pointer_dirs)
    if archive_dir is not None:
        return archive_dir
    if not category_root.exists() and not model_root.exists():
        return build_baseline_dir(
            model=model,
            split_root="params",
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            baseline_root=baseline_root,
        ).parent

    if category_root.exists():
        try:
            return resolve_unique_condition_dir(
                root=category_root,
                metadata_path_builder=baseline_metadata_path,
                matches_payload=lambda payload: (
                    int(payload.get("iteration", -1)) == int(iteration)
                    and int(payload.get("num_topics", -1)) == int(num_topics)
                    and str(payload.get("category")) == str(category)
                    and str(payload.get("data_run", "default")) == str(data_run)
                    and _baseline_payload_matches_component(
                        payload,
                        model=model,
                        num_components=metadata_num_components,
                    )
                ),
                missing_detail=(
                    "Expected a unique baseline condition directory matching the requested axes."
                ),
                ambiguous_detail=(
                    "Multiple baseline condition directories match the requested axes. "
                    "Specify condition_id to disambiguate."
                ),
            )
        except MissingArtifactError:
            pass

    return resolve_unique_condition_dir(
        root=model_root,
        metadata_path_builder=baseline_metadata_path,
        matches_payload=lambda payload: (
            int(payload.get("iteration", -1)) == int(iteration)
            and int(payload.get("num_topics", -1)) == int(num_topics)
            and str(payload.get("category")) == str(category)
            and str(payload.get("data_run", "default")) == str(data_run)
            and _baseline_payload_matches_component(
                payload,
                model=model,
                num_components=metadata_num_components,
            )
        ),
        missing_detail=(
            "Expected a unique baseline condition directory matching the requested axes."
        ),
        ambiguous_detail=(
            "Multiple baseline condition directories match the requested axes. "
            "Specify condition_id to disambiguate."
        ),
    )

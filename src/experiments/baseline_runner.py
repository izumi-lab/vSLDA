from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from src.baselines.params import baseline_params_to_options
from src.baselines.runners import get_runner_spec
from src.core.artifacts import (
    artifact_refs_to_string_map,
    build_artifact_refs,
    load_artifact_json,
)
from src.core.result_identity import build_execution_id
from src.core.runtime import BaselineRuntimeContext, CorpusSelection, PreprocessRuntime
from src.experiments.config import BaselineConfig, resolve_targets
from src.experiments.job_planning import CategoryJob
from src.experiments.summary_schema import BaselineSummary
from src.models import ModelRunRequest, run_model_request
from src.utils.random import DEFAULT_RANDOM_SEED

SerializedArtifactPaths = Dict[str, str]


def resolve_effective_random_state(
    *,
    params: Dict[str, object],
    iteration: int,
    extra_options: Dict[str, object] | None,
) -> int:
    random_state = params.get("random_state")
    if random_state is not None:
        return int(random_state)

    options = {} if extra_options is None else dict(extra_options)
    seed = options.get("seed")
    if seed is not None:
        return int(seed)

    seed_base = options.get("seed_base")
    if seed_base is not None:
        return int(seed_base) + int(iteration)

    return int(DEFAULT_RANDOM_SEED) + int(iteration)


def build_baseline_summary(
    *,
    display_name: str,
    runner_key: str,
    runner_family: str,
    paths: SerializedArtifactPaths,
) -> BaselineSummary:
    metadata_path = paths.get("metadata")
    if metadata_path is None:
        return BaselineSummary(
            name=display_name,
            paths=paths,
            runner_key=runner_key,
            runner_family=runner_family,
        )
    metadata_payload = load_artifact_json(Path(metadata_path))
    if not isinstance(metadata_payload, dict):
        return BaselineSummary(
            name=display_name,
            paths=paths,
            runner_key=runner_key,
            runner_family=runner_family,
        )
    return BaselineSummary(
        name=display_name,
        paths=paths,
        runner_key=str(metadata_payload.get("runner_key", runner_key)),
        runner_family=str(metadata_payload.get("runner_family", runner_family)),
        parameter_variant=metadata_payload.get("parameter_variant"),
        preprocessing_variant=metadata_payload.get("preprocessing_variant"),
        baseline_params=(
            dict(metadata_payload["baseline_params"])
            if isinstance(metadata_payload.get("baseline_params"), dict)
            else None
        ),
    )


def run_baselines_for_category(
    *,
    category: str,
    dataset: str,
    runtime: BaselineRuntimeContext,
    extra_options: Dict[str, object] | None,
    num_topics: int,
    iteration: int,
    baselines: Iterable[BaselineConfig],
    logger: object,
) -> list[BaselineSummary]:
    results: list[BaselineSummary] = []
    for baseline in baselines:
        spec = get_runner_spec(baseline.runner)
        name = spec.key
        params = baseline_params_to_options(baseline.params)
        model_options: Dict[str, object] = {
            **runtime.to_model_options(),
            **({} if extra_options is None else dict(extra_options)),
            **params,
        }
        if name in {"bertopic_kmeans", "etm", "sentlda"}:
            model_options["effective_random_state"] = resolve_effective_random_state(
                params=params,
                iteration=iteration,
                extra_options=extra_options,
            )
        if name in {"etm", "sentlda"}:
            model_options["random_state"] = model_options["effective_random_state"]
        if name == "bertopic_kmeans":
            model_options["doc_topic_source"] = "umap_kmeans_centroid_softmax"
            model_options["doc_topic_space"] = "umap"
        logger.info(
            f"[baseline:{name}] category={category} topics={num_topics} it={iteration}"
        )
        request = ModelRunRequest(
            name=name,
            category=category,
            dataset=dataset,
            num_topics=num_topics,
            iteration=iteration,
            options=model_options,
        )
        artifacts = run_model_request(request)
        artifact_refs = build_artifact_refs(artifacts.as_dict())
        serialized_paths = artifact_refs_to_string_map(artifact_refs)
        results.append(
            build_baseline_summary(
                display_name=spec.display_name,
                runner_key=spec.key,
                runner_family=spec.family,
                paths=serialized_paths,
            )
        )
    return results


def run_baseline_jobs(
    *, job: CategoryJob, logger: object, started_at: str
) -> list[BaselineSummary]:
    baseline_list = job.baselines
    if job.selected_models is not None:
        baseline_list = [
            baseline
            for baseline in baseline_list
            if baseline.runner.lower() in job.selected_models
        ]
    if not baseline_list:
        return []

    cfg = job.config
    resolved_targets = resolve_targets(
        cfg.dataset,
        cfg.preprocess,
        job.category,
        job.targets,
    )
    baseline_runtime = BaselineRuntimeContext(
        corpus=CorpusSelection(
            train_csvs=tuple(job.train_csvs),
            test_csvs=tuple(job.test_csvs),
            targets=(
                None
                if resolved_targets is None
                else tuple(str(target) for target in resolved_targets)
            ),
        ),
        preprocess=PreprocessRuntime(
            text_column=cfg.preprocess.text_column,
            target_column=cfg.preprocess.target_column,
            delimiter=cfg.preprocess.delimiter,
            language=cfg.preprocess.language,
            segmenter=cfg.preprocess.segmenter,
            tokenizer=cfg.preprocess.tokenizer,
            legacy_preprocessing=cfg.preprocess.legacy_preprocessing,
            ja_replace_num=cfg.preprocess.ja_replace_num,
            ja_stopwords_path=cfg.preprocess.ja_stopwords_path,
            ja_dicdir=cfg.preprocess.ja_dicdir,
            ja_require_unidic=cfg.preprocess.ja_require_unidic,
        ),
        encoder_device=job.parallelism.encoder_device,
        runtime_num_workers=job.parallelism.baseline_num_workers,
    )
    return run_baselines_for_category(
        category=job.category,
        dataset=cfg.dataset.name,
        runtime=baseline_runtime,
        extra_options={
            "data_run": job.data_run_name,
            "started_at": started_at,
            "execution_id": build_execution_id(prefix="baseline"),
            "seed": job.seed,
            "seed_base": job.seed_base,
        },
        num_topics=job.num_topics,
        iteration=job.iteration,
        baselines=baseline_list,
        logger=logger,
    )

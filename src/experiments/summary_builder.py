from __future__ import annotations

from src.core.artifacts import artifact_refs_to_string_map, build_artifact_refs
from src.core.paths import build_vmf_display_key
from src.experiments.job_planning import CategoryJob
from src.experiments.summary_schema import (
    BaselineSummary,
    ExecutionSummary,
    PerformanceSummary,
    SummaryAxes,
    SummaryRecord,
)


def build_summary_record(
    *,
    job: CategoryJob,
    axes,
    vmf_result: dict[str, object] | None,
    baseline_results: list[BaselineSummary],
) -> SummaryRecord:
    vmf_payload = None
    if vmf_result is not None:
        vmf_payload = artifact_refs_to_string_map(build_artifact_refs(vmf_result))
    baseline_payload = list(baseline_results) if baseline_results else None
    return SummaryRecord(
        data_run=job.data_run_name,
        condition_id=build_vmf_display_key(
            iteration=job.iteration,
            num_topics=job.num_topics,
            num_components=job.config.train.num_components,
        ),
        fiscal_years=(
            None if job.fiscal_years is None else [int(y) for y in job.fiscal_years]
        ),
        train_csvs=[str(p) for p in job.train_csvs],
        test_csvs=[str(p) for p in job.test_csvs],
        category=job.category,
        num_topics=job.num_topics,
        iteration=job.iteration,
        axes=SummaryAxes(
            dataset=axes.dataset,
            model_family=axes.model_family,
            algorithm_variant=axes.algorithm_variant,
            encoder_model=axes.encoder_model,
            embedding_preprocess_variant=axes.embedding_preprocess_variant,
            num_topics=axes.num_topics,
            iteration=axes.iteration,
            category=axes.category,
            data_run=axes.data_run,
        ),
        execution=ExecutionSummary(
            requested_num_workers=job.parallelism.requested_num_workers,
            category_num_workers=job.parallelism.category_num_workers,
            baseline_num_workers=job.parallelism.baseline_num_workers,
            encoder_device=job.parallelism.encoder_device,
            run_vmf=job.parallelism.run_vmf,
            uses_cuda=job.parallelism.uses_cuda,
            reason=job.parallelism.reason,
        ),
        performance=PerformanceSummary(
            elapsed_sec=0.0,
            rss_mb_before=None,
            rss_mb_after=None,
            peak_rss_mb_before=None,
            peak_rss_mb_after=None,
            peak_rss_mb_delta=None,
        ),
        vmf=vmf_payload,
        baselines=baseline_payload,
    )

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from src.experiments import baseline_runner as baseline_runner_module
from src.experiments import vmf_runner as vmf_runner_module
from src.experiments.config import resolve_targets
from src.experiments.job_planning import CategoryJob
from src.experiments.performance import measure_runtime
from src.experiments.summary_builder import build_summary_record
from src.experiments.summary_schema import PerformanceSummary, SummaryRecord
from src.models import run_model_request
from src.utils.logging import get_logger
from src.utils.random import set_global_seed

build_experiment_axes = vmf_runner_module.build_experiment_axes
_build_vmf_run_options = vmf_runner_module.build_vmf_run_options


def run_baselines_for_category(*args, **kwargs):
    baseline_runner_module.run_model_request = run_model_request
    return baseline_runner_module.run_baselines_for_category(*args, **kwargs)


def _process_category_impl(job: CategoryJob) -> SummaryRecord:
    logger = get_logger(
        f"comparison-{job.category}-it{job.iteration}-k{job.num_topics}"
    )

    if job.seed is not None:
        set_global_seed(int(job.seed), deterministic_torch=False)
    elif job.seed_base is not None:
        set_global_seed(
            int(job.seed_base) + int(job.iteration), deterministic_torch=False
        )

    vmf_result = None
    baseline_results = []
    axes = build_experiment_axes(job)
    run_vmf = job.selected_models is None or "vmf_sentence_lda" in job.selected_models
    vmf_execution = None
    baseline_runner_module.run_model_request = run_model_request
    baseline_runner_module.resolve_targets = resolve_targets
    vmf_runner_module.run_model_request = run_model_request
    vmf_runner_module.resolve_targets = resolve_targets
    if run_vmf:
        vmf_execution = vmf_runner_module.run_vmf_job(job=job, logger=logger)
        vmf_result = vmf_execution.artifacts
        axes = vmf_execution.axes
    started_at = (
        vmf_execution.started_at
        if vmf_execution is not None
        else datetime.now(UTC).isoformat()
    )
    baseline_results = baseline_runner_module.run_baseline_jobs(
        job=job,
        logger=logger,
        started_at=started_at,
    )
    return build_summary_record(
        job=job,
        axes=axes,
        vmf_result=vmf_result,
        baseline_results=baseline_results,
    )


def process_category(job: CategoryJob) -> SummaryRecord:
    record, measurement = measure_runtime(lambda: _process_category_impl(job))
    return replace(
        record,
        performance=PerformanceSummary(
            elapsed_sec=measurement.elapsed_sec,
            rss_mb_before=measurement.rss_mb_before,
            rss_mb_after=measurement.rss_mb_after,
            peak_rss_mb_before=measurement.peak_rss_mb_before,
            peak_rss_mb_after=measurement.peak_rss_mb_after,
            peak_rss_mb_delta=measurement.peak_rss_mb_delta,
        ),
    )

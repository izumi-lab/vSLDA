from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from src.evaluation.schema import build_evaluation_payload

SUMMARY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SummaryAxes:
    dataset: str
    model_family: str
    algorithm_variant: str
    encoder_model: str
    embedding_preprocess_variant: str
    num_topics: int
    iteration: int
    category: str
    data_run: str


@dataclass(frozen=True)
class BaselineSummary:
    name: str
    paths: Dict[str, str]
    runner_key: str | None = None
    runner_family: str | None = None
    parameter_variant: str | None = None
    preprocessing_variant: str | None = None
    baseline_params: Dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionSummary:
    requested_num_workers: int
    category_num_workers: int
    baseline_num_workers: int
    encoder_device: str
    run_vmf: bool
    uses_cuda: bool
    reason: str | None = None


@dataclass(frozen=True)
class PerformanceSummary:
    elapsed_sec: float
    rss_mb_before: float | None
    rss_mb_after: float | None
    peak_rss_mb_before: float | None
    peak_rss_mb_after: float | None
    peak_rss_mb_delta: float | None


@dataclass(frozen=True)
class SummaryRecord:
    data_run: str
    condition_id: str
    fiscal_years: List[int] | None
    train_csvs: List[str]
    test_csvs: List[str]
    category: str
    num_topics: int
    iteration: int
    axes: SummaryAxes
    execution: ExecutionSummary
    performance: PerformanceSummary
    vmf: Dict[str, str] | None = None
    baselines: List[BaselineSummary] | None = None


def summary_record_to_dict(record: SummaryRecord) -> dict:
    payload = asdict(record)
    if payload.get("vmf") is None:
        payload.pop("vmf", None)
    if payload.get("baselines") is None:
        payload.pop("baselines", None)
    return payload


def build_summary_payload(
    *,
    dataset: str,
    summary_path: Path,
    records: List[SummaryRecord],
) -> Mapping[str, object]:
    return build_evaluation_payload(
        meta={
            "schema": "experiment_summary",
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "dataset": dataset,
            "summary_path": str(summary_path),
            "record_count": len(records),
        },
        results={
            "records": [summary_record_to_dict(record) for record in records],
        },
    )

from __future__ import annotations

from pathlib import Path

from src.experiments.summary_schema import (
    SUMMARY_SCHEMA_VERSION,
    BaselineSummary,
    ExecutionSummary,
    PerformanceSummary,
    SummaryAxes,
    SummaryRecord,
    build_summary_payload,
)


def test_build_summary_payload_has_version_and_records() -> None:
    record = SummaryRecord(
        data_run="default",
        condition_id="it0__k20__abcd1234",
        fiscal_years=None,
        train_csvs=["data/train.csv"],
        test_csvs=["data/test.csv"],
        category="science",
        num_topics=20,
        iteration=0,
        axes=SummaryAxes(
            dataset="20newsgroup",
            model_family="vmf_sentence_lda",
            algorithm_variant="components_1__estimate_alpha_every_1",
            encoder_model="sentence-transformers/all-mpnet-base-v2",
            embedding_preprocess_variant="none",
            num_topics=20,
            iteration=0,
            category="science",
            data_run="default",
        ),
        execution=ExecutionSummary(
            requested_num_workers=4,
            category_num_workers=4,
            baseline_num_workers=1,
            encoder_device="cuda",
            run_vmf=True,
            uses_cuda=True,
            reason=None,
        ),
        performance=PerformanceSummary(
            elapsed_sec=12.5,
            rss_mb_before=100.0,
            rss_mb_after=120.0,
            peak_rss_mb_before=130.0,
            peak_rss_mb_after=150.0,
            peak_rss_mb_delta=20.0,
        ),
        vmf={"metrics_path": "results/metrics.json"},
        baselines=[
            BaselineSummary(
                name="ctm",
                paths={"train_path": "results/baselines/ctm/train.pkl"},
                runner_key="ctm",
                runner_family="ctm",
                parameter_variant="num_epochs=50",
                preprocessing_variant="language=english",
                baseline_params={"num_epochs": 50},
            )
        ],
    )

    payload = build_summary_payload(
        dataset="20newsgroup",
        summary_path=Path("results/experiments/20newsgroup/summary.json"),
        records=[record],
    )

    assert payload["_meta"]["schema"] == "experiment_summary"
    assert payload["_meta"]["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert payload["_meta"]["dataset"] == "20newsgroup"
    assert payload["_meta"]["record_count"] == 1
    records = payload["results"]["records"]
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["axes"]["dataset"] == "20newsgroup"
    assert records[0]["execution"]["category_num_workers"] == 4
    assert records[0]["performance"]["peak_rss_mb_delta"] == 20.0
    assert records[0]["baselines"][0]["name"] == "ctm"
    assert records[0]["baselines"][0]["runner_family"] == "ctm"
    assert records[0]["baselines"][0]["baseline_params"]["num_epochs"] == 50

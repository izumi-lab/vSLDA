from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.core.artifacts import load_json, save_json
from src.evaluation.classification.summary import build_summary_report, write_summary
from src.evaluation.classification.summary_coverage import write_summary_coverage_index
from src.evaluation.classification.workflow import (
    build_classification_condition_id,
    build_classification_output_dir,
)
from src.evaluation.reporting import write_evaluation_json


def test_build_summary_report_reads_wrapped_classification_payloads(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={"science": {"ModelA": 50.0, "ModelB": 60.0}},
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "macro": {"ModelA": 40.0, "ModelB": 55.0},
                "micro": {"ModelA": 45.0, "ModelB": 58.0},
            }
        },
        path=iter_dir / "f1_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
    )

    assert report["_meta"]["task"] == "classification_summary"
    assert report["_meta"]["output_kind"] == "tabular"
    assert report["_meta"]["source_meta"]["task"] == "classification"
    assert report["results"]["models"] == ["ModelA", "ModelB"]
    science_row = next(
        row for row in report["results"]["rows"] if row["category"] == "science"
    )
    assert science_row["class_count"] == 4
    assert science_row["values"]["ModelA"] == (
        r"\underline{50.00~\ensuremath{\pm}~0.00}"
    )


def test_build_summary_report_sorts_variant_labels_by_base_model(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "Contextual TM [num_epochs=12]": 60.0,
                "Blei LDA [passes=40]": 55.0,
                "vMF Sentence LDA [SVM]": 70.0,
            }
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "macro": {
                    "Contextual TM [num_epochs=12]": 60.0,
                    "Blei LDA [passes=40]": 55.0,
                    "vMF Sentence LDA [SVM]": 70.0,
                },
                "micro": {
                    "Contextual TM [num_epochs=12]": 60.0,
                    "Blei LDA [passes=40]": 55.0,
                    "vMF Sentence LDA [SVM]": 70.0,
                },
            }
        },
        path=iter_dir / "f1_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
    )

    assert report["results"]["models"] == [
        "Blei LDA [passes=40]",
        "Contextual TM [num_epochs=12]",
        "vMF Sentence LDA [SVM]",
    ]


def test_build_summary_report_uses_configured_model_order(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "vMF Sentence LDA [SVM]": 70.0,
                "ETM [glove100]": 65.0,
                "Gaussian k-means [SVM]": 62.0,
                "sentLDA [SVM]": 60.0,
                "Blei LDA [SVM]": 58.0,
            }
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
    )

    assert report["results"]["models"] == [
        "Blei LDA [SVM]",
        "sentLDA [SVM]",
        "Gaussian k-means [SVM]",
        "ETM [glove100]",
        "vMF Sentence LDA [SVM]",
    ]


def test_write_summary_prints_complete_latex_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "vMF Sentence LDA [SVM]": 70.0,
                "Blei LDA [SVM]": 58.0,
            }
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )

    write_summary(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
    )

    output = capsys.readouterr().out
    assert output.startswith("\\begin{table}[t]\n")
    assert "\\begin{tabular}{lcc}" in output
    assert "\\caption{Classification summary: 20newsgroup, k=20, acc}" in output
    assert "\\label" not in output
    assert "Category & LDA & vSLDA" in output
    assert "Science &" in output
    assert "Science (4)" not in output
    assert "\\end{table}" in output


def test_write_summary_writes_latex_table_to_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "vMF Sentence LDA [SVM]": 70.0,
                "Blei LDA [SVM]": 58.0,
            }
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )
    output_path = tmp_path / "summaries" / "classification_summary.tex"

    write_summary(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
        output_path=output_path,
    )

    assert capsys.readouterr().out == f"[write] {output_path}\n"
    output = output_path.read_text(encoding="utf-8")
    assert output.startswith("\\begin{table}[t]\n")
    assert "\\caption{Classification summary: 20newsgroup, k=20, acc}" in output
    assert output.endswith("\\end{table}\n")
    coverage = load_json(output_path.with_suffix(".runs.json"))
    assert coverage["expected_runs"] == 1
    assert any(
        row["model"] == "Blei LDA [SVM]"
        and row["category"] == "science"
        and row["status"] == "complete"
        for row in coverage["rows"]
    )
    assert output_path.with_suffix(".runs.csv").exists()


def test_write_summary_writes_missing_coverage_for_zero_run_condition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "summaries" / "classification_summary.tex"

    write_summary(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0, 1],
        result_root=tmp_path,
        selected_models=["sentence_gaussianlda"],
        output_path=output_path,
    )

    assert "[skip] no results for 20newsgroup 20topic\n" in capsys.readouterr().out
    assert output_path.read_text(encoding="utf-8").startswith("% no results")
    coverage = load_json(output_path.with_suffix(".runs.json"))
    assert coverage["rows"] == [
        {
            "metric": "acc",
            "dataset": "20newsgroup",
            "data_run": "default",
            "topics": 20,
            "classifiers": "",
            "vmf_assignment": "hard",
            "embedding_variants": "",
            "selector": "sentence_gaussianlda",
            "model": "",
            "category": "",
            "run_count": 0,
            "expected_runs": 2,
            "missing_runs": 2,
            "status": "missing",
        }
    ]


def test_write_summary_uses_short_model_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "Blei LDA [SVM]": 50.0,
                "sentLDA [SVM]": 51.0,
                "Gaussian k-means [SVM]": 52.0,
                "Spherical k-means [SVM]": 53.0,
                "Gaussian mixture [SVM]": 54.0,
                "movMF [SVM]": 55.0,
                "Gaussian LDA [SVM]": 56.0,
                "MvTM [SVM]": 57.0,
                "ETM [glove100] [SVM]": 58.0,
                "Contextual TM [mpnet] [SVM]": 59.0,
                "SenClu [mpnet] [SVM]": 60.0,
                "BERTopic (UMAP + k-means) [mpnet] [SVM]": 61.0,
                "Sentence LDA [mpnet] [SVM]": 62.0,
                "vMF Sentence LDA [mpnet] [SVM]": 63.0,
            }
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )

    write_summary(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
    )

    output = capsys.readouterr().out
    assert (
        "Category & LDA & SLDA & GCLU & SCLU & MGCLU & MSCLU & GLDA & "
        "vLDA & ETM & CTM & SenClu & BERTopic & GSLDA & vSLDA"
    ) in output


def test_build_summary_report_filters_selected_models_by_short_label(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "Blei LDA [SVM]": 50.0,
                "Contextual TM [mpnet] [SVM]": 60.0,
                "vMF Sentence LDA [mpnet] [SVM]": 70.0,
            }
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
        selected_models=["LDA", "vSLDA"],
    )

    assert report["results"]["models"] == [
        "Blei LDA [SVM]",
        "vMF Sentence LDA [mpnet] [SVM]",
    ]


def test_build_summary_report_filters_sentence_gaussianlda_by_raw_selector(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {
                "Sentence LDA [mpnet_raw] [SVM]": 62.0,
                "Blei LDA [SVM]": 50.0,
            }
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
        selected_models=["sentence_gaussianlda"],
    )

    assert report["results"]["models"] == ["Sentence LDA [mpnet_raw] [SVM]"]


def test_build_summary_report_excludes_selected_categories(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "nyt", "iteration": 0},
        results={
            "science": {"Blei LDA [SVM]": 50.0},
            "sports": {"Blei LDA [SVM]": 60.0},
        },
        path=iter_dir / "acc_nyt_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="nyt",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
        excluded_categories=["science"],
    )

    categories = [row["category"] for row in report["results"]["rows"]]
    assert "science" not in categories
    assert "sports" in categories
    assert report["_meta"]["excluded_categories"] == ["science"]


def test_build_summary_report_excludes_all_category_by_default(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {"Blei LDA [SVM]": 50.0},
            "all": {"Blei LDA [SVM]": 60.0},
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
    )

    categories = [row["category"] for row in report["results"]["rows"]]
    assert "science" in categories
    assert "all" not in categories
    assert report["_meta"]["include_all_category"] is False


def test_build_summary_report_can_include_all_category(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={
            "science": {"Blei LDA [SVM]": 50.0},
            "all": {"Blei LDA [SVM]": 60.0},
        },
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0],
        result_root=tmp_path,
        include_all_category=True,
    )

    categories = [row["category"] for row in report["results"]["rows"]]
    assert "science" in categories
    assert "all" in categories
    assert report["_meta"]["include_all_category"] is True


def test_build_summary_report_aggregates_feature_catalog_by_category(
    tmp_path: Path,
) -> None:
    iter0_dir = tmp_path / "iter0"
    iter1_dir = tmp_path / "iter1"
    meta = {
        "task": "classification",
        "dataset": "20newsgroup",
        "categories": {
            "science": {
                "feature_catalog": [
                    {
                        "feature_name": "Contextual TM [num_epochs=12]",
                        "model_key": "ctm",
                        "runner_family": "ctm",
                        "parameter_variant": "num_epochs=12",
                    }
                ]
            }
        },
    }
    write_evaluation_json(
        meta=meta,
        results={"science": {"Contextual TM [num_epochs=12]": 60.0}},
        path=iter0_dir / "acc_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta=meta,
        results={
            "science": {
                "macro": {"Contextual TM [num_epochs=12]": 60.0},
                "micro": {"Contextual TM [num_epochs=12]": 60.0},
            }
        },
        path=iter0_dir / "f1_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta=meta,
        results={"science": {"Contextual TM [num_epochs=12]": 62.0}},
        path=iter1_dir / "acc_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta=meta,
        results={
            "science": {
                "macro": {"Contextual TM [num_epochs=12]": 62.0},
                "micro": {"Contextual TM [num_epochs=12]": 62.0},
            }
        },
        path=iter1_dir / "f1_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0, 1],
        result_root=tmp_path,
    )

    assert report["_meta"]["feature_catalog_by_category"] == {
        "science": [
            {
                "feature_name": "Contextual TM [num_epochs=12]",
                "model_key": "ctm",
                "runner_family": "ctm",
                "parameter_variant": "num_epochs=12",
            }
        ]
    }


def test_build_summary_report_reads_new_dataset_data_run_layout(
    tmp_path: Path,
) -> None:
    """Confirms summary builder reads the dataset/data_run/category/condition layout.

    This is the canonical layout-resolution case. Strict-mode ambiguity and
    latest-pointer priority have dedicated tests below.
    """
    condition_id, _ = build_classification_condition_id(
        dataset="20newsgroup",
        data_run="fy2024",
        topics=20,
        iteration=0,
        classifiers=["svm"],
        vmf_assignment="soft",
        target_column="target_str",
        label_schema="identity",
    )
    out_dir = build_classification_output_dir(
        result_root=tmp_path,
        dataset="20newsgroup",
        data_run="fy2024",
        condition_id=condition_id,
    )
    write_evaluation_json(
        meta={
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "soft",
            "classifiers": ["svm"],
        },
        results={"science": {"ModelA": 50.0}},
        path=out_dir / "acc_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta={
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "soft",
            "classifiers": ["svm"],
        },
        results={"science": {"macro": {"ModelA": 50.0}, "micro": {"ModelA": 50.0}}},
        path=out_dir / "f1_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        data_run="fy2024",
        topics=20,
        iterations=[0],
        classifiers=["svm"],
        vmf_assignment="soft",
        result_root=tmp_path,
    )

    assert report["_meta"]["data_run"] == "fy2024"
    assert report["_meta"]["vmf_assignment"] == "soft"
    assert report["results"]["models"] == ["ModelA"]


def test_build_summary_report_prefers_latest_pointer_layout(
    tmp_path: Path,
) -> None:
    archive_dir = tmp_path / "archive_run"
    write_evaluation_json(
        meta={
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "soft",
            "classifiers": ["svm"],
            "started_at": "2026-04-10T02:15:30+00:00",
        },
        results={"science": {"ModelLatest": 50.0}},
        path=archive_dir / "acc_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta={
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "soft",
            "classifiers": ["svm"],
            "started_at": "2026-04-10T02:15:30+00:00",
        },
        results={
            "science": {
                "macro": {"ModelLatest": 50.0},
                "micro": {"ModelLatest": 50.0},
            }
        },
        path=archive_dir / "f1_20newsgroup_20topic.json",
    )
    save_json(
        {
            "schema": "latest_result_pointer",
            "schema_version": 1,
            "task": "classification",
            "display_key": "svm_soft_k20_it0",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "category": "all",
            "archive_dir": str(archive_dir),
            "started_at": "2026-04-10T02:15:30+00:00",
            "execution_id": "exec_20260410T021530Z",
            "condition_fingerprint": "abcd1234ef",
            "artifacts": {
                "acc": "acc_20newsgroup_20topic.json",
                "f1": "f1_20newsgroup_20topic.json",
            },
        },
        tmp_path
        / "latest"
        / "20newsgroup"
        / "fy2024"
        / "all"
        / "svm_soft_k20_it0"
        / "CURRENT.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        data_run="fy2024",
        topics=20,
        iterations=[0],
        classifiers=["svm"],
        vmf_assignment="soft",
        result_root=tmp_path,
    )

    assert report["results"]["models"] == ["ModelLatest"]


def test_build_summary_report_collects_selected_latest_pointer_models(
    tmp_path: Path,
) -> None:
    cases = [
        (
            "svm_hard_models-bleilda_k20_it0",
            "archive_bleilda",
            {
                "selected_models": ["bleilda"],
                "embedding_variants": None,
                "started_at": "2026-04-10T01:00:00+00:00",
            },
            {"science": {"Blei LDA [SVM]": 40.0}},
        ),
        (
            "svm_hard_emb-mpnet_models-vmf-sentence-lda_k20_it0",
            "archive_vmf_mpnet",
            {
                "selected_models": ["vmf_sentence_lda"],
                "embedding_variants": ["mpnet"],
                "started_at": "2026-04-10T02:00:00+00:00",
            },
            {"science": {"vMF Sentence LDA [c1_mpnet] [SVM]": 70.0}},
        ),
        (
            "svm_hard_emb-bge_models-vmf-sentence-lda_k20_it0",
            "archive_vmf_bge",
            {
                "selected_models": ["vmf_sentence_lda"],
                "embedding_variants": ["bge"],
                "started_at": "2026-04-10T03:00:00+00:00",
            },
            {"science": {"vMF Sentence LDA [c1_bge] [SVM]": 10.0}},
        ),
    ]
    for display_key, archive_name, meta_extra, results in cases:
        archive_dir = tmp_path / archive_name
        meta = {
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "hard",
            "classifiers": ["svm"],
            **meta_extra,
        }
        write_evaluation_json(
            meta=meta,
            results=results,
            path=archive_dir / "acc_20newsgroup_20topic.json",
        )
        save_json(
            {
                "schema": "latest_result_pointer",
                "schema_version": 1,
                "task": "classification",
                "display_key": display_key,
                "dataset": "20newsgroup",
                "data_run": "fy2024",
                "category": "all",
                "archive_dir": str(archive_dir),
                "started_at": meta_extra["started_at"],
                "execution_id": display_key,
                "condition_fingerprint": display_key,
                "artifacts": {
                    "acc": "acc_20newsgroup_20topic.json",
                },
            },
            tmp_path
            / "latest"
            / "20newsgroup"
            / "fy2024"
            / "all"
            / display_key
            / "CURRENT.json",
        )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        data_run="fy2024",
        topics=20,
        iterations=[0],
        classifiers=["svm"],
        vmf_assignment="hard",
        result_root=tmp_path,
        embedding_variants=["mpnet"],
        selected_models=["bleilda", "vmf_sentence_lda"],
    )

    assert report["results"]["models"] == [
        "Blei LDA [SVM]",
        "vMF Sentence LDA [c1_mpnet] [SVM]",
    ]
    science_row = next(
        row for row in report["results"]["rows"] if row["category"] == "science"
    )
    assert science_row["values"]["Blei LDA [SVM]"] == (
        r"\underline{40.00~\ensuremath{\pm}~0.00}"
    )
    assert science_row["values"]["vMF Sentence LDA [c1_mpnet] [SVM]"] == (
        r"\textbf{70.00~\ensuremath{\pm}~0.00}"
    )


def test_build_summary_report_run_coverage_flags_partial_and_missing_models(
    tmp_path: Path,
) -> None:
    iter_dir = tmp_path / "iter0"
    write_evaluation_json(
        meta={"task": "classification", "dataset": "20newsgroup", "iteration": 0},
        results={"science": {"Blei LDA [SVM]": 50.0}},
        path=iter_dir / "acc_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        topics=20,
        iterations=[0, 1],
        result_root=tmp_path,
        selected_models=["bleilda", "sentence_gaussianlda"],
    )

    coverage = report["_meta"]["run_coverage"]
    assert coverage["expected_runs"] == 2
    coverage_rows = {
        (row["selector"], row["model"], row["category"]): row
        for row in coverage["rows"]
    }
    bleilda_row = coverage_rows[("bleilda", "Blei LDA [SVM]", "science")]
    assert bleilda_row["run_count"] == 1
    assert bleilda_row["missing_runs"] == 1
    assert bleilda_row["status"] == "partial"
    missing_row = coverage_rows[("sentence_gaussianlda", "", "")]
    assert missing_row["run_count"] == 0
    assert missing_row["missing_runs"] == 2
    assert missing_row["status"] == "missing"


def test_build_summary_report_ignores_limited_classification_outputs(
    tmp_path: Path,
) -> None:
    for display_key, archive_name, meta_extra, score in (
        (
            "svm_hard_models-bleilda_k20_it0",
            "archive_full",
            {"started_at": "2026-04-10T01:00:00+00:00"},
            50.0,
        ),
        (
            "svm_hard_models-bleilda_ratio_0-05_stratified_k20_it0",
            "archive_limited",
            {
                "mode": "ratio",
                "value": 0.05,
                "started_at": "2026-04-10T02:00:00+00:00",
            },
            5.0,
        ),
    ):
        archive_dir = tmp_path / archive_name
        write_evaluation_json(
            meta={
                "task": "classification",
                "dataset": "20newsgroup",
                "data_run": "fy2024",
                "iteration": 0,
                "topics": 20,
                "vmf_assignment": "hard",
                "classifiers": ["svm"],
                "selected_models": ["bleilda"],
                **meta_extra,
            },
            results={"science": {"Blei LDA [SVM]": score}},
            path=archive_dir / "acc_20newsgroup_20topic.json",
        )
        save_json(
            {
                "schema": "latest_result_pointer",
                "schema_version": 1,
                "task": "classification",
                "display_key": display_key,
                "dataset": "20newsgroup",
                "data_run": "fy2024",
                "category": "all",
                "archive_dir": str(archive_dir),
                "started_at": meta_extra["started_at"],
                "execution_id": display_key,
                "condition_fingerprint": display_key,
                "artifacts": {"acc": "acc_20newsgroup_20topic.json"},
            },
            tmp_path
            / "latest"
            / "20newsgroup"
            / "fy2024"
            / "all"
            / display_key
            / "CURRENT.json",
        )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        data_run="fy2024",
        topics=20,
        iterations=[0],
        classifiers=["svm"],
        vmf_assignment="hard",
        result_root=tmp_path,
        selected_models=["bleilda"],
    )

    science_row = next(
        row for row in report["results"]["rows"] if row["category"] == "science"
    )
    assert science_row["values"]["Blei LDA [SVM]"] == (
        r"\textbf{50.00~\ensuremath{\pm}~0.00}"
    )


def test_build_summary_report_picks_newest_match_when_multiple_exist(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "20newsgroup" / "fy2024" / "all" / "cond_old"
    second_dir = tmp_path / "20newsgroup" / "fy2024" / "all" / "cond_new"
    write_evaluation_json(
        meta={
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "soft",
            "classifiers": ["svm"],
            "started_at": "2026-04-10T01:00:00+00:00",
        },
        results={"science": {"ModelOld": 40.0}},
        path=first_dir / "acc_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta={
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "soft",
            "classifiers": ["svm"],
            "started_at": "2026-04-10T01:00:00+00:00",
        },
        results={"science": {"macro": {"ModelOld": 40.0}, "micro": {"ModelOld": 40.0}}},
        path=first_dir / "f1_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta={
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "soft",
            "classifiers": ["svm"],
            "started_at": "2026-04-10T03:00:00+00:00",
        },
        results={"science": {"ModelNew": 55.0}},
        path=second_dir / "acc_20newsgroup_20topic.json",
    )
    write_evaluation_json(
        meta={
            "task": "classification",
            "dataset": "20newsgroup",
            "data_run": "fy2024",
            "iteration": 0,
            "topics": 20,
            "vmf_assignment": "soft",
            "classifiers": ["svm"],
            "started_at": "2026-04-10T03:00:00+00:00",
        },
        results={"science": {"macro": {"ModelNew": 55.0}, "micro": {"ModelNew": 55.0}}},
        path=second_dir / "f1_20newsgroup_20topic.json",
    )

    report = build_summary_report(
        metric="acc",
        dataset="20newsgroup",
        data_run="fy2024",
        topics=20,
        iterations=[0],
        classifiers=["svm"],
        vmf_assignment="soft",
        result_root=tmp_path,
    )

    assert report["results"]["models"] == ["ModelNew"]


def test_build_summary_report_strict_mode_rejects_multiple_legacy_matches(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "20newsgroup" / "fy2024" / "all" / "cond_old"
    second_dir = tmp_path / "20newsgroup" / "fy2024" / "all" / "cond_new"
    for out_dir, model_name, started_at in (
        (first_dir, "ModelOld", "2026-04-10T01:00:00+00:00"),
        (second_dir, "ModelNew", "2026-04-10T03:00:00+00:00"),
    ):
        write_evaluation_json(
            meta={
                "task": "classification",
                "dataset": "20newsgroup",
                "data_run": "fy2024",
                "iteration": 0,
                "topics": 20,
                "vmf_assignment": "soft",
                "classifiers": ["svm"],
                "started_at": started_at,
            },
            results={"science": {model_name: 50.0}},
            path=out_dir / "acc_20newsgroup_20topic.json",
        )
        write_evaluation_json(
            meta={
                "task": "classification",
                "dataset": "20newsgroup",
                "data_run": "fy2024",
                "iteration": 0,
                "topics": 20,
                "vmf_assignment": "soft",
                "classifiers": ["svm"],
                "started_at": started_at,
            },
            results={
                "science": {
                    "macro": {model_name: 50.0},
                    "micro": {model_name: 50.0},
                }
            },
            path=out_dir / "f1_20newsgroup_20topic.json",
        )

    with pytest.raises(ValueError, match="Multiple classification metric matches"):
        build_summary_report(
            metric="acc",
            dataset="20newsgroup",
            data_run="fy2024",
            topics=20,
            iterations=[0],
            classifiers=["svm"],
            vmf_assignment="soft",
            result_root=tmp_path,
            resolve_mode="strict",
        )


def test_write_summary_coverage_index_collects_incomplete_rows(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "nyt" / "default" / "svm" / "mpnet" / "acc_nyt.runs.json"
    second_path = (
        tmp_path / "20newsgroup" / "default" / "svm" / "mpnet" / "acc_20news.runs.json"
    )
    save_json(
        {
            "rows": [
                {
                    "metric": "acc",
                    "dataset": "nyt",
                    "data_run": "default",
                    "topics": 10,
                    "classifiers": "svm",
                    "vmf_assignment": "hard",
                    "embedding_variants": "mpnet",
                    "selector": "bleilda",
                    "model": "Blei LDA [SVM]",
                    "category": "sports",
                    "run_count": 5,
                    "expected_runs": 5,
                    "missing_runs": 0,
                    "status": "complete",
                }
            ]
        },
        first_path,
    )
    save_json(
        {
            "rows": [
                {
                    "metric": "acc",
                    "dataset": "20newsgroup",
                    "data_run": "default",
                    "topics": 10,
                    "classifiers": "svm",
                    "vmf_assignment": "hard",
                    "embedding_variants": "mpnet",
                    "selector": "sentence_gaussianlda",
                    "model": "",
                    "category": "",
                    "run_count": 0,
                    "expected_runs": 5,
                    "missing_runs": 5,
                    "status": "missing",
                }
            ]
        },
        second_path,
    )

    output_path, incomplete_path, row_count, incomplete_count = (
        write_summary_coverage_index(summary_root=tmp_path)
    )

    assert row_count == 2
    assert incomplete_count == 1
    with incomplete_path.open(newline="", encoding="utf-8") as handle:
        incomplete_rows = list(csv.DictReader(handle))
    assert incomplete_rows[0]["selector"] == "sentence_gaussianlda"
    assert incomplete_rows[0]["status"] == "missing"
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["summary_path"].endswith("acc_20news.tex")

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib
import pytest

matplotlib.use("Agg")

from src.core.vmf_assignment import DEFAULT_VMF_ASSIGNMENT
from src.evaluation.reports.model_style import canonical_model_key, label_for_key
from src.evaluation.reports.topic_sweep import (
    SWEEP_FIELDS,
    build_topic_sweep_latex,
    collect_topic_sweep_rows,
    derive_topic_quality_rows,
    resolve_categories,
    run_topic_sweep_summary,
)

DATASET = "20newsgroup"
MODELS = ("vmf_sentence_lda", "bleilda")
# Display names as the classification summaries spell them.
CLASSIFICATION_MODELS = {
    "vmf_sentence_lda": "vMF Sentence LDA [c1_minilm] [SVM]",
    "bleilda": "Blei LDA [SVM]",
}
ITERATIONS = [0, 1]
COHERENCE_FIELDS = [
    "dataset",
    "data_run",
    "category",
    "num_topics",
    "model",
    "runner_family",
    "iterations",
    "iteration_count",
    "condition_id",
    "started_at",
    "embedding_variant",
    "effective_embedding_variant",
    "encoder_model",
    "coherence_c_npmi_mean",
    "coherence_c_npmi_values",
    "topic_utilization_mean",
    "topic_utilization_values",
]


def _write_scores_json(
    root: Path,
    *,
    topics: int,
    categories: Sequence[str],
    metric: str = "acc",
    assignment: str = DEFAULT_VMF_ASSIGNMENT,
) -> Path:
    directory = root / DATASET / "default" / "svm" / "minilm"
    directory.mkdir(parents=True, exist_ok=True)
    scores = {
        category: {
            display: [float(topics + index) for index in range(len(ITERATIONS))]
            for display in CLASSIFICATION_MODELS.values()
        }
        for category in categories
    }
    provenance = {
        category: {
            display: {
                "embedding_variant": "c1_minilm" if "vMF" in display else None,
                "source_condition_id": f"it0__k{topics}__{display[:4]}",
            }
            for display in CLASSIFICATION_MODELS.values()
        }
        for category in categories
    }
    path = (
        directory
        / f"{metric}_{DATASET}_default_svm_minilm_{assignment}_{topics}topic.scores.json"
    )
    path.write_text(
        json.dumps(
            {
                "metric": metric,
                "dataset": DATASET,
                "data_run": "default",
                "topics": topics,
                "iterations": ITERATIONS,
                "classifiers": ["svm"],
                "embedding_variants": ["minilm"],
                "models": list(CLASSIFICATION_MODELS.values()),
                "categories": list(categories),
                "scores": scores,
                "provenance": provenance,
            }
        )
    )
    return path


def _write_coherence_summary(
    path: Path,
    *,
    topics: Sequence[int],
    categories: Sequence[str],
    started_at: str = "2026-01-01T00:00:00Z",
    npmi: float = 0.1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for category in categories:
        for topic in topics:
            for model in MODELS:
                # The coherence summary writes the proposed method as ``vmf``
                # with an empty runner family.
                is_vmf = model == "vmf_sentence_lda"
                rows.append(
                    {
                        "dataset": DATASET,
                        "data_run": "default",
                        "category": category,
                        "num_topics": topic,
                        "model": "vmf" if is_vmf else model,
                        "runner_family": "" if is_vmf else model,
                        "iterations": ";".join(str(it) for it in ITERATIONS),
                        "iteration_count": len(ITERATIONS),
                        "condition_id": f"it0__k{topic}__{model}",
                        "started_at": started_at,
                        "embedding_variant": "minilm" if is_vmf else "",
                        "effective_embedding_variant": "minilm" if is_vmf else "",
                        "encoder_model": "minilm" if is_vmf else "",
                        "coherence_c_npmi_mean": npmi,
                        "coherence_c_npmi_values": ";".join(
                            str(npmi + index) for index in range(len(ITERATIONS))
                        ),
                        "topic_utilization_mean": 0.5,
                        "topic_utilization_values": ";".join("0.5" for _ in ITERATIONS),
                    }
                )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COHERENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture()
def sweep_inputs(tmp_path: Path) -> dict[str, Path]:
    classification_root = tmp_path / "classification"
    for topics in (10, 20):
        _write_scores_json(classification_root, topics=topics, categories=["computer"])
    coherence_summary = _write_coherence_summary(
        tmp_path / "coherence" / "summary.csv",
        topics=(10, 20),
        categories=["computer"],
    )
    return {
        "classification_root": classification_root,
        "coherence_summary": coherence_summary,
        "out_root": tmp_path / "out",
    }


def test_canonical_model_key_folds_every_spelling() -> None:
    assert canonical_model_key("vmf") == "vmf_sentence_lda"
    assert (
        canonical_model_key("vMF Sentence LDA [c1_minilm] [SVM]") == "vmf_sentence_lda"
    )
    assert (
        canonical_model_key("vMF Sentence LDA (fold-in counts) [c1_minilm] [LogReg]")
        == "vmf_sentence_lda"
    )
    assert canonical_model_key("Blei LDA [SVM]") == "bleilda"
    assert canonical_model_key("bleilda") == "bleilda"
    assert canonical_model_key("ETM [googlenews300] [LogReg]") == "etm"
    assert label_for_key("bleilda") == "LDA"


def test_collect_joins_both_sources_with_label_counts(sweep_inputs) -> None:
    collection = collect_topic_sweep_rows(
        datasets=[DATASET],
        topics=[10, 20],
        models=MODELS,
        categories=["computer"],
        include_all_category=False,
        classification_root=sweep_inputs["classification_root"],
        coherence_summary=sweep_inputs["coherence_summary"],
        word_based_metrics=["coherence_c_npmi", "topic_utilization"],
        classification_metrics=["acc"],
    )
    # 2 models x 2 K x 2 seeds, for acc plus two word-based metrics.
    assert len(collection.rows) == 2 * 2 * 2 * 3
    assert {row["model"] for row in collection.rows} == set(MODELS)
    assert {row["source"] for row in collection.rows} == {
        "classification",
        "word_based",
    }
    # ``computer`` holds five 20 Newsgroups labels, so K=10 is two per label.
    for row in collection.rows:
        assert row["num_labels"] == 5
        assert row["topics_per_label"] == pytest.approx(row["num_topics"] / 5)
    assert not collection.missing


def test_missing_conditions_are_recorded_not_raised(sweep_inputs) -> None:
    collection = collect_topic_sweep_rows(
        datasets=[DATASET],
        topics=[10, 20, 30],  # K=30 was never run
        models=MODELS,
        categories=["computer"],
        include_all_category=True,  # ``all`` was never run either
        classification_root=sweep_inputs["classification_root"],
        coherence_summary=sweep_inputs["coherence_summary"],
    )
    missing = {(entry["category"], entry["num_topics"]) for entry in collection.missing}
    assert ("computer", 30) in missing
    assert ("all", 10) in missing
    assert ("computer", 10) not in missing
    assert collection.rows


def test_duplicate_conditions_keep_the_newest_execution(tmp_path: Path) -> None:
    older = _write_coherence_summary(
        tmp_path / "old.csv", topics=(10,), categories=["computer"], npmi=0.1
    )
    newer = tmp_path / "summary.csv"
    _write_coherence_summary(
        newer,
        topics=(10,),
        categories=["computer"],
        started_at="2026-06-01T00:00:00Z",
        npmi=0.9,
    )
    # Concatenate both runs of the same condition into one summary, as happens
    # when two hosts compute it under different execution ids.
    with newer.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COHERENCE_FIELDS)
        with older.open(newline="") as old_handle:
            writer.writerows(list(csv.DictReader(old_handle)))

    collection = collect_topic_sweep_rows(
        datasets=[DATASET],
        topics=[10],
        models=["bleilda"],
        categories=["computer"],
        include_all_category=False,
        classification_root=tmp_path / "missing",
        coherence_summary=newer,
        word_based_metrics=["coherence_c_npmi"],
        classification_metrics=[],
    )
    assert len(collection.rows) == len(ITERATIONS)
    assert min(row["value"] for row in collection.rows) == pytest.approx(0.9)


def test_resolve_categories_opts_into_all_explicitly() -> None:
    without = resolve_categories(DATASET, categories=None, include_all_category=False)
    assert "all" not in without
    assert "computer" in without
    with_all = resolve_categories(DATASET, categories=None, include_all_category=True)
    assert with_all[-1] == "all"


def test_latex_marks_the_best_model_per_topic_column() -> None:
    rows = [
        {
            "dataset": DATASET,
            "category": "computer",
            "metric": "acc",
            "model": model,
            "num_topics": topics,
            "value": value,
        }
        for model, topics, value in [
            ("bleilda", 10, 40.0),
            ("vmf_sentence_lda", 10, 50.0),
            ("bleilda", 20, 60.0),
            ("vmf_sentence_lda", 20, 55.0),
        ]
    ]
    table = build_topic_sweep_latex(
        rows, dataset=DATASET, metric="acc", topics=[10, 20]
    )
    assert r"\textbf{50.000" in table
    assert r"\textbf{60.000" in table
    assert "$K=10$" in table and "$K=20$" in table


def test_run_writes_table_provenance_and_figures(sweep_inputs) -> None:
    out_root = run_topic_sweep_summary(
        datasets=[DATASET],
        topics=[10, 20, 30],
        models=MODELS,
        categories=["computer"],
        include_all_category=False,
        classification_root=sweep_inputs["classification_root"],
        coherence_summary=sweep_inputs["coherence_summary"],
        word_based_metrics=["coherence_c_npmi", "topic_utilization"],
        classification_metrics=["acc"],
        out_root=sweep_inputs["out_root"],
        plot=True,
    )
    dataset_dir = Path(out_root) / DATASET

    with (dataset_dir / "topic_sweep.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == SWEEP_FIELDS
        table_rows = list(reader)
    assert len(table_rows) == 2 * 2 * 2 * 3

    payload = json.loads((dataset_dir / "topic_sweep.json").read_text())
    meta = payload["_meta"]
    assert meta["task"] == "topic_sweep_summary"
    assert meta["row_count"] == len(table_rows)
    assert {entry["num_topics"] for entry in meta["missing"]} == {30}
    assert payload["results"]["rows"], "source paths are recorded"

    assert (dataset_dir / "topic_sweep_acc.tex").is_file()
    figures = sorted(path.name for path in (dataset_dir / "figures").glob("*.png"))
    assert figures, "at least one figure is drawn"


def test_derive_topic_quality_multiplies_cv_by_diversity() -> None:
    # NPMI goes negative at large K, so the composite deliberately uses C_V,
    # which stays in [0, 1] and lets diversity penalise duplicated topics.
    base = {
        "dataset": "20newsgroup",
        "data_run": "default",
        "category": "all",
        "model": "etm",
        "num_topics": 300,
        "iteration": 0,
        "source": "word_based",
    }
    rows = [
        {**base, "metric": "coherence_c_v", "value": 0.5},
        {**base, "metric": "diversity", "value": 0.4},
        {**base, "metric": "coherence_c_npmi", "value": 0.037},
        # A seed missing its diversity partner yields no composite.
        {**base, "iteration": 1, "metric": "coherence_c_v", "value": 0.6},
    ]

    derived = derive_topic_quality_rows(rows)

    assert len(derived) == 1
    assert derived[0]["metric"] == "topic_quality"
    assert derived[0]["value"] == 0.2
    assert derived[0]["num_topics"] == 300


def test_sidecars_of_another_vmf_assignment_are_not_pooled(sweep_inputs) -> None:
    """A sidecar of another estimator next to the default one must stay out of the sweep."""

    _write_scores_json(
        sweep_inputs["classification_root"],
        topics=10,
        categories=["computer"],
        assignment="hard",
    )
    hard = collect_topic_sweep_rows(
        datasets=[DATASET],
        topics=[10],
        models=MODELS,
        categories=["computer"],
        include_all_category=False,
        classification_root=sweep_inputs["classification_root"],
        coherence_summary=sweep_inputs["coherence_summary"],
    )
    hard_rows = [
        row
        for row in hard.rows
        if row["source"] == "classification" and row["model"] == "vmf_sentence_lda"
    ]
    assert len(hard_rows) == len(ITERATIONS)
    default_paths = [
        str(source["path"])
        for source in hard.sources
        if source["source"] == "classification"
    ]
    assert default_paths and all(
        f"_{DEFAULT_VMF_ASSIGNMENT}_" in path for path in default_paths
    )

    other = collect_topic_sweep_rows(
        datasets=[DATASET],
        topics=[10],
        models=MODELS,
        categories=["computer"],
        include_all_category=False,
        classification_root=sweep_inputs["classification_root"],
        coherence_summary=sweep_inputs["coherence_summary"],
        vmf_assignment="hard",
    )
    other_paths = [
        str(source["path"])
        for source in other.sources
        if source["source"] == "classification"
    ]
    assert len(other_paths) == 1 and "_hard_" in other_paths[0]


def test_baseline_assignment_reads_baselines_from_the_hard_sidecars(
    sweep_inputs,
) -> None:
    """Under a fold-in estimator vSLDA comes from its own sidecar, the rest from hard."""

    _write_scores_json(
        sweep_inputs["classification_root"],
        topics=10,
        categories=["computer"],
        assignment="foldincounts",
    )
    _write_scores_json(
        sweep_inputs["classification_root"],
        topics=10,
        categories=["computer"],
        assignment="hard",
    )
    merged = collect_topic_sweep_rows(
        datasets=[DATASET],
        topics=[10],
        models=MODELS,
        categories=["computer"],
        include_all_category=False,
        classification_root=sweep_inputs["classification_root"],
        coherence_summary=sweep_inputs["coherence_summary"],
        vmf_assignment="foldincounts",
        baseline_vmf_assignment="hard",
    )
    cls_rows = [row for row in merged.rows if row["source"] == "classification"]
    by_model = {row["model"] for row in cls_rows}
    assert "vmf_sentence_lda" in by_model
    assert len(by_model) == len(
        {row["model"] for row in sweep_inputs_rows(sweep_inputs)}
    )
    paths = [
        str(source["path"])
        for source in merged.sources
        if source["source"] == "classification"
    ]
    assert any("_foldincounts_" in path for path in paths)
    assert any("_hard_" in path for path in paths)
    vmf_sources = [
        source
        for source in merged.sources
        if source["source"] == "classification"
        and "_foldincounts_" in str(source["path"])
    ]
    assert vmf_sources and all(
        source["rows"] == len(ITERATIONS) for source in vmf_sources
    )


def sweep_inputs_rows(sweep_inputs):
    hard = collect_topic_sweep_rows(
        datasets=[DATASET],
        topics=[10],
        models=MODELS,
        categories=["computer"],
        include_all_category=False,
        classification_root=sweep_inputs["classification_root"],
        coherence_summary=sweep_inputs["coherence_summary"],
    )
    return [row for row in hard.rows if row["source"] == "classification"]

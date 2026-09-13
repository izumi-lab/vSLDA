from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from src.evaluation.word_based.summary import (
    SummaryError,
    SummarySource,
    build_arm_comparison,
    build_latex_table,
    build_run_coverage,
    build_scores_payload,
    build_summary_row,
    collect_summary_rows,
    group_summary_rows,
    run_word_based_summary,
    select_rows,
)


def _write_condition(
    coherence_root: Path,
    *,
    dataset: str = "20newsgroup",
    data_run: str = "default",
    category: str = "computer",
    model: str = "vmf",
    num_topics: int = 10,
    iteration: int = 0,
    embedding_variant: str | None = "minilm",
    effective_embedding_variant: str | None = None,
    coherence_c_v: float = 0.5,
    diversity: float = 0.9,
    word2vec: str = "",
    prior_scale: float | None = None,
    started_at: str = "2026-01-01T00:00:00+00:00",
    coherence_reference_num_docs: int = 1_000_000,
    coherence_topn: int = 10,
    condition_id: str | None = None,
    execution_id: str = "exec_1",
    values: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """Write one archived run plus the latest pointer that resolves to it."""
    condition_id = condition_id or f"it{iteration}__k{num_topics}__{model}"
    archive_dir = (
        coherence_root
        / "archive"
        / "2026-01-01"
        / dataset
        / data_run
        / category
        / condition_id
        / execution_id
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    per_iteration = (
        list(values)
        if values is not None
        else [
            {
                "coherence_c_v": coherence_c_v,
                "diversity": diversity,
                "num_topics": float(num_topics),
            }
        ]
    )
    payload = {
        "_meta": {
            "task": "word_based_metrics",
            "dataset": dataset,
            "data_run": data_run,
            "category": category,
            "model": model,
            "num_topics": num_topics,
            "iterations": [iteration],
            "condition_id": condition_id,
            "display_key": condition_id,
            "execution_id": execution_id,
            "started_at": started_at,
            "embedding_variant": embedding_variant,
            "effective_embedding_variant": effective_embedding_variant,
            "prior_scale": prior_scale,
            "model_provenance": {
                "baseline_params": {"word2vec": word2vec, "prior_scale": prior_scale}
            },
            "topic_words": {"coherence_topn": coherence_topn, "diversity_topn": 25},
            "coherence": {
                "metrics": ["c_v"],
                "primary_metric": "c_v",
                "coherence_reference": "wikipedia",
                "coherence_reference_num_docs": coherence_reference_num_docs,
                "topn": coherence_topn,
                "split": "train",
            },
        },
        "results": {
            "aggregate": {
                "coherence_c_v": {"mean": coherence_c_v, "std": 0.0},
                "diversity": {"mean": diversity, "std": 0.0},
            },
            "per_iteration": per_iteration,
        },
    }
    (archive_dir / "metrics_agg.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    pointer_dir = (
        coherence_root / "latest" / dataset / data_run / category / condition_id
    )
    pointer_dir.mkdir(parents=True, exist_ok=True)
    (pointer_dir / "CURRENT.json").write_text(
        json.dumps(
            {
                "schema": "latest_result_pointer",
                "task": "word_based_metrics",
                "archive_dir": str(archive_dir),
                "artifacts": {"metrics": "metrics_agg.json"},
            }
        ),
        encoding="utf-8",
    )
    return archive_dir


def _rows(coherence_root: Path) -> list[dict[str, Any]]:
    rows, warnings = collect_summary_rows(coherence_root=coherence_root)
    assert warnings == []
    return rows


def test_build_summary_row_surfaces_coverage_for_non_mvtm_protocols(
    tmp_path: Path,
) -> None:
    # Only the MvTM protocol folds topic utilisation into ``aggregate``. Every
    # other model records it per iteration under ``coverage_by_iteration``, and
    # the summary has to surface it from there or the columns stay empty.
    metrics_path = tmp_path / "metrics_agg.json"
    metrics_path.write_text(
        json.dumps(
            {
                "_meta": {
                    "dataset": "20newsgroup",
                    "data_run": "default",
                    "category": "all",
                    "model": "vmf",
                    "num_topics": 300,
                    "iterations": [0, 1],
                },
                "results": {
                    "aggregate": {"diversity": {"mean": 0.5, "std": 0.0}},
                    "per_iteration": [{"diversity": 0.5}, {"diversity": 0.5}],
                    "coverage_by_iteration": {
                        "0": {
                            "num_active_topics": 300,
                            "num_empty_topics": 0,
                            "topic_utilization": 1.0,
                        },
                        "1": {
                            "num_active_topics": 280,
                            "num_empty_topics": 20,
                            "topic_utilization": 0.9333,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    row = build_summary_row(SummarySource(metrics_path=metrics_path))

    assert row["num_active_topics_values"] == [300.0, 280.0]
    assert row["num_empty_topics_values"] == [0.0, 20.0]
    assert row["topic_utilization_values"] == [1.0, 0.9333]
    assert row["num_active_topics_mean"] == 290.0


def test_collect_summary_rows_reads_latest_pointers(tmp_path: Path) -> None:
    _write_condition(tmp_path, coherence_c_v=0.42)
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["model"] == "vmf"
    assert rows[0]["iterations"] == [0]
    assert rows[0]["coherence_c_v_values"] == [0.42]


def test_collect_summary_rows_falls_back_to_the_condition_index(
    tmp_path: Path,
) -> None:
    archive_dir = _write_condition(tmp_path)
    # Drop the pointers so only the index remains.
    for pointer in (tmp_path / "latest").rglob("CURRENT.json"):
        pointer.unlink()
    (tmp_path / "condition_index.json").write_text(
        json.dumps(
            {
                "completed": [
                    {"metrics_path": str(archive_dir / "metrics_agg.json")},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert len(_rows(tmp_path)) == 1


def test_encoder_independent_models_join_every_encoder_group(tmp_path: Path) -> None:
    # An encoder-aware model under two encoders, and a word model recorded once
    # under whichever variant its run happened to use.
    for variant in ("minilm", "mpnet"):
        _write_condition(
            tmp_path,
            model="vmf",
            embedding_variant=variant,
            condition_id=f"it0__k10__vmf__{variant}",
        )
    _write_condition(
        tmp_path, model="bleilda", embedding_variant="mpnet", condition_id="it0__lda"
    )
    groups = {g.encoder_variant: g for g in group_summary_rows(_rows(tmp_path))}
    assert set(groups) == {"minilm", "mpnet"}
    for group in groups.values():
        assert group.models == ["bleilda", "vmf"]


def test_encoder_variant_uses_the_short_name(tmp_path: Path) -> None:
    _write_condition(
        tmp_path, model="sentence_gaussianlda", embedding_variant="minilm_raw"
    )
    _write_condition(tmp_path, model="vmf", embedding_variant="minilm")
    groups = group_summary_rows(_rows(tmp_path))
    assert [group.encoder_variant for group in groups] == ["minilm"]
    assert groups[0].models == ["sentence_gaussianlda", "vmf"]


def test_expected_iterations_are_inferred_and_can_be_pinned(tmp_path: Path) -> None:
    for iteration in (0, 1):
        _write_condition(tmp_path, iteration=iteration)
    inferred = group_summary_rows(_rows(tmp_path))[0]
    assert inferred.expected_iterations == [0, 1]

    pinned = group_summary_rows(_rows(tmp_path), iterations=[0, 1, 2, 3, 4])[0]
    assert pinned.expected_iterations == [0, 1, 2, 3, 4]
    coverage = build_run_coverage(pinned, metric="coherence_c_v")[0]
    assert (coverage["run_count"], coverage["expected_runs"]) == (2, 5)
    assert coverage["status"] == "partial"
    assert coverage["missing_runs"] == 3


def test_run_coverage_reports_complete_and_missing(tmp_path: Path) -> None:
    _write_condition(tmp_path, model="vmf", iteration=0)
    # A condition whose run produced no per-iteration value at all.
    _write_condition(
        tmp_path, model="ctm", iteration=0, values=[], condition_id="it0__ctm"
    )
    group = group_summary_rows(_rows(tmp_path), iterations=[0])[0]
    statuses = {
        row["model"]: row["status"]
        for row in build_run_coverage(group, metric="coherence_c_v")
    }
    assert statuses == {"vSLDA": "complete", "ConTM": "missing"}


def test_duplicate_iterations_keep_the_newer_run_and_warn(tmp_path: Path) -> None:
    _write_condition(
        tmp_path,
        condition_id="older",
        started_at="2026-01-01T00:00:00+00:00",
        coherence_c_v=0.1,
    )
    _write_condition(
        tmp_path,
        condition_id="newer",
        started_at="2026-02-01T00:00:00+00:00",
        coherence_c_v=0.9,
    )
    warnings: list[str] = []
    group = group_summary_rows(_rows(tmp_path), warnings=warnings)[0]
    assert any("recorded twice" in message for message in warnings)
    assert group.cell("computer", "vmf").values("coherence_c_v") == [0.9]

    with pytest.raises(SummaryError, match="recorded twice"):
        group_summary_rows(_rows(tmp_path), strict=True)


def test_runs_that_disagree_on_the_protocol_are_rejected(tmp_path: Path) -> None:
    _write_condition(tmp_path, iteration=0, coherence_topn=10)
    _write_condition(tmp_path, iteration=1, coherence_topn=20)
    group = group_summary_rows(_rows(tmp_path))[0]
    with pytest.raises(SummaryError, match="disagree on coherence_topn"):
        build_scores_payload(group, metrics=["coherence_c_v"])


def test_select_rows_excludes_the_prior_scale_sweep_by_default(
    tmp_path: Path,
) -> None:
    _write_condition(
        tmp_path, model="gaussianlda", prior_scale=None, condition_id="default"
    )
    _write_condition(
        tmp_path, model="gaussianlda", prior_scale=1.0, condition_id="psi0-1"
    )
    rows = _rows(tmp_path)
    warnings: list[str] = []
    kept = select_rows(rows, warnings=warnings)
    assert [row["condition_id"] for row in kept] == ["default"]
    assert any("prior-scale sweep" in message for message in warnings)

    swept = select_rows(rows, prior_scale=1.0)
    assert [row["condition_id"] for row in swept] == ["psi0-1"]


def test_select_rows_filters_datasets_and_word2vec(tmp_path: Path) -> None:
    _write_condition(tmp_path, dataset="nyt", category="arts")
    _write_condition(tmp_path, model="etm", word2vec="word2vec-google-news-300")
    _write_condition(
        tmp_path, model="mvtm", word2vec="other-vectors", condition_id="it0__mvtm"
    )
    rows = _rows(tmp_path)
    assert {row["dataset"] for row in select_rows(rows, datasets=["nyt"])} == {"nyt"}
    kept = select_rows(rows, word2vec="word2vec-google-news-300")
    assert "mvtm" not in {row["model"] for row in kept}


def test_scores_payload_carries_raw_values_and_provenance(tmp_path: Path) -> None:
    for iteration, value in enumerate((0.10, 0.20)):
        _write_condition(tmp_path, iteration=iteration, coherence_c_v=value)
    group = group_summary_rows(_rows(tmp_path), iterations=[0, 1])[0]
    payload = build_scores_payload(group, metrics=["coherence_c_v", "diversity"])

    assert payload["task"] == "word_based_summary"
    assert payload["iterations"] == [0, 1]
    assert payload["scores"]["coherence_c_v"]["computer"]["vmf"] == [0.10, 0.20]
    assert payload["run_iterations"]["computer"]["vmf"] == [0, 1]
    provenance = payload["provenance"]["computer"]["vmf"]
    assert provenance["coherence_reference_num_docs"] == 1_000_000
    assert provenance["coherence_topn"] == 10
    assert provenance["diversity_topn"] == 25
    assert provenance["encoder_model"] == "minilm"
    assert len(provenance["condition_ids"]) == 2
    # No aggregate crosses the boundary: consumers compute mean/std themselves.
    assert "mean" not in json.dumps(payload)


def test_latex_table_uses_the_shared_labels_and_marks(tmp_path: Path) -> None:
    _write_condition(tmp_path, model="sentlda", coherence_c_v=0.30)
    _write_condition(tmp_path, model="ctm", coherence_c_v=0.20)
    _write_condition(tmp_path, model="vmf", coherence_c_v=0.50)
    group = group_summary_rows(_rows(tmp_path), iterations=[0])[0]
    table = build_latex_table(group, metric="coherence_c_v")
    assert "SentLDA" in table and "SLDA &" not in table
    assert "ConTM" in table and "CTM &" not in table
    assert r"\textbf{0.5000~\ensuremath{\pm}~0.0000}" in table
    assert r"\underline{0.3000~\ensuremath{\pm}~0.0000}" in table


def test_run_word_based_summary_writes_the_summary_tree(tmp_path: Path) -> None:
    coherence_root = tmp_path / "coherence"
    for iteration in (0, 1):
        _write_condition(coherence_root, iteration=iteration)
        _write_condition(coherence_root, model="bleilda", iteration=iteration)
    _write_condition(coherence_root, category="all", condition_id="it0__all")

    summary_root = run_word_based_summary(
        coherence_root=coherence_root, iterations=[0, 1]
    )
    directory = summary_root / "20newsgroup" / "default" / "minilm"
    scores = json.loads(
        (
            directory / "coherence_20newsgroup_default_minilm_10topic.scores.json"
        ).read_text(encoding="utf-8")
    )
    assert scores["models"] == ["bleilda", "vmf"]
    assert scores["categories"] == ["computer"]  # 'all' is excluded by default
    assert (directory / "coherence_20newsgroup_default_c_v_minilm_10topic.tex").exists()
    assert (
        directory / "coherence_20newsgroup_default_c_v_minilm_10topic.runs.json"
    ).exists()
    assert (summary_root / "run_coverage.csv").exists()
    assert (summary_root / "all_summaries.tex").exists()
    assert (coherence_root / "summary.json").exists()
    assert (coherence_root / "summary.csv").exists()


def test_run_word_based_summary_can_keep_the_all_category(tmp_path: Path) -> None:
    coherence_root = tmp_path / "coherence"
    _write_condition(coherence_root)
    _write_condition(coherence_root, category="all", condition_id="it0__all")
    summary_root = run_word_based_summary(
        coherence_root=coherence_root, iterations=[0], include_all_category=True
    )
    scores = json.loads(
        (
            summary_root
            / "20newsgroup"
            / "default"
            / "minilm"
            / "coherence_20newsgroup_default_minilm_10topic.scores.json"
        ).read_text(encoding="utf-8")
    )
    assert scores["categories"] == ["all", "computer"]


def _arm_row(
    *, model: str, arm: str, score: float, metric: str = "coherence_c_npmi_mean"
) -> dict:
    restricted = arm != "full"
    return {
        "dataset": "20newsgroup",
        "data_run": "default",
        "category": "computer",
        "num_topics": 20,
        "model": model,
        "encoder_model": "minilm",
        "vocabulary_arm": arm,
        "reference_min_df": 50 if restricted else 0,
        "reference_max_df_ratio": 0.30 if restricted else 1.0,
        metric: score,
    }


def test_arm_comparison_pairs_full_and_filtered_rows() -> None:
    rows = [
        _arm_row(model="vmf", arm="full", score=-0.15),
        _arm_row(model="vmf", arm="refdf50-30", score=-0.03),
        _arm_row(model="etm", arm="full", score=0.04),
        _arm_row(model="etm", arm="refdf50-30", score=0.041),
    ]

    comparison = [
        row
        for row in build_arm_comparison(rows)
        if row["metric"] == "coherence_c_npmi_mean"
    ]
    by_model = {row["model"]: row for row in comparison}

    assert by_model["vmf"]["score_full"] == -0.15
    assert by_model["vmf"]["score_filtered"] == -0.03
    assert by_model["vmf"]["delta"] == pytest.approx(0.12)
    assert by_model["etm"]["rank_full"] == 1
    assert by_model["vmf"]["rank_full"] == 2


def test_arm_comparison_reports_rank_changes() -> None:
    rows = [
        _arm_row(model="etm", arm="full", score=0.04),
        _arm_row(model="vmf", arm="full", score=-0.15),
        # The filtered arm lifts vmf past etm.
        _arm_row(model="etm", arm="refdf50-30", score=0.042),
        _arm_row(model="vmf", arm="refdf50-30", score=0.090),
    ]

    comparison = {
        row["model"]: row
        for row in build_arm_comparison(rows)
        if row["metric"] == "coherence_c_npmi_mean"
    }

    assert comparison["vmf"]["rank_full"] == 2
    assert comparison["vmf"]["rank_filtered"] == 1
    assert comparison["vmf"]["rank_delta"] == -1
    assert comparison["etm"]["rank_delta"] == 1


def test_arm_comparison_keeps_rows_missing_one_side() -> None:
    rows = [
        _arm_row(model="vmf", arm="full", score=-0.15),
        _arm_row(model="ctm", arm="full", score=-0.31),
        # Only vmf was re-run under the filtered arm.
        _arm_row(model="vmf", arm="refdf50-30", score=-0.03),
    ]

    comparison = {
        row["model"]: row
        for row in build_arm_comparison(rows)
        if row["metric"] == "coherence_c_npmi_mean"
    }

    assert comparison["ctm"]["score_full"] == -0.31
    assert comparison["ctm"]["score_filtered"] == ""
    assert comparison["ctm"]["delta"] == ""


def test_arm_comparison_is_empty_without_a_filtered_arm() -> None:
    rows = [
        _arm_row(model="vmf", arm="full", score=-0.15),
        _arm_row(model="etm", arm="full", score=0.04),
    ]

    assert build_arm_comparison(rows) == []

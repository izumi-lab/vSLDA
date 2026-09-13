from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.reporting import read_evaluation_json
from src.evaluation.topic_pairs import metrics as metrics_module
from src.evaluation.topic_pairs.metrics import CROSS_MODEL_KEY, run_topic_pair_metrics
from src.evaluation.topic_pairs.summary import (
    TopicPairSummaryError,
    build_topic_pair_scores_payload,
    build_topic_pair_summary_table,
    check_paper_grid,
    collect_topic_pair_conditions,
    encoder_group,
    group_records_for_scores,
    write_topic_pair_summary,
)


@pytest.fixture
def populated_root(toy_runner, monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        metrics_module, "_uses_default_output_layout", lambda _root: True
    )
    run_topic_pair_metrics(
        models=["vmf", "sentlda"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=3,
        categories=["a", "b", "all"],
        out_root=tmp_path,
        embedding_variant="mpnet",
    )
    return tmp_path


def test_collect_conditions_excludes_all_and_filters(populated_root: Path) -> None:
    records = collect_topic_pair_conditions(out_root=populated_root)
    assert len(records) == 6  # (2 models + cross) x 2 categories
    assert {record["category"] for record in records} == {"a", "b"}
    assert all(record["meta"]["task"] == "topic_pair_metrics" for record in records)
    assert (
        len(
            collect_topic_pair_conditions(
                out_root=populated_root, exclude_categories=()
            )
        )
        == 9
    )
    only_vmf = collect_topic_pair_conditions(out_root=populated_root, models=["vmf"])
    assert {record["meta"]["model"] for record in only_vmf} == {"vmf"}
    both = collect_topic_pair_conditions(
        out_root=populated_root, models=["vmf", "sentlda"]
    )
    assert {record["meta"]["model"] for record in both} == {
        "vmf",
        "sentlda",
        CROSS_MODEL_KEY,
    }
    assert collect_topic_pair_conditions(out_root=populated_root, topics=[4]) == []
    assert collect_topic_pair_conditions(out_root=populated_root, split="test") == []


def test_encoder_groups(populated_root: Path) -> None:
    records = collect_topic_pair_conditions(out_root=populated_root)
    groups = {
        record["meta"]["model"]: encoder_group(record["meta"]) for record in records
    }
    # Every metric lives in the requested encoder space, so SentLDA - which
    # reads no embeddings of its own - is grouped by the space it was evaluated
    # in rather than joining every group.
    assert groups == {"vmf": "mpnet", "sentlda": "mpnet", CROSS_MODEL_KEY: "mpnet"}
    assert encoder_group({"model": "sentlda", "embedding_variant": None}) is None
    assert (
        encoder_group(
            {
                "model": "sentence_gaussianlda",
                "effective_embedding_variant": "minilm_raw",
            }
        )
        == "minilm"
    )
    grouped = group_records_for_scores(records)
    assert list(grouped) == [("dummy", "default", "mpnet", 3)]
    assert len(grouped[("dummy", "default", "mpnet", 3)]) == 6


def test_summary_table_pools_iterations(populated_root: Path) -> None:
    records = collect_topic_pair_conditions(out_root=populated_root)
    rows = build_topic_pair_summary_table(records)
    assert [row["model"] for row in rows] == ["sentlda", "vmf"]
    vmf = rows[1]
    assert vmf["embedding_variant"] == "mpnet"
    assert vmf["n_categories"] == 2 and vmf["n_iterations"] == 4
    key = "mean_offdiag_centroid_cosine_soft"
    vmf_records = [r for r in records if r["meta"]["model"] == "vmf"]
    means = [r["results"]["aggregate"][key]["mean"] for r in vmf_records]
    values = [e[key] for r in vmf_records for e in r["results"]["per_iteration"]]
    assert vmf[f"{key}_mean"] == pytest.approx(np.mean(means))
    assert vmf[f"{key}_std"] == pytest.approx(np.std(values, ddof=1))


def test_scores_sidecar_carries_raw_matrices_and_provenance(
    populated_root: Path,
) -> None:
    wide_path = write_topic_pair_summary(out_root=populated_root)
    summaries = populated_root / "summaries"
    assert wide_path == summaries / "topic_pairs_summary_wide.csv"
    with wide_path.open(encoding="utf-8", newline="") as handle:
        assert {row["model"] for row in csv.DictReader(handle)} == {"vmf", "sentlda"}
    meta, results = read_evaluation_json(summaries / "topic_pairs_summary.json")
    assert meta["task"] == "topic_pair_summary" and meta["num_conditions"] == 6
    assert len(results["rows"]) == 2

    path = (
        summaries
        / "dummy"
        / "default"
        / "mpnet"
        / "topic_pairs_dummy_default_mpnet_3topic.scores.json"
    )
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["task"] == "topic_pair_summary"
    assert payload["dataset"] == "dummy" and payload["topics"] == 3
    assert payload["encoder_variant"] == "mpnet"
    assert payload["iterations"] == [0, 1]
    assert payload["categories"] == ["a", "b"]
    assert payload["models"] == ["sentlda", "vmf"]
    assert payload["label_names"] == {
        "a": ["label_x", "label_y"],
        "b": ["label_x", "label_y"],
    }
    assert payload["protocol"]["split"] == "train"
    assert payload["protocol"]["posterior_definition"] == "collapsed_fold_in"
    assert payload["protocol"]["foldin_random_seed"] == 0
    assert "num_sentences" in payload["metrics"]
    assert payload["scores"]["num_sentences"]["a"]["vmf"] == [4, 4]

    per_topic = payload["per_topic"]["a"]["vmf"]
    assert len(per_topic["mass_soft"]) == 2 and len(per_topic["mass_soft"][0]) == 3
    assert np.asarray(per_topic["label_mass"]).shape == (2, 3, 2)
    per_pair = payload["per_pair"]["a"]["vmf"]
    assert set(per_pair) == {
        "centroid_cosine_soft",
        "centroid_cosine_hard",
        "assignment_confusion",
        "label_js_divergence",
    }
    assert np.asarray(per_pair["centroid_cosine_soft"]).shape == (2, 3, 3)
    assert payload["model_own"]["a"]["vmf"] is not None
    assert np.asarray(
        payload["model_own"]["a"]["vmf"]["centroid_cosine_model"]
    ).shape == (2, 3, 3)
    assert payload["model_own"]["a"]["sentlda"] is None
    overlap = np.asarray(payload["cross_model"]["a"]["vmf|sentlda"])
    assert overlap.shape == (2, 3, 3)
    assert overlap[0].sum() == pytest.approx(4.0)
    assert payload["run_iterations"]["a"] == {
        "sentlda": [0, 1],
        "vmf": [0, 1],
        CROSS_MODEL_KEY: [0, 1],
    }

    provenance = payload["provenance"]["a"]["vmf"]
    assert provenance["iterations"] == [0, 1]
    assert provenance["encoder_model"] == "mpnet"
    assert provenance["encoder_model_name"] == "fake-encoder"
    assert provenance["split"] == "train"
    assert provenance["label_source_model"] == "vmf"
    assert provenance["model_provenance"]["model_key"] == "vmf_sentence_lda"
    assert payload["provenance"]["a"]["sentlda"]["encoder_model"] is None
    assert set(payload["provenance"]["a"][CROSS_MODEL_KEY]["model_provenance"]) == {
        "vmf",
        "sentlda",
    }
    # Raw values only: no aggregate crosses the boundary.
    assert '"mean":' not in text and '"std":' not in text
    # Compact by default (no indentation).
    assert "\n  " not in text


def test_indented_sidecar_option(populated_root: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "indented"
    write_topic_pair_summary(
        out_root=populated_root, output_dir=output_dir, compact=False
    )
    path = next(output_dir.glob("dummy/default/mpnet/*.scores.json"))
    assert "\n  " in path.read_text(encoding="utf-8")


def test_scores_payload_rejects_disagreeing_protocols_and_duplicates(
    populated_root: Path,
) -> None:
    records = collect_topic_pair_conditions(out_root=populated_root)
    records[0]["meta"]["foldin_random_seed"] = 7
    with pytest.raises(TopicPairSummaryError, match="foldin_random_seed"):
        build_topic_pair_scores_payload(
            records,
            dataset="dummy",
            data_run="default",
            num_topics=3,
            encoder_variant="mpnet",
        )
    records = collect_topic_pair_conditions(out_root=populated_root)
    with pytest.raises(TopicPairSummaryError, match="two"):
        build_topic_pair_scores_payload(
            [*records, records[0]],
            dataset="dummy",
            data_run="default",
            num_topics=3,
            encoder_variant="mpnet",
        )


def test_paper_grid_check(populated_root: Path) -> None:
    records = collect_topic_pair_conditions(out_root=populated_root)
    check_paper_grid(
        records,
        datasets=("dummy",),
        encoder_variant="mpnet",
        topics=(3,),
        iterations=(0, 1),
        models=("vmf", "sentlda"),
        categories={"dummy": ("a", "b")},
    )
    with pytest.raises(TopicPairSummaryError, match="no condition"):
        check_paper_grid(
            records,
            datasets=("dummy",),
            encoder_variant="mpnet",
            topics=(3,),
            iterations=(0, 1),
            models=("vmf", "sentlda", "sentence_gaussianlda"),
            categories={"dummy": ("a", "b")},
        )
    with pytest.raises(TopicPairSummaryError, match="iterations"):
        check_paper_grid(
            records,
            datasets=("dummy",),
            encoder_variant="mpnet",
            topics=(3,),
            iterations=(0, 1, 2),
            models=("vmf", "sentlda"),
            categories={"dummy": ("a", "b")},
        )
    # The manuscript grid itself is not satisfied by the toy root.
    with pytest.raises(TopicPairSummaryError, match="incomplete"):
        write_topic_pair_summary(out_root=populated_root, paper=True)


def test_write_summary_on_empty_root(tmp_path: Path) -> None:
    wide_path = write_topic_pair_summary(out_root=tmp_path / "nothing")
    assert wide_path.exists()
    with wide_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []

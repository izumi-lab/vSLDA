from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.entropy_based import metrics as metrics_module
from src.evaluation.entropy_based import summary as summary_module
from src.evaluation.entropy_based.entropy_metrics import SUMMARY_METRIC_KEYS
from src.evaluation.entropy_based.inputs import DocTopicLoad
from src.evaluation.entropy_based.metrics import run_entropy_based_metrics
from src.evaluation.entropy_based.summary import (
    EntropySummaryError,
    build_entropy_scores_payload,
    build_entropy_summary_table,
    build_latex_tables,
    build_paper_entropy_figure,
    collect_entropy_conditions,
    group_records_for_scores,
    write_entropy_based_summary,
)
from src.evaluation.reporting import read_evaluation_json

THETA_BY_MODEL = {
    "vmf": np.asarray([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]),
    "bleilda": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.4, 0.6]]),
}


@pytest.fixture
def populated_root(monkeypatch, tmp_path: Path) -> Path:
    def _load(**kwargs) -> DocTopicLoad:
        theta = THETA_BY_MODEL[kwargs["model"]]
        # make iterations differ slightly so pooled std is non-zero
        shift = 0.05 * int(kwargs["iteration"])
        matrix = np.clip(theta + np.asarray([[shift, -shift]]), 1e-6, None)
        return DocTopicLoad(
            theta=matrix / matrix.sum(axis=1, keepdims=True),
            source="doc_topic_soft_file",
            path=tmp_path / "artifact.pkl",
            condition_dir=tmp_path / kwargs["model"],
        )

    monkeypatch.setattr(metrics_module, "load_doc_topic_matrix", _load)
    monkeypatch.setattr(
        metrics_module,
        "provenance_for",
        lambda condition_dir, model: {
            "model_key": model,
            "metadata_path": str(condition_dir),
        },
    )
    monkeypatch.setattr(
        metrics_module, "_uses_default_output_layout", lambda _out_root: True
    )
    run_entropy_based_metrics(
        models=["vmf", "bleilda"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=2,
        categories=["a", "b", "all"],
        out_root=tmp_path,
        doc_topic_source="soft_proxy",
        embedding_variant="mpnet",
    )
    return tmp_path


def test_collect_conditions_excludes_all_and_filters(populated_root: Path) -> None:
    records = collect_entropy_conditions(out_root=populated_root)
    assert len(records) == 4  # 2 models x 2 categories (all excluded)
    assert {record["category"] for record in records} == {"a", "b"}
    assert all(record["meta"]["task"] == "entropy_based_metrics" for record in records)

    with_all = collect_entropy_conditions(
        out_root=populated_root, exclude_categories=()
    )
    assert len(with_all) == 6
    only_vmf = collect_entropy_conditions(out_root=populated_root, models=["vmf"])
    assert {record["meta"]["model"] for record in only_vmf} == {"vmf"}
    assert collect_entropy_conditions(out_root=populated_root, topics=[3]) == []


def test_summary_table_averages_categories_and_pools_iterations(
    populated_root: Path,
) -> None:
    records = collect_entropy_conditions(out_root=populated_root)
    wide_rows, long_rows = build_entropy_summary_table(records)
    assert len(wide_rows) == 2
    vmf_row = next(row for row in wide_rows if row["model"] == "vmf")
    assert vmf_row["embedding_variant"] == "mpnet"
    assert vmf_row["n_categories"] == 2
    assert vmf_row["n_iterations"] == 4
    assert vmf_row["categories"] == "a;b"

    key = "doc_topic_entropy_normalized_mean"
    vmf_records = [record for record in records if record["meta"]["model"] == "vmf"]
    condition_means = [
        record["results"]["aggregate"][key]["mean"] for record in vmf_records
    ]
    per_iteration = [
        entry[key]
        for record in vmf_records
        for entry in record["results"]["per_iteration"]
    ]
    assert vmf_row[f"{key}_mean"] == pytest.approx(np.mean(condition_means))
    assert vmf_row[f"{key}_std"] == pytest.approx(np.std(per_iteration, ddof=1))
    assert vmf_row[f"{key}_std"] > 0.0

    assert len(long_rows) == 2 * len(SUMMARY_METRIC_KEYS)
    long_vmf = [
        row for row in long_rows if row["model"] == "vmf" and row["metric"] == key
    ][0]
    assert long_vmf["mean"] == pytest.approx(vmf_row[f"{key}_mean"])


def test_latex_tables_one_per_dataset_and_k(populated_root: Path) -> None:
    records = collect_entropy_conditions(out_root=populated_root)
    wide_rows, _ = build_entropy_summary_table(records)
    tables = build_latex_tables(wide_rows)
    assert list(tables) == [("dummy", 2)]
    tex = tables[("dummy", 2)]
    assert r"\begin{tabular}" in tex
    assert "vMF Sentence LDA (mpnet)" in tex
    assert "LDA" in tex
    assert r"$\pm$" in tex


def test_write_summary_outputs_csv_json_tex_and_plots(populated_root: Path) -> None:
    wide_path = write_entropy_based_summary(out_root=populated_root)
    summaries = populated_root / "summaries"
    assert wide_path == summaries / "entropy_summary_wide.csv"
    with wide_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model"] for row in rows} == {"vmf", "bleilda"}
    assert (summaries / "entropy_summary_long.csv").exists()
    meta, results = read_evaluation_json(summaries / "entropy_summary.json")
    assert meta["task"] == "entropy_based_summary"
    assert meta["num_conditions"] == 4
    assert len(results["rows"]) == 2
    assert (summaries / "entropy_dummy_k2.tex").exists()
    plots = sorted((summaries / "plots").glob("entropy_hist_*.png"))
    assert [path.name for path in plots] == [
        "entropy_hist_dummy_k2_a.png",
        "entropy_hist_dummy_k2_b.png",
    ]


def test_write_summary_can_skip_plots_and_tex(
    populated_root: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "custom"
    write_entropy_based_summary(
        out_root=populated_root,
        output_dir=output_dir,
        write_tex=False,
        write_plots=False,
    )
    assert (output_dir / "entropy_summary_wide.csv").exists()
    assert not list(output_dir.glob("*.tex"))
    assert not (output_dir / "plots").exists()


def test_write_summary_on_empty_root(tmp_path: Path) -> None:
    wide_path = write_entropy_based_summary(out_root=tmp_path / "nothing")
    assert wide_path.exists()
    with wide_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_scores_sidecar_carries_raw_values_and_provenance(populated_root: Path) -> None:
    write_entropy_based_summary(out_root=populated_root, write_plots=False)
    path = (
        populated_root
        / "summaries"
        / "dummy"
        / "default"
        / "mpnet"
        / "entropy_dummy_default_mpnet_2topic.scores.json"
    )
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == "entropy_based_summary"
    assert payload["dataset"] == "dummy"
    assert payload["topics"] == 2
    assert payload["encoder_variant"] == "mpnet"
    assert payload["iterations"] == [0, 1]
    assert payload["categories"] == ["a", "b"]
    # The encoder-independent baseline is listed under the encoder group too.
    assert payload["models"] == ["bleilda", "vmf"]

    records = collect_entropy_conditions(out_root=populated_root)
    vmf_a = next(
        r for r in records if r["meta"]["model"] == "vmf" and r["category"] == "a"
    )
    key = "doc_topic_entropy_normalized_mean"
    expected = [entry[key] for entry in vmf_a["results"]["per_iteration"]]
    assert payload["scores"][key]["a"]["vmf"] == pytest.approx(expected)
    assert payload["scores"]["num_documents"]["a"]["vmf"] == [3, 3]
    per_topic = payload["per_topic"]["a"]["vmf"]["topic_doc_entropy_normalized"]
    assert len(per_topic) == 2 and all(len(row) == 2 for row in per_topic)
    assert payload["run_iterations"]["a"]["vmf"] == [0, 1]

    provenance = payload["provenance"]["a"]["vmf"]
    assert provenance["iterations"] == [0, 1]
    assert provenance["encoder_model"] == "mpnet"
    assert provenance["split"] == "test"
    assert provenance["doc_topic_source"] == "soft_proxy"
    assert provenance["diffuse_entropy_threshold"] == pytest.approx(0.95)
    assert provenance["dead_rank1_threshold"] == pytest.approx(0.01)
    assert provenance["condition_ids"] == [vmf_a["meta"]["condition_id"]]
    assert payload["provenance"]["a"]["bleilda"]["encoder_model"] is None
    assert payload["protocol"]["split"] == "test"
    # No aggregate crosses the boundary: consumers compute mean/std themselves.
    assert '"mean":' not in json.dumps(payload)
    assert '"std":' not in json.dumps(payload)


def test_scores_groups_replicate_encoder_independent_models(
    populated_root: Path,
) -> None:
    records = collect_entropy_conditions(out_root=populated_root)
    grouped = group_records_for_scores(records)
    assert list(grouped) == [("dummy", "default", "mpnet", 2)]
    members = grouped[("dummy", "default", "mpnet", 2)]
    assert {r["meta"]["model"] for r in members} == {"vmf", "bleilda"}

    only_lda = [r for r in records if r["meta"]["model"] == "bleilda"]
    assert list(group_records_for_scores(only_lda)) == [
        ("dummy", "default", "no_encoder", 2)
    ]


def test_scores_payload_rejects_disagreeing_protocols(populated_root: Path) -> None:
    records = collect_entropy_conditions(out_root=populated_root)
    records[0]["meta"]["diffuse_entropy_threshold"] = 0.9
    with pytest.raises(EntropySummaryError, match="diffuse_entropy_threshold"):
        build_entropy_scores_payload(
            records,
            dataset="dummy",
            data_run="default",
            num_topics=2,
            encoder_variant="mpnet",
        )


def test_scores_payload_rejects_duplicate_cells(populated_root: Path) -> None:
    records = collect_entropy_conditions(out_root=populated_root)
    duplicated = [*records, records[0]]
    with pytest.raises(EntropySummaryError, match="two conditions"):
        build_entropy_scores_payload(
            duplicated,
            dataset="dummy",
            data_run="default",
            num_topics=2,
            encoder_variant="mpnet",
        )


def test_paper_figure_is_written_in_both_formats(
    populated_root: Path, tmp_path: Path
) -> None:
    records = collect_entropy_conditions(out_root=populated_root)
    written = build_paper_entropy_figure(
        records, output_dir=tmp_path / "figures", num_topics=2, encoder_variant="mpnet"
    )
    assert [path.name for path in written] == [
        "entropy_boxplot_2topic_mpnet.png",
        "entropy_boxplot_2topic_mpnet.pdf",
    ]
    assert all(path.stat().st_size > 0 for path in written)
    # Nothing at the requested topic count: no file, no error.
    assert (
        build_paper_entropy_figure(
            records,
            output_dir=tmp_path / "none",
            num_topics=20,
            encoder_variant="minilm",
        )
        == []
    )


def test_write_summary_paper_flag_writes_the_figure(populated_root: Path) -> None:
    write_entropy_based_summary(
        out_root=populated_root,
        write_plots=False,
        write_tex=False,
        paper=True,
        paper_num_topics=2,
        paper_encoder="mpnet",
    )
    figures = populated_root / "summaries" / "figures"
    assert (figures / "entropy_boxplot_2topic_mpnet.pdf").exists()


def test_paper_figure_can_take_the_vmf_boxes_from_another_source(
    populated_root: Path, monkeypatch, tmp_path: Path
) -> None:
    """--paper-vmf-doc-topic-source draws vMF from its fold-in conditions, sidecars unchanged."""

    run_entropy_based_metrics(
        models=["vmf"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=2,
        categories=["a", "b"],
        out_root=populated_root,
        embedding_variant="mpnet",
        doc_topic_source="foldincounts",
    )
    captured: dict[str, list] = {}

    def _capture(records, **kwargs):
        captured["records"] = list(records)
        return []

    monkeypatch.setattr(summary_module, "build_paper_entropy_figure", _capture)
    write_entropy_based_summary(
        out_root=populated_root,
        output_dir=tmp_path / "out",
        write_plots=False,
        write_tex=False,
        doc_topic_source="soft_proxy",
        paper=True,
        paper_num_topics=2,
        paper_encoder="mpnet",
        paper_vmf_doc_topic_source="foldincounts",
    )
    sources_by_model = {
        (str(r["meta"]["model"]), str(r["meta"]["doc_topic_source"]))
        for r in captured["records"]
    }
    assert sources_by_model == {("vmf", "foldincounts"), ("bleilda", "soft_proxy")}
    # The sidecars still hold the soft_proxy protocol for every model.
    for sidecar in (tmp_path / "out").glob("*.scores.json"):
        payload = json.loads(sidecar.read_text())
        for cell in payload["provenance"].values():
            for prov in cell.values():
                assert prov["doc_topic_source"] == "soft_proxy"
    with pytest.raises(EntropySummaryError):
        write_entropy_based_summary(
            out_root=populated_root,
            output_dir=tmp_path / "out2",
            write_plots=False,
            write_tex=False,
            doc_topic_source="soft_proxy",
            paper=True,
            paper_num_topics=2,
            paper_encoder="mpnet",
            paper_vmf_doc_topic_source="hard",
        )

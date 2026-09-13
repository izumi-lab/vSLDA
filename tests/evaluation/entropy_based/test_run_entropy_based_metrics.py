from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.core.artifacts import load_json
from src.evaluation.entropy_based import metrics as metrics_module
from src.evaluation.entropy_based.entropy_metrics import SUMMARY_METRIC_KEYS
from src.evaluation.entropy_based.inputs import DocTopicLoad
from src.evaluation.entropy_based.metrics import (
    run_entropy_based_metrics,
    summary_fieldnames,
)
from src.evaluation.reporting import read_evaluation_json

THETA = np.asarray(
    [
        [0.97, 0.01, 0.01, 0.01],
        [0.50, 0.50, 0.00, 0.00],
        [0.25, 0.25, 0.25, 0.25],
        [0.10, 0.10, 0.10, 0.70],
    ]
)


def _set_fixed_now(monkeypatch, iso_timestamp: str) -> None:
    fixed_dt = datetime.fromisoformat(iso_timestamp).astimezone(UTC)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_dt
            return fixed_dt.astimezone(tz)

    monkeypatch.setattr(metrics_module, "datetime", _FixedDateTime)


def _fake_loader(
    tmp_path: Path, *, theta: np.ndarray = THETA, calls: list | None = None
):
    def _load(**kwargs) -> DocTopicLoad:
        if calls is not None:
            calls.append(dict(kwargs))
        num_topics = int(kwargs["num_topics"])
        matrix = theta[:, :num_topics]
        return DocTopicLoad(
            theta=matrix / matrix.sum(axis=1, keepdims=True),
            source="doc_topic_soft_file",
            path=tmp_path / "artifact.pkl",
            condition_dir=tmp_path / str(kwargs["model"]),
            raw_doc_indices=[10, 11, 12, 13],
        )

    return _load


def _fake_provenance(condition_dir: Path, *, model: str) -> dict[str, object]:
    return {
        "model_key": "vmf_sentence_lda" if model == "vmf" else model,
        "metadata_path": str(condition_dir / "metadata.json"),
    }


@pytest.fixture
def patched(monkeypatch, tmp_path: Path):
    calls: list[dict] = []
    monkeypatch.setattr(
        metrics_module, "load_doc_topic_matrix", _fake_loader(tmp_path, calls=calls)
    )
    monkeypatch.setattr(metrics_module, "provenance_for", _fake_provenance)
    return calls


def test_run_writes_flat_layout_and_summary(patched, tmp_path: Path) -> None:
    calls = patched
    summary_path = run_entropy_based_metrics(
        models=["vmf_sentence_lda"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=4,
        categories=["all"],
        out_root=tmp_path,
        embedding_variant="mpnet",
    )
    assert summary_path == tmp_path / "summary.csv"
    assert len(calls) == 2
    assert calls[0]["model"] == "vmf"
    assert calls[0]["embedding_variant"] == "mpnet"
    assert calls[0]["split"] == "test"
    # The default source is "auto", which a vMF run resolves to the fold-in counts.
    assert calls[0]["doc_topic_source"] == "foldincounts"

    out_path = next(
        (tmp_path / "dummy" / "default" / "all").glob(
            "it0__k4__vmf__mpnet__*/entropy_metrics_agg.json"
        )
    )
    meta, results = read_evaluation_json(out_path)
    assert meta["task"] == "entropy_based_metrics"
    assert meta["model"] == "vmf"
    assert meta["split"] == "test"
    assert meta["doc_topic_source"] == "foldincounts"
    assert meta["doc_topic_source_resolved"] == {
        "0": "doc_topic_soft_file",
        "1": "doc_topic_soft_file",
    }
    assert meta["effective_embedding_variant"] == "mpnet"
    assert meta["log_base"] == "e"
    assert meta["model_provenance"]["model_key"] == "vmf_sentence_lda"
    assert meta["iterations"] == [0, 1]
    assert (out_path.parent / "metadata.json").exists()

    per_iteration = results["per_iteration"]
    assert [entry["iteration"] for entry in per_iteration] == [0, 1]
    assert per_iteration[0]["num_documents"] == 4
    assert per_iteration[0]["num_topics"] == 4
    assert set(per_iteration[0]["per_topic"]) == {
        "topic_doc_entropy",
        "topic_doc_entropy_normalized",
        "topic_rank1_doc_fraction",
    }
    assert sum(
        per_iteration[0]["per_topic"]["topic_rank1_doc_fraction"]
    ) == pytest.approx(1.0)
    for key in SUMMARY_METRIC_KEYS:
        assert key in results["aggregate"]
        assert results["aggregate"][key]["std"] == pytest.approx(0.0)

    for iteration in (0, 1):
        doc_csv = out_path.parent / f"iter{iteration}" / "doc_metrics.csv"
        topic_csv = out_path.parent / f"iter{iteration}" / "topic_metrics.csv"
        assert doc_csv.exists() and topic_csv.exists()
    with (out_path.parent / "iter0" / "doc_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["raw_doc_index"] for row in rows] == ["10", "11", "12", "13"]
    assert float(rows[2]["doc_topic_entropy_normalized"]) == pytest.approx(1.0)
    with (out_path.parent / "iter0" / "topic_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        topic_rows = list(csv.DictReader(handle))
    assert len(topic_rows) == 4
    assert sum(
        float(row["topic_rank1_doc_fraction"]) for row in topic_rows
    ) == pytest.approx(1.0)

    summary_meta, summary_results = read_evaluation_json(tmp_path / "summary.json")
    assert summary_meta["task"] == "entropy_based_metrics_summary"
    assert summary_results["columns"] == summary_fieldnames()
    row = summary_results["rows"][0]
    assert row["model"] == "vmf"
    assert row["embedding_variant"] == "mpnet"
    assert row["num_iterations"] == 2
    assert row["num_documents_mean"] == 4


def test_run_accepts_topic_list_and_word_embedding_variant(
    patched, tmp_path: Path
) -> None:
    calls = patched
    run_entropy_based_metrics(
        models=["etm", "bleilda"],
        dataset="dummy",
        iterations=[0],
        num_topics=[2, 3],
        categories=["a", "b"],
        out_root=tmp_path,
        embedding_variant="mpnet",
    )
    # 2 models x 2 topic counts x 2 categories
    assert len(calls) == 8
    etm_dirs = sorted(
        (tmp_path / "dummy" / "default" / "a").glob("it0__k*__etm__googlenews300__*")
    )
    assert len(etm_dirs) == 2
    bleilda_dirs = sorted(
        (tmp_path / "dummy" / "default" / "a").glob("it0__k*__bleilda__*")
    )
    assert len(bleilda_dirs) == 2
    meta, _ = read_evaluation_json(etm_dirs[0] / "entropy_metrics_agg.json")
    assert meta["effective_embedding_variant"] == "googlenews300"
    assert meta["embedding_variant"] == "mpnet"

    _, summary_results = read_evaluation_json(tmp_path / "summary.json")
    assert len(summary_results["rows"]) == 8


def test_run_records_prior_scale_in_condition_id(patched, tmp_path: Path) -> None:
    run_entropy_based_metrics(
        models=["gaussianlda"],
        dataset="dummy",
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        prior_scale=0.1,
    )
    out_dir = next(
        (tmp_path / "dummy" / "default" / "all").glob(
            "it0__k2__gaussianlda__googlenews300__psi0-0p1__*"
        )
    )
    meta, _ = read_evaluation_json(out_dir / "entropy_metrics_agg.json")
    assert meta["parameter_variant"] == "psi0-0p1"
    assert meta["prior_scale"] == pytest.approx(0.1)


def test_skip_existing_reuses_output_without_reloading(
    patched, monkeypatch, tmp_path: Path
) -> None:
    run_entropy_based_metrics(
        models=["vmf"],
        dataset="dummy",
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
    )

    def _boom(**_kwargs):
        raise AssertionError("loader must not be called when skipping existing output")

    monkeypatch.setattr(metrics_module, "load_doc_topic_matrix", _boom)
    summary_path = run_entropy_based_metrics(
        models=["vmf"],
        dataset="dummy",
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        skip_existing=True,
    )
    _, summary_results = read_evaluation_json(summary_path.with_suffix(".json"))
    assert len(summary_results["rows"]) == 1
    assert summary_results["rows"][0]["model"] == "vmf"


def test_isolate_policy_records_failures_and_continues(
    monkeypatch, tmp_path: Path
) -> None:
    good = _fake_loader(tmp_path)

    def _load(**kwargs):
        if kwargs["model"] == "bleilda":
            raise FileNotFoundError("missing bleilda artifact")
        return good(**kwargs)

    monkeypatch.setattr(metrics_module, "load_doc_topic_matrix", _load)
    monkeypatch.setattr(metrics_module, "provenance_for", _fake_provenance)

    with pytest.raises(FileNotFoundError):
        run_entropy_based_metrics(
            models=["bleilda"],
            dataset="dummy",
            iterations=[0],
            num_topics=2,
            categories=["all"],
            out_root=tmp_path / "fail_fast",
        )

    run_entropy_based_metrics(
        models=["bleilda", "vmf"],
        dataset="dummy",
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        condition_failure_policy="isolate",
    )
    failures = load_json(tmp_path / "failed_conditions.json")
    assert len(failures["failures"]) == 1
    assert failures["failures"][0]["model"] == "bleilda"
    assert failures["failures"][0]["error_type"] == "FileNotFoundError"
    _, summary_results = read_evaluation_json(tmp_path / "summary.json")
    assert [row["model"] for row in summary_results["rows"]] == ["vmf"]


def test_isolate_policy_also_isolates_artifact_shape_errors(
    monkeypatch, tmp_path: Path
) -> None:
    # The loader rejects mismatched artifacts with ValueError (doc-topic width,
    # negative theta, unsupported payload). Letting those escape aborted a whole
    # sweep and discarded every condition already computed.
    good = _fake_loader(tmp_path)

    def _load(**kwargs):
        if kwargs["model"] == "bleilda":
            raise ValueError("Doc-topic width 5 does not match num_topics 2")
        return good(**kwargs)

    monkeypatch.setattr(metrics_module, "load_doc_topic_matrix", _load)
    monkeypatch.setattr(metrics_module, "provenance_for", _fake_provenance)

    run_entropy_based_metrics(
        models=["bleilda", "vmf"],
        dataset="dummy",
        iterations=[0],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        condition_failure_policy="isolate",
    )

    failures = load_json(tmp_path / "failed_conditions.json")
    assert [f["error_type"] for f in failures["failures"]] == ["ValueError"]
    _, summary_results = read_evaluation_json(tmp_path / "summary.json")
    assert [row["model"] for row in summary_results["rows"]] == ["vmf"]


def test_default_out_root_uses_archive_latest_layout(
    patched, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        metrics_module, "_uses_default_output_layout", lambda _out_root: True
    )
    _set_fixed_now(monkeypatch, "2026-04-13T00:00:00+00:00")

    run_entropy_based_metrics(
        models=["vmf"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
    )

    archive_root = tmp_path / "archive" / "2026-04-13" / "dummy" / "default" / "all"
    out_path = next(
        archive_root.glob(
            "it0__k2__vmf__*/exec_20260413T000000Z/entropy_metrics_agg.json"
        )
    )
    meta, _ = read_evaluation_json(out_path)
    assert meta["display_key"] == meta["condition_id"]
    assert meta["execution_id"] == "exec_20260413T000000Z"
    assert meta["archive_dir"].endswith("/exec_20260413T000000Z")
    assert meta["latest_dir"].endswith(f"/{meta['display_key']}")

    latest_pointer = load_json(
        tmp_path
        / "latest"
        / "dummy"
        / "default"
        / "all"
        / meta["display_key"]
        / "CURRENT.json"
    )
    assert latest_pointer["task"] == "entropy_based_metrics"
    assert latest_pointer["execution_id"] == "exec_20260413T000000Z"
    assert latest_pointer["artifacts"]["metrics"] == "entropy_metrics_agg.json"
    assert latest_pointer["artifacts"]["metadata"] == "metadata.json"
    assert (
        latest_pointer["artifacts"]["doc_metrics_csv_iter0"] == "iter0/doc_metrics.csv"
    )
    assert (
        latest_pointer["artifacts"]["topic_metrics_csv_iter1"]
        == "iter1/topic_metrics.csv"
    )

    # skip-existing must resolve through the latest pointer in the default layout.
    monkeypatch.setattr(
        metrics_module,
        "load_doc_topic_matrix",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must skip")),
    )
    run_entropy_based_metrics(
        models=["vmf"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=2,
        categories=["all"],
        out_root=tmp_path,
        skip_existing=True,
    )


def test_rejects_invalid_options(patched, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="doc_topic_source"):
        run_entropy_based_metrics(
            models=["vmf"],
            dataset="dummy",
            iterations=[0],
            num_topics=2,
            categories=["all"],
            out_root=tmp_path,
            doc_topic_source="bogus",
        )
    with pytest.raises(ValueError, match="condition_failure_policy"):
        run_entropy_based_metrics(
            models=["vmf"],
            dataset="dummy",
            iterations=[0],
            num_topics=2,
            categories=["all"],
            out_root=tmp_path,
            condition_failure_policy="bogus",
        )
    with pytest.raises(ValueError, match="mismatch"):
        run_entropy_based_metrics(
            models=["vmf"],
            dataset="dummy",
            iterations=[0],
            num_topics=2,
            categories=["all"],
            out_root=tmp_path,
            embedding_variant="bge",
            encoder_model="sentence-transformers/all-mpnet-base-v2",
        )
    with pytest.raises(ValueError, match="diffuse_entropy_threshold"):
        run_entropy_based_metrics(
            models=["vmf"],
            dataset="dummy",
            iterations=[0],
            num_topics=2,
            categories=["all"],
            out_root=tmp_path,
            diffuse_entropy_threshold=1.5,
        )
    with pytest.raises(ValueError, match="dead_rank1_threshold"):
        run_entropy_based_metrics(
            models=["vmf"],
            dataset="dummy",
            iterations=[0],
            num_topics=2,
            categories=["all"],
            out_root=tmp_path,
            dead_rank1_threshold=-0.1,
        )


def test_thresholds_are_recorded_and_change_the_condition_id(
    patched, tmp_path: Path
) -> None:
    common = dict(
        models=["vmf"],
        dataset="dummy",
        iterations=[0],
        num_topics=4,
        categories=["all"],
        out_root=tmp_path,
    )
    run_entropy_based_metrics(**common)
    run_entropy_based_metrics(**common, diffuse_entropy_threshold=0.5)
    dirs = sorted((tmp_path / "dummy" / "default" / "all").glob("it0__k4__vmf__*"))
    assert len(dirs) == 2, "different thresholds must not overwrite each other"

    metas = [read_evaluation_json(d / "entropy_metrics_agg.json")[0] for d in dirs]
    thresholds = sorted(m["diffuse_entropy_threshold"] for m in metas)
    assert thresholds == pytest.approx([0.5, 0.95])
    assert all(m["dead_rank1_threshold"] == pytest.approx(0.01) for m in metas)

    fractions = {}
    for d in dirs:
        meta, results = read_evaluation_json(d / "entropy_metrics_agg.json")
        fractions[meta["diffuse_entropy_threshold"]] = results["per_iteration"][0][
            "topic_diffuse_fraction"
        ]
    assert fractions[0.5] >= fractions[0.95]


def test_topic_metrics_csv_carries_flag_columns(patched, tmp_path: Path) -> None:
    run_entropy_based_metrics(
        models=["vmf"],
        dataset="dummy",
        iterations=[0],
        num_topics=4,
        categories=["all"],
        out_root=tmp_path,
    )
    out_dir = next((tmp_path / "dummy" / "default" / "all").glob("it0__k4__vmf__*"))
    with (out_dir / "iter0" / "topic_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "topic",
            "topic_doc_entropy",
            "topic_doc_entropy_normalized",
            "topic_rank1_doc_fraction",
            "is_diffuse",
            "is_dead",
            "empty",
        ]
        rows = list(reader)
    _, results = read_evaluation_json(out_dir / "entropy_metrics_agg.json")
    entry = results["per_iteration"][0]
    dead = sum(row["is_dead"] == "True" for row in rows) / len(rows)
    diffuse = sum(row["is_diffuse"] == "True" for row in rows) / len(rows)
    assert entry["topic_dead_fraction"] == pytest.approx(dead)
    assert entry["topic_diffuse_fraction"] == pytest.approx(diffuse)


def test_auto_source_is_resolved_per_model_and_recorded(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        metrics_module, "load_doc_topic_matrix", _fake_loader(tmp_path, calls=calls)
    )
    run_entropy_based_metrics(
        models=["vmf", "bleilda"],
        dataset="dummy",
        iterations=[0],
        num_topics=4,
        categories=["all"],
        out_root=tmp_path,
        doc_topic_source="auto",
    )
    by_model = {call["model"]: call["doc_topic_source"] for call in calls}
    assert by_model == {"vmf": "foldincounts", "bleilda": "soft_proxy"}
    recorded = {}
    for path in (tmp_path / "dummy" / "default" / "all").glob(
        "*/entropy_metrics_agg.json"
    ):
        meta, _ = read_evaluation_json(path)
        recorded[meta["model"]] = meta["doc_topic_source"]
    assert recorded == {"vmf": "foldincounts", "bleilda": "soft_proxy"}

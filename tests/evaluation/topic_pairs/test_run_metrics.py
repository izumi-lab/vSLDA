from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.core.artifacts import load_json
from src.evaluation.reporting import read_evaluation_json
from src.evaluation.topic_pairs import metrics as metrics_module
from src.evaluation.topic_pairs.metrics import (
    CROSS_MODEL_KEY,
    METRICS_FILENAME,
    run_topic_pair_metrics,
    summary_fieldnames,
)
from src.evaluation.topic_pairs.numerics import PER_PAIR_KEYS, PER_TOPIC_KEYS


def _set_fixed_now(monkeypatch, iso_timestamp: str) -> None:
    fixed_dt = datetime.fromisoformat(iso_timestamp).astimezone(UTC)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_dt
            return fixed_dt.astimezone(tz)

    monkeypatch.setattr(metrics_module, "datetime", _FixedDateTime)


def _condition_paths(root: Path, category: str = "a") -> dict[str, Path]:
    found = {}
    for path in (root / "dummy" / "default" / category).glob(f"*/{METRICS_FILENAME}"):
        meta, _ = read_evaluation_json(path)
        found[str(meta["model"])] = path
    return found


def test_run_writes_model_and_cross_conditions(toy_runner, tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    summary_path = run_topic_pair_metrics(
        models=["vmf_sentence_lda", "sentlda"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=3,
        categories=["a"],
        out_root=out_root,
        embedding_variant="mpnet",
    )
    assert summary_path == out_root / "summary.csv"
    assert [call["model"] for call in toy_runner] == [
        "vmf",
        "sentlda",
        "vmf",
        "sentlda",
    ]

    paths = _condition_paths(out_root)
    assert set(paths) == {"vmf", "sentlda", CROSS_MODEL_KEY}
    assert "it0__k3__vmf__mpnet__" in paths["vmf"].parent.name
    assert "it0__k3__sentlda__" in paths["sentlda"].parent.name
    assert "it0__k3__cross__mpnet__" in paths[CROSS_MODEL_KEY].parent.name

    meta, results = read_evaluation_json(paths["vmf"])
    assert meta["task"] == "topic_pair_metrics"
    assert meta["split"] == "train"
    assert meta["embedding_space"] == "l2_normalized_raw_encoder_output"
    assert meta["posterior_definition"] == "collapsed_fold_in"
    assert meta["foldin_burn_in_sweeps"] == 20 and meta["foldin_retained_samples"] == 20
    assert meta["kappa_estimator"] == "banerjee_2005_approx"
    assert meta["label_names"] == ["label_x", "label_y"]
    assert meta["label_source_model"] == "vmf"
    assert meta["effective_embedding_variant"] == "mpnet"
    assert meta["encoder_model_name"] == "fake-encoder"
    assert meta["num_sentences"] == 4
    assert meta["model_provenance"]["model_key"] == "vmf_sentence_lda"
    assert meta["source_condition_dirs"] == [
        str(tmp_path / "runs" / "vmf" / "k3_it0"),
        str(tmp_path / "runs" / "vmf" / "k3_it1"),
    ]
    assert meta["posterior_metadata"] == {"posterior_kind": "toy"}
    assert (paths["vmf"].parent / "metadata.json").exists()

    entries = results["per_iteration"]
    assert [entry["iteration"] for entry in entries] == [0, 1]
    entry = entries[0]
    assert entry["num_documents"] == 2 and entry["num_sentences"] == 4
    assert entry["embedding_dim"] == 3 and entry["num_topics"] == 3
    assert set(entry["per_topic"]) == set(PER_TOPIC_KEYS) | {"label_mass"}
    assert set(entry["per_pair"]) == set(PER_PAIR_KEYS)
    assert len(entry["per_topic"]["mass_soft"]) == 3
    assert np.asarray(entry["per_topic"]["label_mass"]).shape == (3, 2)
    assert np.asarray(entry["per_pair"]["centroid_cosine_soft"]).shape == (3, 3)
    assert sum(entry["per_topic"]["mass_soft"]) == pytest.approx(4.0)
    assert entry["model_own"] is not None
    assert len(entry["model_own"]["kappa_model"]) == 3
    assert np.asarray(entry["model_own"]["centroid_cosine_model"]).shape == (3, 3)
    assert entry["source_condition_dir"].endswith("vmf/k3_it0")
    for key in ("mean_offdiag_centroid_cosine_soft", "num_empty_topics"):
        assert key in results["aggregate"]

    _, sentlda_results = read_evaluation_json(paths["sentlda"])
    assert sentlda_results["per_iteration"][0]["model_own"] is None

    cross_meta, cross_results = read_evaluation_json(paths[CROSS_MODEL_KEY])
    assert cross_meta["model"] == CROSS_MODEL_KEY
    assert cross_meta["models"] == ["vmf", "sentlda"]
    assert cross_meta["pairs"] == ["vmf|sentlda"]
    assert cross_meta["split"] == "train"
    assert set(cross_meta["model_provenance"]) == {"vmf", "sentlda"}
    cross_entry = cross_results["per_iteration"][0]
    overlap = np.asarray(cross_entry["overlap"]["vmf|sentlda"])
    assert overlap.shape == (3, 3)
    assert overlap.sum() == pytest.approx(4.0)
    assert cross_entry["num_topics"] == {"vmf": 3, "sentlda": 3}

    for iteration in (0, 1):
        iter_dir = paths["vmf"].parent / f"iter{iteration}"
        assert (iter_dir / "topic_metrics.csv").exists()
        assert (iter_dir / "label_mass.csv").exists()
        assert (iter_dir / "centroid_cosine_soft.csv").exists()
        assert (iter_dir / "assignment_confusion.csv").exists()
        assert (iter_dir / "label_js_divergence.csv").exists()
        assert (iter_dir / "centroid_cosine_model.csv").exists()
        assert (
            paths[CROSS_MODEL_KEY].parent
            / f"iter{iteration}"
            / "overlap_vmf__sentlda.csv"
        ).exists()
    with (paths["vmf"].parent / "iter0" / "label_mass.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["topic", "label_x", "label_y"]
        assert len(list(reader)) == 3

    summary_meta, summary_results = read_evaluation_json(out_root / "summary.json")
    assert summary_meta["task"] == "topic_pair_metrics_summary"
    assert summary_results["columns"] == summary_fieldnames()
    assert [row["model"] for row in summary_results["rows"]] == [
        "vmf",
        "sentlda",
        CROSS_MODEL_KEY,
    ]
    assert summary_results["rows"][0]["num_iterations"] == 2


def test_skip_existing_reuses_output_without_recomputing(
    toy_runner, monkeypatch, tmp_path: Path
) -> None:
    out_root = tmp_path / "out"
    common = dict(
        models=["vmf", "sentlda"],
        dataset="dummy",
        iterations=[0],
        num_topics=[3],
        categories=["a"],
        out_root=out_root,
        embedding_variant="mpnet",
    )
    run_topic_pair_metrics(**common)
    calls_before = len(toy_runner)

    def _boom(**_kwargs):
        raise AssertionError("posteriors must not be recomputed when skipping")

    monkeypatch.setattr(metrics_module, "compute_model_posterior", _boom)
    monkeypatch.setattr(
        metrics_module,
        "load_category_context",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no context needed")),
    )
    summary_path = run_topic_pair_metrics(**common, skip_existing=True)
    assert len(toy_runner) == calls_before
    _, summary_results = read_evaluation_json(summary_path.with_suffix(".json"))
    assert [row["model"] for row in summary_results["rows"]] == [
        "vmf",
        "sentlda",
        CROSS_MODEL_KEY,
    ]


def test_isolate_policy_records_failures_and_keeps_the_rest(
    toy_runner, monkeypatch, tmp_path: Path
) -> None:
    good = metrics_module.compute_model_posterior

    def _posterior(**kwargs):
        if kwargs["model"] == "sentlda":
            raise FileNotFoundError("missing sentlda artifact")
        return good(**kwargs)

    monkeypatch.setattr(metrics_module, "compute_model_posterior", _posterior)
    with pytest.raises(FileNotFoundError):
        run_topic_pair_metrics(
            models=["vmf", "sentlda"],
            dataset="dummy",
            iterations=[0],
            num_topics=3,
            categories=["a"],
            out_root=tmp_path / "fail_fast",
            embedding_variant="mpnet",
        )

    out_root = tmp_path / "isolate"
    run_topic_pair_metrics(
        models=["vmf", "sentlda"],
        dataset="dummy",
        iterations=[0],
        num_topics=3,
        categories=["a"],
        out_root=out_root,
        embedding_variant="mpnet",
        condition_failure_policy="isolate",
    )
    failures = load_json(out_root / "failed_conditions.json")["failures"]
    assert [failure["model"] for failure in failures] == ["sentlda", CROSS_MODEL_KEY]
    assert failures[0]["error_type"] == "FileNotFoundError"
    assert set(_condition_paths(out_root)) == {"vmf"}
    _, summary_results = read_evaluation_json(out_root / "summary.json")
    assert [row["model"] for row in summary_results["rows"]] == ["vmf"]


def test_category_context_failure_isolates_the_whole_category(
    toy_runner, monkeypatch, tmp_path: Path
) -> None:
    good = metrics_module.load_category_context

    def _context(**kwargs):
        if kwargs["category"] == "b":
            raise FileNotFoundError("no reference run for b")
        return good(**kwargs)

    monkeypatch.setattr(metrics_module, "load_category_context", _context)
    out_root = tmp_path / "out"
    run_topic_pair_metrics(
        models=["vmf"],
        dataset="dummy",
        iterations=[0],
        num_topics=[3, 4],
        categories=["a", "b"],
        out_root=out_root,
        embedding_variant="mpnet",
        condition_failure_policy="isolate",
    )
    failures = load_json(out_root / "failed_conditions.json")["failures"]
    assert {failure["category"] for failure in failures} == {"b"}
    assert {failure["model"] for failure in failures} == {"category_context", "vmf"}
    assert set(_condition_paths(out_root, "a")) == {"vmf"}
    assert _condition_paths(out_root, "b") == {}
    assert [call["condition_dir"].name for call in toy_runner] == ["k3_it0", "k4_it0"]


def test_default_out_root_uses_archive_latest_layout(
    toy_runner, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        metrics_module, "_uses_default_output_layout", lambda _root: True
    )
    _set_fixed_now(monkeypatch, "2026-08-28T00:00:00+00:00")
    run_topic_pair_metrics(
        models=["vmf", "sentlda"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=3,
        categories=["a"],
        out_root=tmp_path,
        embedding_variant="mpnet",
    )
    archive_root = tmp_path / "archive" / "2026-08-28" / "dummy" / "default" / "a"
    out_path = next(
        archive_root.glob(f"it0__k3__vmf__*/exec_20260828T000000Z/{METRICS_FILENAME}")
    )
    meta, _ = read_evaluation_json(out_path)
    assert meta["display_key"] == meta["condition_id"]
    assert meta["archive_dir"].endswith("/exec_20260828T000000Z")
    pointer = load_json(
        tmp_path
        / "latest"
        / "dummy"
        / "default"
        / "a"
        / meta["display_key"]
        / "CURRENT.json"
    )
    assert pointer["task"] == "topic_pair_metrics"
    assert pointer["artifacts"]["metrics"] == METRICS_FILENAME
    assert pointer["artifacts"]["topic_metrics_csv_iter1"] == "iter1/topic_metrics.csv"
    cross_pointer = next(
        (tmp_path / "latest" / "dummy" / "default" / "a").glob(
            "it0__k3__cross__*/CURRENT.json"
        )
    )
    assert load_json(cross_pointer)["artifacts"]["overlap_vmf__sentlda_csv_iter0"] == (
        "iter0/overlap_vmf__sentlda.csv"
    )

    monkeypatch.setattr(
        metrics_module,
        "compute_model_posterior",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must skip")),
    )
    run_topic_pair_metrics(
        models=["vmf", "sentlda"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=3,
        categories=["a"],
        out_root=tmp_path,
        embedding_variant="mpnet",
        skip_existing=True,
    )


def test_sentlda_condition_id_separates_encoder_spaces(
    toy_runner, tmp_path: Path
) -> None:
    # SentLDA stores no embedding variant of its own, but every metric is
    # computed in the requested encoder space, so one condition id for all
    # encoders let a later run reuse or overwrite another encoder's numbers.
    def _run(variant: str) -> str:
        out_root = tmp_path / f"out_{variant}"
        run_topic_pair_metrics(
            models=["sentlda"],
            dataset="dummy",
            iterations=[0],
            num_topics=3,
            categories=["a"],
            out_root=out_root,
            embedding_variant=variant,
        )
        meta, _ = read_evaluation_json(_condition_paths(out_root)["sentlda"])
        return str(meta["condition_id"])

    assert _run("minilm") != _run("mpnet")


def test_rejects_invalid_options(toy_runner, tmp_path: Path) -> None:
    common = dict(
        models=["vmf"],
        dataset="dummy",
        iterations=[0],
        num_topics=3,
        categories=["a"],
        out_root=tmp_path,
    )
    with pytest.raises(ValueError, match="condition_failure_policy"):
        run_topic_pair_metrics(**common, condition_failure_policy="bogus")
    with pytest.raises(ValueError, match="split"):
        run_topic_pair_metrics(**common, split="dev")
    with pytest.raises(ValueError, match="mismatch"):
        run_topic_pair_metrics(
            **common,
            embedding_variant="bge",
            encoder_model="sentence-transformers/all-mpnet-base-v2",
        )
    with pytest.raises(ValueError, match="Unsupported model"):
        run_topic_pair_metrics(**{**common, "models": ["etm"]})

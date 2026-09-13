from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.core.artifacts import load_json, save_json
from src.evaluation.foldin import runner as runner_module
from src.evaluation.foldin import theta as theta_module
from src.evaluation.foldin.runner import (
    iter_vmf_run_pointers,
    parse_display_key,
    run_vmf_foldin_theta,
    select_runs,
)
from src.evaluation.topic_pairs.inputs import encoder_fingerprint
from src.evaluation.word_based.topic_assignment import CollapsedFoldInConfig
from tests.evaluation.foldin.conftest import (
    ENCODER_CONFIG,
    toy_embeddings,
    toy_log_likelihoods,
    write_vmf_run,
)

FAST = CollapsedFoldInConfig(burn_in_sweeps=1, retained_samples=2)


def _write_pointer(
    results_root: Path,
    *,
    dataset: str = "dummy",
    category: str = "cat",
    display_key: str,
    parameter_variant: str | None = None,
    with_metadata: bool = True,
) -> Path:
    archive_dir = (
        results_root
        / dataset
        / "default"
        / "vmf_sentence_lda"
        / "archive"
        / "2026-01-01"
        / category
        / display_key
        / "vmf_exec"
    )
    write_vmf_run(archive_dir, with_metadata=with_metadata)
    pointer_path = (
        results_root
        / dataset
        / "default"
        / "vmf_sentence_lda"
        / "latest"
        / category
        / display_key
        / "CURRENT.json"
    )
    payload = {
        "schema": "latest_result_pointer",
        "schema_version": 1,
        "task": "vmf_experiment",
        "display_key": display_key,
        "dataset": dataset,
        "data_run": "default",
        "category": category,
        "archive_dir": str(archive_dir),
        "started_at": "2026-01-01T00:00:00+00:00",
        "execution_id": "vmf_exec",
        "condition_fingerprint": "cond-fp",
        "embedding_variant": "minilm",
        "artifacts": {
            "train_path": "doc_topic_train.pkl",
            "infer_path": "doc_topic_test.pkl",
        },
    }
    if parameter_variant is not None:
        payload["parameter_variant"] = parameter_variant
    save_json(payload, pointer_path)
    return pointer_path


def _patch_encoder(monkeypatch) -> list[str]:
    calls: list[str] = []

    def _fake_encode(
        corpus,
        *,
        encoder_config,
        cache_root,
        dataset,
        data_run,
        category,
        split,
        device,
        encode_batch_size=None,
        encoder_factory=None,
    ):
        calls.append(f"{category}/{split}")
        return toy_embeddings(
            corpus,
            cache_dir=Path(cache_root) / category / split,
            encoder_fp=encoder_fingerprint(encoder_config),
        )

    monkeypatch.setattr(runner_module, "encode_train_corpus", _fake_encode)
    monkeypatch.setattr(
        theta_module, "vmf_sentence_log_likelihoods", toy_log_likelihoods
    )
    monkeypatch.setattr(
        runner_module, "resolve_topic_word_encoder_device", lambda value: "cpu"
    )
    return calls


def test_parse_display_key_reads_k_and_iteration() -> None:
    assert parse_display_key("k20_it3_c1_minilm") == (20, 3)
    assert parse_display_key("k100_it0_c1_minilm_t-1") == (100, 0)
    with pytest.raises(ValueError):
        parse_display_key("svm_hard_k20")


def test_select_runs_filters_and_hides_variants_by_default(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _write_pointer(root, display_key="k2_it0_c1_minilm")
    _write_pointer(root, display_key="k2_it1_c1_minilm")
    _write_pointer(root, display_key="k2_it0_c1_minilm_t-1", parameter_variant="t-1")
    pointers = iter_vmf_run_pointers(datasets=["dummy"], results_root=root)
    # Sorted by K and iteration, a variant next to its main run.
    assert [item.display_key for item in pointers] == [
        "k2_it0_c1_minilm",
        "k2_it0_c1_minilm_t-1",
        "k2_it1_c1_minilm",
    ]
    assert [p.display_key for p in select_runs(pointers)] == [
        "k2_it0_c1_minilm",
        "k2_it1_c1_minilm",
    ]
    assert [p.display_key for p in select_runs(pointers, iterations=[1])] == [
        "k2_it1_c1_minilm"
    ]
    assert [p.display_key for p in select_runs(pointers, vmf_variants=["t-1"])] == [
        "k2_it0_c1_minilm_t-1"
    ]
    assert len(select_runs(pointers, include_vmf_variants=True)) == 3
    assert len(select_runs(pointers, iterations=[7], all_vmf_runs=True)) == 3
    assert select_runs(pointers, embedding_variants=["bge"]) == []


def test_run_encodes_each_unit_once_and_writes_every_run(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "experiments"
    first = _write_pointer(root, display_key="k2_it0_c1_minilm")
    second = _write_pointer(root, display_key="k2_it1_c1_minilm")
    calls = _patch_encoder(monkeypatch)

    summary = run_vmf_foldin_theta(
        datasets=["dummy"],
        splits=["test", "train"],
        cache_root=tmp_path / "cache",
        foldin_config=FAST,
        summary_path=tmp_path / "summary.csv",
        results_root=root,
    )

    # One encoding per (unit, split), shared by the two runs.
    assert calls == ["cat/test", "cat/train"]
    for pointer_path in (first, second):
        payload = load_json(pointer_path)
        archive_dir = Path(payload["archive_dir"])
        for split in ("test", "train"):
            assert (archive_dir / f"doc_topic_{split}_foldin.pkl").exists()
            assert (archive_dir / f"doc_topic_{split}_foldin_counts.pkl").exists()
        meta = load_json(archive_dir / "foldin_meta.json")
        assert set(meta["splits"]) == {"train", "test"}
        # The pointer gained the fold-in keys and kept the others.
        assert payload["artifacts"]["train_path"] == "doc_topic_train.pkl"
        assert (
            payload["artifacts"]["test_doc_topic_foldin"] == "doc_topic_test_foldin.pkl"
        )
        assert (
            payload["artifacts"]["train_doc_topic_foldin"]
            == "doc_topic_train_foldin.pkl"
        )
        assert payload["started_at"] == "2026-01-01T00:00:00+00:00"
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"computed"}
    assert rows[0]["num_documents"] == "3"
    assert not summary.with_suffix(".failures.json").exists()

    # A second invocation skips everything whose settings and inputs match.
    calls.clear()
    run_vmf_foldin_theta(
        datasets=["dummy"],
        splits=["test"],
        cache_root=tmp_path / "cache",
        foldin_config=FAST,
        summary_path=tmp_path / "summary2.csv",
        results_root=root,
    )
    with (tmp_path / "summary2.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["status"] for row in rows} == {"skipped"}
    # Different sampler settings invalidate the artifact.
    run_vmf_foldin_theta(
        datasets=["dummy"],
        splits=["test"],
        cache_root=tmp_path / "cache",
        foldin_config=CollapsedFoldInConfig(burn_in_sweeps=2, retained_samples=2),
        summary_path=tmp_path / "summary3.csv",
        results_root=root,
    )
    with (tmp_path / "summary3.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["status"] for row in rows} == {"computed"}


def test_no_update_pointer_leaves_current_json_alone(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "experiments"
    pointer_path = _write_pointer(root, display_key="k2_it0_c1_minilm")
    before = pointer_path.read_text(encoding="utf-8")
    _patch_encoder(monkeypatch)
    run_vmf_foldin_theta(
        datasets=["dummy"],
        splits=["test"],
        cache_root=tmp_path / "cache",
        foldin_config=FAST,
        update_pointer=False,
        summary_path=tmp_path / "summary.csv",
        results_root=root,
    )
    assert pointer_path.read_text(encoding="utf-8") == before
    archive_dir = Path(json.loads(before)["archive_dir"])
    assert (archive_dir / "doc_topic_test_foldin.pkl").exists()


def test_isolate_policy_records_a_broken_run_and_continues(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "experiments"
    _write_pointer(root, display_key="k2_it0_c1_minilm", with_metadata=False)
    good = _write_pointer(root, display_key="k2_it1_c1_minilm")
    _patch_encoder(monkeypatch)

    with pytest.raises(FileNotFoundError):
        run_vmf_foldin_theta(
            datasets=["dummy"],
            splits=["test"],
            cache_root=tmp_path / "cache",
            foldin_config=FAST,
            summary_path=tmp_path / "summary.csv",
            results_root=root,
        )

    summary = run_vmf_foldin_theta(
        datasets=["dummy"],
        splits=["test"],
        cache_root=tmp_path / "cache",
        foldin_config=FAST,
        condition_failure_policy="isolate",
        summary_path=tmp_path / "summary_isolate.csv",
        results_root=root,
    )
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = {row["display_key"]: row for row in csv.DictReader(handle)}
    assert rows["k2_it0_c1_minilm"]["status"] == "failed"
    assert rows["k2_it1_c1_minilm"]["status"] == "computed"
    failures = load_json(summary.with_suffix(".failures.json"))["failures"]
    assert [item["display_key"] for item in failures] == ["k2_it0_c1_minilm"]
    assert (Path(load_json(good)["archive_dir"]) / "doc_topic_test_foldin.pkl").exists()


def test_rejects_unknown_split_and_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_vmf_foldin_theta(datasets=["dummy"], splits=["dev"], results_root=tmp_path)
    with pytest.raises(ValueError):
        run_vmf_foldin_theta(
            datasets=["dummy"], condition_failure_policy="ignore", results_root=tmp_path
        )
    theta = np.zeros((1, 2))
    assert theta.shape == (1, 2)

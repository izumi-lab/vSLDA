"""The vMF hyperparameter-sweep runs are separate conditions in every evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.classification.feature_registry import _vmf_variant_matches
from src.evaluation.classification.summary import _matches_metric_meta
from src.evaluation.classification.workflow import ClassificationCondition
from src.evaluation.entropy_based import inputs as entropy_inputs
from src.evaluation.word_based import summary as wb_summary
from src.evaluation.word_based.reporting import build_output_condition_id


def test_feature_registry_keeps_only_the_requested_vmf_variant() -> None:
    default = {"parameter_variant": None}
    swept = {"parameter_variant": "kappa0-100"}
    assert _vmf_variant_matches(
        model_key="vmf_sentence_lda",
        pointer_payload=default,
        metadata={},
        vmf_variant=None,
    )
    assert not _vmf_variant_matches(
        model_key="vmf_sentence_lda",
        pointer_payload=swept,
        metadata={},
        vmf_variant=None,
    )
    assert _vmf_variant_matches(
        model_key="vmf_sentence_lda",
        pointer_payload=swept,
        metadata={},
        vmf_variant="kappa0-100",
    )
    assert not _vmf_variant_matches(
        model_key="vmf_sentence_lda",
        pointer_payload=default,
        metadata={},
        vmf_variant="kappa0-100",
    )
    # metadata is the fallback when the pointer predates the key
    assert _vmf_variant_matches(
        model_key="vmf_sentence_lda",
        pointer_payload={},
        metadata={"parameter_variant": "zeta-40"},
        vmf_variant="zeta-40",
    )
    # other models never carry the label
    assert _vmf_variant_matches(
        model_key="sentlda", pointer_payload=swept, metadata={}, vmf_variant=None
    )


def _condition(**overrides: object) -> ClassificationCondition:
    base = dict(
        dataset="20newsgroup",
        topics=20,
        iteration=0,
        classifiers=["logreg"],
        vmf_assignment="hard",
        target_column="target_str",
        label_schema="identity",
        embedding_variants=["minilm"],
        selected_models=["vmf_sentence_lda"],
    )
    base.update(overrides)
    return ClassificationCondition(**base)


def test_classification_condition_labels_and_records_the_variant() -> None:
    default = _condition()
    swept = _condition(vmf_variant="kappa0-100")
    assert default.display_key() == _condition(vmf_variant=None).display_key()
    assert swept.display_key().endswith("_kappa0-100_k20_it0")
    assert "vmf_variant" not in default.payload()
    assert swept.payload()["vmf_variant"] == "kappa0-100"
    assert default.condition_id() != swept.condition_id()
    meta_default = default.meta_payload() if hasattr(default, "meta_payload") else None
    if meta_default is not None:
        assert meta_default["vmf_variant"] is None


def test_classification_summary_filters_on_the_recorded_variant() -> None:
    common = dict(
        iteration=0,
        topics=20,
        vmf_assignment="hard",
        data_run="default",
        alignment_mode="intersection",
        classifiers=None,
        embedding_variants=None,
        feature_resolve_mode="all",
        selected_models=None,
    )
    meta = {
        "iteration": 0,
        "topics": 20,
        "vmf_assignment": "hard",
        "data_run": "default",
        "alignment_mode": "intersection",
        "feature_resolve_mode": "all",
    }
    assert _matches_metric_meta(dict(meta), **common)
    assert _matches_metric_meta(dict(meta), **common, vmf_variant=None)
    assert not _matches_metric_meta(dict(meta), **common, vmf_variant="kappa0-100")
    assert _matches_metric_meta(
        {**meta, "vmf_variant": "kappa0-100"}, **common, vmf_variant="kappa0-100"
    )
    assert not _matches_metric_meta({**meta, "vmf_variant": "kappa0-100"}, **common)


def test_entropy_resolver_passes_the_variant_to_the_vmf_resolver(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        entropy_inputs,
        "resolve_vmf_experiment_dir",
        lambda **kwargs: calls.append(kwargs) or tmp_path,
    )
    entropy_inputs.resolve_condition_dir(
        model="vmf",
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=20,
        category="all",
        embedding_variant="minilm",
        vmf_variant="t-5",
    )
    assert calls[0]["parameter_variant"] == "t-5"


def test_coherence_condition_id_names_a_vmf_variant_only() -> None:
    def cid(model: str, variant: str | None) -> str:
        condition_id, _ = build_output_condition_id(
            model=model,
            dataset="dummy",
            data_run="default",
            category="all",
            iterations=[0],
            num_topics=20,
            coherence="c_v",
            coherences=["c_v"],
            coherence_topn=10,
            coherence_window_size=None,
            coherence_implementation="palmetto-cv",
            coherence_min_window_count=None,
            coherence_reference="wikipedia",
            coherence_reference_path=None,
            coherence_reference_format="tokenized_jsonl",
            coherence_reference_max_docs=None,
            coherence_reference_min_doc_tokens=1,
            coherence_reference_streaming=False,
            diversity_topn=25,
            coherence_split="train",
            topic_word_source="posthoc",
            embedding_variant="minilm",
            metric_names=["coherence_c_v"],
            parameter_variant=variant,
        )
        return condition_id

    assert "__kappa0-100" in cid("vmf", "kappa0-100")
    assert "kappa0" not in cid("vmf", None)
    assert cid("vmf", None) != cid("vmf", "kappa0-100")
    # a baseline-style variant is not a vMF label and is not printed for vmf
    assert "passes-40" not in cid("vmf", "passes-40")


def _write_vmf_condition(
    root: Path, *, variant: str | None, condition_id: str, c_v: float
) -> None:
    archive = (
        root
        / "archive"
        / "2026-09-02"
        / "20newsgroup"
        / "default"
        / "computer"
        / condition_id
        / "exec"
    )
    archive.mkdir(parents=True)
    provenance = {
        "model_family": "vmf_sentence_lda",
        "parameter_variant": variant,
        "vmf_hyperparameters": {
            "kappa0": 100.0 if variant else 10.0,
            "gibbs_sweeps": 20,
            "num_samples": 8,
            "num_iterations": 10,
        },
    }
    payload = {
        "_meta": {
            "task": "word_based_metrics",
            "dataset": "20newsgroup",
            "data_run": "default",
            "category": "computer",
            "model": "vmf",
            "num_topics": 20,
            "iterations": [0],
            "condition_id": condition_id,
            "display_key": condition_id,
            "execution_id": "exec",
            "started_at": "2026-09-02T00:00:00+00:00",
            "embedding_variant": "minilm",
            "effective_embedding_variant": "minilm",
            "prior_scale": None,
            "model_provenance": provenance,
            "topic_words": {"coherence_topn": 10, "diversity_topn": 25},
            "coherence": {
                "metrics": ["c_v"],
                "primary_metric": "c_v",
                "coherence_reference": "wikipedia",
                "coherence_reference_num_docs": 1000000,
                "topn": 10,
                "split": "train",
            },
        },
        "results": {
            "aggregate": {
                "coherence_c_v": {"mean": c_v, "std": 0.0},
                "diversity": {"mean": 0.9, "std": 0.0},
            },
            "per_iteration": [
                {"coherence_c_v": c_v, "diversity": 0.9, "num_topics": 20.0}
            ],
        },
    }
    (archive / "metrics_agg.json").write_text(json.dumps(payload), encoding="utf-8")
    pointer = root / "latest" / "20newsgroup" / "default" / "computer" / condition_id
    pointer.mkdir(parents=True)
    (pointer / "CURRENT.json").write_text(
        json.dumps(
            {
                "schema": "latest_result_pointer",
                "task": "word_based_metrics",
                "archive_dir": str(archive),
                "artifacts": {"metrics": "metrics_agg.json"},
            }
        ),
        encoding="utf-8",
    )


def test_coherence_summary_selects_a_variant_and_suffixes_its_sidecar(
    tmp_path: Path,
) -> None:
    _write_vmf_condition(
        tmp_path, variant=None, condition_id="it0__k20__vmf__minilm", c_v=0.2
    )
    _write_vmf_condition(
        tmp_path,
        variant="kappa0-100",
        condition_id="it0__k20__vmf__minilm__kappa0-100",
        c_v=0.3,
    )
    rows, warnings = wb_summary.collect_summary_rows(coherence_root=tmp_path)
    assert {row["vmf_variant"] for row in rows} == {"", "kappa0-100"}

    kept = wb_summary.select_rows(rows, warnings=warnings)
    assert [row["condition_id"] for row in kept] == ["it0__k20__vmf__minilm"]
    assert any("hyperparameter variants" in message for message in warnings)
    swept = wb_summary.select_rows(rows, vmf_variant="kappa0-100")
    assert [row["condition_id"] for row in swept] == [
        "it0__k20__vmf__minilm__kappa0-100"
    ]
    with pytest.raises(ValueError):
        wb_summary.select_rows(rows, vmf_variant="kappa-100")

    summary_root = tmp_path / "summaries"
    wb_summary.run_word_based_summary(
        coherence_root=tmp_path,
        iterations=[0],
        summary_root=summary_root,
        metrics=["coherence_c_v"],
        vmf_variant="kappa0-100",
        write_flat_summary=False,
    )
    sidecar = (
        summary_root
        / "20newsgroup"
        / "default"
        / "minilm"
        / "coherence_20newsgroup_default_minilm_20topic_kappa0-100.scores.json"
    )
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["scores"]["coherence_c_v"]["computer"]["vmf"] == [0.3]
    provenance = payload["provenance"]["computer"]["vmf"]
    assert provenance["vmf_variant"] == "kappa0-100"
    assert json.loads(provenance["vmf_hyperparameters"])["kappa0"] == 100.0
    assert not (
        summary_root
        / "20newsgroup"
        / "default"
        / "minilm"
        / "coherence_20newsgroup_default_minilm_20topic.scores.json"
    ).exists()

    wb_summary.run_word_based_summary(
        coherence_root=tmp_path,
        iterations=[0],
        summary_root=summary_root,
        metrics=["coherence_c_v"],
        write_flat_summary=False,
    )
    default = json.loads(
        (
            summary_root
            / "20newsgroup"
            / "default"
            / "minilm"
            / "coherence_20newsgroup_default_minilm_20topic.scores.json"
        ).read_text()
    )
    assert default["scores"]["coherence_c_v"]["computer"]["vmf"] == [0.2]
    assert default["provenance"]["computer"]["vmf"]["vmf_variant"] is None

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.artifacts import save_pickle
from src.evaluation.entropy_based import inputs as inputs_module
from src.evaluation.entropy_based.inputs import (
    DOC_TOPIC_SOURCES,
    SOURCE_FOLDIN_COUNTS_FILE,
    SOURCE_FOLDIN_FILE,
    SOURCE_HARD_FILE,
    SOURCE_SENTENCE_AGGREGATE,
    SOURCE_SOFT_FILE,
    effective_embedding_variant,
    load_doc_topic_matrix,
    normalize_model_name,
    parameter_variant_for,
    resolve_condition_dir,
    resolve_doc_topic_source,
)


def test_normalize_model_name_maps_aliases_and_rejects_unknown() -> None:
    assert normalize_model_name("vmf_sentence_lda") == "vmf"
    assert normalize_model_name("gaussian") == "sentence_gaussianlda"
    assert normalize_model_name("SentLDA") == "sentlda"
    with pytest.raises(ValueError):
        normalize_model_name("bertopic_kmeans")


@pytest.mark.parametrize(
    ("model", "variant", "expected"),
    [
        ("vmf", "mpnet", "mpnet"),
        ("ctm", "bge", "bge"),
        ("senclu", "minilm", "minilm"),
        ("sentence_gaussianlda", "mpnet", "mpnet_norm"),
        ("sentence_gaussianlda", "mpnet_norm", "mpnet_norm"),
        ("sentence_gaussianlda", "mpnet_raw", "mpnet_raw"),
        ("etm", "mpnet", "googlenews300"),
        ("gaussianlda", None, "googlenews300"),
        ("mvtm", "bge", "googlenews300"),
        ("bleilda", "mpnet", None),
        ("sentlda", "mpnet", None),
        ("vmf", None, None),
    ],
)
def test_effective_embedding_variant(model: str, variant: str | None, expected) -> None:
    assert effective_embedding_variant(model, variant) == expected


def test_effective_embedding_variant_honours_word_embedding_override() -> None:
    assert effective_embedding_variant(
        "etm", "mpnet", word_embedding_variant="glove100"
    ) == ("glove100")
    assert (
        effective_embedding_variant("etm", "mpnet", word_embedding_variant=None) is None
    )


def test_parameter_variant_only_for_gaussian_family() -> None:
    assert parameter_variant_for("gaussianlda", 0.1) == "psi0-0p1"
    assert parameter_variant_for("sentence_gaussianlda", 1.0) == "psi0-1"
    assert parameter_variant_for("bleilda", 0.1) is None
    assert parameter_variant_for("gaussianlda", None) is None


def test_resolve_condition_dir_forwards_prior_scale_and_variant(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_resolve(**kwargs):
        captured.update(kwargs)
        return tmp_path / "condition"

    monkeypatch.setattr(inputs_module, "resolve_baseline_condition_dir", _fake_resolve)
    resolved = resolve_condition_dir(
        model="sentence_gaussianlda",
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=4,
        category="cat",
        embedding_variant="mpnet_raw",
        prior_scale=0.1,
    )
    assert resolved == tmp_path / "condition"
    assert captured["model"] == "sentence_gaussianlda"
    assert captured["embedding_variant"] == "mpnet_raw"
    assert captured["parameter_variant"] == "psi0-0p1"


def _write_condition(
    tmp_path: Path,
    *,
    model: str,
    category: str = "cat",
    soft: np.ndarray | None = None,
    sentence_soft: list[np.ndarray] | None = None,
    hard: np.ndarray | None = None,
    raw_doc_indices: list[int] | None = None,
    foldin: np.ndarray | None = None,
    foldin_counts: np.ndarray | None = None,
) -> Path:
    condition_dir = tmp_path / model / "condition"
    if model == "vmf":
        base = condition_dir
        soft_name, sentence_name, hard_name, foldin_name, counts_name = (
            "doc_topic_test_soft.pkl",
            "sentence_topic_test_soft.pkl",
            "doc_topic_test.pkl",
            "doc_topic_test_foldin.pkl",
            "doc_topic_test_foldin_counts.pkl",
        )
    else:
        base = condition_dir / "infer"
        soft_name, sentence_name, hard_name, foldin_name, counts_name = (
            f"{category}_doc_topic_soft.pkl",
            f"{category}_sentence_topic_soft.pkl",
            f"{category}.pkl",
            f"{category}_doc_topic_foldin.pkl",
            f"{category}_doc_topic_foldin_counts.pkl",
        )
    base.mkdir(parents=True, exist_ok=True)
    if foldin is not None:
        save_pickle(foldin, base / foldin_name)
    if foldin_counts is not None:
        save_pickle(foldin_counts, base / counts_name)
    if soft is not None:
        save_pickle(soft, base / soft_name)
    if sentence_soft is not None:
        save_pickle(sentence_soft, base / sentence_name)
    if hard is not None:
        save_pickle(hard, base / hard_name)
    if raw_doc_indices is not None:
        # The vMF runner writes both splits into one file; baselines write the
        # flat per-split form.
        payload = (
            {
                "train": {"raw_doc_indices": []},
                "test": {"raw_doc_indices": raw_doc_indices},
            }
            if model == "vmf"
            else {"raw_doc_indices": raw_doc_indices}
        )
        (base / "preprocessing_selection.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return condition_dir


def _patch_resolver(monkeypatch, condition_dir: Path) -> None:
    monkeypatch.setattr(
        inputs_module,
        "resolve_condition_dir",
        lambda **_kwargs: condition_dir,
    )


def _load(model: str, **overrides):
    # The tests below exercise the soft_proxy resolution; the CLI default is "auto".
    kwargs = dict(
        model=model,
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="cat",
        doc_topic_source="soft_proxy",
    )
    kwargs.update(overrides)
    return load_doc_topic_matrix(**kwargs)


def test_soft_proxy_prefers_soft_doc_file(monkeypatch, tmp_path: Path) -> None:
    condition_dir = _write_condition(
        tmp_path,
        model="mvtm",
        soft=np.asarray([[0.2, 0.8], [0.6, 0.4]]),
        hard=np.asarray([[0, 5], [3, 0]]),
        raw_doc_indices=[0, 2],
    )
    _patch_resolver(monkeypatch, condition_dir)
    load = _load("mvtm")
    assert load.source == SOURCE_SOFT_FILE
    assert np.allclose(load.theta, [[0.2, 0.8], [0.6, 0.4]])
    assert load.raw_doc_indices == [0, 2]
    assert load.condition_dir == condition_dir


def test_raw_doc_indices_are_read_from_the_vmf_combined_selection(
    monkeypatch, tmp_path: Path
) -> None:
    # The vMF runner stores {"train": ..., "test": ...} in one file; reading
    # only the flat form left the provenance column empty for every vMF run.
    condition_dir = _write_condition(
        tmp_path,
        model="vmf",
        soft=np.asarray([[0.2, 0.8], [0.6, 0.4]]),
        raw_doc_indices=[4, 7],
    )
    _patch_resolver(monkeypatch, condition_dir)
    load = _load("vmf")
    assert load.raw_doc_indices == [4, 7]


def test_soft_proxy_falls_back_to_sentence_posterior_mean(
    monkeypatch, tmp_path: Path
) -> None:
    sentence_soft = [
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        np.asarray([[0.9, 0.1]]),
        np.zeros((0, 2)),
    ]
    condition_dir = _write_condition(
        tmp_path,
        model="sentlda",
        sentence_soft=sentence_soft,
        hard=np.asarray([[1, 2], [1, 0], [0, 0]]),
    )
    _patch_resolver(monkeypatch, condition_dir)
    load = _load("sentlda")
    assert load.source == SOURCE_SENTENCE_AGGREGATE
    assert np.allclose(load.theta[0], [0.5, 0.5])
    assert np.allclose(load.theta[1], [0.9, 0.1])
    assert np.allclose(load.theta[2], [0.0, 0.0])
    assert load.raw_doc_indices is None


def test_soft_proxy_falls_back_to_hard_counts(monkeypatch, tmp_path: Path) -> None:
    condition_dir = _write_condition(
        tmp_path, model="gaussianlda", hard=np.asarray([[3, 1], [0, 2]])
    )
    _patch_resolver(monkeypatch, condition_dir)
    load = _load("gaussianlda")
    assert load.source == SOURCE_HARD_FILE
    assert np.allclose(load.theta, [[0.75, 0.25], [0.0, 1.0]])


def test_explicit_hard_and_soft_sources(monkeypatch, tmp_path: Path) -> None:
    condition_dir = _write_condition(
        tmp_path,
        model="etm",
        soft=np.asarray([[0.3, 0.7]]),
        hard=np.asarray([[1.0, 0.0]]),
    )
    _patch_resolver(monkeypatch, condition_dir)
    assert _load("etm", doc_topic_source="hard").source == SOURCE_HARD_FILE
    assert _load("etm", doc_topic_source="soft").source == SOURCE_SOFT_FILE

    only_hard = _write_condition(
        tmp_path / "b", model="etm", hard=np.asarray([[1.0, 0.0]])
    )
    _patch_resolver(monkeypatch, only_hard)
    with pytest.raises(FileNotFoundError):
        _load("etm", doc_topic_source="soft")


def test_vmf_uses_condition_dir_level_files(monkeypatch, tmp_path: Path) -> None:
    condition_dir = _write_condition(
        tmp_path,
        model="vmf",
        soft=np.asarray([[0.1, 0.9]]),
        hard=np.asarray([[0.1, 0.9]]),
    )
    _patch_resolver(monkeypatch, condition_dir)
    load = _load("vmf_sentence_lda")
    assert load.source == SOURCE_SOFT_FILE
    assert load.path == condition_dir / "doc_topic_test_soft.pkl"


def test_baseline_rejects_train_split_and_missing_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    condition_dir = _write_condition(tmp_path, model="bleilda")
    _patch_resolver(monkeypatch, condition_dir)
    with pytest.raises(ValueError, match="test split"):
        _load("bleilda", split="train")
    with pytest.raises(FileNotFoundError):
        _load("bleilda")
    with pytest.raises(ValueError, match="doc_topic_source"):
        _load("bleilda", doc_topic_source="bogus")


def test_width_mismatch_is_rejected(monkeypatch, tmp_path: Path) -> None:
    condition_dir = _write_condition(
        tmp_path, model="ctm", hard=np.asarray([[0.2, 0.3, 0.5]])
    )
    _patch_resolver(monkeypatch, condition_dir)
    with pytest.raises(ValueError, match="num_topics"):
        _load("ctm")


def test_sam_is_a_supported_document_level_model() -> None:
    """SAM has doc-topic output but no encoder, so it needs no variant suffix."""

    assert inputs_module.normalize_model_name("sam") == "sam"
    assert "sam" not in inputs_module.SENTENCE_ENCODER_MODELS
    assert "sam" not in inputs_module.WORD_EMBEDDING_MODELS
    assert "sam" not in inputs_module.GAUSSIAN_PRIOR_SCALE_MODELS
    assert effective_embedding_variant(model="sam", embedding_variant="minilm") is None
    assert parameter_variant_for("sam", 0.1) is None


def test_sam_tf_is_registered_like_sam() -> None:
    assert inputs_module.normalize_model_name("sam_tf") == "sam_tf"
    assert "sam_tf" in inputs_module.SUPPORTED_MODELS
    assert "sam_tf" not in inputs_module.SENTENCE_ENCODER_MODELS
    assert "sam_tf" not in inputs_module.WORD_EMBEDDING_MODELS
    assert (
        effective_embedding_variant(model="sam_tf", embedding_variant="minilm") is None
    )
    assert parameter_variant_for("sam_tf", 0.1) is None


def test_foldin_source_reads_only_the_foldin_file(monkeypatch, tmp_path: Path) -> None:
    assert "foldin" in DOC_TOPIC_SOURCES
    condition_dir = _write_condition(
        tmp_path,
        model="vmf",
        soft=np.asarray([[0.9, 0.1]]),
        hard=np.asarray([[1.0, 0.0]]),
        foldin=np.asarray([[0.6, 0.4]]),
    )
    _patch_resolver(monkeypatch, condition_dir)
    load = _load("vmf", doc_topic_source="foldin")
    assert load.source == SOURCE_FOLDIN_FILE
    assert np.allclose(load.theta, [[0.6, 0.4]])
    # soft_proxy never picks the fold-in file up.
    assert _load("vmf").source == SOURCE_SOFT_FILE

    without = _write_condition(
        tmp_path / "without", model="vmf", soft=np.asarray([[0.9, 0.1]])
    )
    _patch_resolver(monkeypatch, without)
    with pytest.raises(FileNotFoundError, match="vmf-foldin-theta"):
        _load("vmf", doc_topic_source="foldin")


def test_foldincounts_source_row_normalizes_the_expected_counts(
    monkeypatch, tmp_path: Path
) -> None:
    assert "foldincounts" in DOC_TOPIC_SOURCES
    condition_dir = _write_condition(
        tmp_path,
        model="vmf",
        soft=np.asarray([[0.9, 0.1]]),
        foldin=np.asarray([[0.6, 0.4]]),
        foldin_counts=np.asarray([[3.0, 1.0]]),
    )
    _patch_resolver(monkeypatch, condition_dir)
    load = _load("vmf", doc_topic_source="foldincounts")
    assert load.source == inputs_module.SOURCE_FOLDIN_COUNTS_FILE
    assert np.allclose(load.theta, [[0.75, 0.25]])
    # The smoothed fold-in file is not a substitute for the counts file.
    without = _write_condition(
        tmp_path / "without", model="vmf", foldin=np.asarray([[0.6, 0.4]])
    )
    _patch_resolver(monkeypatch, without)
    with pytest.raises(FileNotFoundError, match="vmf-foldin-theta"):
        _load("vmf", doc_topic_source="foldincounts")


def test_auto_source_resolves_per_model() -> None:
    assert "auto" in DOC_TOPIC_SOURCES
    assert resolve_doc_topic_source("vmf", "auto") == "foldincounts"
    assert resolve_doc_topic_source("vmf_sentence_lda", "auto") == "foldincounts"
    # MvTM (vLDA) shares the vMF estimators and writes the fold-in files too.
    assert resolve_doc_topic_source("mvtm", "auto") == "foldincounts"
    assert resolve_doc_topic_source("bleilda", "auto") == "soft_proxy"
    assert resolve_doc_topic_source("sentence_gaussianlda", "auto") == "soft_proxy"
    assert resolve_doc_topic_source("vmf", "hard") == "hard"
    with pytest.raises(ValueError):
        resolve_doc_topic_source("vmf", "argmax")


def test_auto_reads_the_fold_in_counts_of_a_vmf_run_and_soft_proxy_elsewhere(
    monkeypatch, tmp_path: Path
) -> None:
    vmf_dir = tmp_path / "vmf" / "condition"
    vmf_dir.mkdir(parents=True)
    save_pickle(np.asarray([[0.9, 0.1]]), vmf_dir / "doc_topic_test_soft.pkl")
    save_pickle(np.asarray([[3.0, 1.0]]), vmf_dir / "doc_topic_test_foldin_counts.pkl")
    _patch_resolver(monkeypatch, vmf_dir)
    load = _load("vmf", doc_topic_source="auto")
    assert load.source == SOURCE_FOLDIN_COUNTS_FILE
    assert np.allclose(load.theta, [[0.75, 0.25]])

    baseline_dir = _write_condition(
        tmp_path / "base", model="etm", soft=np.asarray([[0.2, 0.8]])
    )
    _patch_resolver(monkeypatch, baseline_dir)
    assert _load("etm", doc_topic_source="auto").source == SOURCE_SOFT_FILE


def test_auto_reads_the_fold_in_counts_of_an_mvtm_run_on_both_splits(
    monkeypatch, tmp_path: Path
) -> None:
    """MvTM keeps its fold-in files under ``infer/`` (test) and ``params/`` (train),
    the only baseline artifacts the train split may be loaded from."""

    condition_dir = _write_condition(
        tmp_path / "base",
        model="mvtm",
        soft=np.asarray([[0.2, 0.8]]),
        foldin_counts=np.asarray([[1.0, 3.0]]),
    )
    save_pickle(
        np.asarray([[2.0, 2.0]]),
        condition_dir / "params" / "cat_doc_topic_foldin_counts.pkl",
    )
    _patch_resolver(monkeypatch, condition_dir)
    test_load = _load("mvtm", doc_topic_source="auto")
    assert test_load.source == SOURCE_FOLDIN_COUNTS_FILE
    assert test_load.path == condition_dir / "infer" / "cat_doc_topic_foldin_counts.pkl"
    assert np.allclose(test_load.theta, [[0.25, 0.75]])
    train_load = _load("mvtm", doc_topic_source="foldincounts", split="train")
    assert (
        train_load.path == condition_dir / "params" / "cat_doc_topic_foldin_counts.pkl"
    )
    assert np.allclose(train_load.theta, [[0.5, 0.5]])
    # The other sources of a baseline stay test-only.
    with pytest.raises(ValueError, match="test split"):
        _load("mvtm", doc_topic_source="soft", split="train")

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from gensim.models import KeyedVectors

from src.core.artifacts import load_json, save_json, save_pickle
from src.evaluation.foldin import mvtm as mvtm_module
from src.evaluation.foldin.artifacts import (
    FoldInLayout,
    doc_topic_filename,
    expected_counts_filename,
    foldin_fingerprint,
    foldin_is_current,
    read_foldin_meta,
    run_collapsed_fold_in_chunked,
    token_corpus_fingerprint,
    write_foldin_artifacts,
)
from src.evaluation.foldin.mvtm import (
    compute_mvtm_foldin_for_run,
    load_token_corpus,
    mvtm_encoder_fingerprint,
    mvtm_foldin_fingerprint,
    normalized_word_rows,
    token_type_table,
)
from src.evaluation.foldin.runner import (
    iter_vmf_run_pointers,
    normalize_foldin_model,
    run_vmf_foldin_theta,
)
from src.evaluation.word_based.topic_assignment import CollapsedFoldInConfig
from src.models.vmf_encoding import VMFDocumentEncoder
from tests.evaluation.foldin.conftest import make_document

FAST = CollapsedFoldInConfig(burn_in_sweeps=1, retained_samples=2)
# Two topics on the plane: topic 0 concentrates on e1, topic 1 on e2.
TOY_ALPHA = [0.5, 0.5]
TOY_DOCS: list[list[str]] = [
    ["alpha alpha missing"],
    ["beta"],
    ["gamma alpha"],
    ["missing"],
]


def toy_vectors() -> KeyedVectors:
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        ["alpha", "beta", "gamma"],
        np.asarray([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32),
    )
    return vectors


def write_mvtm_run(
    archive_dir: Path,
    *,
    category: str = "cat",
    condition_fingerprint: str = "cond-fp",
    docs: list[list[str]] | None = None,
) -> Path:
    """A minimal MvTM run: metadata, frozen topics, both splits' corpora, doc-topic files."""

    documents = [make_document(item) for item in (docs or TOY_DOCS)]
    params_dir = archive_dir / "params"
    infer_dir = archive_dir / "infer"
    params_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "condition_fingerprint": condition_fingerprint,
            "baseline_params": {
                "word2vec": "toy-vectors",
                "wikientvec_cache_dir": None,
            },
        },
        archive_dir / "metadata.json",
    )
    save_json({"alpha": TOY_ALPHA, "num_topics": 2}, params_dir / "params.json")
    save_pickle(np.asarray([[1.0], [1.0]]), params_dir / "mixture_weights.pkl")
    save_pickle(
        np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]]), params_dir / "component_means.pkl"
    )
    save_pickle(np.asarray([8.0, 8.0]), params_dir / "kappa_per_topic.pkl")
    for split_dir in (params_dir, infer_dir):
        save_pickle(documents, split_dir / "preprocessed_corpus.pkl")
        save_json(
            {"raw_doc_indices": list(range(len(documents)))},
            split_dir / "preprocessing_selection.json",
        )
    save_pickle(
        np.full((len(documents), 2), 0.5), params_dir / "table_counts_per_doc.pkl"
    )
    save_pickle(np.zeros((len(documents), 2)), infer_dir / f"{category}.pkl")
    return archive_dir


def test_normalize_foldin_model_accepts_aliases() -> None:
    assert normalize_foldin_model("mvtm") == "mvtm"
    assert normalize_foldin_model("vLDA") == "mvtm"
    assert normalize_foldin_model("vmf") == "vmf_sentence_lda"
    with pytest.raises(ValueError, match="Unsupported fold-in model"):
        normalize_foldin_model("etm")


def test_mvtm_layout_names_the_split_by_directory() -> None:
    layout = FoldInLayout.for_model("mvtm", category="science")
    assert doc_topic_filename("train", layout) == "params/science_doc_topic_foldin.pkl"
    assert doc_topic_filename("test", layout) == "infer/science_doc_topic_foldin.pkl"
    assert (
        expected_counts_filename("test", layout)
        == "infer/science_doc_topic_foldin_counts.pkl"
    )
    # The vSLDA names are unchanged.
    assert doc_topic_filename("test") == "doc_topic_test_foldin.pkl"
    with pytest.raises(ValueError, match="category"):
        FoldInLayout.for_model("mvtm")


def test_token_type_table_drops_out_of_vocabulary_tokens_like_training() -> None:
    documents = [make_document(item) for item in TOY_DOCS]
    words, ids = token_type_table(documents, supported=toy_vectors().key_to_index)
    assert words == ["alpha", "beta", "gamma"]
    assert [item.tolist() for item in ids] == [[0, 0], [1], [2, 0], []]


def test_normalized_word_rows_match_the_trainer_encoding() -> None:
    """The unit vectors the fold-in scores are those the trainer observed."""

    from src.baselines.models.mvtm import WordVectorEncoder

    vectors = toy_vectors()
    encoder = VMFDocumentEncoder(
        encoder=WordVectorEncoder(vectors),
        embedding_size=2,
        pre_normalize_transform="none",
        whitening_eps=1e-6,
        log=None,
    )
    expected = encoder.encode_and_normalize(["alpha", "beta", "gamma"])
    rows = normalized_word_rows(vectors, ["alpha", "beta", "gamma"])
    assert rows.dtype == np.float64
    assert np.array_equal(rows, np.asarray(expected, dtype=np.float64))
    assert normalized_word_rows(vectors, []).shape == (0, 2)


def test_chunked_fold_in_is_deterministic_and_counts_every_token() -> None:
    table = np.asarray([[8.0, 0.0], [0.0, 8.0], [4.0, 4.0]])
    ids = [
        np.asarray([0, 0]),
        np.asarray([1]),
        np.asarray([2, 0]),
        np.asarray([], dtype=int),
    ]
    alpha = np.asarray(TOY_ALPHA)
    first = run_collapsed_fold_in_chunked(
        alpha=alpha,
        assignment_unit_type="token",
        config=FAST,
        log_likelihood_by_type=table,
        type_ids_by_doc=ids,
        chunk_docs=2,
    )
    second = run_collapsed_fold_in_chunked(
        alpha=alpha,
        assignment_unit_type="token",
        config=FAST,
        log_likelihood_by_type=table,
        type_ids_by_doc=ids,
        chunk_docs=2,
    )
    expected, num_units, metadata = first
    assert np.array_equal(expected, second[0])
    assert num_units.tolist() == [2, 1, 2, 0]
    assert np.allclose(expected.sum(axis=1), num_units)
    assert expected[0, 0] > expected[0, 1] and expected[1, 1] > expected[1, 0]
    assert metadata["chunk_docs"] == 2 and metadata["assignment_unit_type"] == "token"
    with pytest.raises(ValueError, match="chunk_docs"):
        run_collapsed_fold_in_chunked(
            alpha=alpha,
            assignment_unit_type="token",
            config=FAST,
            log_likelihood_by_type=table,
            type_ids_by_doc=ids,
            chunk_docs=0,
        )


def test_token_fingerprints_extend_the_sentence_one_only_for_token_units() -> None:
    base = dict(config=FAST, encoder_fp="enc", corpus_sha1="sha", condition_fp="cond")
    sentence = foldin_fingerprint(**base)
    assert foldin_fingerprint(**base, assignment_unit_type="sentence") == sentence
    token = foldin_fingerprint(**base, assignment_unit_type="token", chunk_docs=256)
    assert token != sentence
    assert (
        foldin_fingerprint(**base, assignment_unit_type="token", chunk_docs=64) != token
    )
    docs = [make_document(item) for item in TOY_DOCS]
    assert token_corpus_fingerprint(docs) == token_corpus_fingerprint(docs)
    assert token_corpus_fingerprint(docs) != token_corpus_fingerprint(docs[::-1])


def test_compute_mvtm_foldin_for_run_writes_the_baseline_layout(tmp_path: Path) -> None:
    run_dir = write_mvtm_run(tmp_path / "run")
    corpus = load_token_corpus(run_dir, split="test")
    assert corpus.num_documents == 4
    result = compute_mvtm_foldin_for_run(
        run_dir,
        split="test",
        dataset="dummy",
        data_run="default",
        category="cat",
        foldin_config=FAST,
        chunk_docs=3,
        corpus=corpus,
        vectors=toy_vectors(),
    )
    assert result.theta.shape == (4, 2)
    assert np.allclose(result.theta.sum(axis=1), 1.0)
    # E[n_dk] sums to the in-vocabulary token count; the empty document gets the prior.
    assert np.allclose(result.expected_counts.sum(axis=1), [2.0, 1.0, 2.0, 0.0])
    assert np.allclose(result.theta[3], [0.5, 0.5])
    assert result.theta[0, 0] > result.theta[0, 1]
    assert result.theta[1, 1] > result.theta[1, 0]
    assert result.metadata["model"] == "mvtm"
    assert result.metadata["assignment_unit_type"] == "token"
    assert result.metadata["num_covered_word_types"] == 3
    assert result.fingerprint == mvtm_foldin_fingerprint(
        run_dir, split="test", config=FAST, chunk_docs=3, corpus=corpus
    )
    assert result.fingerprint != mvtm_foldin_fingerprint(
        run_dir, split="test", config=FAST, chunk_docs=4, corpus=corpus
    )
    assert mvtm_encoder_fingerprint(run_dir) == mvtm_encoder_fingerprint(run_dir)

    layout = FoldInLayout.for_model("mvtm", category="cat")
    assert not foldin_is_current(
        run_dir, split="test", fingerprint=result.fingerprint, layout=layout
    )
    artifacts = write_foldin_artifacts(run_dir, result=result)
    assert artifacts == {
        "test_doc_topic_foldin": "infer/cat_doc_topic_foldin.pkl",
        "test_doc_topic_foldin_counts": "infer/cat_doc_topic_foldin_counts.pkl",
    }
    assert (run_dir / "infer" / "cat_doc_topic_foldin.pkl").exists()
    assert (run_dir / "infer" / "cat_doc_topic_foldin_counts.pkl").exists()
    meta = read_foldin_meta(run_dir)
    assert meta["model"] == "mvtm"
    assert meta["splits"]["test"]["fingerprint"] == result.fingerprint
    assert foldin_is_current(
        run_dir, split="test", fingerprint=result.fingerprint, layout=layout
    )
    # The vSLDA layout does not see the MvTM files.
    assert not foldin_is_current(run_dir, split="test", fingerprint=result.fingerprint)
    # Per-unit posteriors are not kept by the chunked path.
    with pytest.raises(ValueError, match="per-unit posteriors"):
        write_foldin_artifacts(run_dir, result=result, write_sentence_posteriors=True)

    # A corpus of another split or run is rejected.
    with pytest.raises(ValueError, match="corpus describes"):
        compute_mvtm_foldin_for_run(
            run_dir,
            split="train",
            dataset="dummy",
            data_run="default",
            category="cat",
            foldin_config=FAST,
            corpus=corpus,
            vectors=toy_vectors(),
        )


def _write_mvtm_pointer(results_root: Path, *, display_key: str) -> Path:
    archive_dir = (
        results_root
        / "dummy"
        / "default"
        / "mvtm"
        / "archive"
        / "2026-01-01"
        / "cat"
        / display_key
        / "baseline_exec"
    )
    write_mvtm_run(archive_dir)
    pointer_path = (
        results_root / "dummy" / "default" / "mvtm" / "latest" / "cat" / display_key
    ) / "CURRENT.json"
    save_json(
        {
            "schema": "latest_result_pointer",
            "schema_version": 1,
            "task": "baseline_mvtm",
            "display_key": display_key,
            "dataset": "dummy",
            "data_run": "default",
            "category": "cat",
            "archive_dir": str(archive_dir),
            "started_at": "2026-01-01T00:00:00+00:00",
            "execution_id": "baseline_exec",
            "condition_fingerprint": "cond-fp",
            "embedding_variant": "toy",
            "parameter_variant": None,
            "artifacts": {
                "train_path": "params/table_counts_per_doc.pkl",
                "infer_path": "infer/cat.pkl",
            },
        },
        pointer_path,
    )
    return pointer_path


def test_runner_handles_mvtm_runs_and_shares_the_word_vectors(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "baselines"
    first = _write_mvtm_pointer(root, display_key="k2_it0_c1_toy")
    second = _write_mvtm_pointer(root, display_key="k2_it1_c1_toy")
    loads: list[str] = []

    def _fake_load(condition_dir: Path):
        loads.append(str(condition_dir))
        return toy_vectors()

    monkeypatch.setattr(mvtm_module, "load_mvtm_word_vectors", _fake_load)

    pointers = iter_vmf_run_pointers(
        datasets=["dummy"], results_root=root, model="mvtm"
    )
    assert [item.display_key for item in pointers] == ["k2_it0_c1_toy", "k2_it1_c1_toy"]
    assert {item.model for item in pointers} == {"mvtm"}

    summary = run_vmf_foldin_theta(
        model="mvtm",
        datasets=["dummy"],
        splits=["test", "train"],
        foldin_config=FAST,
        chunk_docs=2,
        summary_path=tmp_path / "summary.csv",
        results_root=root,
    )
    # One word-vector load for both runs and splits.
    assert len(loads) == 1
    for pointer_path in (first, second):
        payload = load_json(pointer_path)
        archive_dir = Path(payload["archive_dir"])
        assert (archive_dir / "infer" / "cat_doc_topic_foldin.pkl").exists()
        assert (archive_dir / "infer" / "cat_doc_topic_foldin_counts.pkl").exists()
        assert (archive_dir / "params" / "cat_doc_topic_foldin.pkl").exists()
        assert (archive_dir / "params" / "cat_doc_topic_foldin_counts.pkl").exists()
        assert payload["artifacts"]["train_path"] == "params/table_counts_per_doc.pkl"
        assert (
            payload["artifacts"]["test_doc_topic_foldin_counts"]
            == "infer/cat_doc_topic_foldin_counts.pkl"
        )
        assert (
            payload["artifacts"]["train_doc_topic_foldin"]
            == "params/cat_doc_topic_foldin.pkl"
        )
        meta = load_json(archive_dir / "foldin_meta.json")
        assert meta["model"] == "mvtm"
        assert set(meta["splits"]) == {"train", "test"}
        assert meta["splits"]["train"]["posterior_metadata"]["chunk_docs"] == 2
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"computed"}
    assert {row["model"] for row in rows} == {"mvtm"}
    assert rows[0]["total_sentences"] == "5"

    # A second invocation with the same settings skips every run.
    run_vmf_foldin_theta(
        model="mvtm",
        datasets=["dummy"],
        splits=["test"],
        foldin_config=FAST,
        chunk_docs=2,
        summary_path=tmp_path / "summary2.csv",
        results_root=root,
    )
    with (tmp_path / "summary2.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["status"] for row in rows} == {"skipped"}
    assert len(loads) == 1
    # A different chunk size is a different fingerprint.
    run_vmf_foldin_theta(
        model="mvtm",
        datasets=["dummy"],
        splits=["test"],
        foldin_config=FAST,
        chunk_docs=3,
        summary_path=tmp_path / "summary3.csv",
        results_root=root,
    )
    with (tmp_path / "summary3.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["status"] for row in rows} == {"computed"}


def test_runner_default_summary_name_depends_on_the_model(
    tmp_path: Path, monkeypatch
) -> None:
    from src.evaluation.foldin import runner as runner_module

    monkeypatch.setattr(runner_module, "DEFAULT_SUMMARY_ROOT", tmp_path / "summaries")
    summary = run_vmf_foldin_theta(
        model="mvtm", datasets=["nothing"], results_root=tmp_path / "empty"
    )
    assert summary == tmp_path / "summaries" / "mvtm_foldin_theta.csv"
    assert json.loads("[]") == []

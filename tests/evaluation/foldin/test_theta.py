from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.foldin import theta as theta_module
from src.evaluation.foldin.theta import (
    FoldInRunResult,
    compute_foldin_for_run,
    foldin_is_current,
    foldin_theta_from_posterior,
    read_foldin_meta,
    write_foldin_artifacts,
)
from src.evaluation.topic_pairs.inputs import (
    SentenceAlignmentError,
    encoder_fingerprint,
    load_train_corpus,
)
from src.evaluation.word_based.topic_assignment import CollapsedFoldInConfig
from tests.evaluation.foldin.conftest import (
    ENCODER_CONFIG,
    TOY_ALPHA,
    toy_embeddings,
    toy_log_likelihoods,
    write_vmf_run,
)


def test_theta_is_the_dirichlet_smoothed_expected_counts() -> None:
    posterior = [
        np.asarray([[1.0, 0.0], [0.5, 0.5]]),  # E[n] = (1.5, 0.5), N = 2
        np.asarray([[0.2, 0.8]]),  # E[n] = (0.2, 0.8), N = 1
    ]
    alpha = np.asarray([0.5, 1.5])
    theta = foldin_theta_from_posterior(posterior, alpha)
    assert np.allclose(theta[0], [(1.5 + 0.5) / 4.0, (0.5 + 1.5) / 4.0])
    assert np.allclose(theta[1], [(0.2 + 0.5) / 3.0, (0.8 + 1.5) / 3.0])
    assert np.allclose(theta.sum(axis=1), 1.0)


def test_empty_document_gets_the_prior_mean() -> None:
    theta = foldin_theta_from_posterior([np.empty((0, 3))], np.asarray([1.0, 2.0, 1.0]))
    assert np.allclose(theta[0], [0.25, 0.5, 0.25])


def test_one_hot_posteriors_reduce_to_smoothed_counts() -> None:
    posterior = [np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])]
    alpha = np.asarray([0.1, 0.1])
    theta = foldin_theta_from_posterior(posterior, alpha)
    counts = np.asarray([2.0, 1.0])
    assert np.allclose(theta[0], (counts + alpha) / (3.0 + 0.2))


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        foldin_theta_from_posterior([np.zeros((1, 2))], np.asarray([1.0, -1.0]))
    with pytest.raises(ValueError):
        foldin_theta_from_posterior([np.zeros((1, 3))], np.asarray([1.0, 1.0]))


def _patch_model(monkeypatch) -> None:
    monkeypatch.setattr(
        theta_module, "vmf_sentence_log_likelihoods", toy_log_likelihoods
    )


def test_compute_foldin_for_run_is_deterministic_and_uses_the_document_prior(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = write_vmf_run(tmp_path / "run")
    _patch_model(monkeypatch)
    corpus = load_train_corpus(run_dir, model="vmf", split="test")
    embeddings = toy_embeddings(
        corpus,
        cache_dir=tmp_path / "cache",
        encoder_fp=encoder_fingerprint(ENCODER_CONFIG),
    )
    kwargs = dict(
        split="test",
        dataset="dummy",
        data_run="default",
        category="cat",
        cache_root=tmp_path / "cache",
        encoder_device="cpu",
        encode_batch_size=None,
        foldin_config=CollapsedFoldInConfig(burn_in_sweeps=2, retained_samples=3),
        corpus=corpus,
        reference_corpus=corpus,
        embeddings=embeddings,
    )
    first = compute_foldin_for_run(run_dir, **kwargs)
    second = compute_foldin_for_run(run_dir, **kwargs)

    assert first.theta.shape == (3, 2)
    assert np.array_equal(first.theta, second.theta)
    assert first.fingerprint == second.fingerprint
    assert np.allclose(first.theta.sum(axis=1), 1.0)
    # Document 0 (two e1 sentences) leans to topic 0, document 1 to topic 1,
    # and the empty document 2 is exactly the prior mean.
    assert first.theta[0, 0] > first.theta[0, 1]
    assert first.theta[1, 1] > first.theta[1, 0]
    assert np.allclose(first.theta[2], TOY_ALPHA / TOY_ALPHA.sum())
    assert first.num_empty_documents == 1
    assert first.total_sentences == 3
    assert first.metadata["config"]["burn_in_sweeps"] == 2
    assert first.metadata["condition_fingerprint"] == "cond-fp"
    assert np.allclose(first.expected_counts.sum(axis=1), [2.0, 1.0, 0.0])
    assert np.allclose(
        first.theta,
        (first.expected_counts + TOY_ALPHA)
        / (first.expected_counts.sum(axis=1, keepdims=True) + TOY_ALPHA.sum()),
    )
    # The unsmoothed estimator of the manuscript (``foldincounts``) is the
    # row-normalized expected count E[n_dk] / N_d; the loaders divide by the row
    # sum, so non-empty rows are proper distributions.
    totals = first.expected_counts.sum(axis=1, keepdims=True)
    proportions = first.expected_counts[:2] / totals[:2]
    assert np.allclose(proportions.sum(axis=1), 1.0)
    assert proportions[0, 0] > proportions[0, 1]
    assert proportions[1, 1] > proportions[1, 0]


def test_compute_foldin_for_run_rejects_foreign_embeddings(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = write_vmf_run(tmp_path / "run")
    _patch_model(monkeypatch)
    corpus = load_train_corpus(run_dir, model="vmf", split="test")
    embeddings = toy_embeddings(
        corpus, cache_dir=tmp_path / "cache", encoder_fp="other"
    )
    with pytest.raises(SentenceAlignmentError):
        compute_foldin_for_run(
            run_dir,
            split="test",
            dataset="dummy",
            data_run="default",
            category="cat",
            cache_root=tmp_path / "cache",
            encoder_device="cpu",
            encode_batch_size=None,
            corpus=corpus,
            embeddings=embeddings,
        )


def test_compute_foldin_for_run_checks_the_runs_own_document_count(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = write_vmf_run(tmp_path / "run")
    # A doc_topic_test.pkl with the wrong number of rows must be noticed.
    np.save  # keep numpy referenced for readers
    from src.core.artifacts import save_pickle

    save_pickle(np.zeros((5, 2)), run_dir / "doc_topic_test.pkl")
    _patch_model(monkeypatch)
    corpus = load_train_corpus(run_dir, model="vmf", split="test")
    embeddings = toy_embeddings(
        corpus,
        cache_dir=tmp_path / "cache",
        encoder_fp=encoder_fingerprint(ENCODER_CONFIG),
    )
    with pytest.raises(SentenceAlignmentError):
        compute_foldin_for_run(
            run_dir,
            split="test",
            dataset="dummy",
            data_run="default",
            category="cat",
            cache_root=tmp_path / "cache",
            encoder_device="cpu",
            encode_batch_size=None,
            corpus=corpus,
            embeddings=embeddings,
        )


def test_write_foldin_artifacts_and_currency(tmp_path: Path, monkeypatch) -> None:
    run_dir = write_vmf_run(tmp_path / "run")
    _patch_model(monkeypatch)
    corpus = load_train_corpus(run_dir, model="vmf", split="test")
    embeddings = toy_embeddings(
        corpus,
        cache_dir=tmp_path / "cache",
        encoder_fp=encoder_fingerprint(ENCODER_CONFIG),
    )
    result = compute_foldin_for_run(
        run_dir,
        split="test",
        dataset="dummy",
        data_run="default",
        category="cat",
        cache_root=tmp_path / "cache",
        encoder_device="cpu",
        encode_batch_size=None,
        corpus=corpus,
        embeddings=embeddings,
    )
    assert not foldin_is_current(run_dir, split="test", fingerprint=result.fingerprint)

    artifacts = write_foldin_artifacts(run_dir, result=result)
    assert artifacts == {
        "test_doc_topic_foldin": "doc_topic_test_foldin.pkl",
        "test_doc_topic_foldin_counts": "doc_topic_test_foldin_counts.pkl",
    }
    assert (run_dir / "doc_topic_test_foldin.pkl").exists()
    assert (run_dir / "doc_topic_test_foldin_counts.pkl").exists()
    assert not (run_dir / "sentence_topic_test_foldin.pkl").exists()
    meta = read_foldin_meta(run_dir)
    assert meta["splits"]["test"]["fingerprint"] == result.fingerprint
    assert meta["splits"]["test"]["num_documents"] == 3
    assert "created_at" in meta["splits"]["test"]
    assert foldin_is_current(run_dir, split="test", fingerprint=result.fingerprint)
    assert not foldin_is_current(run_dir, split="test", fingerprint="other")
    assert not foldin_is_current(run_dir, split="train", fingerprint=result.fingerprint)

    # A second split adds its own block without touching the first.
    train = FoldInRunResult(
        split="train",
        theta=result.theta,
        sentence_posteriors=result.sentence_posteriors,
        alpha=result.alpha,
        fingerprint="train-fp",
        metadata=dict(result.metadata, split="train"),
        timing={},
    )
    artifacts = write_foldin_artifacts(
        run_dir, result=train, write_sentence_posteriors=True
    )
    assert artifacts["train_sentence_topic_foldin"] == "sentence_topic_train_foldin.pkl"
    meta = read_foldin_meta(run_dir)
    assert set(meta["splits"]) == {"train", "test"}
    assert meta["splits"]["test"]["fingerprint"] == result.fingerprint

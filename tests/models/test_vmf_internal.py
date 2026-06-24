from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.core.artifacts import load_json, load_pickle
from src.models.vmf_artifacts import (
    build_vmf_model_artifact_payload,
    build_vmf_run_output_payload,
    save_vmf_model_artifacts,
    save_vmf_run_outputs,
)
from src.models.vmf_inference import VMFTopicInferencer
from src.models.vmf_numba import (
    ACCUMULATE_DOC_ASSIGNMENT_STATISTICS_BACKEND,
    ACCUMULATE_DOC_AVG_LL_BACKEND,
    SAMPLE_DOC_TOPIC_ASSIGNMENTS_BACKEND,
    _accumulate_doc_assignment_statistics_python,
    _accumulate_doc_average_log_likelihood_python,
    _sample_doc_topic_assignments_python,
    accumulate_doc_assignment_statistics,
    accumulate_doc_average_log_likelihood,
    sample_doc_topic_assignments,
)
from src.models.vmf_sentence_lda import VMFEmbeddingCacheReport, VMFLDATrainer
from src.utils.embedding_preprocess import EmbeddingPreprocessor
from src.utils.evaluation import calculate_avg_ll_vmf, calculate_avg_ll_vmf_from_encoded


class DummyEncoder:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, ...]] = []

    def encode(self, sentences) -> np.ndarray:
        batch = tuple(sentences)
        self.calls.append(batch)
        if not batch:
            return np.zeros((0, 2), dtype=np.float64)
        return np.vstack([self.mapping[text] for text in batch]).astype(np.float64)

    def get_sentence_embedding_dimension(self) -> int:
        return 2


def test_vmf_trainer_caches_training_encodings_across_iterations() -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c"]],
        encoder=encoder,
        num_topics=1,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-cache"),
    )

    trainer.sample(
        num_iterations=2,
        num_sweeps=1,
        num_samples=1,
        estimate_alpha=False,
    )

    assert encoder.calls == [("a", "b"), ("c",)]
    assert len(trainer.average_ll) == 2
    assert len(trainer.iteration_diagnostics) == 2
    assert trainer.iteration_diagnostics[0].iteration == 0
    assert trainer.iteration_diagnostics[0].alpha_updated is False
    assert trainer.iteration_diagnostics[0].e_step_sec >= 0.0
    assert trainer.iteration_diagnostics[0].m_step_sec >= 0.0
    assert trainer.iteration_diagnostics[0].alpha_update_sec >= 0.0
    assert trainer.iteration_diagnostics[0].avg_log_likelihood_sec >= 0.0
    assert trainer.iteration_diagnostics[0].iteration_elapsed_sec >= 0.0
    assert trainer.training_corpus_encoding_sec >= 0.0
    assert all(
        isinstance(doc_topics, np.ndarray) for doc_topics in trainer.topic_assignments
    )
    assert all(doc_topics.dtype == np.int32 for doc_topics in trainer.topic_assignments)
    assert trainer.encoded_corpus[0].dtype == np.float32
    assert trainer.topic_means.dtype == np.float32
    assert trainer.component_means.dtype == np.float32
    assert trainer.e_step_kernel_backend in {"python", "numba"}
    assert trainer.m_step_statistics_kernel_backend in {"python", "numba"}
    assert trainer.avg_ll_kernel_backend in {"python", "numba"}
    report = trainer.assert_valid_state()
    cache_report = trainer.build_embedding_cache_report()
    assert report.is_valid is True
    assert report.total_sentences == 3
    assert report.assigned_sentences == 3
    assert cache_report == VMFEmbeddingCacheReport(
        strategy="preencoded_training_corpus",
        num_documents=2,
        total_sentences=3,
        embedding_size=2,
        pre_normalize_transform="none",
        reused_for_training_iterations=True,
        reused_for_avg_log_likelihood=True,
    )


def test_vmf_sample_does_not_recompute_doc_topic_counts_each_sweep(
    monkeypatch,
) -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c"]],
        encoder=encoder,
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-no-recompute"),
    )

    def _fail_recompute() -> None:
        raise AssertionError("_recompute_doc_topic_counts should not be called")

    monkeypatch.setattr(trainer, "_recompute_doc_topic_counts", _fail_recompute)

    trainer.sample(
        num_iterations=1,
        num_sweeps=2,
        num_samples=1,
        estimate_alpha=False,
        repair_empty_topics=False,
    )

    assert trainer.assert_valid_state().is_valid is True


def test_vmf_sample_supports_periodic_ll_and_invariant_checks(
    monkeypatch,
) -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c"]],
        encoder=encoder,
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-periodic"),
    )

    calls = {"count": 0}
    original_assert_valid_state = trainer.assert_valid_state

    def _count_assert_valid_state():
        calls["count"] += 1
        return original_assert_valid_state()

    monkeypatch.setattr(trainer, "assert_valid_state", _count_assert_valid_state)

    trainer.sample(
        num_iterations=3,
        num_sweeps=1,
        num_samples=1,
        estimate_alpha=False,
        avg_log_likelihood_every=2,
        invariant_check_every=2,
    )

    assert len(trainer.average_ll) == 2
    assert trainer.iteration_diagnostics[0].avg_log_likelihood_sec == 0.0
    assert trainer.iteration_diagnostics[1].avg_log_likelihood_sec >= 0.0
    assert trainer.iteration_diagnostics[2].avg_log_likelihood_sec >= 0.0
    assert calls["count"] == 2


def test_vmf_sample_with_multiple_samples_keeps_state_invariant() -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
            "d": np.array([1.0, -1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c", "d"]],
        encoder=encoder,
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-multi-samples"),
    )

    trainer.sample(
        num_iterations=2,
        num_sweeps=3,
        num_samples=2,
        estimate_alpha=False,
    )

    assert trainer.assert_valid_state().is_valid is True


def test_vmf_doc_sampling_kernel_matches_python_reference() -> None:
    assignments_ref = np.array([0, 1, 0, 1], dtype=np.int32)
    assignments_kernel = assignments_ref.copy()
    counts_ref = np.array([2, 2], dtype=np.int32)
    counts_kernel = counts_ref.copy()
    log_lik_doc = np.array(
        [
            [0.2, -0.3],
            [0.4, 0.1],
            [-0.2, 0.7],
            [0.3, 0.5],
        ],
        dtype=np.float64,
    )
    alpha = np.array([1.0, 1.0], dtype=np.float64)
    uniforms = np.array([0.1, 0.8, 0.4, 0.7], dtype=np.float64)

    _sample_doc_topic_assignments_python(
        assignments=assignments_ref,
        counts=counts_ref,
        log_lik_doc=log_lik_doc,
        alpha=alpha,
        uniforms=uniforms,
    )
    sample_doc_topic_assignments(
        assignments=assignments_kernel,
        counts=counts_kernel,
        log_lik_doc=log_lik_doc,
        alpha=alpha,
        uniforms=uniforms,
    )

    assert SAMPLE_DOC_TOPIC_ASSIGNMENTS_BACKEND in {"python", "numba"}
    assert np.array_equal(assignments_kernel, assignments_ref)
    assert np.array_equal(counts_kernel, counts_ref)


def test_vmf_invariant_report_detects_broken_topic_counts() -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a"], ["b"]],
        encoder=encoder,
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-invariant"),
    )

    trainer.topic_counts[0] += 1

    report = trainer.build_invariant_report()

    assert report.topic_counts_match_assignments is False
    assert report.is_valid is False


def test_vmf_alpha_fixed_point_applies_configured_floor() -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c"]],
        encoder=encoder,
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-alpha-floor"),
    )

    trainer.alpha[:] = 1e-12
    trainer._update_alpha_fixed_point(min_alpha=1e-3)

    assert np.all(trainer.alpha >= 1e-3)


def test_vmf_empty_topic_detection_uses_doc_topic_counts() -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a"], ["b"]],
        encoder=encoder,
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-empty-detection"),
    )

    trainer.topic_counts_per_doc[:] = np.array([[1, 1], [0, 0]], dtype=np.int32)

    assert trainer._get_empty_topics(min_count=1).tolist() == [1]


def test_vmf_empty_topic_repair_reassigns_sentences_and_recomputes_statistics() -> None:
    np.random.seed(0)
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
            "d": np.array([1.0, -1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c", "d"]],
        encoder=encoder,
        num_topics=3,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-empty-repair"),
    )
    for assignments in trainer.topic_assignments:
        assignments[:] = 0
    trainer._recompute_doc_topic_counts()

    report = trainer._repair_empty_topics(min_topic_count_for_repair=1)
    nk, nk_comp, r = trainer._compute_assignment_statistics(
        encoded_docs=trainer.encoded_corpus,
    )

    assert report["num_repaired"] == 2
    assert report["num_failed"] == 0
    assert np.count_nonzero(trainer.topic_counts) == trainer.num_topics
    assert np.allclose(nk, trainer.topic_counts.astype(float))
    assert np.allclose(nk_comp.sum(axis=1), trainer.topic_counts.astype(float))
    assert r.shape == (
        trainer.num_topics,
        trainer.num_components,
        trainer.embedding_size,
    )
    assert trainer.assert_valid_state().is_valid is True


def test_vmf_sample_records_empty_topic_repair_diagnostics() -> None:
    np.random.seed(0)
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c"]],
        encoder=encoder,
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-repair-diagnostics"),
    )

    trainer.sample(
        num_iterations=1,
        num_sweeps=1,
        num_samples=1,
        estimate_alpha=True,
        alpha_min_value=1e-3,
        repair_empty_topics=True,
        min_topic_count_for_repair=1,
    )

    diagnostic = trainer.iteration_diagnostics[0]
    assert diagnostic.repair_enabled is True
    assert diagnostic.repair_sec >= 0.0
    assert diagnostic.alpha_min >= 1e-3
    assert diagnostic.alpha_floor_count >= 0
    assert isinstance(diagnostic.empty_topics, list)
    assert trainer.assert_valid_state().is_valid is True


def test_accumulate_doc_assignment_statistics_kernel_matches_python() -> None:
    encoded_doc = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
        dtype=np.float32,
    )
    assignments = np.array([0, 1, 0], dtype=np.int32)
    mixture_weights = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    component_means = np.array(
        [
            [[1.2, 0.1], [0.8, 0.4]],
            [[0.2, 1.1], [0.5, 0.9]],
        ],
        dtype=np.float64,
    )

    nk_ref = np.zeros(2, dtype=np.float64)
    nk_comp_ref = np.zeros((2, 2), dtype=np.float64)
    r_ref = np.zeros((2, 2, 2), dtype=np.float64)
    _accumulate_doc_assignment_statistics_python(
        encoded_doc=encoded_doc,
        assignments=assignments,
        log_mixture_weights=np.log(mixture_weights),
        scaled_component_means=component_means,
        nk=nk_ref,
        nk_comp=nk_comp_ref,
        r=r_ref,
    )

    nk_kernel = np.zeros(2, dtype=np.float64)
    nk_comp_kernel = np.zeros((2, 2), dtype=np.float64)
    r_kernel = np.zeros((2, 2, 2), dtype=np.float64)
    accumulate_doc_assignment_statistics(
        encoded_doc=encoded_doc,
        assignments=assignments,
        log_mixture_weights=np.log(mixture_weights),
        scaled_component_means=component_means,
        nk=nk_kernel,
        nk_comp=nk_comp_kernel,
        r=r_kernel,
    )

    assert ACCUMULATE_DOC_ASSIGNMENT_STATISTICS_BACKEND in {"python", "numba"}
    assert np.allclose(nk_kernel, nk_ref)
    assert np.allclose(nk_comp_kernel, nk_comp_ref)
    assert np.allclose(r_kernel, r_ref)


def test_accumulate_doc_average_log_likelihood_kernel_matches_python() -> None:
    encoded_doc = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
        dtype=np.float32,
    )
    assignments = np.array([0, -1, 1], dtype=np.int32)
    log_c_per_topic = np.array([-1.5, -0.7], dtype=np.float64)
    scaled_topic_means = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)

    ll_ref, count_ref = _accumulate_doc_average_log_likelihood_python(
        encoded_doc=encoded_doc,
        assignments=assignments,
        log_c_per_topic=log_c_per_topic,
        scaled_topic_means=scaled_topic_means,
    )
    ll_kernel, count_kernel = accumulate_doc_average_log_likelihood(
        encoded_doc=encoded_doc,
        assignments=assignments,
        log_c_per_topic=log_c_per_topic,
        scaled_topic_means=scaled_topic_means,
    )

    assert ACCUMULATE_DOC_AVG_LL_BACKEND in {"python", "numba"}
    assert np.isclose(ll_kernel, ll_ref)
    assert count_kernel == count_ref


def test_calculate_avg_ll_from_encoded_matches_encoder_version() -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }
    )
    corpus = [["a"], ["b"]]
    encoded_corpus = [
        np.array([[1.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.0]], dtype=np.float64),
    ]
    topic_assignments = [[0], [1]]
    topic_means = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    kappa_per_topic = np.array([2.0, 3.0], dtype=np.float64)

    ll_from_encoder = calculate_avg_ll_vmf(
        corpus=corpus,
        topic_assignments=topic_assignments,
        encoder=encoder,
        topic_means=topic_means,
        kappa_per_topic=kappa_per_topic,
    )
    ll_from_encoded = calculate_avg_ll_vmf_from_encoded(
        encoded_corpus=encoded_corpus,
        topic_assignments=topic_assignments,
        topic_means=topic_means,
        kappa_per_topic=kappa_per_topic,
    )

    assert np.isclose(ll_from_encoded, ll_from_encoder)


def test_vmf_density_matrix_matches_rowwise_density() -> None:
    encoder = DummyEncoder(
        {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
        }
    )
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c"]],
        encoder=encoder,
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-density-matrix"),
    )

    enc = trainer.encoded_corpus[0]
    matrix_scores = trainer.log_vmf_density_matrix(enc)
    rowwise_scores = np.vstack([trainer.log_vmf_density_tables(x) for x in enc])

    assert np.allclose(matrix_scores, rowwise_scores)
    assert isinstance(trainer.topic_assignments[0], np.ndarray)


def test_vmf_topic_inferencer_reuses_shared_softmax_logic() -> None:
    inferencer = VMFTopicInferencer(
        num_topics=2,
        embedding_size=2,
        encode_document=lambda doc: np.asarray(doc, dtype=np.float64),
        log_vmf_density=lambda x: np.asarray([x[0], x[1]], dtype=np.float64),
    )

    corpus = [
        [[3.0, 0.0], [0.0, 2.0]],
        [[1.0, 1.0]],
    ]

    counts = inferencer.infer_document_topic_counts(corpus)
    sentence_posteriors = inferencer.infer_sentence_topic_distribution_soft(corpus)
    doc_posteriors = inferencer.infer_document_topic_distribution_soft(corpus)

    assert counts.tolist() == [[1, 1], [1, 0]]
    assert sentence_posteriors[0].shape == (2, 2)
    assert np.allclose(doc_posteriors.sum(axis=1), np.array([1.0, 1.0]))


def test_vmf_topic_inferencer_prefers_batch_log_likelihood_when_available() -> None:
    batch_calls: list[tuple[int, int]] = []
    row_calls: list[np.ndarray] = []

    def _row_log_lik(x: np.ndarray) -> np.ndarray:
        row_calls.append(np.asarray(x, dtype=np.float64))
        return np.asarray([x[0], x[1]], dtype=np.float64)

    def _batch_log_lik(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        batch_calls.append(arr.shape)
        return arr[:, :2]

    inferencer = VMFTopicInferencer(
        num_topics=2,
        embedding_size=2,
        encode_document=lambda doc: np.asarray(doc, dtype=np.float64),
        log_vmf_density=_row_log_lik,
        log_vmf_density_batch=_batch_log_lik,
    )

    corpus = [
        [[3.0, 0.0], [0.0, 2.0]],
        [[1.0, 1.0]],
    ]

    counts = inferencer.infer_document_topic_counts(corpus)
    sentence_posteriors = inferencer.infer_sentence_topic_distribution_soft(corpus)

    assert counts.tolist() == [[1, 1], [1, 0]]
    assert len(batch_calls) == 4
    assert row_calls == []
    assert sentence_posteriors[0].shape == (2, 2)


def test_vmf_topic_inferencer_combines_counts_and_soft_outputs_in_single_pass() -> None:
    batch_calls: list[tuple[int, int]] = []

    def _batch_log_lik(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        batch_calls.append(arr.shape)
        return arr[:, :2]

    inferencer = VMFTopicInferencer(
        num_topics=2,
        embedding_size=2,
        encode_document=lambda doc: np.asarray(doc, dtype=np.float64),
        log_vmf_density=lambda x: np.asarray([x[0], x[1]], dtype=np.float64),
        log_vmf_density_batch=_batch_log_lik,
    )

    corpus = [
        [[3.0, 0.0], [0.0, 2.0]],
        [[1.0, 1.0]],
    ]

    outputs = inferencer.infer_corpus_topic_outputs(
        corpus,
        include_counts=True,
        include_sentence_posteriors=True,
        include_document_posteriors=True,
    )

    assert outputs.counts is not None
    assert outputs.sentence_posteriors is not None
    assert outputs.document_posteriors is not None
    assert outputs.counts.tolist() == [[1, 1], [1, 0]]
    assert len(batch_calls) == 2
    assert outputs.sentence_posteriors[0].shape == (2, 2)
    assert np.allclose(outputs.document_posteriors.sum(axis=1), np.array([1.0, 1.0]))


def test_vmf_topic_inferencer_can_reuse_preencoded_corpus() -> None:
    encode_calls: list[object] = []

    def _encode_document(doc: object) -> np.ndarray:
        encode_calls.append(doc)
        return np.asarray(doc, dtype=np.float64)

    inferencer = VMFTopicInferencer(
        num_topics=2,
        embedding_size=2,
        encode_document=_encode_document,
        log_vmf_density=lambda x: np.asarray([x[0], x[1]], dtype=np.float64),
    )

    encoded_corpus = [
        np.asarray([[3.0, 0.0], [0.0, 2.0]], dtype=np.float64),
        np.asarray([[1.0, 1.0]], dtype=np.float64),
    ]

    outputs = inferencer.infer_encoded_corpus_topic_outputs(
        encoded_corpus,
        include_counts=True,
        include_sentence_posteriors=True,
        include_document_posteriors=True,
    )

    assert encode_calls == []
    assert outputs.counts is not None
    assert outputs.sentence_posteriors is not None
    assert outputs.document_posteriors is not None
    assert outputs.counts.tolist() == [[1, 1], [1, 0]]
    assert outputs.sentence_posteriors[0].shape == (2, 2)
    assert np.allclose(outputs.document_posteriors.sum(axis=1), np.array([1.0, 1.0]))


def test_vmf_artifact_payload_saves_params_and_arrays(tmp_path: Path) -> None:
    preprocessor = EmbeddingPreprocessor(mode="mean_center", whitening_eps=1e-5)
    preprocessor.mean_ = np.array([0.1, -0.1], dtype=np.float64)
    preprocessor._fitted = True

    payload = build_vmf_model_artifact_payload(
        average_ll=[-1.2],
        iteration_diagnostics=[
            {
                "iteration": 0,
                "num_sweeps": 1,
                "num_samples": 1,
                "active_topics": 2,
                "alpha_updated": True,
                "alpha_converged": True,
                "avg_log_likelihood": -1.2,
            }
        ],
        embedding_cache={
            "strategy": "preencoded_training_corpus",
            "num_documents": 2,
            "total_sentences": 3,
            "embedding_size": 2,
            "pre_normalize_transform": "mean_center",
            "reused_for_training_iterations": True,
            "reused_for_avg_log_likelihood": True,
        },
        alpha=np.array([0.5, 0.5], dtype=np.float64),
        num_topics=2,
        kappa_default=10.0,
        num_components=1,
        pre_normalize_transform="mean_center",
        whitening_eps=1e-5,
        algorithm_variant="components_1__estimate_alpha_every_1",
        topic_counts=np.array([2, 1], dtype=np.int32),
        topic_counts_per_doc=np.array([[1, 1], [1, 0]], dtype=np.int32),
        topic_means=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        sum_topic_vectors=np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        kappa_per_topic=np.array([2.0, 3.0], dtype=np.float64),
        mixture_weights=np.array([[1.0], [1.0]], dtype=np.float64),
        component_means=np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float64),
        embedding_preprocessor=preprocessor,
    )

    saved = save_vmf_model_artifacts(payload, tmp_path)

    assert saved["params"] == tmp_path / "params.json"
    assert load_json(saved["params"])["pre_normalize_transform"] == "mean_center"
    assert (
        load_json(saved["params"])["embedding_cache"]["strategy"]
        == "preencoded_training_corpus"
    )
    assert (
        load_json(saved["params"])["algorithm_variant"]
        == "components_1__estimate_alpha_every_1"
    )
    assert np.allclose(
        load_pickle(saved["embedding_transform_mean"]),
        np.array([0.1, -0.1], dtype=np.float64),
    )


def test_vmf_run_output_payload_saves_pickles_and_metrics(tmp_path: Path) -> None:
    payload = build_vmf_run_output_payload(
        theta_train=np.array([[0.7, 0.3]], dtype=np.float64),
        theta_test=np.array([[0.2, 0.8]], dtype=np.float64),
        theta_train_soft=np.array([[0.6, 0.4]], dtype=np.float64),
        theta_test_soft=np.array([[0.1, 0.9]], dtype=np.float64),
        sentence_topic_train_soft=[np.array([[0.6, 0.4]], dtype=np.float64)],
        sentence_topic_test_soft=[np.array([[0.1, 0.9]], dtype=np.float64)],
        counts_train=np.array([[3, 1]], dtype=np.int32),
        metrics={
            "category": "all",
            "num_topics": 2,
            "algorithm_variant": "components_1__fixed_alpha",
        },
        embedding_cache={
            "strategy": "preencoded_training_corpus",
            "num_documents": 1,
            "total_sentences": 1,
            "embedding_size": 2,
            "pre_normalize_transform": "none",
            "reused_for_training_iterations": True,
            "reused_for_avg_log_likelihood": True,
        },
    )

    saved = save_vmf_run_outputs(payload, tmp_path)

    assert saved["doc_topic_train"] == tmp_path / "doc_topic_train.pkl"
    assert saved["doc_topic_test"] == tmp_path / "doc_topic_test.pkl"
    assert saved["metrics_path"] == tmp_path / "metrics.json"
    assert (
        load_json(saved["metrics_path"])["algorithm_variant"]
        == "components_1__fixed_alpha"
    )
    assert load_json(saved["metrics_path"])["embedding_cache"]["total_sentences"] == 1
    assert np.allclose(
        load_pickle(saved["doc_topic_train"]),
        np.array([[0.7, 0.3]], dtype=np.float64),
    )
    assert np.array_equal(
        load_pickle(saved["table_counts_per_doc"]),
        np.array([[3, 1]], dtype=np.int32),
    )

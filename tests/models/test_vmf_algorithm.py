from __future__ import annotations

import logging
import math

import numpy as np
from scipy.special import digamma, ive, logsumexp

from src.core.progress import NullProgressReporter
from src.models.vmf_numba import sample_doc_topic_assignments
from src.models.vmf_sentence_lda import VMFLDATrainer


class _FixedEncoder:
    def __init__(self, mapping: dict[str, np.ndarray], dim: int) -> None:
        self._mapping = mapping
        self._dim = dim

    def encode(self, sentences) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self._dim), dtype=np.float64)
        return np.vstack([self._mapping[item] for item in sentences]).astype(np.float64)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def _normalize(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _manual_log_vmf_const(kappa: np.ndarray, dim: int) -> np.ndarray:
    kappa_arr = np.asarray(kappa, dtype=np.float64)
    kappa_safe = np.clip(kappa_arr, 1e-12, None)
    order = float(dim) / 2.0 - 1.0
    ive_value = np.maximum(ive(order, kappa_safe), 1e-300)
    log_iv = np.log(ive_value) + kappa_safe
    return (
        order * np.log(kappa_safe)
        - (float(dim) / 2.0) * math.log(2.0 * math.pi)
        - log_iv
    )


def _manual_sample_doc_topics(
    assignments: np.ndarray,
    counts: np.ndarray,
    log_lik_doc: np.ndarray,
    alpha: np.ndarray,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected_assignments = np.asarray(assignments, dtype=np.int32).copy()
    expected_counts = np.asarray(counts, dtype=np.int32).copy()
    num_topics = int(expected_counts.shape[0])

    for sentence_index in range(expected_assignments.shape[0]):
        old_topic = int(expected_assignments[sentence_index])
        if 0 <= old_topic < num_topics:
            expected_counts[old_topic] -= 1

        weights = (expected_counts.astype(np.float64) + alpha) * np.exp(
            log_lik_doc[sentence_index]
        )
        total = float(weights.sum())
        threshold = float(uniforms[sentence_index]) * total
        cumulative = 0.0
        new_topic = num_topics - 1
        for topic_index, weight in enumerate(weights):
            cumulative += float(weight)
            if threshold <= cumulative:
                new_topic = int(topic_index)
                break

        expected_assignments[sentence_index] = new_topic
        expected_counts[new_topic] += 1

    return expected_assignments, expected_counts


def _make_trainer(
    *,
    num_topics: int,
    num_components: int,
    dim: int = 3,
) -> VMFLDATrainer:
    mapping = {
        "a": _normalize(np.array([1.0, 0.0, 0.0])),
        "b": _normalize(np.array([0.0, 1.0, 0.0])),
        "c": _normalize(np.array([0.0, 0.0, 1.0])),
    }
    return VMFLDATrainer(
        corpus=[["a"], ["b"], ["c"]],
        encoder=_FixedEncoder(mapping, dim=dim),
        num_topics=num_topics,
        alpha=1.0,
        kappa=2.0,
        num_components=num_components,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-algorithm"),
        progress=NullProgressReporter(),
    )


def test_doc_topic_sampling_matches_hand_computed_posterior() -> None:
    assignments = np.array([0, 1, 2], dtype=np.int32)
    counts = np.array([1, 1, 1], dtype=np.int32)
    log_lik_doc = np.array(
        [
            [0.0, np.log(2.0), np.log(4.0)],
            [np.log(3.0), 0.0, np.log(1.5)],
            [np.log(0.5), np.log(2.5), 0.0],
        ],
        dtype=np.float64,
    )
    alpha = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    uniforms = np.array([0.1, 0.6, 0.9], dtype=np.float64)
    expected_assignments, expected_counts = _manual_sample_doc_topics(
        assignments=assignments,
        counts=counts,
        log_lik_doc=log_lik_doc,
        alpha=alpha,
        uniforms=uniforms,
    )

    actual_assignments = assignments.copy()
    actual_counts = counts.copy()
    sample_doc_topic_assignments(
        assignments=actual_assignments,
        counts=actual_counts,
        log_lik_doc=log_lik_doc,
        alpha=alpha,
        uniforms=uniforms,
    )

    assert np.array_equal(actual_assignments, expected_assignments)
    assert np.array_equal(actual_counts, expected_counts)
    assert int(actual_counts.sum()) == int(assignments.shape[0])
    assert np.all((actual_assignments >= 0) & (actual_assignments < 3))


def test_single_vmf_density_matches_hand_computed_formula() -> None:
    trainer = _make_trainer(num_topics=2, num_components=1)
    trainer.topic_means[:] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    trainer.component_means[:, 0, :] = trainer.topic_means
    trainer.mixture_weights[:] = 1.0
    trainer.kappa_per_topic[:] = np.array([2.0, 4.0], dtype=np.float64)
    trainer._refresh_density_caches()

    x = _normalize(np.array([1.0, 1.0, 0.0]))
    expected = _manual_log_vmf_const(trainer.kappa_per_topic, dim=3) + (
        trainer.kappa_per_topic * (trainer.topic_means.astype(np.float64) @ x)
    )

    row_scores = trainer.log_vmf_density_tables(x)
    batch = np.vstack([x, _normalize(np.array([1.0, 0.0, 1.0]))])
    expected_batch = np.vstack(
        [
            _manual_log_vmf_const(trainer.kappa_per_topic, dim=3)
            + trainer.kappa_per_topic * (trainer.topic_means.astype(np.float64) @ row)
            for row in batch
        ]
    )

    assert np.allclose(row_scores, expected)
    assert np.allclose(trainer.log_vmf_density_matrix(batch), expected_batch)
    assert trainer.log_vmf_density_matrix(batch).shape == (2, 2)


def test_mixture_vmf_density_matches_hand_computed_logsumexp() -> None:
    trainer = _make_trainer(num_topics=2, num_components=2)
    trainer.mixture_weights[:] = np.array(
        [
            [0.25, 0.75],
            [0.60, 0.40],
        ],
        dtype=np.float64,
    )
    trainer.component_means[:] = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    trainer.kappa_per_topic[:] = np.array([3.0, 5.0], dtype=np.float64)
    trainer._refresh_density_caches()

    x = _normalize(np.array([1.0, 2.0, 0.5]))
    expected = []
    for topic_index in range(trainer.num_topics):
        component_scores = np.log(trainer.mixture_weights[topic_index]) + (
            trainer.kappa_per_topic[topic_index]
            * (trainer.component_means[topic_index].astype(np.float64) @ x)
        )
        expected.append(
            _manual_log_vmf_const(trainer.kappa_per_topic[[topic_index]], dim=3)[0]
            + logsumexp(component_scores)
        )
    expected_array = np.asarray(expected, dtype=np.float64)

    batch = np.vstack([x, _normalize(np.array([0.2, 0.1, 1.0]))])
    expected_batch = np.vstack(
        [
            np.asarray(
                [
                    _manual_log_vmf_const(
                        trainer.kappa_per_topic[[topic_index]],
                        dim=3,
                    )[0]
                    + logsumexp(
                        np.log(trainer.mixture_weights[topic_index])
                        + trainer.kappa_per_topic[topic_index]
                        * (
                            trainer.component_means[topic_index].astype(np.float64)
                            @ row
                        )
                    )
                    for topic_index in range(trainer.num_topics)
                ],
                dtype=np.float64,
            )
            for row in batch
        ]
    )

    assert np.allclose(trainer.log_vmf_density_tables(x), expected_array)
    assert np.allclose(trainer.log_vmf_density_matrix(batch), expected_batch)
    assert np.all(np.isfinite(trainer.log_vmf_density_matrix(batch)))


def test_component_responsibilities_match_softmax_over_components() -> None:
    trainer = _make_trainer(num_topics=2, num_components=2)
    trainer.mixture_weights[:] = np.array(
        [
            [0.25, 0.75],
            [0.60, 0.40],
        ],
        dtype=np.float64,
    )
    trainer.component_means[:] = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    trainer.kappa_per_topic[:] = np.array([3.0, 5.0], dtype=np.float64)
    trainer._refresh_density_caches()

    x = _normalize(np.array([1.0, 2.0, 0.5]))
    topic_index = 0
    expected = _softmax(
        np.log(trainer.mixture_weights[topic_index])
        + trainer.kappa_per_topic[topic_index]
        * (trainer.component_means[topic_index].astype(np.float64) @ x)
    )

    actual = trainer._component_responsibilities(topic_index, x)

    assert np.allclose(actual, expected)
    assert np.isclose(actual.sum(), 1.0)

    single_component = _make_trainer(num_topics=2, num_components=1)
    assert np.array_equal(
        single_component._component_responsibilities(0, x),
        np.array([1.0], dtype=np.float64),
    )


def test_alpha_fixed_point_update_matches_one_manual_iteration() -> None:
    trainer = _make_trainer(num_topics=2, num_components=1)
    counts = np.array(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 4.0],
        ],
        dtype=np.float64,
    )
    initial_alpha = np.array([0.8, 1.2], dtype=np.float64)
    min_alpha = 1e-3
    trainer.topic_counts_per_doc[:] = counts.astype(np.int32)
    trainer.alpha[:] = initial_alpha

    doc_lengths = counts.sum(axis=0)
    alpha0 = float(initial_alpha.sum())
    denom = np.sum(digamma(doc_lengths + alpha0) - digamma(alpha0))
    numer = np.sum(
        digamma(counts + initial_alpha[:, None]) - digamma(initial_alpha)[:, None],
        axis=1,
    )
    expected = np.clip(initial_alpha * (numer / denom), min_alpha, None)

    converged = trainer._update_alpha_fixed_point(
        max_iter=1,
        tol=0.0,
        min_alpha=min_alpha,
    )

    assert converged is False
    assert np.allclose(trainer.alpha, expected)
    assert np.all(np.isfinite(trainer.alpha))
    assert np.all(trainer.alpha >= min_alpha)


def test_m_step_updates_parameters_from_fixed_sufficient_statistics() -> None:
    trainer = _make_trainer(num_topics=2, num_components=2)
    nk = np.array([4.0, 3.0], dtype=np.float64)
    nk_comp = np.array(
        [
            [1.0, 3.0],
            [1.2, 1.8],
        ],
        dtype=np.float64,
    )
    r = np.array(
        [
            [[0.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
            [[0.0, 0.6, 0.0], [0.0, 0.0, 0.9]],
        ],
        dtype=np.float64,
    )
    expected_weights = nk_comp / nk[:, None]
    expected_weights = expected_weights / expected_weights.sum(axis=1, keepdims=True)
    expected_component_means = np.zeros_like(r)
    expected_kappa = np.zeros(trainer.num_topics, dtype=np.float64)
    expected_topic_means = np.zeros((trainer.num_topics, trainer.embedding_size))
    for topic_index in range(trainer.num_topics):
        lengths = []
        for component_index in range(trainer.num_components):
            norm = np.linalg.norm(r[topic_index, component_index])
            expected_component_means[topic_index, component_index] = (
                r[topic_index, component_index] / norm
            )
            lengths.append(float(norm))
        r_k = float(np.clip(sum(lengths) / nk[topic_index], 1e-6, 1.0 - 1e-6))
        expected_kappa[topic_index] = (r_k * 3.0 - r_k**3) / (1.0 - r_k**2)
        effective_mean = (
            expected_weights[topic_index, :, None]
            * expected_component_means[topic_index]
        ).sum(axis=0)
        expected_topic_means[topic_index] = effective_mean / np.linalg.norm(
            effective_mean
        )

    trainer._apply_m_step_updates(nk=nk, nk_comp=nk_comp, r=r)

    assert np.allclose(trainer.mixture_weights, expected_weights)
    assert np.allclose(trainer.component_means, expected_component_means)
    assert np.allclose(trainer.topic_means, expected_topic_means)
    assert np.allclose(trainer.kappa_per_topic, expected_kappa)
    assert np.allclose(trainer.sum_topic_vectors, r.sum(axis=1))


def test_m_step_reinitializes_empty_topic_with_valid_parameters() -> None:
    np.random.seed(0)
    trainer = _make_trainer(num_topics=2, num_components=2)
    nk = np.array([3.0, 0.0], dtype=np.float64)
    nk_comp = np.array(
        [
            [1.0, 2.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    r = np.array(
        [
            [[0.3, 0.0, 0.0], [0.0, 1.2, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    trainer._apply_m_step_updates(nk=nk, nk_comp=nk_comp, r=r)

    assert np.allclose(trainer.mixture_weights[0], np.array([1.0 / 3.0, 2.0 / 3.0]))
    assert np.isclose(trainer.mixture_weights[1].sum(), 1.0)
    assert np.all(trainer.mixture_weights[1] >= 0.0)
    assert np.allclose(
        np.linalg.norm(trainer.component_means[1], axis=1),
        np.ones(trainer.num_components),
        atol=2e-6,
    )
    assert trainer.kappa_per_topic[1] == trainer.kappa_default
    assert np.all(np.isfinite(trainer.topic_means[1]))


def test_mixture_vmf_sentence_lda_small_golden_output() -> None:
    np.random.seed(0)
    raw_vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.9, 0.1, 0.0],
        "c": [0.0, 1.0, 0.0],
        "d": [0.0, 0.9, 0.1],
        "e": [0.0, 0.0, 1.0],
        "f": [0.1, 0.0, 0.9],
    }
    mapping = {
        key: _normalize(np.asarray(value, dtype=np.float64))
        for key, value in raw_vectors.items()
    }
    trainer = VMFLDATrainer(
        corpus=[["a", "b"], ["c", "d"], ["e", "f"]],
        encoder=_FixedEncoder(mapping, dim=3),
        num_topics=2,
        alpha=1.0,
        kappa=1.0,
        num_components=2,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-mixture-golden"),
        progress=NullProgressReporter(),
    )

    trainer.sample(
        num_iterations=2,
        num_sweeps=2,
        num_samples=1,
        estimate_alpha=False,
        repair_empty_topics=True,
    )

    assert trainer.assert_valid_state().is_valid is True
    assert [item.tolist() for item in trainer.topic_assignments] == [
        [0, 0],
        [0, 0],
        [1, 1],
    ]
    assert trainer.topic_counts.tolist() == [4, 2]
    assert trainer.topic_counts_per_doc.T.tolist() == [[2, 0], [2, 0], [0, 2]]
    assert np.allclose(
        trainer.mixture_weights,
        np.array(
            [
                [0.50072324, 0.49927676],
                [0.49003375, 0.50996625],
            ],
            dtype=np.float64,
        ),
        atol=1e-8,
    )
    assert np.allclose(
        trainer.kappa_per_topic,
        np.array([3.80281675, 653.99039635], dtype=np.float64),
        atol=1e-8,
    )
    assert np.allclose(
        trainer.average_ll,
        np.array([-1.8036809093475255, 0.18826899505806352], dtype=np.float64),
        atol=1e-12,
    )
    assert np.allclose(trainer.mixture_weights.sum(axis=1), np.ones(2))
    assert np.allclose(
        np.linalg.norm(trainer.component_means, axis=2),
        np.ones((2, 2)),
        atol=1e-6,
    )
    assert np.all(np.isfinite(trainer.average_ll))
    assert any(
        not np.allclose(row, np.full(trainer.num_components, 0.5))
        for row in trainer.mixture_weights
    )

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.word_based.topic_word_model_adapters import (
    lda_log_likelihood_by_type,
    normalize_ctm_decoder_scores,
    restrict_scores_to_evaluation_vocabulary,
    sample_etm_theta,
    sentlda_log_likelihood_by_doc,
    vmf_mixture_log_likelihood,
)


def test_lda_adapter_is_log_phi_transposed() -> None:
    phi = np.array([[0.75, 0.25], [0.1, 0.9]])
    np.testing.assert_allclose(lda_log_likelihood_by_type(phi), np.log(phi).T)


def test_vmf_adapter_includes_weights_and_normalization() -> None:
    observations = np.array([[1.0, 0.0]])
    means = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    equal = vmf_mixture_log_likelihood(
        observations,
        mixture_weights=np.ones((2, 1)),
        component_means=means,
        kappa_per_topic=np.array([2.0, 2.0]),
    )
    assert equal[0, 0] - equal[0, 1] == pytest.approx(2.0)
    unequal = vmf_mixture_log_likelihood(
        observations,
        mixture_weights=np.ones((2, 1)),
        component_means=means,
        kappa_per_topic=np.array([2.0, 20.0]),
    )
    # The normalization constant makes this differ from a bare dot product.
    assert unequal[0, 0] - unequal[0, 1] != pytest.approx(2.0)


def test_vmf_normalizer_matches_training_clamp_for_small_kappa() -> None:
    from src.evaluation.word_based.topic_word_model_adapters import (
        _log_vmf_normalizer,
    )

    dimension = 768
    kappa = np.array([0.0, 10.0, 500.0])
    result = _log_vmf_normalizer(kappa, dimension)
    assert np.all(np.isfinite(result))

    # Reference: VmfSentenceLda._log_vmf_normalization_const clamping rules.
    from scipy.special import ive

    kappa_safe = np.clip(kappa, 1e-12, None)
    order = dimension / 2.0 - 1.0
    log_bessel = np.log(np.maximum(ive(order, kappa_safe), 1e-300)) + kappa_safe
    expected = (
        order * np.log(kappa_safe)
        - (dimension / 2.0) * np.log(2.0 * np.pi)
        - log_bessel
    )
    np.testing.assert_array_equal(result, expected)


def test_sentlda_predictive_does_not_mutate_global_counts() -> None:
    counts = np.array([[3, 1], [1, 3]], dtype=np.int64)
    before = counts.copy()
    likelihoods = sentlda_log_likelihood_by_doc(
        [[[(0, 2)]]], topic_word_counts=counts, beta=0.5
    )
    assert likelihoods[0][0, 0] > likelihoods[0][0, 1]
    np.testing.assert_array_equal(counts, before)


def test_scores_are_mapped_to_v_eval_before_ranking() -> None:
    mapped, covered = restrict_scores_to_evaluation_vocabulary(
        np.array([[9.0, 8.0, 7.0]]),
        model_vocabulary=["x", "a", "b"],
        evaluation_vocabulary=["a", "missing", "b"],
    )
    np.testing.assert_array_equal(covered, [True, False, True])
    np.testing.assert_allclose(mapped[:, [0, 2]], [[8.0, 7.0]])
    assert np.isneginf(mapped[0, 1])


def test_ctm_softmax_and_etm_sampling_are_normalized_and_reproducible() -> None:
    decoder = normalize_ctm_decoder_scores(
        np.array([[2.0, 0.0], [0.0, 2.0]]), already_probabilities=False
    )
    np.testing.assert_allclose(decoder.sum(axis=1), 1.0)
    first = sample_etm_theta(
        mu=np.zeros((2, 3)), logsigma=np.zeros((2, 3)), num_samples=4, seed=5
    )
    second = sample_etm_theta(
        mu=np.zeros((2, 3)), logsigma=np.zeros((2, 3)), num_samples=4, seed=5
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=2), 1.0)

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.word_based import topic_assignment
from src.evaluation.word_based.topic_assignment import (
    CollapsedFoldInConfig,
    EmptyTopicError,
    TopicWordStatistics,
    compute_coverage,
    compute_etm_expected_counts,
    compute_expected_topic_word_counts,
    rank_topic_words,
    run_collapsed_fold_in,
    stable_top_word_indices,
    topic_word_probabilities,
    word_topic_npmi,
)


def _config(backend: str = "python") -> CollapsedFoldInConfig:
    return CollapsedFoldInConfig(
        num_chains=2,
        burn_in_sweeps=2,
        retained_samples=4,
        thinning=2,
        random_seed=19,
        backend=backend,  # type: ignore[arg-type]
    )


def test_collapsed_fold_in_is_reproducible_and_log_stable() -> None:
    kwargs = dict(
        alpha=np.array([0.3, 0.7]),
        assignment_unit_type="sentence",
        config=_config(),
        log_likelihood_by_doc=[
            np.array([[10_000.0, -10_000.0], [-10_000.0, 10_000.0]])
        ],
    )
    first = run_collapsed_fold_in(**kwargs)
    second = run_collapsed_fold_in(**kwargs)
    np.testing.assert_array_equal(
        first.posterior_mean_by_doc[0], second.posterior_mean_by_doc[0]
    )
    np.testing.assert_allclose(first.posterior_mean_by_doc[0].sum(axis=1), 1.0)
    assert np.isfinite(first.posterior_mean_by_doc[0]).all()
    assert first.metadata["initialization"] == "seeded_alpha_likelihood_sample"


def test_type_likelihood_path_matches_expanded_path_and_backends() -> None:
    table = np.array([[0.0, -1.0], [-2.0, 0.0]])
    ids = [np.array([0, 1, 0], dtype=np.int64)]
    typed = run_collapsed_fold_in(
        alpha=np.ones(2),
        assignment_unit_type="token",
        config=_config("numba"),
        log_likelihood_by_type=table,
        type_ids_by_doc=ids,
    )
    expanded = run_collapsed_fold_in(
        alpha=np.ones(2),
        assignment_unit_type="token",
        config=_config("python"),
        log_likelihood_by_doc=[table[ids[0]]],
    )
    np.testing.assert_array_equal(
        typed.posterior_mean_by_doc[0], expanded.posterior_mean_by_doc[0]
    )


def test_expected_counts_conserve_covered_mass() -> None:
    posterior = [np.array([[0.75, 0.25], [0.1, 0.9]])]
    unit_counts = [[[(0, 2), (1, 1)], [(1, 3)]]]
    stats = compute_expected_topic_word_counts(
        assignment_probabilities_by_doc=posterior,
        unit_word_counts_by_doc=unit_counts,
        vocab_size=3,
        expected_covered_word_counts=np.array([2.0, 4.0, 0.0]),
    )
    np.testing.assert_allclose(stats.word_counts, [2.0, 4.0, 0.0])
    np.testing.assert_allclose(topic_word_probabilities(stats).sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="mass is not conserved"):
        compute_expected_topic_word_counts(
            assignment_probabilities_by_doc=posterior,
            unit_word_counts_by_doc=unit_counts,
            vocab_size=3,
            expected_covered_word_counts=np.array([2.0, 5.0, 0.0]),
        )


def test_expected_counts_reports_empty_topics_as_isolatable_condition() -> None:
    with pytest.raises(EmptyTopicError) as caught:
        compute_expected_topic_word_counts(
            assignment_probabilities_by_doc=[np.array([[1.0, 0.0]])],
            unit_word_counts_by_doc=[[[(0, 1)]]],
            vocab_size=1,
        )

    assert caught.value.topic_ids == [1]


def test_partial_expected_counts_preserve_nonempty_rankings() -> None:
    stats = compute_expected_topic_word_counts(
        assignment_probabilities_by_doc=[np.array([[1.0, 0.0], [1.0, 0.0]])],
        unit_word_counts_by_doc=[[[(0, 2)], [(1, 1)]]],
        vocab_size=2,
        expected_covered_word_counts=np.array([2.0, 1.0]),
        allow_empty_topics=True,
    )

    assert stats.empty_topic_ids == (1,)
    np.testing.assert_allclose(stats.word_counts, [2.0, 1.0])
    probability_words = rank_topic_words(
        topic_word_probabilities(stats, allow_empty_topics=True),
        ["alpha", "beta"],
        topn=2,
        empty_topic_ids=stats.empty_topic_ids,
    )
    npmi_words = rank_topic_words(
        word_topic_npmi(stats, allow_empty_topics=True),
        ["alpha", "beta"],
        topn=2,
        empty_topic_ids=stats.empty_topic_ids,
    )
    assert [word for word, _score in probability_words[0]] == ["alpha", "beta"]
    assert len(npmi_words[0]) == 2
    assert probability_words[1] == []
    assert npmi_words[1] == []


def test_sparse_and_dense_unit_counts_produce_identical_expected_counts() -> None:
    posterior = [np.array([[0.75, 0.25], [0.1, 0.9]])]
    sparse_units = [[[(0, 2), (1, 1)], [(1, 3)]]]
    dense_units = [np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0]])]
    sparse_stats = compute_expected_topic_word_counts(
        assignment_probabilities_by_doc=posterior,
        unit_word_counts_by_doc=sparse_units,
        vocab_size=3,
    )
    dense_stats = compute_expected_topic_word_counts(
        assignment_probabilities_by_doc=posterior,
        unit_word_counts_by_doc=dense_units,
        vocab_size=3,
    )
    np.testing.assert_allclose(
        sparse_stats.expected_counts, dense_stats.expected_counts
    )


def test_word_topic_npmi_uses_joint_probability_normalizer() -> None:
    counts = np.array([[3.0, 1.0], [1.0, 3.0]])
    stats = TopicWordStatistics(
        expected_counts=counts,
        topic_counts=counts.sum(axis=1),
        word_counts=counts.sum(axis=0),
        total_count=float(counts.sum()),
    )
    actual = word_topic_npmi(stats)
    joint = 3.0 / 8.0
    smoothed_joint = joint + 1e-12
    expected = np.log(smoothed_joint / (0.5 * 0.5)) / -np.log(smoothed_joint)
    assert actual[0, 0] == pytest.approx(expected)


def test_word_topic_npmi_smooths_zero_joint_counts() -> None:
    counts = np.array([[1.0, 0.0], [0.0, 1.0]])
    stats = TopicWordStatistics(
        expected_counts=counts,
        topic_counts=counts.sum(axis=1),
        word_counts=counts.sum(axis=0),
        total_count=float(counts.sum()),
    )

    actual = word_topic_npmi(stats)
    expected = np.log(1e-12 / (0.5 * 0.5)) / -np.log(1e-12)

    assert actual[0, 1] == pytest.approx(expected)
    assert np.all(np.isfinite(actual))


def test_word_topic_npmi_excludes_zero_marginal_words() -> None:
    counts = np.array([[1.0, 0.0], [1.0, 0.0]])
    stats = TopicWordStatistics(
        expected_counts=counts,
        topic_counts=counts.sum(axis=1),
        word_counts=counts.sum(axis=0),
        total_count=float(counts.sum()),
    )

    actual = word_topic_npmi(stats)

    assert np.all(np.isnan(actual[:, 1]))


def test_word_topic_npmi_does_not_underflow_for_tiny_positive_counts() -> None:
    tiny = np.nextafter(0.0, 1.0)
    counts = np.array([[tiny, 1.0], [1.0, 1.0]])
    stats = TopicWordStatistics(
        expected_counts=counts,
        topic_counts=counts.sum(axis=1),
        word_counts=counts.sum(axis=0),
        total_count=float(counts.sum()),
    )

    with np.errstate(all="raise"):
        scores = word_topic_npmi(stats)

    assert np.isfinite(scores[0, 0])


def test_npmi_ranking_excludes_zero_counts_and_uses_vocab_index_ties() -> None:
    scores = np.array([[0.5, np.nan, 0.5]])
    ids = stable_top_word_indices(scores, topn=2)
    assert ids[0].tolist() == [0, 2]
    assert rank_topic_words(scores, ["a", "b", "c"], topn=2)[0] == [
        ("a", 0.5),
        ("c", 0.5),
    ]
    with pytest.raises(ValueError, match="only 2 eligible"):
        stable_top_word_indices(scores, topn=3)


def test_coverage_reports_type_and_occurrence_rates() -> None:
    coverage = compute_coverage(
        covered_word_counts=np.array([2.0, 0.0, 1.0]),
        corpus_word_counts=np.array([2.0, 3.0, 1.0]),
    )
    assert coverage["type_coverage"] == pytest.approx(2 / 3)
    assert coverage["token_coverage"] == pytest.approx(0.5)


def test_etm_responsibilities_conserve_token_mass() -> None:
    theta = np.array(
        [
            [[0.8, 0.2]],
            [[0.6, 0.4]],
        ]
    )
    beta = np.array([[0.9, 0.1], [0.2, 0.8]])
    stats, metadata = compute_etm_expected_counts(
        theta_samples=theta,
        beta=beta,
        corpus_bow=[[(0, 2), (1, 1)]],
    )
    np.testing.assert_allclose(stats.word_counts, [2.0, 1.0])
    assert stats.total_count == pytest.approx(3.0)
    assert metadata["posterior_kind"] == "variational_token_topic_posterior_mean"


# --------------------------------------------------------------------------- #
# SAM: signed topic directions and the positive-part protocol
# --------------------------------------------------------------------------- #


def test_stable_top_word_indices_ranks_signed_scores_algebraically() -> None:
    """SAM topic directions carry negative weights, and must rank by value.

    The eligibility gate is ``np.isfinite`` and the sort key is ``-values``, so a
    strongly negative weight is a legal candidate that ranks last rather than a
    large-magnitude one that ranks first.
    """

    scores = np.array([[0.1, -0.9, 0.5, -0.2, 0.3]])
    vocabulary = ["a", "b", "c", "d", "e"]
    ranked = topic_assignment.rank_topic_words(scores, vocabulary, topn=5)
    assert [word for word, _ in ranked[0]] == ["c", "e", "a", "d", "b"]


def test_sam_positive_topic_profiles_projects_onto_the_simplex() -> None:
    directions = np.array([[0.8, -0.6, 0.0], [0.0, 0.6, 0.8]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    beta, informative = topic_assignment.sam_positive_topic_profiles(directions)

    assert np.allclose(beta.sum(axis=1), 1.0)
    assert np.all(beta >= 0.0)
    assert beta[0, 1] == 0.0  # the negative entry is dropped, not folded in
    assert informative.tolist() == [True, True, True]


def test_sam_positive_topic_profiles_flags_words_no_topic_likes() -> None:
    directions = np.array([[0.8, -0.6, 0.0], [0.6, -0.8, 0.0]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    _, informative = topic_assignment.sam_positive_topic_profiles(directions)
    assert informative.tolist() == [True, False, False]


def test_sam_positive_topic_profiles_rejects_a_wholly_non_positive_topic() -> None:
    directions = np.array([[0.8, 0.6], [-0.6, -0.8]])
    with pytest.raises(ValueError, match="no positive weight"):
        topic_assignment.sam_positive_topic_profiles(directions)


def test_sam_positive_topic_profiles_requires_unit_directions() -> None:
    with pytest.raises(ValueError, match="unit vectors"):
        topic_assignment.sam_positive_topic_profiles(np.array([[1.0, 1.0]]))


def test_sam_expected_counts_conserve_token_mass() -> None:
    directions = np.array([[0.8, -0.6, 0.0], [0.0, 0.6, 0.8]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    doc_topic = np.array([[0.7, 0.3], [0.2, 0.8]])
    corpus_bow = [[(0, 2), (1, 1)], [(1, 3), (2, 1)]]

    statistics, metadata = topic_assignment.compute_sam_expected_counts(
        doc_topic=doc_topic, topic_directions=directions, corpus_bow=corpus_bow
    )
    assert np.all(statistics.expected_counts >= 0.0)
    assert statistics.total_count == pytest.approx(7.0)
    assert metadata["retained_token_fraction"] == pytest.approx(1.0)
    assert metadata["posterior_kind"] == (
        "sam_positive_part_document_mixture_responsibility"
    )
    # Word 0 is positive only under topic 0, word 2 only under topic 1.
    assert statistics.expected_counts[1, 0] == pytest.approx(0.0)
    assert statistics.expected_counts[0, 2] == pytest.approx(0.0)


def test_sam_expected_counts_exclude_words_no_topic_likes() -> None:
    """Words negative under every topic carry no positive-part signal.

    They are dropped from the statistics rather than attributed by document
    composition alone, and the dropped mass is reported so the size of the
    approximation is auditable per run.
    """

    directions = np.array([[0.8, -0.6, 0.0], [0.6, -0.8, 0.0]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    statistics, metadata = topic_assignment.compute_sam_expected_counts(
        doc_topic=np.array([[0.5, 0.5]]),
        topic_directions=directions,
        corpus_bow=[[(0, 3), (1, 5)]],
    )
    assert statistics.total_count == pytest.approx(3.0)
    assert metadata["informative_word_count"] == 1
    assert metadata["retained_token_fraction"] == pytest.approx(3.0 / 8.0)


def test_sam_expected_counts_report_positive_mass_fraction() -> None:
    directions = np.array([[0.6, -0.8, 0.0], [1.0, 0.0, 0.0]])
    statistics, metadata = topic_assignment.compute_sam_expected_counts(
        doc_topic=np.array([[0.5, 0.5]]),
        topic_directions=directions,
        corpus_bow=[[(0, 4)]],
    )
    assert metadata["positive_mass_fraction_by_topic"] == pytest.approx(
        [0.6 / 1.4, 1.0]
    )
    assert statistics.total_count == pytest.approx(4.0)


def test_resolve_topic_word_protocol_covers_sam() -> None:
    assert topic_assignment.resolve_topic_word_protocol("sam") == (
        "sam_positive_part_topics_with_document_mixture_responsibilities"
    )
    assert "sam" not in topic_assignment.COLLAPSED_MODELS
    # The tf variant shares SAM's artifacts and therefore its protocol.
    assert topic_assignment.resolve_topic_word_protocol(
        "sam_tf"
    ) == topic_assignment.resolve_topic_word_protocol("sam")
    assert "sam_tf" not in topic_assignment.COLLAPSED_MODELS

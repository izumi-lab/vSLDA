from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.topic_pairs.numerics import (
    PER_PAIR_KEYS,
    PER_TOPIC_KEYS,
    banerjee_kappa,
    compute_topic_pair_metrics,
    cross_model_overlap,
    jensen_shannon_divergence_matrix,
    model_reference_metrics,
    normalized_entropy,
    offdiagonal_values,
)


def _trainer_kappa(r_bar: float, dim: int) -> float:
    # src/models/vmf_sentence_lda.py, estimate from the mean resultant length
    l_k = float(np.clip(r_bar, 1e-6, 1.0 - 1e-6))
    return l_k * (dim - l_k**2) / ((1.0 - l_k**2) + 1e-12)


def test_banerjee_kappa_matches_the_trainer_formula() -> None:
    values = np.asarray([0.1, 0.5, 0.9, 0.999, 1.0, 0.0, np.nan])
    result = banerjee_kappa(values, 384)
    for r_bar, kappa in zip(values, result):
        if np.isnan(r_bar):
            assert np.isnan(kappa)
        else:
            assert kappa == pytest.approx(_trainer_kappa(r_bar, 384))
    assert result[4] > result[3] > result[2]
    assert result[5] == pytest.approx(_trainer_kappa(1e-6, 384))


def test_banerjee_kappa_respects_max_kappa() -> None:
    capped = banerjee_kappa(np.asarray([0.9, 0.999]), 384, max_kappa=2000.0)
    assert capped[0] == pytest.approx(_trainer_kappa(0.9, 384))
    assert capped[0] < 2000.0
    assert capped[1] == pytest.approx(2000.0)


def test_normalized_entropy_bounds() -> None:
    rows = np.asarray([[1.0, 1.0], [2.0, 0.0], [0.0, 0.0]])
    result = normalized_entropy(rows)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
    assert np.isnan(result[2])
    assert normalized_entropy(np.asarray([[3.0], [0.0]])).tolist()[0] == 0.0


def test_jensen_shannon_matrix_is_symmetric_and_bounded() -> None:
    rows = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    result = jensen_shannon_divergence_matrix(rows)
    assert result[0, 1] == pytest.approx(np.log(2.0))
    assert result[0, 2] == pytest.approx(0.0)
    assert result[0, 0] == pytest.approx(0.0)
    assert np.allclose(result[:3, :3], result[:3, :3].T)
    assert np.isnan(result[3]).all() and np.isnan(result[:, 3]).all()


@pytest.fixture
def toy():
    embeddings = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # one-hot posteriors: sentences 0,1 -> topic 0; 2 -> topic 1; 3 -> topic 2;
    # topic 3 is empty
    probs = np.zeros((4, 4))
    probs[[0, 1, 2, 3], [0, 0, 1, 2]] = 1.0
    labels = np.asarray([0, 0, 1, 1])
    return embeddings, probs, labels


def test_one_hot_posteriors_give_identity_confusion_and_unit_resultants(toy) -> None:
    embeddings, probs, labels = toy
    result = compute_topic_pair_metrics(embeddings, probs, labels, num_labels=2)
    assert result.num_topics == 4 and result.num_sentences == 4
    assert result.num_empty_topics == 1
    assert set(result.per_topic) == set(PER_TOPIC_KEYS) | {"label_mass"}
    assert set(result.per_pair) == set(PER_PAIR_KEYS)

    per_topic = result.per_topic
    assert per_topic["mass_soft"].tolist() == [2.0, 1.0, 1.0, 0.0]
    assert per_topic["count_hard"].tolist() == [2.0, 1.0, 1.0, 0.0]
    assert per_topic["resultant_length_soft"][:3] == pytest.approx([1.0, 1.0, 1.0])
    assert np.isnan(per_topic["resultant_length_soft"][3])
    assert per_topic["kappa_soft"][:3] == pytest.approx(
        banerjee_kappa(np.ones(3), 3).tolist()
    )
    assert np.isnan(per_topic["kappa_hard"][3])
    assert per_topic["label_entropy_normalized"][:3] == pytest.approx([0.0, 0.0, 0.0])
    assert per_topic["label_dominant"].tolist() == [0.0, 1.0, 1.0, -1.0]
    assert per_topic["label_dominant_share"][:3] == pytest.approx([1.0, 1.0, 1.0])
    assert per_topic["label_mass"].tolist() == [
        [2.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]

    per_pair = result.per_pair
    cosine = per_pair["centroid_cosine_soft"]
    assert np.allclose(cosine[:3, :3], np.eye(3))
    assert np.isnan(cosine[3]).all() and np.isnan(cosine[:, 3]).all()
    assert np.allclose(per_pair["centroid_cosine_hard"][:3, :3], np.eye(3))
    confusion = per_pair["assignment_confusion"]
    assert np.allclose(confusion[:3], np.eye(4)[:3])
    assert np.isnan(confusion[3]).all()
    js = per_pair["label_js_divergence"]
    assert js[0, 1] == pytest.approx(np.log(2.0))
    assert js[1, 2] == pytest.approx(0.0)
    assert np.allclose(js[:3, :3], js[:3, :3].T)

    scalars = result.scalar_summary()
    assert scalars["num_empty_topics"] == 1.0
    assert scalars["mean_offdiag_centroid_cosine_soft"] == pytest.approx(0.0)
    assert scalars["max_offdiag_centroid_cosine_soft"] == pytest.approx(0.0)
    assert scalars["mean_offdiag_assignment_confusion"] == pytest.approx(0.0)
    assert offdiagonal_values(cosine).shape == (3,)


def test_soft_posteriors_change_centroids_and_confusion(toy) -> None:
    embeddings, _, labels = toy
    probs = np.asarray(
        [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]]
    )
    result = compute_topic_pair_metrics(embeddings, probs, labels, num_labels=2)
    confusion = result.per_pair["assignment_confusion"]
    assert confusion.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert confusion[0, 1] == pytest.approx(0.1)
    assert confusion[1, 0] == pytest.approx(0.2)
    cosine = result.per_pair["centroid_cosine_soft"]
    assert 0.0 < cosine[0, 1] < 1.0
    assert cosine[0, 1] == pytest.approx(cosine[1, 0])
    assert cosine[0, 2] == pytest.approx(0.0)
    assert result.per_topic["resultant_length_soft"][0] < 1.0
    assert result.per_topic["kappa_soft"][0] < result.per_topic["kappa_soft"][2]
    assert result.per_topic["mass_soft"].sum() == pytest.approx(4.0)
    assert result.per_topic["label_mass"].sum() == pytest.approx(4.0)


def test_input_validation() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    probs = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="aligned"):
        compute_topic_pair_metrics(
            embeddings, probs[:1], np.asarray([0, 0]), num_labels=1
        )
    with pytest.raises(ValueError, match="normalised"):
        compute_topic_pair_metrics(
            embeddings * 2.0, probs, np.asarray([0, 0]), num_labels=1
        )
    with pytest.raises(ValueError, match="sum to one"):
        compute_topic_pair_metrics(
            embeddings, probs * 0.5, np.asarray([0, 0]), num_labels=1
        )
    with pytest.raises(ValueError, match="out of range"):
        compute_topic_pair_metrics(embeddings, probs, np.asarray([0, 3]), num_labels=2)


def test_cross_model_overlap_marginals_are_the_soft_masses() -> None:
    probs_a = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    probs_b = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    overlap = cross_model_overlap(probs_a, probs_b)
    assert overlap.shape == (2, 3)
    assert overlap.sum(axis=1) == pytest.approx(probs_a.sum(axis=0))
    assert overlap.sum(axis=0) == pytest.approx(probs_b.sum(axis=0))
    with pytest.raises(ValueError, match="aligned"):
        cross_model_overlap(probs_a, probs_b[:2])


def test_model_reference_metrics_normalises_the_means() -> None:
    means = np.asarray([[2.0, 0.0], [1.0, 1.0]])
    result = model_reference_metrics(means, np.asarray([10.0, 20.0]))
    cosine = result["centroid_cosine_model"]
    assert cosine[0, 0] == pytest.approx(1.0)
    assert cosine[0, 1] == pytest.approx(np.sqrt(0.5))
    assert result["kappa_model"].tolist() == [10.0, 20.0]
    with pytest.raises(ValueError, match="kappa shape"):
        model_reference_metrics(means, np.asarray([1.0]))

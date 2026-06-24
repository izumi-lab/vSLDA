from __future__ import annotations

import numpy as np
import pytest
from gensim.models import KeyedVectors

from src.utils.encoder import SentenceEncoder
from src.utils.encoder_profiles import resolve_encoder_settings
from src.utils.usif_encoder import UsifSentenceEncoder


def _tiny_vectors() -> KeyedVectors:
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        ["alpha", "beta", "gamma"],
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    return vectors


def test_usif_profile_defaults_to_backend_and_glove_100() -> None:
    resolved = resolve_encoder_settings(model_name="usif")

    assert resolved.backend == "usif"
    assert resolved.embedding_variant == "usif"
    assert resolved.encode_batch_size == 128
    assert resolved.model_kwargs["word2vec"] == "glove-wiki-gigaword-100"


def test_usif_weighted_average_uses_train_probabilities() -> None:
    encoder = UsifSentenceEncoder(
        word2vec=_tiny_vectors(),
        alpha=1.0,
        component_policy="none",
    )
    encoder.fit_tokenized([["alpha", "beta"], ["alpha"]])

    encoded = encoder.encode_tokenized([["alpha", "beta"], ["missing"]])

    alpha_weight = 1.0 / ((2.0 / 3.0) + 0.5)
    beta_weight = 1.0 / ((1.0 / 3.0) + 0.5)
    expected = np.asarray(
        [
            [alpha_weight / 2.0, beta_weight / 2.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(encoded, expected, rtol=1e-6, atol=1e-6)


def test_usif_estimates_alpha_without_sif_upper_cap() -> None:
    encoder = UsifSentenceEncoder(
        word2vec=_tiny_vectors(),
        component_policy="none",
    )
    encoder.fit_tokenized([["alpha"] for _ in range(8)] + [["beta"], ["gamma"]])

    assert encoder.state is not None
    assert encoder.state.word_probability_threshold == pytest.approx(1.0 / 3.0)
    assert encoder.state.alpha_hat == pytest.approx(1.0 / 3.0)
    assert encoder.state.alpha == pytest.approx(4.0 / 3.0)


def test_usif_alpha_sentence_length_uses_input_tokens_not_only_fitted_vocab() -> None:
    encoder = UsifSentenceEncoder(
        word2vec=_tiny_vectors(),
        component_policy="none",
    )
    encoder.fit_tokenized(
        [["alpha", "missing"] for _ in range(4)]
        + [["beta", "missing"] for _ in range(3)]
        + [["gamma", "missing"] for _ in range(3)]
    )

    assert encoder.state is not None
    assert encoder.state.average_sentence_length == pytest.approx(2.0)
    assert encoder.state.word_probability_threshold == pytest.approx(5.0 / 9.0)


def test_usif_component_weights_are_normalized_over_selected_components() -> None:
    encoder = UsifSentenceEncoder(
        word2vec=_tiny_vectors(),
        alpha=1.0,
        component_policy="fixed",
        n_components=1,
    )
    encoder.fit_tokenized([["alpha"], ["beta"], ["gamma"]])

    assert encoder.state is not None
    np.testing.assert_allclose(encoder.state.component_weights, [1.0])


def test_usif_requires_fit_before_encode() -> None:
    encoder = UsifSentenceEncoder(word2vec=_tiny_vectors())

    with pytest.raises(ValueError, match="fitted"):
        encoder.encode_tokenized([["alpha"]])


def test_sentence_encoder_uses_usif_backend_with_tokenized_inputs() -> None:
    encoder = SentenceEncoder(
        "usif",
        device="cpu",
        model_kwargs={
            "word2vec": _tiny_vectors(),
            "alpha": 1.0,
            "component_policy": "none",
        },
    )
    assert encoder.requires_fit is True
    assert encoder.accepts_tokenized is True

    encoder.fit_tokenized([["alpha", "beta"], ["alpha"]])
    encoded = encoder.encode_tokenized([["beta"]])

    np.testing.assert_allclose(encoded, [[0.0, 1.2]], rtol=1e-6, atol=1e-6)
    assert encoder.encoder_config["embedding_variant"] == "usif"

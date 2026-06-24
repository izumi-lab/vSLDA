from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baselines.models.sentence_gaussian_helpers import (
    SentenceGaussianLdaModel,
    build_sentence_gaussian_encoder,
    load_sentence_gaussianlda_model,
)
from src.core.artifacts import save_json, save_pickle


class DummyEncoder:
    def __init__(self, embeddings: dict[str, np.ndarray] | None = None) -> None:
        self._embeddings = embeddings or {}

    def encode(self, sentences, **_kwargs):
        rows = []
        for sentence in sentences:
            rows.append(self._embeddings[str(sentence)])
        return np.asarray(rows, dtype=np.float64)

    def get_sentence_embedding_dimension(self) -> int:
        return 2


def test_build_sentence_gaussian_encoder_uses_repo_owned_wrapper(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceEncoder:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "src.baselines.models.sentence_gaussian_helpers.SentenceEncoder",
        FakeSentenceEncoder,
    )

    encoder = build_sentence_gaussian_encoder(
        "sentence-transformers/all-minilm-l6-v2",
        device="cpu",
        encode_prefix="query: ",
        strip_terminal_normalize=False,
    )

    assert isinstance(encoder, FakeSentenceEncoder)
    assert captured == {
        "model_name": "sentence-transformers/all-minilm-l6-v2",
        "device": "cpu",
        "encode_prefix": "query: ",
        "strip_terminal_normalize": False,
    }


def test_sentence_gaussianlda_model_scores_and_samples_with_repo_owned_model() -> None:
    encoder = DummyEncoder(
        embeddings={
            "s1": np.asarray([1.0, 0.0]),
            "s2": np.asarray([0.0, 1.0]),
        }
    )
    model = SentenceGaussianLdaModel(
        prior_mu=np.asarray([0.5, 0.5]),
        encoder=encoder,
        num_tables=1,
        alpha=0.1,
        kappa=0.1,
        table_counts=np.asarray([2.0]),
        table_means=np.asarray([[0.5, 0.5]]),
        log_determinants=np.asarray([0.0]),
        table_cholesky_ltriangular_mat=np.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
    )

    scores = model.log_multivariate_tdensity_tables(np.asarray([1.0, 0.0]))
    topics = model.sample(["s1", "s2"], num_iterations=1)

    assert scores.shape == (1,)
    assert np.isfinite(scores[0])
    assert topics == [0, 0]
    assert model.table_density_kernel_backend in {"python", "numba"}
    assert model.posterior_sampling_kernel_backend in {"python", "numba"}


def test_load_sentence_gaussianlda_model_reads_repo_owned_artifacts(
    tmp_path: Path,
) -> None:
    encoder = DummyEncoder()
    save_json(
        {
            "alpha": 0.1,
            "num_tables": 1,
            "kappa": 0.1,
            "average_ll": [0.0],
        },
        tmp_path / "params.json",
    )
    save_pickle(np.asarray([1.0]), tmp_path / "table_counts.pkl")
    save_pickle(np.asarray([[0.5, 0.5]]), tmp_path / "table_means.pkl")
    save_pickle(np.asarray([0.0]), tmp_path / "log_determinants.pkl")
    save_pickle(
        np.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
        tmp_path / "table_cholesky_ltriangular_mat.pkl",
    )
    save_pickle(np.asarray([0.5, 0.5]), tmp_path / "prior_mu.pkl")

    persisted = load_sentence_gaussianlda_model(
        param_dir=tmp_path,
        encoder=encoder,
    )

    assert persisted.encoder is encoder
    assert persisted.model.num_tables == 1
    assert persisted.model.embedding_size == 2
    assert persisted.model.table_density_kernel_backend in {"python", "numba"}
    assert persisted.model.posterior_sampling_kernel_backend in {"python", "numba"}

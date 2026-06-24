from __future__ import annotations

from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from src.baselines.models.gaussian_helpers import (
    GaussianLdaScorer,
    load_gaussian_word_vectors,
    load_gaussianlda_model,
)
from src.core.artifacts import save_json, save_pickle


def _build_keyed_vectors() -> KeyedVectors:
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        ["alpha", "beta"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    return vectors


def test_load_gaussian_word_vectors_prefers_persisted_local_kv(tmp_path: Path) -> None:
    kv = _build_keyed_vectors()
    kv_path = tmp_path / "local_word2vec.kv"
    kv.save(kv_path.as_posix())

    loaded = load_gaussian_word_vectors("glove-wiki-gigaword-100", param_dir=tmp_path)

    assert list(loaded.key_to_index.keys()) == ["alpha", "beta"]
    assert loaded.vector_size == 2


def test_load_gaussian_word_vectors_supports_wikientvec_specs(
    monkeypatch,
) -> None:
    expected = _build_keyed_vectors()

    monkeypatch.setattr(
        "src.baselines.models.gaussian_helpers.load_wikientvec",
        lambda spec, cache_dir=None: expected,
    )

    loaded = load_gaussian_word_vectors(
        "wikientvec:20190520:jawiki.word_vectors.100d.txt.bz2"
    )

    assert loaded is expected


def test_load_gaussianlda_model_reads_repo_owned_artifacts(tmp_path: Path) -> None:
    kv = _build_keyed_vectors()
    (tmp_path / "local_word2vec.kv").parent.mkdir(parents=True, exist_ok=True)
    kv.save((tmp_path / "local_word2vec.kv").as_posix())
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

    persisted = load_gaussianlda_model(
        param_dir=tmp_path,
        word2vec="glove-wiki-gigaword-100",
    )

    assert persisted.vocab == ["alpha", "beta"]
    assert persisted.embeddings.shape == (2, 2)
    assert persisted.model.num_tables == 1
    assert persisted.model.table_density_kernel_backend in {"python", "numba"}
    assert persisted.model.posterior_sampling_kernel_backend in {"python", "numba"}


def test_gaussianlda_scorer_can_sample_with_vocab_ids() -> None:
    model = GaussianLdaScorer(
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        vocab=["alpha", "beta"],
        num_tables=1,
        alpha=0.1,
        kappa=0.1,
        table_counts=np.asarray([2.0]),
        table_means=np.asarray([[0.5, 0.5]]),
        log_determinants=np.asarray([0.0]),
        table_cholesky_ltriangular_mat=np.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
    )

    topics = model.sample([0, 1], num_iterations=1)

    assert topics == [0, 0]
    assert model.table_density_kernel_backend in {"python", "numba"}
    assert model.posterior_sampling_kernel_backend in {"python", "numba"}

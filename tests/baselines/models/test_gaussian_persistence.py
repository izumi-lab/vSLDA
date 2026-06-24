from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gensim.models import KeyedVectors

from src.baselines.models.gaussian_persistence import (
    build_gaussian_params_payload,
    build_gaussian_state_specs,
    persist_gaussian_family_run,
)
from src.baselines.models.gaussian_state import snapshot_gaussian_trainer
from src.core.artifacts import load_artifact_json, load_artifact_pickle


def _build_trainer() -> SimpleNamespace:
    return SimpleNamespace(
        alpha=0.1,
        num_tables=2,
        average_ll=[1.0, 2.0],
        prior=SimpleNamespace(kappa=0.2, mu=np.asarray([0.5, 0.5])),
        table_density_kernel_backend="numba",
        posterior_sampling_kernel_backend="numba",
        avg_ll_kernel_backend="python",
        table_counts=np.asarray([1, 2]),
        table_means=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        table_inverse_covariances=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        log_determinants=np.asarray([0.0, 0.1]),
        sum_table_customers=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        sum_squared_table_customers=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        table_cholesky_ltriangular_mat=np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
        ),
        training_corpus_preencoded=True,
        training_corpus_encoding_sec=0.25,
    )


def test_build_gaussian_params_payload_reads_common_fields() -> None:
    payload = build_gaussian_params_payload(_build_trainer())

    assert payload == {
        "average_ll": [1.0, 2.0],
        "alpha": 0.1,
        "num_tables": 2,
        "kappa": 0.2,
        "table_density_kernel_backend": "numba",
        "posterior_sampling_kernel_backend": "numba",
        "avg_ll_kernel_backend": "python",
        "training_corpus_preencoded": True,
        "training_corpus_encoding_sec": 0.25,
    }


def test_build_gaussian_state_specs_includes_common_artifacts() -> None:
    specs = build_gaussian_state_specs(
        train_doc_topic=np.asarray([[1.0, 0.0]]),
        infer_doc_topic=np.asarray([[0.5, 0.5]]),
        trainer=_build_trainer(),
        category="all",
    )

    assert [spec.name for spec in specs[:3]] == [
        "train_path",
        "infer_path",
        "table_counts",
    ]
    assert specs[0].filename == "table_counts_per_doc.pkl"
    assert specs[1].filename == "all.pkl"


def test_persist_gaussian_family_run_writes_common_and_extra_artifacts(
    tmp_path: Path,
) -> None:
    kv = KeyedVectors(vector_size=2)
    kv.add_vectors(["alpha"], np.asarray([[1.0, 0.0]], dtype=np.float32))

    artifacts = persist_gaussian_family_run(
        trainer=_build_trainer(),
        train_doc_topic=np.asarray([[1.0, 0.0]]),
        infer_doc_topic=np.asarray([[0.5, 0.5]]),
        train_dir=tmp_path / "params" / "all",
        infer_dir=tmp_path / "infer",
        category="all",
        additional_specs=[
            SimpleNamespace(
                name="extra_train",
                filename="extra_train.pkl",
                payload=np.asarray([[1.0]]),
                split="train",
            )
        ],
        extra_saved_artifact_names=["extra_train"],
        local_word_vectors=kv,
    )

    assert artifacts.train_path.name == "table_counts_per_doc.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert artifacts.extras["params_json"].name == "params.json"
    assert artifacts.extras["extra_train"].name == "extra_train.pkl"
    assert artifacts.extras["local_word2vec"].name == "local_word2vec.kv"
    assert load_artifact_json(artifacts.extras["params_json"])["num_tables"] == 2
    assert load_artifact_pickle(artifacts.extras["extra_train"]).shape == (1, 1)


def test_persist_gaussian_family_run_accepts_state_without_prior_mu(
    tmp_path: Path,
) -> None:
    trainer_state = snapshot_gaussian_trainer(_build_trainer(), include_prior_mu=False)

    artifacts = persist_gaussian_family_run(
        trainer=trainer_state,
        train_doc_topic=np.asarray([[1.0, 0.0]]),
        infer_doc_topic=np.asarray([[0.5, 0.5]]),
        train_dir=tmp_path / "params" / "all",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    assert artifacts.train_path.name == "table_counts_per_doc.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert artifacts.extras["params_json"].name == "params.json"

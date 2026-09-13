from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from gensim.models import KeyedVectors

from src.baselines.contracts import BaselineArtifacts
from src.baselines.models.gaussian_state import (
    GaussianTrainerLike,
    GaussianTrainerState,
    coerce_gaussian_trainer_state,
    is_reduced_covariance_state,
    validate_gaussian_trainer_state,
)
from src.core.artifacts import PickleArtifactSpec, save_json, save_split_pickles


def build_gaussian_params_payload(
    trainer: GaussianTrainerState | GaussianTrainerLike,
) -> dict[str, object]:
    trainer_state = validate_gaussian_trainer_state(
        coerce_gaussian_trainer_state(trainer)
    )
    payload: dict[str, object] = {
        "average_ll": list(trainer_state.average_ll),
        "alpha": trainer_state.alpha,
        "num_tables": trainer_state.num_tables,
        "kappa": trainer_state.prior_kappa,
    }
    if trainer_state.prior_scale is not None:
        payload["prior_scale"] = trainer_state.prior_scale
    if trainer_state.prior_nu is not None:
        payload["prior_nu"] = trainer_state.prior_nu
    if trainer_state.covariance_type is not None:
        payload["covariance_type"] = str(trainer_state.covariance_type)
    if trainer_state.table_density_kernel_backend is not None:
        payload["table_density_kernel_backend"] = (
            trainer_state.table_density_kernel_backend
        )
    if trainer_state.posterior_sampling_kernel_backend is not None:
        payload["posterior_sampling_kernel_backend"] = (
            trainer_state.posterior_sampling_kernel_backend
        )
    if trainer_state.avg_ll_kernel_backend is not None:
        payload["avg_ll_kernel_backend"] = trainer_state.avg_ll_kernel_backend
    if trainer_state.training_corpus_preencoded is not None:
        payload["training_corpus_preencoded"] = bool(
            trainer_state.training_corpus_preencoded
        )
    if trainer_state.training_corpus_encoding_sec is not None:
        payload["training_corpus_encoding_sec"] = float(
            trainer_state.training_corpus_encoding_sec
        )
    if trainer_state.iteration_diagnostics:
        payload["iteration_diagnostics"] = [
            dict(item) for item in trainer_state.iteration_diagnostics
        ]
    if trainer_state.training_elapsed_sec is not None:
        payload["training_elapsed_sec"] = float(trainer_state.training_elapsed_sec)
    return payload


def build_gaussian_state_specs(
    *,
    train_doc_topic: Any,
    infer_doc_topic: Any,
    trainer: GaussianTrainerState | GaussianTrainerLike,
    category: str,
) -> list[PickleArtifactSpec]:
    trainer_state = validate_gaussian_trainer_state(
        coerce_gaussian_trainer_state(trainer)
    )
    if is_reduced_covariance_state(trainer_state):
        covariance_specs = [
            PickleArtifactSpec(
                name="sum_squared_table_customers_diag",
                filename="sum_squared_table_customers_diag.pkl",
                payload=trainer_state.sum_squared_table_customers_diag,
                split="train",
            ),
            PickleArtifactSpec(
                name="table_scaled_variances",
                filename="table_scaled_variances.pkl",
                payload=trainer_state.table_scaled_variances,
                split="train",
            ),
        ]
    else:
        covariance_specs = [
            PickleArtifactSpec(
                name="table_inverse_covariances",
                filename="table_inverse_covariances.pkl",
                payload=trainer_state.table_inverse_covariances,
                split="train",
            ),
            PickleArtifactSpec(
                name="sum_squared_table_customers",
                filename="sum_squared_table_customers.pkl",
                payload=trainer_state.sum_squared_table_customers,
                split="train",
            ),
            PickleArtifactSpec(
                name="table_cholesky_ltriangular_mat",
                filename="table_cholesky_ltriangular_mat.pkl",
                payload=trainer_state.table_cholesky_ltriangular_mat,
                split="train",
            ),
        ]
    return [
        PickleArtifactSpec(
            name="train_path",
            filename="table_counts_per_doc.pkl",
            payload=train_doc_topic,
            split="train",
        ),
        PickleArtifactSpec(
            name="infer_path",
            filename=f"{category}.pkl",
            payload=infer_doc_topic,
            split="infer",
        ),
        PickleArtifactSpec(
            name="table_counts",
            filename="table_counts.pkl",
            payload=trainer_state.table_counts,
            split="train",
        ),
        PickleArtifactSpec(
            name="table_means",
            filename="table_means.pkl",
            payload=trainer_state.table_means,
            split="train",
        ),
        PickleArtifactSpec(
            name="log_determinants",
            filename="log_determinants.pkl",
            payload=trainer_state.log_determinants,
            split="train",
        ),
        PickleArtifactSpec(
            name="sum_table_customers",
            filename="sum_table_customers.pkl",
            payload=trainer_state.sum_table_customers,
            split="train",
        ),
        *covariance_specs,
    ]


def persist_gaussian_family_run(
    *,
    trainer: GaussianTrainerState | GaussianTrainerLike,
    train_doc_topic: Any,
    infer_doc_topic: Any,
    train_dir: Path,
    infer_dir: Path,
    category: str,
    additional_specs: Iterable[PickleArtifactSpec] = (),
    extra_saved_artifact_names: Iterable[str] = (),
    local_word_vectors: KeyedVectors | None = None,
    additional_extras: Mapping[str, Path] | None = None,
) -> BaselineArtifacts:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = validate_gaussian_trainer_state(
        coerce_gaussian_trainer_state(trainer)
    )

    params_path = train_dir / "params.json"
    save_json(build_gaussian_params_payload(trainer_state), params_path)
    saved = save_split_pickles(
        [
            *build_gaussian_state_specs(
                train_doc_topic=train_doc_topic,
                infer_doc_topic=infer_doc_topic,
                trainer=trainer_state,
                category=category,
            ),
            *list(additional_specs),
        ],
        train_dir=train_dir,
        infer_dir=infer_dir,
    )

    extras: dict[str, Path] = {"params_json": params_path}
    for name in extra_saved_artifact_names:
        extras[str(name)] = saved[str(name)]
    if local_word_vectors is not None:
        kv_path = train_dir / "local_word2vec.kv"
        local_word_vectors.save(kv_path.as_posix())
        extras["local_word2vec"] = kv_path
    if additional_extras is not None:
        extras.update({str(key): path for key, path in additional_extras.items()})

    return BaselineArtifacts(
        train_path=saved["train_path"],
        infer_path=saved["infer_path"],
        extras=extras,
    )

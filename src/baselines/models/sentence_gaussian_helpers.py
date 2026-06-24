from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.baselines.models.gaussian_numerics import (
    GAUSSIAN_POSTERIOR_SAMPLING_BACKEND,
    GAUSSIAN_TABLE_DENSITY_BACKEND,
    build_gaussian_nu,
    build_scaled_cholesky,
    log_multivariate_tdensity,
    log_multivariate_tdensity_tables,
    sample_doc_topic_assignments,
)
from src.core.artifacts import load_artifact_json, load_artifact_pickle
from src.utils.encoder import SentenceEncoder


def build_sentence_gaussian_encoder(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    *,
    device: str = "cuda",
    encode_prefix: str | None = None,
    backend: str = "auto",
    pooling: str | None = None,
    encode_prompt: str | None = None,
    encode_prompt_name: str | None = None,
    encode_batch_size: int | None = None,
    model_kwargs: dict | None = None,
    tokenizer_kwargs: dict | None = None,
    normalize_embeddings: bool | None = None,
    truncate_dim: int | None = None,
    strip_terminal_normalize: bool = True,
) -> SentenceEncoder:
    kwargs = {
        "model_name": model_name,
        "device": device,
        "encode_prefix": encode_prefix,
        "strip_terminal_normalize": strip_terminal_normalize,
    }
    if backend != "auto":
        kwargs["backend"] = backend
    if pooling is not None:
        kwargs["pooling"] = pooling
    if encode_prompt is not None:
        kwargs["encode_prompt"] = encode_prompt
    if encode_prompt_name is not None:
        kwargs["encode_prompt_name"] = encode_prompt_name
    if encode_batch_size is not None:
        kwargs["encode_batch_size"] = encode_batch_size
    if model_kwargs is not None:
        kwargs["model_kwargs"] = model_kwargs
    if tokenizer_kwargs is not None:
        kwargs["tokenizer_kwargs"] = tokenizer_kwargs
    if normalize_embeddings is not None:
        kwargs["normalize_embeddings"] = normalize_embeddings
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    return SentenceEncoder(**kwargs)


class SentenceGaussianLdaModel:
    def __init__(
        self,
        *,
        prior_mu: np.ndarray,
        encoder: SentenceEncoder,
        num_tables: int,
        alpha: float,
        kappa: float,
        table_counts: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        table_cholesky_ltriangular_mat: np.ndarray,
    ) -> None:
        self.alpha = float(alpha)
        self.encoder = encoder
        self.embedding_size = encoder.get_sentence_embedding_dimension()
        self.num_tables = int(num_tables)
        self.table_counts = np.asarray(table_counts, dtype=np.float64)
        self.table_means = np.asarray(table_means, dtype=np.float64)
        self.log_determinants = np.asarray(log_determinants, dtype=np.float64)
        self.table_cholesky_ltriangular_mat = np.asarray(
            table_cholesky_ltriangular_mat,
            dtype=np.float64,
        )
        self.prior_mu = np.asarray(prior_mu, dtype=np.float64)
        self.kappa = float(kappa)

        self.nu = build_gaussian_nu(
            table_counts=self.table_counts,
            embedding_size=self.embedding_size,
        )
        self.scaled_table_cholesky_ltriangular_mat = build_scaled_cholesky(
            table_counts=self.table_counts,
            kappa=self.kappa,
            embedding_size=self.embedding_size,
            table_cholesky_ltriangular_mat=self.table_cholesky_ltriangular_mat,
        )
        self.table_density_kernel_backend = GAUSSIAN_TABLE_DENSITY_BACKEND
        self.posterior_sampling_kernel_backend = GAUSSIAN_POSTERIOR_SAMPLING_BACKEND

    def log_multivariate_tdensity(
        self,
        x: np.ndarray,
        table_id: int,
    ) -> np.ndarray:
        return log_multivariate_tdensity(
            x,
            table_id=table_id,
            embedding_size=self.embedding_size,
            nu=self.nu,
            table_means=self.table_means,
            log_determinants=self.log_determinants,
            scaled_table_cholesky_ltriangular_mat=(
                self.scaled_table_cholesky_ltriangular_mat
            ),
        )

    def log_multivariate_tdensity_tables(self, x: np.ndarray) -> np.ndarray:
        return log_multivariate_tdensity_tables(
            np.asarray(x, dtype=np.float64),
            embedding_size=self.embedding_size,
            nu=self.nu,
            table_means=self.table_means,
            log_determinants=self.log_determinants,
            scaled_table_cholesky_ltriangular_mat=(
                self.scaled_table_cholesky_ltriangular_mat
            ),
        )

    def sample(
        self,
        doc: list[str] | np.ndarray,
        num_iterations: int,
    ) -> list[int]:
        if len(doc) == 0:
            return []

        if isinstance(doc, np.ndarray):
            if doc.ndim == 1:
                encoded_doc = doc[np.newaxis, :]
            elif doc.ndim == 2:
                encoded_doc = doc
            else:
                raise ValueError("Encoded doc array must be 1D or 2D.")
        else:
            encoded_doc = np.asarray(self.encoder.encode(doc))
        if encoded_doc.shape[0] == 0:
            return []

        table_assignments = list(
            np.random.randint(self.num_tables, size=len(encoded_doc))
        )
        doc_table_counts = np.bincount(
            table_assignments, minlength=self.num_tables
        ).astype(
            np.int32,
            copy=False,
        )
        log_likelihood_matrix = np.vstack(
            [
                self.log_multivariate_tdensity_tables(embedding)
                for embedding in encoded_doc
            ]
        ).astype(np.float64, copy=False)

        for _iteration in range(num_iterations):
            assignment_array = np.asarray(table_assignments, dtype=np.int32)
            uniforms = np.random.random(len(encoded_doc)).astype(np.float64, copy=False)
            sample_doc_topic_assignments(
                assignment_array,
                doc_table_counts,
                log_likelihood_matrix,
                alpha=self.alpha,
                uniforms=uniforms,
            )
            table_assignments = assignment_array.tolist()
        return table_assignments


@dataclass(frozen=True)
class PersistedSentenceGaussianLdaModel:
    model: SentenceGaussianLdaModel
    encoder: SentenceEncoder


def load_sentence_gaussianlda_model(
    *,
    param_dir: Path,
    encoder: SentenceEncoder,
) -> PersistedSentenceGaussianLdaModel:
    params = load_artifact_json(param_dir / "params.json")
    model = SentenceGaussianLdaModel(
        prior_mu=load_artifact_pickle(param_dir / "prior_mu.pkl"),
        encoder=encoder,
        num_tables=int(params["num_tables"]),
        alpha=float(params["alpha"]),
        kappa=float(params["kappa"]),
        table_counts=load_artifact_pickle(param_dir / "table_counts.pkl"),
        table_means=load_artifact_pickle(param_dir / "table_means.pkl"),
        log_determinants=load_artifact_pickle(param_dir / "log_determinants.pkl"),
        table_cholesky_ltriangular_mat=load_artifact_pickle(
            param_dir / "table_cholesky_ltriangular_mat.pkl"
        ),
    )
    return PersistedSentenceGaussianLdaModel(model=model, encoder=encoder)

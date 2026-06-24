from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gensim
import gensim.downloader
import numpy as np
from gensim.models import KeyedVectors

from src.baselines.models.gaussian_numerics import (
    GAUSSIAN_POSTERIOR_SAMPLING_BACKEND,
    GAUSSIAN_TABLE_DENSITY_BACKEND,
    build_gaussian_nu,
    build_scaled_cholesky,
    log_multivariate_tdensity,
    log_multivariate_tdensity_tables,
    sample_doc_topic_assignments,
)
from src.baselines.preprocessing.wikientvec import is_wikientvec_spec, load_wikientvec
from src.core.artifacts import load_artifact_json, load_artifact_pickle

_LOCAL_WORD2VEC_FILENAME = "local_word2vec.kv"


class GaussianLdaScorer:
    def __init__(
        self,
        *,
        embeddings: np.ndarray,
        vocab: list[str],
        num_tables: int,
        alpha: float,
        kappa: float,
        table_counts: np.ndarray,
        table_means: np.ndarray,
        log_determinants: np.ndarray,
        table_cholesky_ltriangular_mat: np.ndarray,
    ) -> None:
        self.vocab = list(vocab)
        self._vocab_index = {word: idx for idx, word in enumerate(self.vocab)}
        self._topic_word_pdf_cache: dict[int, np.ndarray] = {}
        self.embedding_size = embeddings.shape[1]
        self.vocab_embeddings = np.asarray(embeddings, dtype=np.float64)
        self.num_tables = int(num_tables)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.table_counts = np.asarray(table_counts, dtype=np.float64)
        self.table_means = np.asarray(table_means, dtype=np.float64)
        self.log_determinants = np.asarray(log_determinants, dtype=np.float64)
        self.table_cholesky_ltriangular_mat = np.asarray(
            table_cholesky_ltriangular_mat,
            dtype=np.float64,
        )
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

    def sample(
        self,
        doc: list[object],
        num_iterations: int,
        oovs_as_nones: bool = False,
    ) -> list[int | None]:
        if len(doc) == 0:
            return []

        normalized_doc = [
            (
                token
                if isinstance(token, np.ndarray) or type(token) is int
                else self._vocab_index.get(str(token))
            )
            for token in doc
        ]
        unknown_word_locs = [
            index for index, token in enumerate(normalized_doc) if token is None
        ]
        working_doc = [token for token in normalized_doc if token is not None]
        if not working_doc:
            if oovs_as_nones:
                return [None for _token in normalized_doc]
            return []

        table_assignments = list(
            np.random.randint(self.num_tables, size=len(working_doc))
        )
        doc_table_counts = np.bincount(
            table_assignments, minlength=self.num_tables
        ).astype(
            np.int32,
            copy=False,
        )
        log_likelihood_matrix = np.vstack(
            [self.log_multivariate_tdensity_tables(token) for token in working_doc]
        ).astype(np.float64, copy=False)

        for _iteration in range(num_iterations):
            assignment_array = np.asarray(table_assignments, dtype=np.int32)
            uniforms = np.random.random(len(working_doc)).astype(np.float64, copy=False)
            sample_doc_topic_assignments(
                assignment_array,
                doc_table_counts,
                log_likelihood_matrix,
                alpha=self.alpha,
                uniforms=uniforms,
            )
            table_assignments = assignment_array.tolist()

        if oovs_as_nones:
            for index in unknown_word_locs:
                table_assignments.insert(index, None)
        return table_assignments

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

    def log_multivariate_tdensity_tables(self, x: np.ndarray | int) -> np.ndarray:
        if type(x) is int:
            if x not in self._topic_word_pdf_cache:
                self._topic_word_pdf_cache[x] = self.log_multivariate_tdensity_tables(
                    self.vocab_embeddings[x]
                )
            return self._topic_word_pdf_cache[x]

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


@dataclass(frozen=True)
class PersistedGaussianLdaModel:
    model: object
    embeddings: np.ndarray
    vocab: list[str]
    word_vectors: KeyedVectors


def load_word_vectors(
    word2vec: str | KeyedVectors,
    *,
    wikientvec_cache_dir: str | None = None,
) -> KeyedVectors:
    if not isinstance(word2vec, str):
        return word2vec
    if is_wikientvec_spec(word2vec):
        return load_wikientvec(word2vec, cache_dir=wikientvec_cache_dir)
    local_path = Path(word2vec).expanduser()
    if local_path.exists():
        if local_path.suffix == ".kv":
            return KeyedVectors.load(local_path.as_posix())
        binary = local_path.suffix == ".bin" or local_path.name.endswith(".bin.gz")
        return KeyedVectors.load_word2vec_format(local_path.as_posix(), binary=binary)
    return gensim.downloader.load(word2vec)


def should_use_external_vectors(word2vec: str | KeyedVectors) -> bool:
    if not isinstance(word2vec, str):
        return True
    if is_wikientvec_spec(word2vec):
        return True
    return Path(word2vec).expanduser().exists()


def build_local_word2vec(
    token_docs: Sequence[Sequence[str]],
) -> KeyedVectors:
    model = gensim.models.Word2Vec(
        sentences=list(token_docs),
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=20,
        seed=0,
    )
    return model.wv


def to_index_docs(
    token_docs: Sequence[Sequence[str]],
    vocab: dict[str, int],
) -> list[list[int]]:
    return [[vocab[token] for token in doc if token in vocab] for doc in token_docs]


def load_gaussian_word_vectors(
    word2vec: str | KeyedVectors,
    *,
    param_dir: Path | None = None,
    wikientvec_cache_dir: str | None = None,
) -> KeyedVectors:
    if param_dir is not None:
        local_kv_path = param_dir / _LOCAL_WORD2VEC_FILENAME
        if local_kv_path.exists():
            return KeyedVectors.load(local_kv_path.as_posix())
    return load_word_vectors(
        word2vec,
        wikientvec_cache_dir=wikientvec_cache_dir,
    )


def load_gaussianlda_model(
    *,
    param_dir: Path,
    word2vec: str | KeyedVectors,
    wikientvec_cache_dir: str | None = None,
) -> PersistedGaussianLdaModel:
    word_vectors = load_gaussian_word_vectors(
        word2vec,
        param_dir=param_dir,
        wikientvec_cache_dir=wikientvec_cache_dir,
    )
    vocab = [str(word) for word in word_vectors.key_to_index.keys()]
    embeddings = np.asarray(word_vectors.vectors, dtype=float)
    params = load_artifact_json(param_dir / "params.json")
    model = GaussianLdaScorer(
        embeddings=embeddings,
        vocab=vocab,
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
    return PersistedGaussianLdaModel(
        model=model,
        embeddings=embeddings,
        vocab=vocab,
        word_vectors=word_vectors,
    )

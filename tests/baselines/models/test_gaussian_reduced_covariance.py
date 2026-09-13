"""Tests of the diagonal / spherical covariance variants of the sentence Gaussian LDA."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import multivariate_t
from scipy.stats import t as student_t

from src.baselines.models.gaussian_internal.prior import Wishart
from src.baselines.models.gaussian_numerics import (
    build_gaussian_nu,
    build_scaled_cholesky,
    log_multivariate_tdensity_tables,
)
from src.baselines.models.gaussian_persistence import (
    build_gaussian_params_payload,
    build_gaussian_state_specs,
    persist_gaussian_family_run,
)
from src.baselines.models.gaussian_reduced_numba import (
    NUMBA_AVAILABLE,
    _accumulate_reduced_log_likelihood_encoded_python,
    _log_reduced_tdensity_tables_python,
    accumulate_reduced_log_likelihood_encoded_kernel,
    log_reduced_tdensity_tables_kernel,
)
from src.baselines.models.gaussian_reduced_numerics import (
    ReducedCovarianceTables,
    compute_reduced_table_parameters,
    log_reduced_tdensity_tables,
    reduced_prior_nu,
)
from src.baselines.models.gaussian_state import snapshot_gaussian_trainer
from src.baselines.models.sentence_gaussian_helpers import (
    load_sentence_gaussianlda_model,
)
from src.baselines.models.sentence_gaussian_trainer import (
    GaussianLDATrainer as SentenceGaussianTrainer,
)
from src.baselines.params import (
    format_covariance_variant,
    normalize_covariance_type,
    parse_sentence_gaussianlda_params,
)
from src.core.artifacts import PickleArtifactSpec

pytest.importorskip("choldate")

DIM = 6
NUM_TABLES = 3


class ArrayEncoder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, sentences, **_kwargs) -> np.ndarray:
        rows = [np.asarray(row, dtype=np.float64) for row in sentences]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack(rows)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


def _corpus(seed: int = 0, num_docs: int = 12) -> list[list[tuple[float, ...]]]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(NUM_TABLES, DIM))
    corpus = []
    for _ in range(num_docs):
        count = int(rng.integers(2, 6))
        labels = rng.integers(0, NUM_TABLES, size=count)
        rows = centers[labels] + 0.3 * rng.normal(size=(count, DIM))
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        corpus.append([tuple(row) for row in rows])
    return corpus


def _train(
    covariance_type: str, *, seed: int = 3, sweeps: int = 2
) -> SentenceGaussianTrainer:
    np.random.seed(seed)
    trainer = SentenceGaussianTrainer(
        _corpus(),
        ArrayEncoder(DIM),
        NUM_TABLES,
        0.1,
        0.1,
        save_path=None,
        prior_scale=0.1,
        covariance_type=covariance_type,
    )
    trainer.sample(sweeps)
    return trainer


# --------------------------------------------------------------------------- params


def test_covariance_type_normalization_and_labels() -> None:
    assert normalize_covariance_type(None) == "full"
    assert normalize_covariance_type("ISO") == "spherical"
    assert normalize_covariance_type("diagonal") == "diag"
    assert format_covariance_variant("full") is None
    assert format_covariance_variant("spherical") == "cov-iso"
    assert format_covariance_variant("diag") == "cov-diag"
    with pytest.raises(ValueError):
        normalize_covariance_type("tied")
    params = parse_sentence_gaussianlda_params({"covariance_type": "iso"})
    assert params.covariance_type == "spherical"
    assert parse_sentence_gaussianlda_params({}).covariance_type == "full"


# --------------------------------------------------------------------------- numerics


@pytest.mark.parametrize("covariance_type", ["spherical", "diag"])
def test_reduced_densities_match_scipy(covariance_type: str) -> None:
    trainer = _train(covariance_type)
    tables = trainer._reduced
    assert isinstance(tables, ReducedCovarianceTables)
    x = trainer.encoded_corpus[1][0]
    ours = trainer.log_multivariate_tdensity_tables(x)
    reference = []
    for table_id in range(NUM_TABLES):
        if covariance_type == "spherical":
            reference.append(
                multivariate_t(
                    loc=trainer.table_means[table_id],
                    shape=tables.scaled_variances[table_id] * np.eye(DIM),
                    df=tables.nu[table_id],
                ).logpdf(x)
            )
        else:
            reference.append(
                np.sum(
                    student_t(
                        df=tables.nu[table_id],
                        loc=trainer.table_means[table_id],
                        scale=np.sqrt(tables.scaled_variances[table_id]),
                    ).logpdf(x)
                )
            )
    np.testing.assert_allclose(ours, np.asarray(reference), rtol=0, atol=1e-10)
    # single-table access agrees with the all-table kernel, also for batched rows
    np.testing.assert_allclose(
        trainer.log_multivariate_tdensity(x, 1), ours[1], atol=1e-12
    )
    batch = trainer.encoded_corpus[1][:2]
    single = trainer.log_multivariate_tdensity(batch, 2)
    assert single.shape == (2,)


@pytest.mark.parametrize("covariance_type", ["spherical", "diag"])
def test_incremental_updates_match_recomputation(covariance_type: str) -> None:
    trainer = _train(covariance_type, sweeps=3)
    tables = trainer._reduced
    params = compute_reduced_table_parameters(
        covariance_type=covariance_type,
        table_counts=trainer.table_counts,
        sum_table_customers=trainer.sum_table_customers,
        sum_squared_table_customers_diag=tables.sum_squared_table_customers_diag,
        prior_mu=trainer.prior.mu,
        kappa=trainer.prior.kappa,
        prior_nu=trainer.prior_nu,
        prior_scale=trainer.prior.scale_sigma,
    )
    np.testing.assert_allclose(params.table_means, trainer.table_means, atol=1e-10)
    np.testing.assert_allclose(params.nu, tables.nu, atol=1e-10)
    np.testing.assert_allclose(
        params.scaled_variances, tables.scaled_variances, atol=1e-10
    )
    np.testing.assert_allclose(
        params.log_determinants, trainer.log_determinants, atol=1e-10
    )
    # squared sums accumulated through remove/add equal the from-scratch sums
    scratch = np.zeros((NUM_TABLES, DIM))
    for doc, assignments in zip(trainer.encoded_corpus, trainer.table_assignments):
        for row, table_id in zip(doc, assignments):
            scratch[table_id] += row * row
    np.testing.assert_allclose(
        scratch, tables.sum_squared_table_customers_diag, atol=1e-10
    )
    assert int(trainer.table_counts.sum()) == sum(
        len(doc) for doc in trainer.encoded_corpus
    )


def test_spherical_empty_table_matches_full_niw_predictive() -> None:
    """With nu_0' = 1 the empty-table predictive of the spherical variant is the full one."""
    rng = np.random.default_rng(1)
    mu0 = rng.normal(size=DIM)
    mu0 /= np.linalg.norm(mu0)
    prior = Wishart(mu0, DIM, kappa=0.1, scale_sigma=0.1)
    counts = np.zeros(NUM_TABLES)
    chol = np.stack([prior.chol_sigma.copy() for _ in range(NUM_TABLES)])
    scaled = build_scaled_cholesky(
        table_counts=counts,
        kappa=0.1,
        embedding_size=DIM,
        table_cholesky_ltriangular_mat=chol,
    )
    log_det_full = np.array(
        [np.sum(np.log(np.diagonal(scaled[k]))) for k in range(NUM_TABLES)]
    )
    x = rng.normal(size=DIM)
    full = log_multivariate_tdensity_tables(
        x,
        embedding_size=DIM,
        nu=build_gaussian_nu(table_counts=counts, embedding_size=DIM),
        table_means=np.tile(mu0, (NUM_TABLES, 1)),
        log_determinants=log_det_full,
        scaled_table_cholesky_ltriangular_mat=scaled,
    )
    params = compute_reduced_table_parameters(
        covariance_type="spherical",
        table_counts=counts,
        sum_table_customers=np.zeros((NUM_TABLES, DIM)),
        sum_squared_table_customers_diag=np.zeros((NUM_TABLES, DIM)),
        prior_mu=mu0,
        kappa=0.1,
        prior_nu=reduced_prior_nu(DIM),
        prior_scale=0.1,
    )
    spherical = log_reduced_tdensity_tables(
        x,
        covariance_type="spherical",
        embedding_size=DIM,
        nu=params.nu,
        table_means=params.table_means,
        log_determinants=params.log_determinants,
        scaled_variances=params.scaled_variances,
    )
    np.testing.assert_allclose(spherical, full, atol=1e-10)
    assert params.nu[0] == pytest.approx(1.0)
    assert params.scaled_variances[0] == pytest.approx(0.1 * 1.1 / 0.1)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
@pytest.mark.parametrize("covariance_type", ["spherical", "diag"])
def test_numba_kernels_match_python_reference(covariance_type: str) -> None:
    trainer = _train(covariance_type)
    tables = trainer._reduced
    doc = trainer.encoded_corpus[0]
    assignments = np.asarray(trainer.table_assignments[0], dtype=np.int64)
    kwargs = dict(
        embedding_size=DIM,
        nu=tables.nu,
        table_means=trainer.table_means,
        log_determinants=trainer.log_determinants,
        scaled_variances=tables.scaled_variances,
    )
    numba_tables = log_reduced_tdensity_tables_kernel(
        doc[0], covariance_type=covariance_type, **kwargs
    )
    python_tables = _log_reduced_tdensity_tables_python(
        doc[0], spherical=covariance_type == "spherical", **kwargs
    )
    np.testing.assert_allclose(numba_tables, python_tables, atol=1e-12)
    numba_ll = accumulate_reduced_log_likelihood_encoded_kernel(
        doc, assignments, covariance_type=covariance_type, **kwargs
    )
    python_ll = _accumulate_reduced_log_likelihood_encoded_python(
        doc, assignments, spherical=covariance_type == "spherical", **kwargs
    )
    assert numba_ll[1] == python_ll[1] == len(doc)
    assert numba_ll[0] == pytest.approx(python_ll[0], abs=1e-10)


# --------------------------------------------------------------------------- trainer


def test_full_covariance_trainer_is_unchanged_by_the_option() -> None:
    explicit = _train("full")
    np.random.seed(3)
    implicit = SentenceGaussianTrainer(
        _corpus(),
        ArrayEncoder(DIM),
        NUM_TABLES,
        0.1,
        0.1,
        save_path=None,
        prior_scale=0.1,
    )
    implicit.sample(2)
    assert np.array_equal(explicit.table_counts, implicit.table_counts)
    assert np.array_equal(explicit.table_means, implicit.table_means)
    assert explicit.average_ll == implicit.average_ll
    assert explicit.table_cholesky_ltriangular_mat is not None
    assert explicit._reduced is None
    assert explicit.prior_nu == pytest.approx(float(DIM))


@pytest.mark.parametrize("covariance_type", ["spherical", "diag"])
def test_reduced_trainer_state_and_artifacts(covariance_type: str, tmp_path) -> None:
    trainer = _train(covariance_type)
    assert trainer.table_cholesky_ltriangular_mat is None
    assert trainer.sum_squared_table_customers is None
    assert trainer.prior_nu == pytest.approx(1.0)
    assert len(trainer.average_ll) == 2 and all(np.isfinite(trainer.average_ll))

    state = snapshot_gaussian_trainer(trainer, include_prior_mu=True)
    assert state.covariance_type == covariance_type
    assert state.prior_nu == pytest.approx(1.0)
    payload = build_gaussian_params_payload(state)
    assert payload["covariance_type"] == covariance_type
    assert payload["prior_nu"] == pytest.approx(1.0)
    names = {
        spec.filename
        for spec in build_gaussian_state_specs(
            train_doc_topic=np.zeros((1, NUM_TABLES)),
            infer_doc_topic=np.zeros((1, NUM_TABLES)),
            trainer=state,
            category="all",
        )
    }
    assert {
        "sum_squared_table_customers_diag.pkl",
        "table_scaled_variances.pkl",
    } <= names
    assert "table_cholesky_ltriangular_mat.pkl" not in names

    persist_gaussian_family_run(
        trainer=state,
        train_doc_topic=trainer.table_counts_per_doc.T,
        infer_doc_topic=np.zeros((1, NUM_TABLES)),
        train_dir=tmp_path / "params",
        infer_dir=tmp_path / "infer",
        category="all",
        additional_specs=[
            PickleArtifactSpec(
                name="prior_mu",
                filename="prior_mu.pkl",
                payload=state.prior_mu,
                split="train",
            )
        ],
    )
    written = json.loads((tmp_path / "params" / "params.json").read_text())
    assert written["covariance_type"] == covariance_type
    loaded = load_sentence_gaussianlda_model(
        param_dir=tmp_path / "params", encoder=ArrayEncoder(DIM)
    ).model
    x = trainer.encoded_corpus[2][0]
    np.testing.assert_allclose(
        loaded.log_multivariate_tdensity_tables(x),
        trainer.log_multivariate_tdensity_tables(x),
        atol=1e-10,
    )
    assert loaded.covariance_type == covariance_type
    assert loaded.table_cholesky_ltriangular_mat is None


def test_full_state_specs_keep_the_nine_pickles() -> None:
    trainer = _train("full")
    state = snapshot_gaussian_trainer(trainer, include_prior_mu=True)
    assert state.covariance_type == "full"
    names = [
        spec.filename
        for spec in build_gaussian_state_specs(
            train_doc_topic=np.zeros((1, NUM_TABLES)),
            infer_doc_topic=np.zeros((1, NUM_TABLES)),
            trainer=state,
            category="all",
        )
    ]
    assert len(names) == 9
    assert {
        "table_inverse_covariances.pkl",
        "sum_squared_table_customers.pkl",
        "table_cholesky_ltriangular_mat.pkl",
    } <= set(names)
    assert "table_scaled_variances.pkl" not in names

"""Result-path identity of the sentence Gaussian LDA covariance variants."""

from __future__ import annotations

import json

import pytest

from src.baselines.adapter_runtime import (
    _baseline_archive_root,
    _baseline_parameter_variant,
    _build_baseline_identity,
    compose_gaussian_parameter_variant,
)
from src.baselines.contracts import BaselineRunRequest
from src.baselines.params import parse_sentence_gaussianlda_params
from src.core.errors import MissingArtifactError
from src.core.path_resolution import resolve_baseline_condition_dir
from src.evaluation.entropy_based.inputs import parameter_variant_for
from src.experiments.config import apply_gaussian_covariance_type_override

BASE_OPTIONS = {
    "train_csvs": ["train.csv"],
    "test_csvs": ["test.csv"],
    "language": "english",
    "text_column": "data",
    "target_column": "target_str",
    "started_at": "2026-09-01T00:00:00+00:00",
    "execution_id": "baseline_20260901T000000Z",
}


def _request(**options: object) -> BaselineRunRequest:
    return BaselineRunRequest(
        name="sentence_gaussianlda",
        category="all",
        dataset="dummy",
        num_topics=5,
        iteration=1,
        options={**BASE_OPTIONS, **options},
    )


def test_parameter_variant_labels() -> None:
    assert (
        compose_gaussian_parameter_variant(
            runner="sentence_gaussianlda", prior_scale=0.1
        )
        == "psi0-0p1"
    )
    assert (
        compose_gaussian_parameter_variant(
            runner="sentence_gaussianlda", prior_scale=0.1, covariance_type="spherical"
        )
        == "psi0-0p1_cov-iso"
    )
    assert (
        compose_gaussian_parameter_variant(
            runner="sentence_gaussianlda", prior_scale=3.0, covariance_type="diag"
        )
        == "psi0-3_cov-diag"
    )
    # the word-level Gaussian LDA never carries a covariance label
    assert (
        compose_gaussian_parameter_variant(
            runner="gaussianlda", prior_scale=0.1, covariance_type="diag"
        )
        == "psi0-0p1"
    )
    assert (
        _baseline_parameter_variant(
            runner="sentence_gaussianlda",
            baseline_params=parse_sentence_gaussianlda_params(
                {"covariance_type": "iso"}
            ),
        )
        == "psi0-0p1_cov-iso"
    )
    assert (
        _baseline_parameter_variant(
            runner="sentence_gaussianlda",
            baseline_params=parse_sentence_gaussianlda_params({}),
        )
        == "psi0-0p1"
    )


def test_full_condition_id_is_unchanged_by_an_explicit_default() -> None:
    implicit, fp_implicit = _build_baseline_identity(
        model="sentence_gaussianlda", request=_request()
    )
    explicit, fp_explicit = _build_baseline_identity(
        model="sentence_gaussianlda", request=_request(covariance_type="full")
    )
    assert implicit == explicit
    assert fp_implicit == fp_explicit
    reduced, fp_reduced = _build_baseline_identity(
        model="sentence_gaussianlda", request=_request(covariance_type="spherical")
    )
    assert reduced != implicit
    assert fp_reduced != fp_implicit


def test_reduced_variants_use_separate_archive_dirs() -> None:
    full = _baseline_archive_root(
        "sentence_gaussianlda", request=_request(prior_scale=0.1)
    )
    iso = _baseline_archive_root(
        "sentence_gaussianlda",
        request=_request(prior_scale=0.1, covariance_type="spherical"),
    )
    diag = _baseline_archive_root(
        "sentence_gaussianlda",
        request=_request(prior_scale=0.1, covariance_type="diag"),
    )
    assert full.parent.name.endswith("_psi0-0p1")
    assert not full.parent.name.endswith("_cov-iso")
    assert iso.parent.name.endswith("_psi0-0p1_cov-iso")
    assert diag.parent.name.endswith("_psi0-0p1_cov-diag")
    assert len({full, iso, diag}) == 3


def test_entropy_parameter_variant_mirror() -> None:
    assert parameter_variant_for("sentence_gaussianlda", 0.1) == "psi0-0p1"
    assert (
        parameter_variant_for("sentence_gaussianlda", 0.1, "diag")
        == "psi0-0p1_cov-diag"
    )
    assert parameter_variant_for("gaussianlda", 0.1, "diag") == "psi0-0p1"
    assert parameter_variant_for("sentlda", 0.1, "diag") is None


def test_reduced_variants_do_not_resolve_to_the_full_covariance_run(tmp_path) -> None:
    """psi0-0p1 has a legacy fallback to the suffix-less dir; a reduced variant must not."""
    latest_root = (
        tmp_path / "dummy" / "default" / "sentence_gaussianlda" / "latest" / "all"
    )
    archive = tmp_path / "archive" / "full"
    archive.mkdir(parents=True)
    full_key = "k5_it0_minilm_norm_psi0-0p1"
    (latest_root / full_key).mkdir(parents=True)
    (latest_root / full_key / "CURRENT.json").write_text(
        json.dumps({"display_key": full_key, "archive_dir": str(archive)})
    )

    resolved_full = resolve_baseline_condition_dir(
        model="sentence_gaussianlda",
        dataset="dummy",
        iteration=0,
        num_topics=5,
        category="all",
        data_run="default",
        embedding_variant="minilm_norm",
        parameter_variant="psi0-0p1",
        baseline_root=tmp_path,
    )
    assert resolved_full == archive

    # The reduced variant has no run, and the psi0-0p1 legacy fallback must not hand it
    # the full-covariance one.
    with pytest.raises(MissingArtifactError):
        resolve_baseline_condition_dir(
            model="sentence_gaussianlda",
            dataset="dummy",
            iteration=0,
            num_topics=5,
            category="all",
            data_run="default",
            embedding_variant="minilm_norm",
            parameter_variant="psi0-0p1_cov-diag",
            baseline_root=tmp_path,
        )


def test_config_override_targets_only_the_sentence_gaussian_lda() -> None:
    raw = {
        "baselines": [
            {"name": "Sentence LDA", "runner": "sentence_gaussianlda"},
            {
                "name": "Gaussian LDA",
                "runner": "gaussianlda",
                "params": {"prior_scale": 0.1},
            },
        ]
    }
    out = apply_gaussian_covariance_type_override(raw, covariance_type="iso")
    assert out["baselines"][0]["params"] == {"covariance_type": "spherical"}
    assert out["baselines"][1]["params"] == {"prior_scale": 0.1}
    assert apply_gaussian_covariance_type_override(raw, covariance_type=None) is raw
    with pytest.raises(ValueError):
        apply_gaussian_covariance_type_override(raw, covariance_type="tied")

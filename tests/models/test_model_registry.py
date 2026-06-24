from __future__ import annotations

from pathlib import Path

from src.models.contracts import ModelArtifacts, ModelRunRequest
from src.models.registry import VMF_RUNNER, VmfRequestOptions, get_model_runner_spec


def test_get_model_runner_spec_includes_vmf_runner() -> None:
    spec = get_model_runner_spec("vmf_sentence_lda")
    assert spec.key == VMF_RUNNER.key
    assert spec.display_name == "vMF Sentence LDA"


def test_get_model_runner_spec_adapts_baseline_registry() -> None:
    spec = get_model_runner_spec("gaussianlda")
    assert spec.display_name == "Gaussian LDA"
    assert spec.family == "gaussianlda"


def test_get_model_runner_spec_adapts_mvtm_baseline_registry() -> None:
    spec = get_model_runner_spec("mvtm")
    assert spec.display_name == "MvTM"
    assert spec.family == "mvtm"


def test_get_model_runner_spec_preserves_baseline_method_kind() -> None:
    spec = get_model_runner_spec("bertopic_kmeans")
    assert spec.display_name == "BERTopic (UMAP + k-means)"
    assert spec.method_kind == "clustering"
    assert spec.is_clustering


def test_model_artifacts_as_dict_preserves_paths() -> None:
    artifacts = ModelArtifacts(
        train_path=Path("train.pkl"),
        infer_path=Path("infer.pkl"),
        extras={"metrics_path": Path("metrics.json")},
    )

    assert artifacts.as_dict() == {
        "train_path": Path("train.pkl"),
        "infer_path": Path("infer.pkl"),
        "metrics_path": Path("metrics.json"),
    }


def test_model_artifacts_expose_shared_dirs_and_lookup() -> None:
    artifacts = ModelArtifacts(
        train_path=Path("results/train.pkl"),
        infer_path=Path("results/test.pkl"),
        extras={
            "train_dir": Path("results/train"),
            "infer_dir": Path("results/infer"),
            "metrics_path": Path("results/metrics.json"),
        },
    )

    assert artifacts.train_dir == Path("results/train")
    assert artifacts.infer_dir == Path("results/infer")
    assert artifacts.get_path("metrics_path") == Path("results/metrics.json")


def test_model_run_request_defaults_options() -> None:
    request = ModelRunRequest(
        name="vmf_sentence_lda",
        category="all",
        dataset="dummy",
        num_topics=10,
        iteration=0,
    )
    assert request.options == {}


def test_vmf_request_options_from_request_options_applies_defaults() -> None:
    logger = object()
    options = VmfRequestOptions.from_request_options(
        {
            "train_csvs": ["train.csv"],
            "test_csvs": ["test.csv"],
            "output_dir": "results/run",
            "logger": logger,
            "encoder_name": "dummy-encoder",
            "encoder_device": "cpu",
            "kappa_default": 10.0,
            "encoder_pre_normalize_transform": "none",
            "encoder_whitening_eps": 1e-5,
            "num_iterations": 15,
            "gibbs_sweeps": 8,
            "num_samples": 4,
            "estimate_alpha": True,
            "alpha_update_every": 2,
            "alpha_max_iter": 50,
            "alpha_tol": 1e-5,
        }
    )

    assert options.train_csvs == ["train.csv"]
    assert options.test_csvs == ["test.csv"]
    assert options.output_dir == Path("results/run")
    assert options.logger is logger
    assert options.language == "english"
    assert options.text_column == "data"
    assert options.target_column == "target_str"
    assert options.num_components == 1
    assert options.soft_temperature == 1.0

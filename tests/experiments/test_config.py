from __future__ import annotations

import typing
from pathlib import Path

import pytest
import yaml

from src.baselines.params import (
    BertopicKMeansParams,
    BleiLdaParams,
    CtmParams,
    EtmParams,
    GaussianKMeansParams,
    GaussianLdaParams,
    GaussianMixtureParams,
    MovMFParams,
    MvTMParams,
    SenCluParams,
    SentenceGaussianLdaParams,
    SentLdaParams,
    SphericalKMeansParams,
)
from src.experiments.config import (
    load_config,
    resolve_model_selection,
    resolve_run_selection,
    resolve_targets,
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def _minimal_config_payload(*, dataset: dict, encoder: dict | None = None) -> dict:
    return {
        "dataset": dataset,
        "train": {
            "num_topics": 20,
            "num_iterations": 10,
        },
        "encoder": encoder or {},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }


def test_load_config_resolves_common_fields() -> None:
    config_path = Path("configs/experiments/20newsgroup.example.yaml")
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_config(config_path)

    assert cfg.dataset.name == raw_config["dataset"]["name"]
    assert cfg.dataset.train_csv.name == Path(raw_config["dataset"]["train_csv"]).name
    assert cfg.encoder.model_name == raw_config["encoder"]["model_name"]
    assert cfg.output_root.name == raw_config["dataset"]["name"]
    assert cfg.preset.kind == raw_config["preset"]["kind"]
    assert cfg.preset.purpose == raw_config["preset"]["purpose"]
    assert cfg.preprocess.segmenter == "delimiter"
    assert cfg.preprocess.tokenizer == "simple"
    assert cfg.train.num_components == raw_config["train"]["num_components"]
    assert cfg.runtime.seed_base == raw_config["runtime"]["seed_base"]
    assert cfg.runtime.num_workers == raw_config["runtime"]["num_workers"]
    assert (
        cfg.vmf.inference.soft_temperature
        == raw_config["vmf"]["inference"]["soft_temperature"]
    )


def test_resolve_run_selection_rejects_unknown_category(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        _minimal_config_payload(
            dataset={
                "name": "dummy",
                "train_csv": "data/dummy/train.csv",
                "test_csv": "data/dummy/test.csv",
                "categories": {
                    "known": ["target-a", "target-b"],
                },
            },
        ),
    )
    cfg = load_config(config_path)

    with pytest.raises(ValueError, match="Unknown category 'missing'"):
        resolve_run_selection(cfg, categories=["missing"])


def test_sentence_embedding_baseline_encoder_defaults_follow_configured_encoder(
    tmp_path: Path,
) -> None:
    payload = _minimal_config_payload(
        dataset={
            "name": "japanese_corpus",
            "train_csv": "data/japanese_corpus/train.csv",
            "test_csv": "data/japanese_corpus/test.csv",
            "categories": {"all": None},
            "language": "ja",
            "text_column": "data",
            "target_column": "target_str",
            "has_labels": True,
        },
        encoder={
            "model_name": "cl-nagoya/ruri-v3-130m",
            "encode_prefix": "topic: ",
        },
    )
    payload["baselines"] = [
        {"name": "Contextual TM", "runner": "ctm"},
        {"name": "Sentence LDA", "runner": "sentence_gaussianlda"},
    ]
    config_path = _write_config(tmp_path, payload)
    cfg = load_config(config_path)
    params_by_runner = {baseline.runner: baseline.params for baseline in cfg.baselines}

    assert cfg.encoder.model_name == "cl-nagoya/ruri-v3-130m"
    assert "senclu" not in params_by_runner
    assert params_by_runner["ctm"].contextual_model_name == cfg.encoder.model_name
    assert params_by_runner["ctm"].contextual_encode_prefix == cfg.encoder.encode_prefix
    assert (
        params_by_runner["sentence_gaussianlda"].encoder_model_name
        == cfg.encoder.model_name
    )
    assert (
        params_by_runner["sentence_gaussianlda"].encode_prefix
        == cfg.encoder.encode_prefix
    )


def test_baseline_encoder_params_override_global_encoder(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "encoder": {
            "model_name": "shared-encoder",
            "encode_prefix": "query: ",
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [
            {
                "runner": "ctm",
                "params": {
                    "contextual_model_name": "explicit-ctm-encoder",
                    "contextual_encode_prefix": "explicit: ",
                },
            },
        ],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert isinstance(cfg.baselines[0].params, CtmParams)
    assert cfg.baselines[0].params.contextual_model_name == "explicit-ctm-encoder"
    assert cfg.baselines[0].params.contextual_encode_prefix == "explicit: "


def test_resolve_run_selection_type_hints_are_evaluable() -> None:
    hints = typing.get_type_hints(resolve_run_selection)

    assert "return" in hints


def test_load_config_supports_runtime_and_vmf_inference_blocks(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "runtime": {"seed_base": 99, "num_workers": 6},
        "vmf": {"inference": {"soft_temperature": 0.7}},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.runtime.seed_base == 99
    assert cfg.runtime.num_workers == 6
    assert cfg.vmf.inference.soft_temperature == 0.7


def test_load_config_supports_extends_override(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    override_path = tmp_path / "override.yaml"
    base_payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"science": ["sci.space"], "sports": ["rec.sport.baseball"]},
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "encoder": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "device": "cpu",
        },
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments/dummy",
    }
    override_payload = {
        "extends": str(base_path),
        "dataset": {"categories": {"__replace__": True, "all": None}},
        "train": {"num_topics": 50, "alpha": 0.1},
    }
    base_path.write_text(yaml.safe_dump(base_payload), encoding="utf-8")
    override_path.write_text(yaml.safe_dump(override_payload), encoding="utf-8")

    cfg = load_config(override_path)

    assert cfg.dataset.name == "dummy"
    assert list(cfg.dataset.categories.keys()) == ["all"]
    assert cfg.train.num_topics == [50]
    assert cfg.train.alpha == 0.1


def test_load_config_from_smoke_directory(tmp_path: Path) -> None:
    payload = {
        "preset": {"kind": "smoke", "purpose": "quantitative"},
        "dataset": {
            "name": "japanese_smoke",
            "train_csv": "data/japanese_smoke/train.csv",
            "test_csv": "data/japanese_smoke/test.csv",
            "categories": {"all": None},
            "delimiter": " / ",
            "language": "ja",
            "text_column": "data",
            "target_column": "target_str",
            "has_labels": True,
        },
        "train": {
            "num_topics": 5,
            "num_iterations": 3,
            "kappa_default": 10.0,
            "gibbs_sweeps": 2,
            "num_samples": 1,
        },
        "encoder": {
            "model_name": "cl-nagoya/ruri-v3-130m",
            "device": "cuda",
            "encode_prefix": "トピック: ",
            "pre_normalize_transform": "mean_center",
        },
        "experiments": {"iterations": [0]},
        "selection": {"models": ["vmf_sentence_lda"]},
        "evaluation": {"tasks": ["classification"], "classifiers": ["svm"]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path = tmp_path / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)

    assert cfg.dataset.name == "japanese_smoke"
    assert cfg.dataset.train_csv.name == "train.csv"
    assert cfg.train.num_topics == [5]
    assert cfg.preset.kind == "smoke"


def test_load_config_supports_qualitative_preset(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "preset": {"kind": "qualitative_allfit", "purpose": "qualitative"},
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.preset.kind == "qualitative_allfit"
    assert cfg.preset.purpose == "qualitative"


def test_load_config_supports_preprocess_block(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "preprocess": {
            "language": "ja",
            "delimiter": " | ",
            "text_column": "body",
            "target_column": None,
            "has_labels": False,
            "segmenter": "pysbd",
            "tokenizer": "mecab",
            "legacy_preprocessing": True,
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.preprocess.language == "ja"
    assert cfg.preprocess.tokenizer == "mecab"
    assert cfg.preprocess.legacy_preprocessing is True


def test_load_config_preprocess_overrides_dataset_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
            "delimiter": " / ",
            "language": "english",
            "text_column": "body",
            "target_column": "label",
            "has_labels": True,
        },
        "preprocess": {
            "delimiter": " | ",
            "language": "ja",
            "text_column": "text",
            "target_column": None,
            "has_labels": False,
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.preprocess.delimiter == " | "
    assert cfg.preprocess.language == "ja"
    assert cfg.preprocess.text_column == "text"
    assert cfg.preprocess.target_column is None
    assert cfg.preprocess.has_labels is False


def test_load_config_keeps_dataset_and_preprocess_responsibilities_separate(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "preprocess": {
            "language": "ja",
            "delimiter": " | ",
            "text_column": "body",
            "target_column": None,
            "has_labels": False,
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.dataset.name == "dummy"
    assert cfg.dataset.categories == {"all": None}
    assert not hasattr(cfg.dataset, "language")
    assert not hasattr(cfg.dataset, "delimiter")
    assert cfg.preprocess.language == "ja"
    assert cfg.preprocess.target_column is None
    assert cfg.preprocess.has_labels is False


def test_resolve_targets_uses_preprocess_target_column_and_has_labels(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": str(tmp_path / "train.csv"),
            "test_csv": str(tmp_path / "test.csv"),
            "categories": {"all": None},
        },
        "preprocess": {
            "target_column": None,
            "has_labels": False,
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    (tmp_path / "train.csv").write_text("data,target_str\nx,a\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("data,target_str\ny,a\n", encoding="utf-8")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert resolve_targets(cfg.dataset, cfg.preprocess, "all", None) is None


def test_load_config_normalizes_default_japanese_tokenizer(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
            "language": "ja",
        },
        "preprocess": {
            "language": "ja",
            "tokenizer": "default",
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.preprocess.tokenizer == "mecab"


def test_load_config_preserves_etm_random_state_semantics(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [
            {"runner": "etm"},
            {"runner": "etm", "params": {"random_state": "7"}},
        ],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert isinstance(cfg.baselines[0].params, EtmParams)
    assert cfg.baselines[0].params.random_state is None
    assert isinstance(cfg.baselines[1].params, EtmParams)
    assert cfg.baselines[1].params.random_state == 7


def test_load_config_supports_num_components_variant(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {
            "num_topics": 20,
            "num_iterations": 3,
            "num_components": 4,
            "estimate_alpha": False,
        },
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.train.num_components == 4
    assert cfg.train.estimate_alpha is False


def test_load_config_supports_periodic_vmf_diagnostics(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {
            "num_topics": 20,
            "num_iterations": 3,
            "avg_log_likelihood_every": 2,
            "invariant_check_every": 3,
        },
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.train.avg_log_likelihood_every == 2
    assert cfg.train.invariant_check_every == 3


def test_load_config_supports_encoder_encode_batch_size(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "encoder": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "device": "cpu",
            "encode_batch_size": 32,
        },
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.encoder.encode_batch_size == 32


def test_load_config_supports_selection_block(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"science": ["sci.space"], "sports": ["rec.sport.baseball"]},
        },
        "selection": {
            "models": ["vmf_sentence_lda", "ctm"],
            "categories": ["science"],
            "topics": [10, 20],
            "iterations": [1, 3],
        },
        "evaluation": {
            "tasks": ["classification"],
            "classifiers": ["svm", "logreg"],
            "alignment_mode": "strict_skip",
            "embedding_variants": ["mpnet", "e5"],
            "feature_resolve_mode": "strict",
        },
        "train": {"num_topics": 50, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)
    categories, topics, iterations = resolve_run_selection(cfg)

    assert cfg.selection.models == ["vmf_sentence_lda", "ctm"]
    assert cfg.selection.categories == ["science"]
    assert cfg.selection.topics == [10, 20]
    assert cfg.selection.iterations == [1, 3]
    assert cfg.evaluation.tasks == ["classification"]
    assert cfg.evaluation.classifiers == ["svm", "logreg"]
    assert cfg.evaluation.alignment_mode == "strict_skip"
    assert cfg.evaluation.embedding_variants == ["mpnet", "e5"]
    assert cfg.evaluation.feature_resolve_mode == "strict"
    assert list(categories.keys()) == ["science"]
    assert topics == [10, 20]
    assert iterations == [1, 3]


def test_load_config_normalizes_evaluation_task_names(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "evaluation": {"tasks": ["word-based-metrics", "classification_summary"]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.evaluation.tasks == ["word_based_metrics", "classification_summary"]


def test_load_config_normalizes_known_baseline_param_types(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "encoder": {
            "model_name": "shared-encoder",
            "encode_prefix": "query: ",
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [
            {"runner": "bleilda", "params": {"passes": "21", "num_iterations": "60"}},
            {
                "runner": "ctm",
                "params": {
                    "num_epochs": "12",
                    "num_samples": "7",
                    "batch_size_cap": "32",
                },
            },
            {
                "runner": "senclu",
                "params": {
                    "alpha": "0.25",
                    "num_epochs": "9",
                    "soft_temperature": "0.8",
                    "verbose": True,
                    "embedding_cache_dir": 123,
                },
            },
            {
                "runner": "gaussianlda",
                "params": {
                    "word2vec": "wikientvec:20190520:jawiki.word_vectors.100d.txt.bz2",
                    "wikientvec_cache_dir": 123,
                    "num_iterations": "15",
                },
            },
            {
                "runner": "sentence_gaussianlda",
                "params": {
                    "encoder_model_name": "sentence-transformers/all-minilm-l6-v2",
                    "num_iterations": "11",
                    "num_gibbs_iters": "9",
                    "encode_batch_size": "32",
                    "preencode_corpus": False,
                    "soft_temperature": "0.7",
                    "strip_terminal_normalize": "false",
                },
            },
            {
                "runner": "sentlda",
                "params": {
                    "num_iterations": "25",
                    "alpha": "0.2",
                    "beta": "0.05",
                    "random_state": "7",
                    "infer_num_iterations": "14",
                    "save_phi": False,
                    "backend": "python",
                },
            },
            {
                "runner": "bertopic_kmeans",
                "params": {
                    "umap_n_neighbors": "9",
                    "umap_n_components": "3",
                    "umap_min_dist": "0.05",
                    "kmeans_n_init": "4",
                    "soft_temperature": "0.6",
                    "random_state": None,
                },
            },
            {
                "runner": "spherical_kmeans",
                "params": {"n_init": "3", "max_iter": "44", "tol": "0.01"},
            },
            {
                "runner": "gaussian_kmeans",
                "params": {
                    "n_init": "4",
                    "max_iter": "55",
                    "tol": "0.02",
                    "strip_terminal_normalize": False,
                },
            },
            {
                "runner": "movmf",
                "params": {
                    "n_init": "2",
                    "max_iter": "33",
                    "tol": "0.03",
                    "min_kappa": "0.2",
                    "max_kappa": "20",
                },
            },
            {
                "runner": "gaussian_mixture",
                "params": {
                    "n_init": "6",
                    "max_iter": "66",
                    "tol": "0.04",
                    "covariance_type": "spherical",
                    "reg_covar": "0.001",
                    "strip_terminal_normalize": True,
                },
            },
        ],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert isinstance(cfg.baselines[0].params, BleiLdaParams)
    assert cfg.baselines[0].params.passes == 21
    assert cfg.baselines[0].params.num_iterations == 60
    assert isinstance(cfg.baselines[1].params, CtmParams)
    assert cfg.baselines[1].params.contextual_model_name == "shared-encoder"
    assert cfg.baselines[1].params.contextual_encode_prefix == "query: "
    assert cfg.baselines[1].params.num_epochs == 12
    assert cfg.baselines[1].params.num_samples == 7
    assert cfg.baselines[1].params.batch_size_cap == 32
    assert isinstance(cfg.baselines[2].params, SenCluParams)
    assert cfg.baselines[2].params.encoder_model_name == "shared-encoder"
    assert cfg.baselines[2].params.encode_prefix == "query: "
    assert cfg.baselines[2].params.alpha == 0.25
    assert cfg.baselines[2].params.num_epochs == 9
    assert cfg.baselines[2].params.soft_temperature == 0.8
    assert cfg.baselines[2].params.verbose is True
    assert cfg.baselines[2].params.embedding_cache_dir == "123"
    assert isinstance(cfg.baselines[3].params, GaussianLdaParams)
    assert (
        cfg.baselines[3].params.word2vec
        == "wikientvec:20190520:jawiki.word_vectors.100d.txt.bz2"
    )
    assert cfg.baselines[3].params.wikientvec_cache_dir == "123"
    assert cfg.baselines[3].params.num_iterations == 15
    assert isinstance(cfg.baselines[4].params, SentenceGaussianLdaParams)
    assert (
        cfg.baselines[4].params.encoder_model_name
        == "sentence-transformers/all-minilm-l6-v2"
    )
    assert cfg.baselines[4].params.encode_prefix == "query: "
    assert cfg.baselines[4].params.num_iterations == 11
    assert cfg.baselines[4].params.num_gibbs_iters == 9
    assert cfg.baselines[4].params.encode_batch_size == 32
    assert cfg.baselines[4].params.preencode_corpus is False
    assert cfg.baselines[4].params.soft_temperature == 0.7
    assert cfg.baselines[4].params.strip_terminal_normalize is False
    assert isinstance(cfg.baselines[5].params, SentLdaParams)
    assert cfg.baselines[5].params.num_iterations == 25
    assert cfg.baselines[5].params.alpha == 0.2
    assert cfg.baselines[5].params.beta == 0.05
    assert cfg.baselines[5].params.random_state == 7
    assert cfg.baselines[5].params.infer_num_iterations == 14
    assert cfg.baselines[5].params.save_phi is False
    assert cfg.baselines[5].params.backend == "python"
    assert isinstance(cfg.baselines[6].params, BertopicKMeansParams)
    assert cfg.baselines[6].params.encoder_model_name == "shared-encoder"
    assert cfg.baselines[6].params.encode_prefix == "query: "
    assert cfg.baselines[6].params.encode_batch_size == 128
    assert cfg.baselines[6].params.umap_n_neighbors == 9
    assert cfg.baselines[6].params.umap_n_components == 3
    assert cfg.baselines[6].params.umap_min_dist == 0.05
    assert cfg.baselines[6].params.kmeans_n_init == 4
    assert cfg.baselines[6].params.soft_temperature == 0.6
    assert cfg.baselines[6].params.random_state is None
    assert isinstance(cfg.baselines[7].params, SphericalKMeansParams)
    assert cfg.baselines[7].params.encoder_model_name == "shared-encoder"
    assert cfg.baselines[7].params.encode_prefix == "query: "
    assert cfg.baselines[7].params.n_init == 3
    assert cfg.baselines[7].params.max_iter == 44
    assert cfg.baselines[7].params.tol == 0.01
    assert isinstance(cfg.baselines[8].params, GaussianKMeansParams)
    assert cfg.baselines[8].params.n_init == 4
    assert cfg.baselines[8].params.max_iter == 55
    assert cfg.baselines[8].params.tol == 0.02
    assert cfg.baselines[8].params.strip_terminal_normalize is False
    assert isinstance(cfg.baselines[9].params, MovMFParams)
    assert cfg.baselines[9].params.n_init == 2
    assert cfg.baselines[9].params.max_iter == 33
    assert cfg.baselines[9].params.tol == 0.03
    assert cfg.baselines[9].params.min_kappa == 0.2
    assert cfg.baselines[9].params.max_kappa == 20
    assert isinstance(cfg.baselines[10].params, GaussianMixtureParams)
    assert cfg.baselines[10].params.n_init == 6
    assert cfg.baselines[10].params.max_iter == 66
    assert cfg.baselines[10].params.tol == 0.04
    assert cfg.baselines[10].params.covariance_type == "spherical"
    assert cfg.baselines[10].params.reg_covar == 0.001
    assert cfg.baselines[10].params.strip_terminal_normalize is True


def test_load_config_applies_encoder_overrides_to_default_baseline_params(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "encoder": {"model_name": "sentence-transformers/all-mpnet-base-v2"},
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [
            {"runner": "sentence_gaussianlda"},
            {"runner": "gaussian_kmeans"},
            {"runner": "gaussian_mixture"},
        ],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(
        config_path,
        encoder_model="baai/bge-base-en-v1.5",
        strip_terminal_normalize=False,
    )

    assert cfg.encoder.model_name == "baai/bge-base-en-v1.5"
    assert cfg.encoder.strip_terminal_normalize is False
    assert cfg.baselines[0].params.encoder_model_name == "baai/bge-base-en-v1.5"
    assert cfg.baselines[0].params.strip_terminal_normalize is False
    assert cfg.baselines[1].params.encoder_model_name == "baai/bge-base-en-v1.5"
    assert cfg.baselines[1].params.strip_terminal_normalize is False
    assert cfg.baselines[2].params.encoder_model_name == "baai/bge-base-en-v1.5"
    assert cfg.baselines[2].params.strip_terminal_normalize is False


def test_load_config_parses_mvtm_params(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "train": {"num_topics": 20, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [
            {
                "runner": "mvtm",
                "params": {
                    "word2vec": "glove-wiki-gigaword-50",
                    "num_components": "2",
                    "num_iterations": "7",
                    "alpha": None,
                    "estimate_alpha": False,
                    "gibbs_sweeps": "3",
                    "num_samples": "2",
                },
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert isinstance(cfg.baselines[0].params, MvTMParams)
    assert cfg.baselines[0].params.word2vec == "glove-wiki-gigaword-50"
    assert cfg.baselines[0].params.num_components == 2
    assert cfg.baselines[0].params.num_iterations == 7
    assert cfg.baselines[0].params.alpha is None
    assert cfg.baselines[0].params.estimate_alpha is False
    assert cfg.baselines[0].params.gibbs_sweeps == 3
    assert cfg.baselines[0].params.num_samples == 2


def test_run_selection_prefers_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"science": ["sci.space"], "sports": ["rec.sport.baseball"]},
        },
        "selection": {
            "categories": ["science"],
            "topics": [10],
            "iterations": [1],
        },
        "train": {"num_topics": 50, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)
    categories, topics, iterations = resolve_run_selection(
        cfg,
        categories=["all"],
        num_topics=[30],
        iterations=[2],
    )

    assert list(categories.keys()) == ["all"]
    assert categories["all"] is None
    assert topics == [30]
    assert iterations == [2]


def test_resolve_model_selection_uses_config_when_cli_is_unset(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "selection": {
            "models": ["vmf_sentence_lda", "ctm"],
        },
        "train": {"num_topics": 50, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert resolve_model_selection(cfg) == {"vmf_sentence_lda", "ctm"}


def test_resolve_model_selection_prefers_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "dataset": {
            "name": "dummy",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "categories": {"all": None},
        },
        "selection": {
            "models": ["vmf_sentence_lda"],
        },
        "train": {"num_topics": 50, "num_iterations": 3},
        "experiments": {"iterations": [0]},
        "baselines": [],
        "output_root": "results/experiments",
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(config_path)

    assert resolve_model_selection(cfg, models="ctm,gaussianlda") == {
        "ctm",
        "gaussianlda",
    }

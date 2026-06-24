from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.cli.workflows import run_smoke_workflow
from src.experiments.config import load_config


def _build_japanese_smoke_payload(
    *,
    train_csv: Path,
    test_csv: Path,
    output_root: Path,
) -> dict:
    return {
        "preset": {"kind": "smoke", "purpose": "quantitative"},
        "dataset": {
            "name": "japanese_smoke",
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
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
        "output_root": str(output_root),
    }


def _write_fixture_backed_config(
    *,
    config_path: Path,
    payload: dict,
    train_csv: Path,
    test_csv: Path,
    output_root: Path,
) -> Path:
    payload = dict(payload)
    payload["dataset"] = dict(payload["dataset"])
    payload["dataset"]["train_csv"] = str(train_csv)
    payload["dataset"]["test_csv"] = str(test_csv)
    payload["output_root"] = str(output_root)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_materialize_japanese_smoke_fixture_has_expected_columns(
    materialize_smoke_fixture,
) -> None:
    dataset_dir = materialize_smoke_fixture(
        "japanese_smoke",
        "data/japanese_smoke",
    )

    train = pd.read_csv(dataset_dir / "train.csv")
    test = pd.read_csv(dataset_dir / "test.csv")

    assert list(train.columns) == ["data", "target_str"]
    assert list(test.columns) == list(train.columns)
    assert train["target_str"].notna().all()


def test_load_smoke_configs_with_committed_fixture_data(
    tmp_path: Path,
    materialize_smoke_fixture,
) -> None:
    dataset_dir = materialize_smoke_fixture("japanese_smoke", "data/japanese_smoke")

    config_path = _write_fixture_backed_config(
        config_path=tmp_path / "japanese_smoke.yaml",
        payload=_build_japanese_smoke_payload(
            train_csv=dataset_dir / "train.csv",
            test_csv=dataset_dir / "test.csv",
            output_root=tmp_path / "results" / "japanese_smoke",
        ),
        train_csv=dataset_dir / "train.csv",
        test_csv=dataset_dir / "test.csv",
        output_root=tmp_path / "results" / "japanese_smoke",
    )

    cfg = load_config(config_path)

    assert cfg.preset.kind == "smoke"
    assert cfg.dataset.train_csv.exists()
    assert cfg.dataset.test_csv.exists()
    assert cfg.preprocess.language == "ja"


def test_run_smoke_workflow_accepts_fixture_backed_config(
    tmp_path: Path,
    monkeypatch,
    materialize_smoke_fixture,
) -> None:
    dataset_dir = materialize_smoke_fixture(
        "japanese_smoke",
        "data/japanese_smoke",
    )
    config_path = _write_fixture_backed_config(
        config_path=tmp_path / "japanese_smoke.yaml",
        payload=_build_japanese_smoke_payload(
            train_csv=dataset_dir / "train.csv",
            test_csv=dataset_dir / "test.csv",
            output_root=tmp_path / "results" / "smoke",
        ),
        train_csv=dataset_dir / "train.csv",
        test_csv=dataset_dir / "test.csv",
        output_root=tmp_path / "results" / "smoke",
    )

    captured: dict[str, object] = {}

    def _fake_run_comparison(**kwargs):
        cfg = load_config(kwargs["config_path"])
        captured["dataset"] = cfg.dataset.name
        captured["train_csv"] = cfg.dataset.train_csv
        captured["test_csv"] = cfg.dataset.test_csv
        return tmp_path / "results" / "smoke" / "summary.json"

    monkeypatch.setattr("src.cli.workflows.run_comparison", _fake_run_comparison)

    summary_path = run_smoke_workflow(
        config=config_path,
        models="vmf_sentence_lda",
        seed=7,
        num_workers=1,
        category=["computer"],
        topic=[20],
        iteration=[0],
    )

    assert summary_path == tmp_path / "results" / "smoke" / "summary.json"
    assert captured["dataset"] == "japanese_smoke"
    assert Path(captured["train_csv"]).exists()
    assert Path(captured["test_csv"]).exists()

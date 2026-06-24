from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from src.baselines.adapter_runtime import (
    _baseline_embedding_variant,
    _build_baseline_identity,
    _word_embedding_alias,
)
from src.baselines.adapters import (
    _build_artifacts,
    _build_split_artifacts,
    _save_runner_metadata,
    run_bleilda,
    run_ctm,
    run_etm,
    run_gaussianlda,
    run_mvtm,
    run_senclu,
    run_sentence_gaussianlda,
    run_sentlda,
)
from src.baselines.contracts import (
    BaselineArtifacts,
    BaselineRunnerSpec,
    BaselineRunRequest,
)
from src.baselines.model_kinds import CLUSTERING_RUNNER_KEYS
from src.baselines.params import GaussianKMeansParams
from src.baselines.runners import (
    RUNNERS,
    get_runner_spec,
    run_baseline,
    run_baseline_request,
)
from src.core.artifacts import load_json


@dataclass
class _RunnerFixture:
    name: str
    runner_fn: Callable[[BaselineRunRequest], BaselineArtifacts]
    parse_module_path: str
    train_module_path: str
    infer_module_path: str
    persist_module_path: str
    train_artifact_name: str
    infer_artifact_name: str
    extra_artifacts: dict[str, str]
    expected_pointer_path_parts: tuple[str, ...]
    train_kwargs_to_assert: dict[str, object] = field(default_factory=dict)
    infer_kwargs_to_assert: dict[str, object] = field(default_factory=dict)
    request_options_extra: dict[str, object] = field(default_factory=dict)
    pointer_payload_to_assert: dict[str, object] = field(default_factory=dict)
    absent_infer_kwargs: tuple[str, ...] = ()


def _make_persist_run(fixture: _RunnerFixture, captured: dict[str, object]):
    def _persist(**kwargs):
        captured["persist_kwargs"] = kwargs
        train_dir = kwargs["train_dir"]
        infer_dir = kwargs["infer_dir"]
        train_dir.mkdir(parents=True, exist_ok=True)
        infer_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / fixture.train_artifact_name).write_bytes(b"x")
        (infer_dir / fixture.infer_artifact_name).write_bytes(b"z")
        extras_paths: dict[str, Path] = {}
        for key, filename in fixture.extra_artifacts.items():
            target_dir = infer_dir if key.startswith("test_") else train_dir
            artifact_path = target_dir / filename
            if filename.endswith(".json"):
                artifact_path.write_text("{}", encoding="utf-8")
            elif artifact_path.suffix == "":
                artifact_path.mkdir(parents=True, exist_ok=True)
            else:
                artifact_path.write_bytes(b"y")
            extras_paths[key] = artifact_path
        return BaselineArtifacts(
            train_path=train_dir / fixture.train_artifact_name,
            infer_path=infer_dir / fixture.infer_artifact_name,
            extras=extras_paths,
        )

    return _persist


_SENTENCE_TOPIC_SOFT_EXTRAS = {
    "params_json": "params.json",
    "train_sentence_topic_soft": "all_sentence_topic_soft.pkl",
    "test_sentence_topic_soft": "all_sentence_topic_soft.pkl",
}


_RUNNER_FIXTURES: list[_RunnerFixture] = [
    _RunnerFixture(
        name="ctm",
        runner_fn=run_ctm,
        parse_module_path="src.baselines.adapters.parse_ctm_params",
        train_module_path="src.baselines.adapters.train_ctm",
        infer_module_path="src.baselines.adapters.infer_ctm",
        persist_module_path="src.baselines.adapters.persist_ctm_run",
        train_artifact_name="ctm.pkl",
        infer_artifact_name="all.pkl",
        extra_artifacts={
            "topic_preparation": "tp.pkl",
            "model_dir": "contextualized_topic_model_dummy",
        },
        expected_pointer_path_parts=(
            "dummy",
            "default",
            "ctm",
            "latest",
            "all",
            "k5_it1",
        ),
        train_kwargs_to_assert={"num_topics": 5},
        infer_kwargs_to_assert={"num_topics": 5},
        request_options_extra={
            "data_run": "default",
            "started_at": "2026-04-10T02:15:30+00:00",
            "execution_id": "baseline_20260410T021530Z",
        },
    ),
    _RunnerFixture(
        name="bleilda",
        runner_fn=run_bleilda,
        parse_module_path="src.baselines.adapters.parse_bleilda_params",
        train_module_path="src.baselines.adapters.train_bleilda",
        infer_module_path="src.baselines.adapters.infer_bleilda",
        persist_module_path="src.baselines.adapters.persist_bleilda_run",
        train_artifact_name="lda_comp.pkl",
        infer_artifact_name="all.pkl",
        extra_artifacts={"model_path": "model.gensim"},
        expected_pointer_path_parts=(
            "dummy",
            "default",
            "bleilda",
            "latest",
            "all",
            "k5_it1",
        ),
        train_kwargs_to_assert={"num_topics": 5},
        infer_kwargs_to_assert={"num_topics": 5},
    ),
    _RunnerFixture(
        name="senclu",
        runner_fn=run_senclu,
        parse_module_path="src.baselines.adapters.parse_senclu_params",
        train_module_path="src.baselines.adapters.train_senclu",
        infer_module_path="src.baselines.adapters.infer_senclu",
        persist_module_path="src.baselines.adapters.persist_senclu_run",
        train_artifact_name="all.pkl",
        infer_artifact_name="all.pkl",
        extra_artifacts={
            "train_sentence_topic_soft": "all_sentence_topic_soft.pkl",
            "test_sentence_topic_soft": "all_sentence_topic_soft.pkl",
        },
        expected_pointer_path_parts=(
            "dummy",
            "default",
            "senclu",
            "latest",
            "all",
            "k5_it1",
        ),
        train_kwargs_to_assert={"encoder_device": "cpu"},
        request_options_extra={"encoder_device": "cpu"},
    ),
    _RunnerFixture(
        name="gaussianlda",
        runner_fn=run_gaussianlda,
        parse_module_path="src.baselines.adapters.parse_gaussianlda_params",
        train_module_path="src.baselines.adapters.train_gaussianlda",
        infer_module_path="src.baselines.adapters.infer_gaussianlda",
        persist_module_path="src.baselines.adapters.persist_gaussianlda_run",
        train_artifact_name="table_counts_per_doc.pkl",
        infer_artifact_name="all.pkl",
        extra_artifacts={"params_json": "params.json"},
        expected_pointer_path_parts=(
            "dummy",
            "default",
            "gaussianlda",
            "latest",
            "all",
            "k5_it1_glove100",
        ),
        train_kwargs_to_assert={"num_topics": 5},
        infer_kwargs_to_assert={"num_topics": 5},
        request_options_extra={"word2vec": "glove-wiki-gigaword-100"},
        pointer_payload_to_assert={"embedding_variant": "glove100"},
    ),
    _RunnerFixture(
        name="mvtm",
        runner_fn=run_mvtm,
        parse_module_path="src.baselines.adapters.parse_mvtm_params",
        train_module_path="src.baselines.adapters.train_mvtm",
        infer_module_path="src.baselines.adapters.infer_mvtm",
        persist_module_path="src.baselines.adapters.persist_mvtm_run",
        train_artifact_name="table_counts_per_doc.pkl",
        infer_artifact_name="all.pkl",
        extra_artifacts={"params_json": "params.json"},
        expected_pointer_path_parts=(
            "dummy",
            "default",
            "mvtm",
            "latest",
            "all",
            "k5_it1_c2_glove100",
        ),
        train_kwargs_to_assert={"num_topics": 5},
        infer_kwargs_to_assert={"num_topics": 5},
        request_options_extra={
            "word2vec": "glove-wiki-gigaword-100",
            "num_components": 2,
        },
        pointer_payload_to_assert={"embedding_variant": "glove100"},
    ),
    _RunnerFixture(
        name="sentence_gaussianlda",
        runner_fn=run_sentence_gaussianlda,
        parse_module_path="src.baselines.adapters.parse_sentence_gaussianlda_params",
        train_module_path="src.baselines.adapters.train_sentence_gaussianlda",
        infer_module_path="src.baselines.adapters.infer_sentence_gaussianlda",
        persist_module_path="src.baselines.adapters.persist_sentence_gaussianlda_run",
        train_artifact_name="table_counts_per_doc.pkl",
        infer_artifact_name="all.pkl",
        extra_artifacts=_SENTENCE_TOPIC_SOFT_EXTRAS,
        expected_pointer_path_parts=(
            "dummy",
            "default",
            "sentence_gaussianlda",
            "latest",
            "all",
            "k5_it1",
        ),
        train_kwargs_to_assert={"encoder_device": "cpu"},
        request_options_extra={"encoder_device": "cpu"},
        absent_infer_kwargs=("encoder_device",),
    ),
    _RunnerFixture(
        name="sentlda",
        runner_fn=run_sentlda,
        parse_module_path="src.baselines.adapters.parse_sentlda_params",
        train_module_path="src.baselines.adapters.train_sentlda",
        infer_module_path="src.baselines.adapters.infer_sentlda",
        persist_module_path="src.baselines.adapters.persist_sentlda_run",
        train_artifact_name="table_counts_per_doc.pkl",
        infer_artifact_name="all.pkl",
        extra_artifacts=_SENTENCE_TOPIC_SOFT_EXTRAS,
        expected_pointer_path_parts=(
            "dummy",
            "default",
            "sentlda",
            "latest",
            "all",
            "k5_it1",
        ),
        train_kwargs_to_assert={"num_topics": 5},
        infer_kwargs_to_assert={"num_topics": 5},
        request_options_extra={"num_iterations": 12},
    ),
]


def test_runner_registry_contains_expected_models() -> None:
    assert {
        "ctm",
        "bleilda",
        "bertopic_kmeans",
        "gaussianlda",
        "etm",
        "mvtm",
        "spherical_kmeans",
        "gaussian_kmeans",
        "movmf",
        "gaussian_mixture",
        "senclu",
        "sentlda",
        "sentence_gaussianlda",
    }.issubset(RUNNERS.keys())


def test_runner_registry_marks_clustering_methods_explicitly() -> None:
    clustering_keys = {key for key, spec in RUNNERS.items() if spec.is_clustering}

    assert clustering_keys == set(CLUSTERING_RUNNER_KEYS)
    assert all(RUNNERS[key].method_kind == "clustering" for key in clustering_keys)
    assert all(
        RUNNERS[key].method_kind == "topic_model"
        for key in RUNNERS.keys() - clustering_keys
    )


def test_get_runner_spec_exposes_display_name() -> None:
    spec = get_runner_spec("ctm")
    assert spec.display_name == "Contextual TM"
    assert spec.family == "ctm"
    assert spec.method_kind == "topic_model"


def test_runner_spec_rejects_unknown_method_kind() -> None:
    with pytest.raises(ValueError, match="Unknown runner method_kind"):
        BaselineRunnerSpec(
            key="dummy",
            display_name="Dummy",
            family="dummy",
            runner=lambda _request: BaselineArtifacts(
                train_path=Path("train"),
                infer_path=Path("infer"),
            ),
            method_kind="not_a_kind",
        )


def test_unknown_runner_raises_value_error() -> None:
    with pytest.raises(ValueError):
        run_baseline(
            "missing",
            category="all",
            dataset="20newsgroup",
            num_topics=10,
            iteration=0,
        )


def test_runner_contract_requires_train_and_infer_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        RUNNERS,
        "dummy_missing_keys",
        BaselineRunnerSpec(
            key="dummy_missing_keys",
            display_name="Dummy Missing Keys",
            family="dummy",
            runner=lambda _request: Path("a"),
        ),
    )
    with pytest.raises(TypeError):
        run_baseline(
            "dummy_missing_keys",
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=0,
        )


def test_runner_contract_requires_path_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        RUNNERS,
        "dummy_wrong_type",
        BaselineRunnerSpec(
            key="dummy_wrong_type",
            display_name="Dummy Wrong Type",
            family="dummy",
            runner=lambda _request: BaselineArtifacts(
                train_path=Path("a"),
                infer_path=Path("b"),
            ),
        ),
    )
    artifacts = run_baseline(
        "dummy_wrong_type",
        category="all",
        dataset="dummy",
        num_topics=5,
        iteration=0,
    )
    assert artifacts["train_path"] == Path("a")


def test_run_baseline_request_returns_structured_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        RUNNERS,
        "dummy_structured",
        BaselineRunnerSpec(
            key="dummy_structured",
            display_name="Dummy Structured",
            family="dummy",
            runner=lambda _request: BaselineArtifacts(
                train_path=Path("train"),
                infer_path=Path("infer"),
                extras={"extra_path": Path("extra")},
            ),
        ),
    )

    artifacts = run_baseline_request(
        BaselineRunRequest(
            name="dummy_structured",
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=0,
        )
    )

    assert isinstance(artifacts, BaselineArtifacts)
    assert artifacts.train_path == Path("train")
    assert artifacts.infer_path == Path("infer")
    assert artifacts.extras == {"extra_path": Path("extra")}


def test_baseline_artifacts_fallback_dirs_use_parent_paths() -> None:
    artifacts = BaselineArtifacts(
        train_path=Path("params/train.pkl"),
        infer_path=Path("infer/test.pkl"),
    )

    assert artifacts.train_dir == Path("params")
    assert artifacts.infer_dir == Path("infer")


def test_adapter_build_artifacts_requires_existing_files(tmp_path: Path) -> None:
    train_path = tmp_path / "train.pkl"
    infer_path = tmp_path / "infer.pkl"
    train_path.write_bytes(b"x")

    with pytest.raises(FileNotFoundError):
        _build_artifacts(
            train_path=train_path,
            infer_path=infer_path,
        )


def test_adapter_build_artifacts_keeps_extras_when_files_exist(tmp_path: Path) -> None:
    train_path = tmp_path / "train.pkl"
    infer_path = tmp_path / "infer.pkl"
    extra_path = tmp_path / "extra.pkl"
    train_path.write_bytes(b"x")
    infer_path.write_bytes(b"y")
    extra_path.write_bytes(b"z")

    artifacts = _build_artifacts(
        train_path=train_path,
        infer_path=infer_path,
        extras={"extra_path": extra_path},
    )

    assert artifacts.train_path == train_path
    assert artifacts.infer_path == infer_path
    assert artifacts.extras == {"extra_path": extra_path}


def test_build_split_artifacts_uses_file_paths_and_split_dirs(tmp_path: Path) -> None:
    train_dir = tmp_path / "params" / "all"
    infer_dir = tmp_path / "infer"
    train_dir.mkdir(parents=True)
    infer_dir.mkdir(parents=True)
    (train_dir / "train.pkl").write_bytes(b"x")
    (infer_dir / "test.pkl").write_bytes(b"y")

    artifacts = _build_split_artifacts(
        train_dir=train_dir,
        infer_dir=infer_dir,
        train_filename="train.pkl",
        infer_filename="test.pkl",
    )

    assert artifacts.train_path == train_dir / "train.pkl"
    assert artifacts.infer_path == infer_dir / "test.pkl"
    assert artifacts.extras["train_dir"] == train_dir
    assert artifacts.extras["infer_dir"] == infer_dir


def test_save_runner_metadata_persists_common_baseline_fields(tmp_path: Path) -> None:
    train_dir = tmp_path / "params" / "all"
    infer_dir = tmp_path / "infer"
    train_dir.mkdir(parents=True)
    infer_dir.mkdir(parents=True)

    metadata_path = _save_runner_metadata(
        request=BaselineRunRequest(
            name="senclu",
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=2,
            options={
                "train_csvs": ["train.csv"],
                "test_csvs": ["test.csv"],
                "targets": ["a", "b"],
                "language": "ja",
                "delimiter": " / ",
                "segmenter": "delimiter",
                "tokenizer": "mecab",
                "text_column": "data",
                "target_column": "target_str",
                "legacy_preprocessing": None,
                "ja_replace_num": False,
                "ja_stopwords_path": "stopwords.txt",
                "ja_dicdir": "neologd",
                "ja_require_unidic": False,
                "encoder_device": "cpu",
                "runtime_num_workers": 1,
                "doc_topic_source": "umap_kmeans_centroid_softmax",
                "doc_topic_space": "umap",
            },
        ),
        runner_family="senclu",
        train_dir=train_dir,
        infer_dir=infer_dir,
    )

    payload = metadata_path.read_text(encoding="utf-8")

    assert metadata_path == train_dir / "metadata.json"
    config_payload = load_json(train_dir / "config.json")
    assert config_payload["model_name"] == "senclu"
    assert config_payload["num_topics"] == 5
    assert config_payload["iteration"] == 2
    assert '"schema": "baseline_artifact_metadata"' in payload
    assert '"schema_version": 1' in payload
    assert '"runner_key": "senclu"' in payload
    assert '"method_kind": "topic_model"' in payload
    assert (
        '"parameter_variant": '
        '"alpha=none__embedding_cache_dir=none__encode_prefix=none__encoder_model_name=sentence-transformers/all-mpnet-base-v2__num_epochs=40__soft_temperature=1.0__verbose=false"'
        in payload
    )
    assert (
        '"preprocessing_variant": "language=ja__delimiter= / __segmenter=delimiter__tokenizer=mecab__legacy_preprocessing=auto__text_column=data__target_column=target_str__ja_replace_num=false__ja_require_unidic=false__ja_stopwords_path=stopwords.txt__ja_dicdir=neologd"'
        in payload
    )
    assert '"encoder_device": "cpu"' in payload
    assert '"runtime_num_workers": 1' in payload
    assert '"doc_topic_source": "umap_kmeans_centroid_softmax"' in payload
    assert '"doc_topic_space": "umap"' in payload


def test_gaussian_embedding_variant_marks_terminal_normalize_mode() -> None:
    stripped = _baseline_embedding_variant(
        runner="gaussian_kmeans",
        baseline_params=GaussianKMeansParams(
            encoder_model_name="sentence-transformers/all-mpnet-base-v2",
            strip_terminal_normalize=True,
        ),
    )
    kept = _baseline_embedding_variant(
        runner="gaussian_kmeans",
        baseline_params=GaussianKMeansParams(
            encoder_model_name="sentence-transformers/all-mpnet-base-v2",
            strip_terminal_normalize=False,
        ),
    )
    kept_from_string = _baseline_embedding_variant(
        runner="gaussian_kmeans",
        baseline_params={
            "encoder_model_name": "sentence-transformers/all-mpnet-base-v2",
            "encoder_backend": "sentence_transformers",
            "strip_terminal_normalize": "false",
        },
    )

    assert stripped == "mpnet_raw"
    assert kept == "mpnet_norm"
    assert kept_from_string == "mpnet_norm"


def test_word_embedding_aliases_use_short_model_labels() -> None:
    assert _word_embedding_alias("glove-wiki-gigaword-100") == "glove100"
    assert _word_embedding_alias("glove-wiki-gigaword-50") == "glove50"
    assert (
        _word_embedding_alias("wikientvec:20190520:jawiki.word_vectors.100d.txt.bz2")
        == "wikient100"
    )
    assert _word_embedding_alias("local") == "local"


def test_word_embedding_baselines_get_embedding_variants() -> None:
    assert (
        _baseline_embedding_variant(
            runner="gaussianlda",
            baseline_params={"word2vec": "glove-wiki-gigaword-100"},
        )
        == "glove100"
    )
    assert (
        _baseline_embedding_variant(
            runner="etm",
            baseline_params={"word2vec": "glove-wiki-gigaword-50"},
        )
        == "glove50"
    )
    assert (
        _baseline_embedding_variant(
            runner="mvtm",
            baseline_params={
                "word2vec": "wikientvec:20190520:jawiki.word_vectors.100d.txt.bz2"
            },
        )
        == "wikient100"
    )


def test_word_embedding_metadata_records_embedding_config(tmp_path: Path) -> None:
    train_dir = tmp_path / "params"
    infer_dir = tmp_path / "infer"
    train_dir.mkdir()
    infer_dir.mkdir()

    metadata_path = _save_runner_metadata(
        request=BaselineRunRequest(
            name="gaussianlda",
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=1,
            options={
                "train_csvs": ["train.csv"],
                "test_csvs": ["test.csv"],
                "language": "english",
                "text_column": "data",
                "target_column": "target_str",
                "word2vec": "glove-wiki-gigaword-100",
            },
        ),
        runner_family="gaussianlda",
        train_dir=train_dir,
        infer_dir=infer_dir,
    )

    payload = load_json(metadata_path)

    assert payload["embedding_variant"] == "glove100"
    assert payload["encoder_config"] == {
        "embedding_type": "word_vectors",
        "word2vec": "glove-wiki-gigaword-100",
        "wikientvec_cache_dir": None,
        "embedding_variant": "glove100",
    }


def test_gaussian_metadata_records_terminal_normalize_setting(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "params"
    infer_dir = tmp_path / "infer"
    train_dir.mkdir()
    infer_dir.mkdir()

    metadata_path = _save_runner_metadata(
        request=BaselineRunRequest(
            name="gaussian_kmeans",
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=1,
            options={
                "train_csvs": ["train.csv"],
                "test_csvs": ["test.csv"],
                "language": "english",
                "text_column": "data",
                "target_column": "target_str",
                "encoder_model_name": "sentence-transformers/all-mpnet-base-v2",
                "strip_terminal_normalize": False,
            },
        ),
        runner_family="gaussian_kmeans",
        train_dir=train_dir,
        infer_dir=infer_dir,
    )

    payload = load_json(metadata_path)

    assert payload["encoder_config"]["strip_terminal_normalize"] is False
    assert payload["baseline_params"]["strip_terminal_normalize"] is False
    assert "strip_terminal_normalize=false" in payload["parameter_variant"]


def test_gaussian_identity_changes_with_terminal_normalize_setting() -> None:
    base_options = {
        "train_csvs": ["train.csv"],
        "test_csvs": ["test.csv"],
        "language": "english",
        "text_column": "data",
        "target_column": "target_str",
        "encoder_model_name": "sentence-transformers/all-mpnet-base-v2",
    }
    stripped, _ = _build_baseline_identity(
        model="gaussian_kmeans",
        request=BaselineRunRequest(
            name="gaussian_kmeans",
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=1,
            options={**base_options, "strip_terminal_normalize": True},
        ),
    )
    kept, _ = _build_baseline_identity(
        model="gaussian_kmeans",
        request=BaselineRunRequest(
            name="gaussian_kmeans",
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=1,
            options={**base_options, "strip_terminal_normalize": False},
        ),
    )

    assert stripped != kept


@pytest.mark.parametrize("fixture", _RUNNER_FIXTURES, ids=lambda f: f.name)
def test_runner_uses_first_party_pipeline(
    fixture: _RunnerFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("src.core.paths.BASELINE_RESULTS_ROOT", tmp_path)

    monkeypatch.setattr(
        "src.baselines.adapters.use_legacy_category_behavior",
        lambda dataset, language: (
            captured.setdefault("legacy", (dataset, language)),
            False,
        )[1],
    )
    monkeypatch.setattr(
        fixture.parse_module_path,
        lambda options: captured.setdefault("params", dict(options)) or object(),
    )
    monkeypatch.setattr(
        fixture.train_module_path,
        lambda **kwargs: captured.setdefault("train_kwargs", kwargs) or object(),
    )
    monkeypatch.setattr(
        fixture.infer_module_path,
        lambda **kwargs: captured.setdefault("infer_kwargs", kwargs) or object(),
    )
    monkeypatch.setattr(
        fixture.persist_module_path,
        _make_persist_run(fixture, captured),
    )
    monkeypatch.setattr(
        "src.baselines.adapters._save_runner_metadata",
        lambda **kwargs: (
            (kwargs["train_dir"] / "metadata.json").write_text("{}", encoding="utf-8"),
            kwargs["train_dir"] / "metadata.json",
        )[1],
    )

    options = {
        "train_csvs": ["train.csv"],
        "test_csvs": ["test.csv"],
        "language": "english",
        "text_column": "data",
        "target_column": "target_str",
        **fixture.request_options_extra,
    }
    artifacts = fixture.runner_fn(
        BaselineRunRequest(
            name=fixture.name,
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=1,
            options=options,
        )
    )

    assert captured["legacy"] == ("dummy", "english")
    for key, expected in fixture.train_kwargs_to_assert.items():
        assert captured["train_kwargs"][key] == expected
    for key, expected in fixture.infer_kwargs_to_assert.items():
        assert captured["infer_kwargs"][key] == expected
    for key in fixture.absent_infer_kwargs:
        assert key not in captured["infer_kwargs"]
    assert artifacts.train_path.name == fixture.train_artifact_name
    assert artifacts.infer_path.name == fixture.infer_artifact_name
    for extras_key, expected_name in fixture.extra_artifacts.items():
        assert artifacts.extras[extras_key].name == expected_name
    latest_pointer = (
        tmp_path.joinpath(*fixture.expected_pointer_path_parts) / "CURRENT.json"
    )
    assert latest_pointer.exists()
    pointer_payload = load_json(latest_pointer)
    assert pointer_payload["display_key"] == fixture.expected_pointer_path_parts[-1]
    for key, expected in fixture.pointer_payload_to_assert.items():
        assert pointer_payload[key] == expected


def test_run_ctm_respects_explicit_legacy_preprocessing_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.baselines.adapters.use_legacy_category_behavior",
        lambda _dataset, _language: (_ for _ in ()).throw(
            AssertionError("auto legacy detection should not be called")
        ),
    )
    monkeypatch.setattr(
        "src.baselines.adapters.parse_ctm_params",
        lambda options: captured.setdefault("params", dict(options)) or object(),
    )
    monkeypatch.setattr(
        "src.baselines.adapters.train_ctm",
        lambda **kwargs: captured.setdefault("train_kwargs", kwargs) or object(),
    )
    monkeypatch.setattr(
        "src.baselines.adapters.infer_ctm",
        lambda **kwargs: captured.setdefault("infer_kwargs", kwargs) or object(),
    )
    monkeypatch.setattr(
        "src.baselines.adapters.persist_ctm_run",
        lambda **_kwargs: BaselineArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
        ),
    )
    monkeypatch.setattr(
        "src.baselines.adapters._save_runner_metadata",
        lambda **_kwargs: Path("metadata.json"),
    )
    monkeypatch.setattr(
        "src.baselines.adapters._build_persisted_artifacts",
        lambda **kwargs: kwargs["artifacts"],
    )

    run_ctm(
        BaselineRunRequest(
            name="ctm",
            category="all",
            dataset="20newsgroup",
            num_topics=5,
            iteration=1,
            options={
                "train_csvs": ["train.csv"],
                "test_csvs": ["test.csv"],
                "language": "english",
                "text_column": "data",
                "target_column": "target_str",
                "legacy_preprocessing": False,
            },
        )
    )

    assert captured["train_kwargs"]["use_legacy"] is False
    assert captured["infer_kwargs"]["use_legacy"] is False


def test_run_etm_passes_device_and_effective_random_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("src.core.paths.BASELINE_RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        "src.baselines.adapters.use_legacy_category_behavior",
        lambda dataset, language: (
            captured.setdefault("legacy", (dataset, language)),
            False,
        )[1],
    )
    monkeypatch.setattr(
        "src.baselines.adapters.parse_etm_params",
        lambda options: captured.setdefault("params", dict(options)) or object(),
    )
    monkeypatch.setattr(
        "src.baselines.adapters.train_etm",
        lambda **kwargs: captured.setdefault("train_kwargs", kwargs) or object(),
    )
    monkeypatch.setattr(
        "src.baselines.adapters.infer_etm",
        lambda **kwargs: captured.setdefault("infer_kwargs", kwargs) or object(),
    )

    def _persist_etm_run(**kwargs):
        captured["persist_kwargs"] = kwargs
        train_dir = kwargs["train_dir"]
        infer_dir = kwargs["infer_dir"]
        train_dir.mkdir(parents=True, exist_ok=True)
        infer_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "etm.pkl").write_bytes(b"x")
        (train_dir / "model_state.pt").write_bytes(b"state")
        (train_dir / "vocabulary.json").write_text("[]", encoding="utf-8")
        (train_dir / "topic_word_scores.pkl").write_bytes(b"scores")
        (infer_dir / "all.pkl").write_bytes(b"y")
        (infer_dir / "all_doc_topic_soft.pkl").write_bytes(b"z")
        return BaselineArtifacts(
            train_path=train_dir / "etm.pkl",
            infer_path=infer_dir / "all.pkl",
            extras={
                "model_state": train_dir / "model_state.pt",
                "vocabulary": train_dir / "vocabulary.json",
                "topic_word_scores": train_dir / "topic_word_scores.pkl",
                "test_doc_topic_soft": infer_dir / "all_doc_topic_soft.pkl",
            },
        )

    monkeypatch.setattr("src.baselines.adapters.persist_etm_run", _persist_etm_run)

    def _save_metadata(**kwargs):
        metadata_path = kwargs["train_dir"] / "metadata.json"
        metadata_path.write_text("{}", encoding="utf-8")
        return metadata_path

    monkeypatch.setattr("src.baselines.adapters._save_runner_metadata", _save_metadata)

    artifacts = run_etm(
        BaselineRunRequest(
            name="etm",
            category="all",
            dataset="dummy",
            num_topics=5,
            iteration=1,
            options={
                "train_csvs": ["train.csv"],
                "test_csvs": ["test.csv"],
                "language": "english",
                "text_column": "data",
                "target_column": "target_str",
                "encoder_device": "cpu",
                "effective_random_state": 123,
            },
        )
    )

    assert captured["legacy"] == ("dummy", "english")
    assert captured["train_kwargs"]["encoder_device"] == "cpu"
    assert captured["train_kwargs"]["effective_random_state"] == 123
    assert captured["infer_kwargs"]["num_topics"] == 5
    assert artifacts.train_path.name == "etm.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert artifacts.extras["test_doc_topic_soft"].name == "all_doc_topic_soft.pkl"
    pointer_path = (
        tmp_path
        / "dummy"
        / "default"
        / "etm"
        / "latest"
        / "all"
        / "k5_it1_glove100"
        / "CURRENT.json"
    )
    assert pointer_path.exists()
    pointer = load_json(pointer_path)
    assert pointer["embedding_variant"] == "glove100"
    assert pointer["encoder_config"]["embedding_type"] == "word_vectors"

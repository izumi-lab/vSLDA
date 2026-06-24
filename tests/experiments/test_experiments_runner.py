from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.params import (
    BertopicKMeansParams,
    CtmParams,
    EtmParams,
    SentLdaParams,
)
from src.core.artifacts import load_json
from src.core.runtime import BaselineRuntimeContext, CorpusSelection, PreprocessRuntime
from src.experiments.comparison_runner import run_comparison
from src.experiments.config import BaselineConfig
from src.experiments.execution import _build_vmf_run_options, run_baselines_for_category
from src.experiments.summary_schema import BaselineSummary
from src.models.contracts import ModelArtifacts
from src.utils.random import DEFAULT_RANDOM_SEED


def test_run_comparison_rejects_non_positive_num_workers() -> None:
    with pytest.raises(ValueError):
        run_comparison(
            config_path="configs/experiments/20newsgroup.example.yaml",
            num_workers=0,
        )


def test_run_comparison_defaults_seed_base(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.experiments.comparison_runner.load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            output_root=Path("results/experiments/dummy"),
            dataset=SimpleNamespace(name="dummy"),
            encoder=SimpleNamespace(device="cpu"),
        ),
    )
    monkeypatch.setattr(
        "src.experiments.comparison_runner.resolve_model_selection",
        lambda _cfg, models=None: {"vmf_sentence_lda"},
    )
    monkeypatch.setattr(
        "src.experiments.comparison_runner.resolve_parallelism_plan",
        lambda **_kwargs: SimpleNamespace(category_num_workers=1),
    )
    monkeypatch.setattr(
        "src.experiments.comparison_runner.resolve_run_selection",
        lambda *_args, **_kwargs: ({"all": None}, [10], [0]),
    )
    monkeypatch.setattr(
        "src.experiments.comparison_runner.resolve_data_runs",
        lambda _cfg: [],
    )
    monkeypatch.setattr(
        "src.experiments.comparison_runner.build_jobs",
        lambda **kwargs: captured.update(kwargs) or [],
    )
    monkeypatch.setattr(
        "src.experiments.comparison_runner.save_json",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.experiments.comparison_runner.build_summary_payload",
        lambda **_kwargs: {},
    )

    run_comparison(config_path="dummy.yaml")

    assert captured["seed"] is None
    assert captured["seed_base"] == DEFAULT_RANDOM_SEED


def test_process_category_uses_explicit_seed_for_each_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[int] = []

    monkeypatch.setattr(
        "src.experiments.execution.set_global_seed",
        lambda seed, deterministic_torch=False: captured.append(seed),
    )

    from src.experiments.execution import _process_category_impl
    from src.experiments.job_planning import CategoryJob, ParallelismPlan

    job = CategoryJob(
        data_run_name="default",
        train_csvs=(Path("train.csv"),),
        test_csvs=(Path("test.csv"),),
        fiscal_years=None,
        category="all",
        targets=None,
        num_topics=10,
        iteration=3,
        baselines=[],
        selected_models=set(),
        seed=77,
        seed_base=DEFAULT_RANDOM_SEED,
        parallelism=ParallelismPlan(
            requested_num_workers=2,
            category_num_workers=2,
            baseline_num_workers=1,
            encoder_device="cpu",
            run_vmf=False,
            uses_cuda=False,
        ),
        config=SimpleNamespace(
            dataset=SimpleNamespace(name="dummy"),
            train=SimpleNamespace(
                num_components=2,
                estimate_alpha=False,
                alpha_update_every=1,
            ),
            encoder=SimpleNamespace(
                model_name="dummy-encoder",
                pre_normalize_transform="none",
            ),
            preprocess=SimpleNamespace(
                delimiter=" / ",
                language="en",
                segmenter="delimiter",
                tokenizer="default",
                text_column="data",
                target_column="target_str",
                has_labels=True,
                ja_replace_num=False,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=False,
            ),
            baselines=[],
        ),
        vmf_soft_temp=1.0,
    )

    _process_category_impl(job)

    assert captured == [77]


def test_run_baselines_for_category_passes_preprocess_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_request(request):
        captured["options"] = request.options
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(
            '{"runner_key":"gaussianlda","runner_family":"gaussianlda","parameter_variant":"default","preprocessing_variant":"language=ja","baseline_params":{}}',
            encoding="utf-8",
        )
        return ModelArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
            extras={"metadata": metadata_path},
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request",
        _fake_run_model_request,
    )

    results = run_baselines_for_category(
        category="all",
        dataset="dummy",
        runtime=BaselineRuntimeContext(
            corpus=CorpusSelection(
                train_csvs=(Path("train.csv"),),
                test_csvs=(Path("test.csv"),),
                targets=None,
            ),
            preprocess=PreprocessRuntime(
                text_column="data",
                target_column="target",
                delimiter=" / ",
                language="ja",
                segmenter="delimiter",
                tokenizer="mecab",
                legacy_preprocessing=None,
                ja_replace_num=False,
                ja_stopwords_path="stopwords.txt",
                ja_dicdir="neologd",
                ja_require_unidic=False,
            ),
            encoder_device="cpu",
            runtime_num_workers=1,
        ),
        extra_options=None,
        num_topics=10,
        iteration=0,
        baselines=[
            BaselineConfig(name="Gaussian LDA", runner="gaussianlda", params={})
        ],
        logger=type("Logger", (), {"info": lambda self, _msg: None})(),
    )

    assert isinstance(results[0], BaselineSummary)
    assert results[0].name == "Gaussian LDA"
    assert results[0].runner_key == "gaussianlda"
    assert results[0].parameter_variant == "default"
    assert captured["options"] == {
        "train_csvs": ["train.csv"],
        "test_csvs": ["test.csv"],
        "targets": None,
        "text_column": "data",
        "target_column": "target",
        "delimiter": " / ",
        "language": "ja",
        "segmenter": "delimiter",
        "tokenizer": "mecab",
        "legacy_preprocessing": None,
        "ja_replace_num": False,
        "ja_stopwords_path": "stopwords.txt",
        "ja_dicdir": "neologd",
        "ja_require_unidic": False,
        "encoder_device": "cpu",
        "runtime_num_workers": 1,
    }


def test_run_baselines_for_category_serializes_typed_baseline_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_request(request):
        captured["options"] = request.options
        metadata_path = tmp_path / "metadata_ctm.json"
        metadata_path.write_text(
            '{"runner_key":"ctm","runner_family":"ctm","parameter_variant":"batch_size_cap=16__contextual_encode_prefix=none__contextual_model_name=sentence-transformers/all-mpnet-base-v2__num_epochs=12__num_samples=4","preprocessing_variant":"language=english","baseline_params":{"num_epochs":12,"num_samples":4,"batch_size_cap":16,"contextual_model_name":"sentence-transformers/all-mpnet-base-v2","contextual_encode_prefix":null}}',
            encoding="utf-8",
        )
        return ModelArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
            extras={"metadata": metadata_path},
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request",
        _fake_run_model_request,
    )

    results = run_baselines_for_category(
        category="all",
        dataset="dummy",
        runtime=BaselineRuntimeContext(
            corpus=CorpusSelection(
                train_csvs=(Path("train.csv"),),
                test_csvs=(Path("test.csv"),),
                targets=None,
            ),
            preprocess=PreprocessRuntime(
                text_column="data",
                target_column="target",
                delimiter=" / ",
                language="english",
                segmenter="delimiter",
                tokenizer="simple",
                legacy_preprocessing=None,
                ja_replace_num=True,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=True,
            ),
            encoder_device="cpu",
            runtime_num_workers=1,
        ),
        extra_options=None,
        num_topics=10,
        iteration=0,
        baselines=[
            BaselineConfig(
                name="Contextual TM",
                runner="ctm",
                params=CtmParams(num_epochs=12, num_samples=4, batch_size_cap=16),
            )
        ],
        logger=type("Logger", (), {"info": lambda self, _msg: None})(),
    )

    assert captured["options"]["num_epochs"] == 12
    assert captured["options"]["num_samples"] == 4
    assert captured["options"]["batch_size_cap"] == 16
    assert captured["options"]["contextual_encode_prefix"] is None
    assert results[0].runner_family == "ctm"
    assert results[0].baseline_params["num_epochs"] == 12


def test_bertopic_kmeans_effective_random_state_uses_seed_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_request(request):
        captured["options"] = request.options
        metadata_path = tmp_path / "metadata_bertopic.json"
        metadata_path.write_text(
            '{"runner_key":"bertopic_kmeans","runner_family":"bertopic_kmeans","parameter_variant":"default","preprocessing_variant":"language=english","baseline_params":{}}',
            encoding="utf-8",
        )
        return ModelArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
            extras={"metadata": metadata_path},
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request",
        _fake_run_model_request,
    )

    run_baselines_for_category(
        category="all",
        dataset="dummy",
        runtime=BaselineRuntimeContext(
            corpus=CorpusSelection(
                train_csvs=(Path("train.csv"),),
                test_csvs=(Path("test.csv"),),
                targets=None,
            ),
            preprocess=PreprocessRuntime(
                text_column="data",
                target_column="target",
                delimiter=" / ",
                language="english",
                segmenter="delimiter",
                tokenizer="simple",
                legacy_preprocessing=None,
                ja_replace_num=True,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=True,
            ),
            encoder_device="cpu",
            runtime_num_workers=1,
        ),
        extra_options={"seed": None, "seed_base": 100},
        num_topics=10,
        iteration=3,
        baselines=[
            BaselineConfig(
                name="BERTopic",
                runner="bertopic_kmeans",
                params=BertopicKMeansParams(random_state=None),
            )
        ],
        logger=type("Logger", (), {"info": lambda self, _msg: None})(),
    )

    assert captured["options"]["effective_random_state"] == 103
    assert captured["options"]["doc_topic_source"] == "umap_kmeans_centroid_softmax"
    assert captured["options"]["doc_topic_space"] == "umap"


def test_bertopic_kmeans_params_random_state_overrides_seed_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_request(request):
        captured["options"] = request.options
        metadata_path = tmp_path / "metadata_bertopic.json"
        metadata_path.write_text(
            '{"runner_key":"bertopic_kmeans","runner_family":"bertopic_kmeans","parameter_variant":"default","preprocessing_variant":"language=english","baseline_params":{}}',
            encoding="utf-8",
        )
        return ModelArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
            extras={"metadata": metadata_path},
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request",
        _fake_run_model_request,
    )

    run_baselines_for_category(
        category="all",
        dataset="dummy",
        runtime=BaselineRuntimeContext(
            corpus=CorpusSelection(
                train_csvs=(Path("train.csv"),),
                test_csvs=(Path("test.csv"),),
                targets=None,
            ),
            preprocess=PreprocessRuntime(
                text_column="data",
                target_column="target",
                delimiter=" / ",
                language="english",
                segmenter="delimiter",
                tokenizer="simple",
                legacy_preprocessing=None,
                ja_replace_num=True,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=True,
            ),
            encoder_device="cpu",
            runtime_num_workers=1,
        ),
        extra_options={"seed": None, "seed_base": 100},
        num_topics=10,
        iteration=3,
        baselines=[
            BaselineConfig(
                name="BERTopic",
                runner="bertopic_kmeans",
                params=BertopicKMeansParams(random_state=7),
            )
        ],
        logger=type("Logger", (), {"info": lambda self, _msg: None})(),
    )

    assert captured["options"]["effective_random_state"] == 7
    assert captured["options"]["doc_topic_source"] == "umap_kmeans_centroid_softmax"
    assert captured["options"]["doc_topic_space"] == "umap"


def test_etm_random_state_uses_seed_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_request(request):
        captured["options"] = request.options
        metadata_path = tmp_path / "metadata_etm.json"
        metadata_path.write_text(
            '{"runner_key":"etm","runner_family":"etm","parameter_variant":"default","preprocessing_variant":"language=english","baseline_params":{}}',
            encoding="utf-8",
        )
        return ModelArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
            extras={"metadata": metadata_path},
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request",
        _fake_run_model_request,
    )

    run_baselines_for_category(
        category="all",
        dataset="dummy",
        runtime=BaselineRuntimeContext(
            corpus=CorpusSelection(
                train_csvs=(Path("train.csv"),),
                test_csvs=(Path("test.csv"),),
                targets=None,
            ),
            preprocess=PreprocessRuntime(
                text_column="data",
                target_column="target",
                delimiter=" / ",
                language="english",
                segmenter="delimiter",
                tokenizer="simple",
                legacy_preprocessing=None,
                ja_replace_num=True,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=True,
            ),
            encoder_device="cpu",
            runtime_num_workers=1,
        ),
        extra_options={"seed": None, "seed_base": 100},
        num_topics=10,
        iteration=3,
        baselines=[
            BaselineConfig(
                name="ETM",
                runner="etm",
                params=EtmParams(random_state=None),
            )
        ],
        logger=type("Logger", (), {"info": lambda self, _msg: None})(),
    )

    assert captured["options"]["effective_random_state"] == 103
    assert captured["options"]["random_state"] == 103


def test_etm_params_random_state_overrides_seed_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_request(request):
        captured["options"] = request.options
        metadata_path = tmp_path / "metadata_etm.json"
        metadata_path.write_text(
            '{"runner_key":"etm","runner_family":"etm","parameter_variant":"default","preprocessing_variant":"language=english","baseline_params":{}}',
            encoding="utf-8",
        )
        return ModelArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
            extras={"metadata": metadata_path},
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request",
        _fake_run_model_request,
    )

    run_baselines_for_category(
        category="all",
        dataset="dummy",
        runtime=BaselineRuntimeContext(
            corpus=CorpusSelection(
                train_csvs=(Path("train.csv"),),
                test_csvs=(Path("test.csv"),),
                targets=None,
            ),
            preprocess=PreprocessRuntime(
                text_column="data",
                target_column="target",
                delimiter=" / ",
                language="english",
                segmenter="delimiter",
                tokenizer="simple",
                legacy_preprocessing=None,
                ja_replace_num=True,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=True,
            ),
            encoder_device="cpu",
            runtime_num_workers=1,
        ),
        extra_options={"seed": None, "seed_base": 100},
        num_topics=10,
        iteration=3,
        baselines=[
            BaselineConfig(
                name="ETM",
                runner="etm",
                params=EtmParams(random_state=7),
            )
        ],
        logger=type("Logger", (), {"info": lambda self, _msg: None})(),
    )

    assert captured["options"]["effective_random_state"] == 7
    assert captured["options"]["random_state"] == 7


def test_sentlda_random_state_uses_seed_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_request(request):
        captured["options"] = request.options
        metadata_path = tmp_path / "metadata_sentlda.json"
        metadata_path.write_text(
            '{"runner_key":"sentlda","runner_family":"sentlda","parameter_variant":"default","preprocessing_variant":"language=english","baseline_params":{}}',
            encoding="utf-8",
        )
        return ModelArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
            extras={"metadata": metadata_path},
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request",
        _fake_run_model_request,
    )

    run_baselines_for_category(
        category="all",
        dataset="dummy",
        runtime=BaselineRuntimeContext(
            corpus=CorpusSelection(
                train_csvs=(Path("train.csv"),),
                test_csvs=(Path("test.csv"),),
                targets=None,
            ),
            preprocess=PreprocessRuntime(
                text_column="data",
                target_column="target",
                delimiter=" / ",
                language="english",
                segmenter="delimiter",
                tokenizer="simple",
                legacy_preprocessing=None,
                ja_replace_num=True,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=True,
            ),
            encoder_device="cpu",
            runtime_num_workers=1,
        ),
        extra_options={"seed": None, "seed_base": 100},
        num_topics=10,
        iteration=3,
        baselines=[
            BaselineConfig(
                name="sentLDA",
                runner="sentlda",
                params=SentLdaParams(random_state=None),
            )
        ],
        logger=type("Logger", (), {"info": lambda self, _msg: None})(),
    )

    assert captured["options"]["effective_random_state"] == 103
    assert captured["options"]["random_state"] == 103


def test_sentlda_params_random_state_overrides_seed_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_model_request(request):
        captured["options"] = request.options
        metadata_path = tmp_path / "metadata_sentlda.json"
        metadata_path.write_text(
            '{"runner_key":"sentlda","runner_family":"sentlda","parameter_variant":"default","preprocessing_variant":"language=english","baseline_params":{}}',
            encoding="utf-8",
        )
        return ModelArtifacts(
            train_path=Path("train.pkl"),
            infer_path=Path("infer.pkl"),
            extras={"metadata": metadata_path},
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request",
        _fake_run_model_request,
    )

    run_baselines_for_category(
        category="all",
        dataset="dummy",
        runtime=BaselineRuntimeContext(
            corpus=CorpusSelection(
                train_csvs=(Path("train.csv"),),
                test_csvs=(Path("test.csv"),),
                targets=None,
            ),
            preprocess=PreprocessRuntime(
                text_column="data",
                target_column="target",
                delimiter=" / ",
                language="english",
                segmenter="delimiter",
                tokenizer="simple",
                legacy_preprocessing=None,
                ja_replace_num=True,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=True,
            ),
            encoder_device="cpu",
            runtime_num_workers=1,
        ),
        extra_options={"seed": None, "seed_base": 100},
        num_topics=10,
        iteration=3,
        baselines=[
            BaselineConfig(
                name="sentLDA",
                runner="sentlda",
                params=SentLdaParams(random_state=7),
            )
        ],
        logger=type("Logger", (), {"info": lambda self, _msg: None})(),
    )

    assert captured["options"]["effective_random_state"] == 7
    assert captured["options"]["random_state"] == 7


def test_build_vmf_run_options_serializes_execution_context() -> None:
    job = SimpleNamespace(
        train_csvs=(Path("train.csv"),),
        test_csvs=(Path("test.csv"),),
        vmf_soft_temp=0.7,
        data_run_name="fy2024",
    )
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            num_iterations=15,
            alpha=None,
            kappa_default=10.0,
            num_components=2,
            gibbs_sweeps=8,
            num_samples=4,
            estimate_alpha=True,
            alpha_update_every=2,
            alpha_max_iter=50,
            alpha_tol=1e-5,
            avg_log_likelihood_every=3,
            invariant_check_every=4,
        ),
        encoder=SimpleNamespace(
            model_name="dummy-encoder",
            device="cpu",
            encode_prefix="prefix: ",
            encode_batch_size=16,
            pre_normalize_transform="whitening",
            whitening_eps=1e-4,
        ),
        preprocess=SimpleNamespace(
            delimiter=" / ",
            language="ja",
            segmenter="delimiter",
            tokenizer="mecab",
            text_column="data",
            target_column="target_str",
            ja_replace_num=False,
            ja_stopwords_path="stopwords.txt",
            ja_dicdir="neologd",
            ja_require_unidic=False,
        ),
    )
    axes = SimpleNamespace(algorithm_variant="components_2__estimate_alpha_every_2")
    logger = object()

    options = _build_vmf_run_options(
        job=job,
        cfg=cfg,
        axes=axes,
        logger=logger,
        vmf_out_dir=Path("results/run"),
        resolved_targets=["a", "b"],
        vmf_condition_id="cond123",
        vmf_condition_fingerprint="fp123",
        started_at="2026-04-10T00:00:00+00:00",
        execution_id="exec123",
    ).to_request_options()

    assert options["targets"] == ["a", "b"]
    assert options["train_csvs"] == ["train.csv"]
    assert options["test_csvs"] == ["test.csv"]
    assert options["algorithm_variant"] == "components_2__estimate_alpha_every_2"
    assert options["encoder_name"] == "dummy-encoder"
    assert options["encoder_encode_batch_size"] == 16
    assert options["soft_temperature"] == 0.7
    assert options["data_run"] == "fy2024"
    assert options["condition_id"] == "cond123"
    assert options["output_dir"] == Path("results/run")
    assert options["logger"] is logger


def test_process_category_impl_writes_vmf_archive_and_latest_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.experiments.execution import _process_category_impl
    from src.experiments.job_planning import CategoryJob, ParallelismPlan

    def _fake_run_model_request(request):
        out_dir = Path(request.options["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "doc_topic_train.pkl").write_bytes(b"x")
        (out_dir / "doc_topic_test.pkl").write_bytes(b"y")
        (out_dir / "doc_topic_train_soft.pkl").write_bytes(b"z")
        (out_dir / "doc_topic_test_soft.pkl").write_bytes(b"w")
        (out_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (out_dir / "params.json").write_text("{}", encoding="utf-8")
        return ModelArtifacts(
            train_path=out_dir / "doc_topic_train.pkl",
            infer_path=out_dir / "doc_topic_test.pkl",
            extras={
                "doc_topic_train_soft": out_dir / "doc_topic_train_soft.pkl",
                "doc_topic_test_soft": out_dir / "doc_topic_test_soft.pkl",
                "metrics_path": out_dir / "metrics.json",
                "params": out_dir / "params.json",
            },
        )

    monkeypatch.setattr(
        "src.experiments.execution.run_model_request", _fake_run_model_request
    )
    monkeypatch.setattr(
        "src.experiments.execution.resolve_targets", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "src.experiments.execution.set_global_seed", lambda *_args, **_kwargs: None
    )

    job = CategoryJob(
        data_run_name="default",
        train_csvs=(Path("train.csv"),),
        test_csvs=(Path("test.csv"),),
        fiscal_years=None,
        category="all",
        targets=None,
        num_topics=10,
        iteration=0,
        baselines=[],
        selected_models={"vmf_sentence_lda"},
        seed=None,
        seed_base=None,
        parallelism=ParallelismPlan(
            requested_num_workers=1,
            category_num_workers=1,
            baseline_num_workers=1,
            encoder_device="cpu",
            run_vmf=True,
            uses_cuda=False,
            reason=None,
        ),
        config=SimpleNamespace(
            dataset=SimpleNamespace(name="dummy"),
            output_root=tmp_path / "dummy",
            train=SimpleNamespace(
                num_components=2,
                estimate_alpha=False,
                alpha_update_every=1,
                num_iterations=5,
                alpha=None,
                kappa_default=1.0,
                gibbs_sweeps=1,
                num_samples=1,
                alpha_max_iter=5,
                alpha_tol=1e-6,
                avg_log_likelihood_every=1,
                invariant_check_every=1,
            ),
            encoder=SimpleNamespace(
                model_name="dummy-encoder",
                device="cpu",
                encode_prefix=None,
                encode_batch_size=None,
                pre_normalize_transform="none",
                whitening_eps=1e-6,
            ),
            preprocess=SimpleNamespace(
                delimiter=" / ",
                language="english",
                segmenter="delimiter",
                tokenizer="default",
                text_column="data",
                target_column="target_str",
                has_labels=True,
                ja_replace_num=False,
                ja_stopwords_path=None,
                ja_dicdir=None,
                ja_require_unidic=False,
                legacy_preprocessing=None,
            ),
            baselines=[],
        ),
        vmf_soft_temp=1.0,
    )

    summary = _process_category_impl(job)

    assert summary.condition_id == "k10_it0_c2"
    latest_pointer = (
        tmp_path
        / "dummy"
        / "default"
        / "vmf_sentence_lda"
        / "latest"
        / "all"
        / "k10_it0_c2"
        / "CURRENT.json"
    )
    assert latest_pointer.exists()
    pointer_payload = load_json(latest_pointer)
    archive_dir = tmp_path / pointer_payload["archive_dir"]
    assert archive_dir.exists()
    config_payload = load_json(archive_dir / "config.json")
    assert config_payload["model_name"] == "vmf_sentence_lda"
    assert config_payload["num_topics"] == 10
    assert config_payload["seed"] == DEFAULT_RANDOM_SEED
    assert (archive_dir / "metadata.json").exists()

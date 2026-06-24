from __future__ import annotations

import bz2
from io import BytesIO
from pathlib import Path

from src.core.artifacts import (
    ARTIFACT_METADATA_SCHEMA_VERSION,
    BASELINE_METADATA_SCHEMA,
    METADATA_FILENAME,
    VMF_METADATA_SCHEMA,
    VMF_METRICS_FILENAME,
    VMF_PARAMS_FILENAME,
    BaselineArtifactMetadata,
    ExperimentAxes,
    PickleArtifactSpec,
    VmfArtifactMetadata,
    artifact_refs_to_path_map,
    artifact_refs_to_string_map,
    build_artifact_refs,
    copy_binary_stream_to_path,
    ensure_artifact_paths_exist,
    extract_bz2_file,
    load_json,
    load_pickle,
    load_text_lines,
    load_yaml,
    save_baseline_metadata,
    save_csv_rows,
    save_json,
    save_pickle,
    save_pickles,
    save_split_pickles,
    save_vmf_metadata,
    save_yaml,
)
from src.core.errors import MissingArtifactError
from src.core.runner_contracts import MODEL_KIND_TOPIC_MODEL
from src.core.runtime import BaselineRuntimeContext, CorpusSelection, PreprocessRuntime


def test_save_vmf_metadata_persists_axes_schema(tmp_path: Path) -> None:
    metadata = VmfArtifactMetadata(
        axes=ExperimentAxes(
            dataset="20newsgroup",
            model_family="vmf_sentence_lda",
            algorithm_variant="components_1__estimate_alpha_every_1",
            encoder_model="sentence-transformers/all-mpnet-base-v2",
            embedding_preprocess_variant="none",
            num_topics=20,
            iteration=0,
            category="science",
            data_run="default",
        ),
        condition_id="it0__k20__abcd1234",
        condition_fingerprint="abcd1234ef",
        started_at="2026-04-02T00:00:00+00:00",
        execution_id="vmf_20260402T000000Z",
        language="english",
        delimiter=" / ",
        segmenter="delimiter",
        tokenizer="default",
        text_column="data",
        target_column="target_str",
        has_labels=True,
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        train_csvs=("data/20newsgroup/train.csv",),
        test_csvs=("data/20newsgroup/test.csv",),
        fiscal_years=None,
    )
    out_path = tmp_path / "metadata.json"

    save_vmf_metadata(metadata, out_path)
    loaded = load_json(out_path)

    assert loaded["schema"] == VMF_METADATA_SCHEMA
    assert loaded["schema_version"] == ARTIFACT_METADATA_SCHEMA_VERSION
    assert loaded["artifact_kind"] == "metadata"
    axes = loaded["axes"]
    assert axes["dataset"] == "20newsgroup"
    assert axes["model_family"] == "vmf_sentence_lda"
    assert axes["algorithm_variant"] == "components_1__estimate_alpha_every_1"
    assert axes["encoder_model"] == "sentence-transformers/all-mpnet-base-v2"
    assert axes["embedding_preprocess_variant"] == "none"
    assert axes["num_topics"] == 20
    assert axes["iteration"] == 0
    assert axes["category"] == "science"
    assert axes["data_run"] == "default"
    assert loaded["condition_id"] == "it0__k20__abcd1234"
    assert loaded["condition_fingerprint"] == "abcd1234ef"
    assert loaded["delimiter"] == " / "
    assert loaded["segmenter"] == "delimiter"
    assert loaded["tokenizer"] == "default"
    assert loaded["has_labels"] is True
    assert loaded["ja_replace_num"] is True


def test_save_json_uses_standard_artifact_filenames() -> None:
    assert METADATA_FILENAME == "metadata.json"
    assert VMF_PARAMS_FILENAME == "params.json"
    assert VMF_METRICS_FILENAME == "metrics.json"


def test_baseline_metadata_schema_is_distinct() -> None:
    assert BASELINE_METADATA_SCHEMA != VMF_METADATA_SCHEMA


def test_save_baseline_metadata_persists_variant_axes(tmp_path: Path) -> None:
    metadata = BaselineArtifactMetadata(
        runner_key="ctm",
        runner_family="ctm",
        method_kind=MODEL_KIND_TOPIC_MODEL,
        data_run="default",
        condition_id="it0__k10__abcd1234",
        condition_fingerprint="abcd1234ef",
        started_at="2026-04-02T00:00:00+00:00",
        execution_id="baseline_20260402T000000Z",
        parameter_variant="batch_size_cap=64__num_epochs=50__num_samples=20",
        preprocessing_variant="language=english__delimiter= / __segmenter=delimiter__tokenizer=default__legacy_preprocessing=auto__text_column=data__target_column=target_str__ja_replace_num=true__ja_require_unidic=true",
        dataset="dummy",
        category="all",
        num_topics=10,
        iteration=0,
        baseline_params={"num_epochs": 50, "num_samples": 20, "batch_size_cap": 64},
        targets=None,
        language="english",
        delimiter=" / ",
        segmenter="delimiter",
        tokenizer="default",
        legacy_preprocessing=None,
        text_column="data",
        target_column="target_str",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        encoder_device="cpu",
        runtime_num_workers=1,
        train_csvs=("train.csv",),
        test_csvs=("test.csv",),
        train_dir="results/baselines/ctm/params",
        infer_dir="results/baselines/ctm/infer",
    )
    out_path = tmp_path / "metadata.json"

    save_baseline_metadata(metadata, out_path)
    loaded = load_json(out_path)

    assert loaded["schema"] == BASELINE_METADATA_SCHEMA
    assert loaded["method_kind"] == MODEL_KIND_TOPIC_MODEL
    assert loaded["data_run"] == "default"
    assert loaded["condition_id"] == "it0__k10__abcd1234"
    assert loaded["parameter_variant"] == metadata.parameter_variant
    assert loaded["preprocessing_variant"] == metadata.preprocessing_variant
    assert loaded["baseline_params"]["num_epochs"] == 50


def test_pickle_roundtrip(tmp_path: Path) -> None:
    payload = {"x": [1, 2, 3], "y": "z"}
    out_path = tmp_path / "artifact.pkl"
    save_pickle(payload, out_path)
    loaded = load_pickle(out_path)
    assert loaded == payload


def test_json_roundtrip_accepts_any_payload(tmp_path: Path) -> None:
    payload = [{"x": 1}, {"y": 2}]
    out_path = tmp_path / "artifact.json"

    save_json(payload, out_path)
    loaded = load_json(out_path)

    assert loaded == payload


def test_yaml_roundtrip(tmp_path: Path) -> None:
    payload = {"dataset": {"name": "dummy"}, "train": {"num_topics": [10, 20]}}
    out_path = tmp_path / "config.yaml"

    save_yaml(payload, out_path)
    loaded = load_yaml(out_path)

    assert loaded == payload


def test_text_lines_roundtrip(tmp_path: Path) -> None:
    out_path = tmp_path / "lines.txt"
    out_path.write_text("a\n# comment\nb\n", encoding="utf-8")

    assert load_text_lines(out_path) == ["a", "# comment", "b"]


def test_save_csv_rows_writes_header_and_rows(tmp_path: Path) -> None:
    out_path = tmp_path / "rows.csv"

    save_csv_rows(
        fieldnames=["dataset", "score"],
        rows=[{"dataset": "dummy", "score": 1.0}],
        path=out_path,
    )

    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "dataset,score",
        "dummy,1.0",
    ]


def test_copy_binary_stream_to_path_writes_bytes(tmp_path: Path) -> None:
    out_path = tmp_path / "blob.bin"

    copy_binary_stream_to_path(BytesIO(b"abc"), out_path)

    assert out_path.read_bytes() == b"abc"


def test_extract_bz2_file_unpacks_archive(tmp_path: Path) -> None:
    src_path = tmp_path / "payload.txt.bz2"
    dst_path = tmp_path / "payload.txt"

    with bz2.open(src_path, "wb") as f:
        f.write(b"hello")

    extract_bz2_file(src_path, dst_path)

    assert dst_path.read_bytes() == b"hello"


def test_save_pickles_returns_saved_paths(tmp_path: Path) -> None:
    saved = save_pickles({"a": [1, 2], "b": {"x": 1}}, tmp_path / "artifacts")

    assert saved["a"] == tmp_path / "artifacts" / "a.pkl"
    assert saved["b"] == tmp_path / "artifacts" / "b.pkl"
    assert load_pickle(saved["a"]) == [1, 2]
    assert load_pickle(saved["b"]) == {"x": 1}


def test_save_split_pickles_routes_outputs_by_split(tmp_path: Path) -> None:
    saved = save_split_pickles(
        [
            PickleArtifactSpec(
                name="train_path",
                filename="train.pkl",
                payload=[1, 2],
                split="train",
            ),
            PickleArtifactSpec(
                name="infer_path",
                filename="test.pkl",
                payload={"x": 1},
                split="infer",
            ),
        ],
        train_dir=tmp_path / "params",
        infer_dir=tmp_path / "infer",
    )

    assert saved["train_path"] == tmp_path / "params" / "train.pkl"
    assert saved["infer_path"] == tmp_path / "infer" / "test.pkl"
    assert load_pickle(saved["train_path"]) == [1, 2]
    assert load_pickle(saved["infer_path"]) == {"x": 1}


def test_ensure_artifact_paths_exist_raises_for_missing_paths(tmp_path: Path) -> None:
    existing = tmp_path / "ok.pkl"
    existing.write_bytes(b"x")

    try:
        ensure_artifact_paths_exist(
            {
                "ok": existing,
                "missing": tmp_path / "missing.pkl",
            }
        )
    except MissingArtifactError as exc:
        assert "missing.pkl" in str(exc)
    else:
        raise AssertionError("Expected MissingArtifactError for missing artifact path.")


def test_artifact_refs_roundtrip_maps() -> None:
    refs = build_artifact_refs(
        {
            "metrics_path": Path("results/metrics.json"),
            "train_doc_topic": Path("results/doc_topic_train.pkl"),
        }
    )

    assert [ref.name for ref in refs] == ["metrics_path", "train_doc_topic"]
    assert artifact_refs_to_path_map(refs)["metrics_path"] == Path(
        "results/metrics.json"
    )
    assert (
        artifact_refs_to_string_map(refs)["train_doc_topic"]
        == "results/doc_topic_train.pkl"
    )


def test_baseline_runtime_context_serializes_model_options() -> None:
    runtime = BaselineRuntimeContext(
        corpus=CorpusSelection(
            train_csvs=(Path("train.csv"),),
            test_csvs=(Path("test.csv"),),
            targets=("a", "b"),
        ),
        preprocess=PreprocessRuntime(
            text_column="data",
            target_column="target_str",
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
    )

    assert runtime.to_model_options() == {
        "train_csvs": ["train.csv"],
        "test_csvs": ["test.csv"],
        "targets": ["a", "b"],
        "text_column": "data",
        "target_column": "target_str",
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

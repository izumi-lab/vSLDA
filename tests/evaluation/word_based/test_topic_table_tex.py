from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.artifacts import save_pickle
from src.evaluation.reporting import read_evaluation_json, write_evaluation_json
from src.evaluation.schema import build_evaluation_meta
from src.evaluation.word_based.topic_word_table import run_topic_table_tex


def test_run_topic_table_tex_writes_sidecar_with_topic_words_provenance(
    tmp_path: Path,
) -> None:
    profile_json = tmp_path / "profile.json"
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="word_based_label_profile",
            model_provenance={
                "model_key": "bleilda",
                "parameter_variant": "passes=20",
            },
        ),
        results={
            "labels": [
                {
                    "label": "science",
                    "top_topics": [
                        {"rank": 1, "topic_id": 0, "score_value": 1.25},
                    ],
                }
            ],
            "global_top_topics": [],
        },
        path=profile_json,
    )

    topic_words_json = tmp_path / "topic_words.json"
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="word_based_metrics",
            model_provenance={
                "model_key": "bleilda",
                "parameter_variant": "passes=20",
                "coherence": "c_v",
            },
        ),
        results={
            "per_iteration": [
                {
                    "iteration": 0,
                    "topics": [
                        {
                            "topic_id": 0,
                            "words": [{"word": "alpha"}, {"word": "beta"}],
                        }
                    ],
                }
            ]
        },
        path=topic_words_json,
    )

    out_tex = tmp_path / "tables" / "topic_profile.tex"
    result = run_topic_table_tex(
        profile_json=profile_json,
        topic_words_json=topic_words_json,
        iteration=0,
        labels=["science"],
        include_score=True,
        out_tex=out_tex,
    )

    assert result == out_tex
    assert out_tex.exists()
    assert "alpha" in out_tex.read_text(encoding="utf-8")

    meta, results = read_evaluation_json(out_tex.with_suffix(".json"))
    assert meta["task"] == "word_based_topic_word_table"
    assert meta["representative_words_source"] == "topic_words_json"
    assert meta["model_provenance"]["profile"]["parameter_variant"] == "passes=20"
    assert meta["model_provenance"]["topic_words"]["coherence"] == "c_v"
    assert results["selected_topic_ids"] == [0]


def test_run_topic_table_tex_can_compute_weighted_tf_and_capture_doc_topic_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "data": ["alpha alpha / beta", "gamma / delta"],
            "target_str": ["science", "science"],
        }
    ).to_csv(dataset_dir / "train.csv", index=False)

    doc_topic_path = tmp_path / "results" / "baselines" / "bleilda" / "train.pkl"
    save_pickle([[0.9, 0.1], [0.2, 0.8]], doc_topic_path)
    (doc_topic_path.parent / "metadata.json").write_text(
        (
            '{"schema":"baseline_artifact_metadata","runner_key":"bleilda",'
            '"runner_family":"bleilda","parameter_variant":"passes=5",'
            '"preprocessing_variant":"language=english","baseline_params":{"passes":5}}'
        ),
        encoding="utf-8",
    )

    profile_json = tmp_path / "profile_weighted.json"
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="word_based_label_profile",
            dataset="dummy_topic_table_tex",
            category="all",
            split="train",
            iteration=0,
            num_topics=2,
            model="bleilda",
            doc_topic_path=str(doc_topic_path),
            model_provenance={
                "model_key": "bleilda",
                "parameter_variant": "passes=5",
            },
        ),
        results={
            "labels": [
                {
                    "label": "science",
                    "top_topics": [{"rank": 1, "topic_id": 0, "mean_weight": 0.55}],
                }
            ],
            "global_top_topics": [],
        },
        path=profile_json,
    )

    monkeypatch.setattr(
        "src.evaluation.word_based.topic_word_table.resolve_dataset_dir",
        lambda _dataset: dataset_dir,
    )
    monkeypatch.setattr(
        "src.data.catalog.resolve_dataset_dir",
        lambda _dataset: dataset_dir,
    )

    out_tex = tmp_path / "tables" / "topic_profile_weighted.tex"
    result = run_topic_table_tex(
        profile_json=profile_json,
        labels=["science"],
        words_per_topic=2,
        language="english",
        out_tex=out_tex,
    )

    assert result == out_tex
    tex = out_tex.read_text(encoding="utf-8")
    assert "alpha" in tex

    meta, results = read_evaluation_json(out_tex.with_suffix(".json"))
    assert meta["representative_words_source"] == "weighted_tf_from_doc_topic"
    assert meta["model_provenance"]["doc_topic"]["parameter_variant"] == "passes=5"
    assert meta["source_meta"]["doc_topic_path"] == str(doc_topic_path)
    assert results["selected_topic_ids"] == [0]


def test_run_topic_table_tex_can_compute_npmi_topic_words(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset_npmi"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "data": [
                "alpha alpha / beta",
                "alpha / beta",
                "gamma gamma / delta",
            ],
            "target_str": ["science", "science", "science"],
        }
    ).to_csv(dataset_dir / "train.csv", index=False)

    doc_topic_path = tmp_path / "results" / "experiments" / "vmf" / "train.pkl"
    save_pickle([[0.95, 0.05], [0.9, 0.1], [0.05, 0.95]], doc_topic_path)
    (doc_topic_path.parent / "metadata.json").write_text(
        (
            '{"schema":"vmf_artifact_metadata","model_family":"vmf_sentence_lda",'
            '"algorithm_variant":"test","encoder_model":"dummy"}'
        ),
        encoding="utf-8",
    )

    profile_json = tmp_path / "profile_npmi.json"
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="word_based_label_profile",
            dataset="dummy_topic_table_npmi",
            category="all",
            split="train",
            iteration=0,
            num_topics=2,
            model="vmf_sentence_lda",
            doc_topic_path=str(doc_topic_path),
            model_provenance={
                "model_key": "vmf_sentence_lda",
                "algorithm_variant": "test",
            },
        ),
        results={
            "labels": [
                {
                    "label": "science",
                    "top_topics": [{"rank": 1, "topic_id": 0, "mean_weight": 0.6}],
                }
            ],
            "global_top_topics": [],
        },
        path=profile_json,
    )

    monkeypatch.setattr(
        "src.evaluation.word_based.topic_word_table.resolve_dataset_dir",
        lambda _dataset: dataset_dir,
    )
    monkeypatch.setattr(
        "src.data.catalog.resolve_dataset_dir",
        lambda _dataset: dataset_dir,
    )

    out_tex = tmp_path / "tables" / "topic_profile_npmi.tex"
    result = run_topic_table_tex(
        profile_json=profile_json,
        labels=["science"],
        words_per_topic=2,
        language="english",
        representative_words_method="npmi",
        out_tex=out_tex,
    )

    assert result == out_tex
    tex = out_tex.read_text(encoding="utf-8")
    assert "alpha" in tex

    meta, results = read_evaluation_json(out_tex.with_suffix(".json"))
    assert meta["representative_words_source"] == "document_topic_proxy_npmi"
    assert meta["representative_words_method"] == "npmi"
    assert meta["source_meta"]["representative_words_method"] == "npmi"
    assert results["selected_topic_ids"] == [0]

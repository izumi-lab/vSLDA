from __future__ import annotations

from pathlib import Path

import pytest

from src.core.artifacts import save_json
from src.evaluation.reporting import read_evaluation_json, write_evaluation_json
from src.evaluation.schema import build_evaluation_meta
from src.evaluation.word_based.topic_word_table import run_topic_table_tex


def _write_profile(path: Path, *, model: str = "bleilda") -> None:
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="word_based_label_profile",
            model=model,
            model_provenance={"model_key": model, "parameter_variant": "test"},
        ),
        results={
            "labels": [
                {
                    "label": "science",
                    "top_topics": [{"rank": 1, "topic_id": 0, "score_value": 1.25}],
                }
            ],
            "global_top_topics": [],
        },
        path=path,
    )


def test_topic_table_reads_display_artifact_only(tmp_path: Path) -> None:
    profile_json = tmp_path / "profile.json"
    _write_profile(profile_json)
    topic_words_json = tmp_path / "topic_words_display_topk.json"
    save_json(
        {
            "topic_word_role": "display",
            "score_mode": "word_topic_npmi",
            "source": "posthoc_word_topic_npmi",
            "model": "bleilda",
            "split": "train",
            "condition_fingerprint": "condition",
            "evaluation_vocabulary_fingerprint": "vocabulary",
            "topics": [
                {
                    "topic_id": 0,
                    "words": [
                        {"word": "alpha", "score": 0.5},
                        {"word": "beta", "score": 0.25},
                    ],
                }
            ],
        },
        topic_words_json,
    )
    out_tex = tmp_path / "tables" / "topic_profile.tex"
    result = run_topic_table_tex(
        profile_json=profile_json,
        topic_words_json=topic_words_json,
        labels=["science"],
        include_score=True,
        out_tex=out_tex,
    )
    assert result == out_tex
    assert "alpha" in out_tex.read_text(encoding="utf-8")
    meta, results = read_evaluation_json(out_tex.with_suffix(".json"))
    assert meta["representative_words_source"] == "posthoc_word_topic_npmi"
    assert meta["topic_word_role"] == "display"
    assert meta["score_mode"] == "word_topic_npmi"
    assert results["selected_topic_ids"] == [0]


def test_topic_table_rejects_evaluation_ranking(tmp_path: Path) -> None:
    profile_json = tmp_path / "profile.json"
    _write_profile(profile_json)
    topic_words_json = tmp_path / "topic_words_evaluation_topk.json"
    save_json(
        {
            "topic_word_role": "evaluation",
            "score_mode": "topic_word_probability",
            "model": "bleilda",
            "topics": [],
        },
        topic_words_json,
    )
    with pytest.raises(ValueError, match="topic_word_role='display'"):
        run_topic_table_tex(
            profile_json=profile_json,
            topic_words_json=topic_words_json,
        )


@pytest.mark.parametrize(
    ("score_mode", "source"),
    [
        (
            "decoder_topic_word_probability",
            "native_ctm_decoder_topic_word_distribution",
        ),
        ("word_topic_npmi", "ctm_variational_word_topic_npmi"),
    ],
)
def test_ctm_table_accepts_current_and_legacy_display_scores(
    tmp_path: Path, score_mode: str, source: str
) -> None:
    profile_json = tmp_path / "profile.json"
    _write_profile(profile_json, model="ctm")
    topic_words_json = tmp_path / "topic_words_display_topk.json"
    save_json(
        {
            "topic_word_role": "display",
            "score_mode": score_mode,
            "source": source,
            "model": "ctm",
            "split": "train",
            "condition_fingerprint": "condition",
            "evaluation_vocabulary_fingerprint": "vocabulary",
            "topics": [{"topic_id": 0, "words": [{"word": "alpha", "score": 0.8}]}],
        },
        topic_words_json,
    )
    tex = run_topic_table_tex(
        profile_json=profile_json,
        topic_words_json=topic_words_json,
    )
    assert isinstance(tex, str) and "alpha" in tex

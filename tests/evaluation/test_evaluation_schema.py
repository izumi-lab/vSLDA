from __future__ import annotations

from src.evaluation.schema import (
    EVALUATION_SCHEMA_NAME,
    EVALUATION_SCHEMA_VERSION,
    build_evaluation_meta,
    build_evaluation_payload,
    split_evaluation_payload,
)


def test_build_evaluation_meta_adds_standard_fields() -> None:
    meta = build_evaluation_meta(task="topic_overlap", dataset="20newsgroup")

    assert meta["task"] == "topic_overlap"
    assert meta["schema"] == EVALUATION_SCHEMA_NAME
    assert meta["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert meta["output_kind"] == "payload"


def test_build_evaluation_payload_wraps_meta_and_results() -> None:
    payload = build_evaluation_payload(
        meta={"task": "topic_overlap", "dataset": "20newsgroup"},
        results={"aggregate": {"diversity_score": {"mean": 0.1, "std": 0.0}}},
    )

    assert payload == {
        "_meta": {
            "task": "topic_overlap",
            "dataset": "20newsgroup",
            "schema": EVALUATION_SCHEMA_NAME,
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "output_kind": "payload",
        },
        "results": {"aggregate": {"diversity_score": {"mean": 0.1, "std": 0.0}}},
    }


def test_split_evaluation_payload_supports_wrapped_and_legacy_shapes() -> None:
    meta, results = split_evaluation_payload(
        {
            "_meta": {"task": "classification"},
            "results": {"science": {"ModelA": 50.0}},
        }
    )
    assert meta["task"] == "classification"
    assert meta["schema"] == EVALUATION_SCHEMA_NAME
    assert meta["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert meta["output_kind"] == "payload"
    assert results == {"science": {"ModelA": 50.0}}

    legacy_meta, legacy_results = split_evaluation_payload(
        {"science": {"ModelA": 50.0}}
    )
    assert legacy_meta == {}
    assert legacy_results == {"science": {"ModelA": 50.0}}

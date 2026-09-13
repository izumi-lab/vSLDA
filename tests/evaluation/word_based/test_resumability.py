from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gensim.corpora import Dictionary

from src.evaluation.word_based import metrics as metrics_module
from src.evaluation.word_based.metrics import (
    WordBasedConditionFailures,
    run_topic_coherence_analysis,
)
from src.evaluation.word_based.resumability import (
    load_topic_word_checkpoint,
    save_topic_word_checkpoint,
)
from src.evaluation.word_based.topic_assignment import (
    DegenerateTopicError,
    InsufficientTopicWordsError,
)
from src.evaluation.word_based.topic_word_runtime import RuntimeTopicWords
from src.evaluation.word_based.topic_words import (
    TopicWordsResult,
    serialize_topic_words,
)


def _runtime(words=None) -> RuntimeTopicWords:
    topic_words = words or [[("alpha", 1.0)], [("beta", 0.5)]]
    return RuntimeTopicWords(
        evaluation=TopicWordsResult(
            topic_words=topic_words,
            topic_word_source="posthoc_expected_p_w_given_topic",
            score_mode="topic_word_probability",
            score_definition="test",
        ),
        display_topic_words=topic_words,
        display_source="posthoc_word_topic_npmi",
        display_score_mode="word_topic_npmi",
        protocol="collapsed_posthoc",
        condition_dir=Path("/tmp/source"),
        source_condition_fingerprint="source-fingerprint",
        vocabulary_fingerprint="vocabulary-fingerprint",
        corpus_fingerprint="corpus-fingerprint",
        coverage={"token_coverage": 1.0},
        posterior_mean_by_doc=[np.asarray([[0.6, 0.4]])],
        posterior_metadata={"seed": 0},
        expected_counts=np.asarray([[2.0, 0.0], [0.0, 1.0]]),
        execution_metadata={
            "effective_device": "cpu",
            "encode_batch_size": 32,
            "encode_batch_size_source": "backend_default",
        },
    )


def test_topic_word_checkpoint_round_trip_and_rejects_incomplete(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    identity = {"dataset": "dummy", "model": "vmf", "iteration": 0}
    checkpoint_dir = tmp_path / "checkpoint"
    save_topic_word_checkpoint(
        checkpoint_dir=checkpoint_dir,
        identity=identity,
        runtime=runtime,
        serialized_evaluation_words=serialize_topic_words(
            runtime.evaluation.topic_words
        ),
        serialized_display_words=serialize_topic_words(runtime.display_topic_words),
    )

    loaded = load_topic_word_checkpoint(
        checkpoint_dir=checkpoint_dir,
        expected_identity=identity,
    )
    assert loaded is not None
    assert loaded.evaluation.topic_words == runtime.evaluation.topic_words
    np.testing.assert_allclose(loaded.expected_counts, runtime.expected_counts)
    np.testing.assert_allclose(
        loaded.posterior_mean_by_doc[0], runtime.posterior_mean_by_doc[0]
    )
    assert loaded.execution_metadata == runtime.execution_metadata

    (checkpoint_dir / "COMPLETE.json").unlink()
    assert (
        load_topic_word_checkpoint(
            checkpoint_dir=checkpoint_dir,
            expected_identity=identity,
        )
        is None
    )


def test_topic_word_checkpoint_round_trip_preserves_empty_topics(
    tmp_path: Path,
) -> None:
    runtime = replace(
        _runtime(words=[[("alpha", 1.0)], []]),
        empty_topic_ids=(1,),
    )
    identity = {"dataset": "dummy", "model": "vmf", "iteration": 0}
    checkpoint_dir = tmp_path / "checkpoint"
    save_topic_word_checkpoint(
        checkpoint_dir=checkpoint_dir,
        identity=identity,
        runtime=runtime,
        serialized_evaluation_words=serialize_topic_words(
            runtime.evaluation.topic_words
        ),
        serialized_display_words=serialize_topic_words(runtime.display_topic_words),
    )

    loaded = load_topic_word_checkpoint(
        checkpoint_dir=checkpoint_dir,
        expected_identity=identity,
    )
    assert loaded is not None
    assert loaded.empty_topic_ids == (1,)
    assert loaded.display_topic_words[1] == []


def test_condition_failure_does_not_prevent_other_condition_save(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    reference_path.write_text("", encoding="utf-8")

    def collect(**kwargs):
        task = kwargs["task"]
        if task.category == "bad":
            raise InsufficientTopicWordsError(
                topic_id=1,
                eligible_words=1,
                requested_topn=25,
            )
        runtime = _runtime()
        return metrics_module.PendingWordBasedGroup(
            data_run=task.data_run,
            model=task.model,
            num_topics=task.num_topics,
            category=task.category,
            iterations=[
                metrics_module.PendingWordBasedIteration(
                    iteration=0,
                    topic_words=runtime.evaluation.topic_words,
                    runtime_payload=runtime,
                )
            ],
            topic_word_source=runtime.evaluation.topic_word_source,
            topic_word_score_mode=runtime.evaluation.score_mode or "",
            topic_word_score_definition=runtime.evaluation.score_definition or "",
        )

    def score(**kwargs):
        group = kwargs["group"]
        return metrics_module.ScoredWordBasedGroup(
            group=group,
            per_iter_metrics=[{"coherence": 0.5, "diversity": 1.0, "num_topics": 2.0}],
            per_iter_topic_words=[
                {
                    "iteration": 0,
                    "topics": serialize_topic_words(group.iterations[0].topic_words),
                }
            ],
            used_iterations=[0],
        )

    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._collect_pending_word_based_group",
        collect,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics._score_pending_word_based_group",
        score,
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.build_shared_reference_counts",
        lambda **kwargs: SimpleNamespace(
            num_docs=1,
            vocab_size=len(kwargs["target_words"]),
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.word_based.metrics.resolve_model_provenance",
        lambda **kwargs: {"model_key": kwargs["model"]},
    )

    with pytest.raises(WordBasedConditionFailures, match="1 word-based condition"):
        run_topic_coherence_analysis(
            models=["vmf"],
            dataset="dummy",
            data_runs=["default"],
            iterations=[0],
            num_topics=2,
            categories=["good", "bad"],
            out_root=tmp_path,
            coherence_reference="wikipedia",
            coherence_reference_path=reference_path,
            condition_failure_policy="continue-and-fail",
        )

    metrics_paths = list(
        (tmp_path / "dummy" / "default" / "good").glob("*/metrics_agg.json")
    )
    assert len(metrics_paths) == 1
    assert (metrics_paths[0].parent / "COMPLETE.json").exists()
    assert (tmp_path / "condition_index.json").exists()
    assert (tmp_path / "failed_conditions.json").exists()


def test_collection_excludes_only_the_degenerate_iteration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dictionary = Dictionary([["alpha", "beta"]])
    runtime = _runtime()
    args = SimpleNamespace(
        dataset="dummy",
        iteration=[0, 1],
        coherence_split="train",
        coherence_min_token_len=1,
        language="english",
        delimiter=" / ",
        ja_replace_num=True,
        ja_dicdir=None,
        ja_require_unidic=True,
        dict_no_below=1,
        dict_no_above=1.0,
        dict_exclude_tokens=frozenset({"<NUM>"}),
        dict_exclude_single_alpha=False,
        dict_exclude_with_digit=False,
        dict_exclude_hiragana_only=False,
        checkpoint_mode="off",
        checkpoint_root=None,
        out_root=tmp_path,
        condition_failure_policy="exclude-condition",
    )
    monkeypatch.setattr(
        metrics_module,
        "_resolve_split_csvs_and_target_column",
        lambda **_kwargs: (None, "target_str"),
    )
    monkeypatch.setattr(
        metrics_module,
        "_get_corpus_bundle_cached",
        lambda **_kwargs: ([["alpha", "beta"]], dictionary, [[(0, 1), (1, 1)]]),
    )
    monkeypatch.setattr(
        metrics_module,
        "_topic_word_checkpoint_identity",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        metrics_module,
        "topic_word_checkpoint_dir",
        lambda **_kwargs: tmp_path / "checkpoint",
    )

    def resolve(*, iteration: int, **_kwargs):
        if iteration == 0:
            raise DegenerateTopicError(
                topic_id=1,
                eligible_word_count=1,
                required_topn=2,
                eligible_words=["<NUM>"],
                eligible_word_counts=[2, 1],
            )
        result = TopicWordsResult(
            topic_words=runtime.evaluation.topic_words,
            topic_word_source=runtime.evaluation.topic_word_source,
            score_mode=runtime.evaluation.score_mode,
            score_definition=runtime.evaluation.score_definition,
            runtime_payload=runtime,
        )
        return result, [["alpha", "beta"]], dictionary, [[(0, 1), (1, 1)]]

    monkeypatch.setattr(metrics_module, "_resolve_topic_words_result", resolve)
    failures: list[dict[str, object]] = []

    group = metrics_module._collect_pending_word_based_group(
        args=args,
        task=metrics_module.PendingWordBasedGroupTask(
            sort_index=0,
            data_run="default",
            model="mvtm",
            num_topics=2,
            category="all",
            progress_start=0,
        ),
        total_conditions=2,
        failure_sink=failures,
    )

    assert group is not None
    assert [item.iteration for item in group.iterations] == [1]
    assert failures[0]["iteration"] == 0
    assert failures[0]["status"] == "insufficient_topic_words"
    assert failures[0]["special_words"] == ["<NUM>"]


def test_collection_persists_partial_words_and_excludes_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dictionary = Dictionary([["alpha", "beta"]])
    runtime = replace(
        _runtime(words=[[("alpha", 1.0)], []]),
        empty_topic_ids=(1,),
    )
    args = SimpleNamespace(
        dataset="dummy",
        iteration=[0],
        coherence_split="train",
        coherence_min_token_len=1,
        language="english",
        delimiter=" / ",
        ja_replace_num=True,
        ja_dicdir=None,
        ja_require_unidic=True,
        dict_no_below=1,
        dict_no_above=1.0,
        dict_exclude_tokens=frozenset({"<NUM>"}),
        dict_exclude_single_alpha=False,
        dict_exclude_with_digit=False,
        dict_exclude_hiragana_only=False,
        checkpoint_mode="off",
        checkpoint_root=None,
        out_root=tmp_path,
        condition_failure_policy="exclude-condition",
        embedding_variant="ruri",
    )
    monkeypatch.setattr(
        metrics_module,
        "_resolve_split_csvs_and_target_column",
        lambda **_kwargs: (None, "target_str"),
    )
    monkeypatch.setattr(
        metrics_module,
        "_get_corpus_bundle_cached",
        lambda **_kwargs: ([["alpha", "beta"]], dictionary, [[(0, 1), (1, 1)]]),
    )
    monkeypatch.setattr(
        metrics_module,
        "_topic_word_checkpoint_identity",
        lambda **_kwargs: {
            "model_provenance": {
                "runner_key": "vmf_sentence_lda",
                "condition_fingerprint": "source-fingerprint",
            }
        },
    )
    monkeypatch.setattr(
        metrics_module,
        "topic_word_checkpoint_dir",
        lambda **_kwargs: tmp_path / "checkpoint",
    )
    monkeypatch.setattr(
        metrics_module,
        "_resolve_topic_words_result",
        lambda **_kwargs: (
            TopicWordsResult(
                topic_words=runtime.display_topic_words,
                topic_word_source=runtime.display_source,
                score_mode=runtime.display_score_mode,
                score_definition="test",
                runtime_payload=runtime,
            ),
            [["alpha", "beta"]],
            dictionary,
            [[(0, 1), (1, 1)]],
        ),
    )
    failures: list[dict[str, object]] = []

    group = metrics_module._collect_pending_word_based_group(
        args=args,
        task=metrics_module.PendingWordBasedGroupTask(
            sort_index=0,
            data_run="default",
            model="vmf",
            num_topics=2,
            category="商社・卸売",
            progress_start=0,
        ),
        total_conditions=1,
        failure_sink=failures,
    )

    assert group is None
    assert failures[0]["error_type"] == "EmptyTopicError"
    assert failures[0]["topic_ids"] == [1]
    current = Path(str(failures[0]["partial_artifact_current"]))
    assert current.is_file()
    pointer = json.loads(current.read_text(encoding="utf-8"))
    display_path = (
        Path(pointer["archive_dir"]) / pointer["artifacts"]["topic_words_display_topk"]
    )
    if not display_path.is_absolute():
        display_path = Path.cwd() / display_path
    display = json.loads(display_path.read_text(encoding="utf-8"))
    assert display["_meta"]["metrics_status"] == "excluded"
    assert display["results"]["per_iteration"][0]["topics"][1]["words"] == []


def test_collection_keeps_mvtm_empty_topics_under_fixed_k_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dictionary = Dictionary([["alpha", "beta"]])
    runtime = replace(
        _runtime(words=[[("alpha", 1.0)], []]),
        empty_topic_ids=(1,),
    )
    args = SimpleNamespace(
        dataset="dummy",
        iteration=[0],
        coherence_split="train",
        coherence_min_token_len=1,
        language="english",
        delimiter=" / ",
        ja_replace_num=True,
        ja_dicdir=None,
        ja_require_unidic=True,
        dict_no_below=1,
        dict_no_above=1.0,
        dict_exclude_tokens=frozenset(),
        dict_exclude_single_alpha=False,
        dict_exclude_with_digit=False,
        dict_exclude_hiragana_only=False,
        checkpoint_mode="off",
        checkpoint_root=None,
        out_root=tmp_path,
        condition_failure_policy="exclude-condition",
        mvtm_empty_topic_policy="fixed-k",
        embedding_variant="minilm",
    )
    monkeypatch.setattr(
        metrics_module,
        "_resolve_split_csvs_and_target_column",
        lambda **_kwargs: (None, "target_str"),
    )
    monkeypatch.setattr(
        metrics_module,
        "_get_corpus_bundle_cached",
        lambda **_kwargs: ([["alpha", "beta"]], dictionary, [[(0, 1), (1, 1)]]),
    )
    monkeypatch.setattr(
        metrics_module,
        "_topic_word_checkpoint_identity",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        metrics_module,
        "topic_word_checkpoint_dir",
        lambda **_kwargs: tmp_path / "checkpoint",
    )
    monkeypatch.setattr(
        metrics_module,
        "_resolve_topic_words_result",
        lambda **_kwargs: (
            TopicWordsResult(
                topic_words=runtime.display_topic_words,
                topic_word_source=runtime.display_source,
                score_mode=runtime.display_score_mode,
                score_definition="test",
                runtime_payload=runtime,
            ),
            [["alpha", "beta"]],
            dictionary,
            [[(0, 1), (1, 1)]],
        ),
    )
    failures: list[dict[str, object]] = []

    group = metrics_module._collect_pending_word_based_group(
        args=args,
        task=metrics_module.PendingWordBasedGroupTask(
            sort_index=0,
            data_run="default",
            model="mvtm",
            num_topics=2,
            category="all",
            progress_start=0,
        ),
        total_conditions=1,
        failure_sink=failures,
    )

    assert group is not None
    assert [item.iteration for item in group.iterations] == [0]
    assert group.iterations[0].topic_words == [[("alpha", 1.0)], []]
    assert failures == []


def test_shared_reference_scoring_applies_mvtm_fixed_k_policy(monkeypatch) -> None:
    runtime = replace(
        _runtime(words=[[("alpha", 1.0)], []]),
        empty_topic_ids=(1,),
    )
    group = metrics_module.PendingWordBasedGroup(
        data_run="default",
        model="mvtm",
        num_topics=2,
        category="all",
        iterations=[
            metrics_module.PendingWordBasedIteration(
                iteration=0,
                topic_words=runtime.display_topic_words,
                runtime_payload=runtime,
            )
        ],
        topic_word_source=runtime.display_source,
        topic_word_score_mode=runtime.display_score_mode,
        topic_word_score_definition="test",
    )

    def score(**kwargs):
        assert kwargs["topic_words"] == [[("alpha", 1.0)]]
        assert kwargs["metric_names"] == ["coherence", "diversity"]
        return {"coherence": 0.6, "diversity": 1.0}

    monkeypatch.setattr(
        metrics_module,
        "compute_shared_reference_coherence_scores",
        score,
    )
    args = SimpleNamespace(
        mvtm_empty_topic_policy="fixed-k",
        coherence_topn=1,
        diversity_topn=1,
        coherence_window_size=None,
        coherence_min_window_count=None,
    )

    scored = metrics_module._score_pending_word_based_group(
        group=group,
        args=args,
        metric_names=metrics_module._fixed_k_metric_names(["c_v"]),
        coherences=["c_v"],
        shared_counts=object(),
    )

    metrics = scored.per_iter_metrics[0]
    assert metrics["coherence"] == 0.3
    assert metrics["coherence_active_only"] == 0.6
    assert metrics["diversity"] == 0.5
    assert metrics["topic_utilization"] == 0.5

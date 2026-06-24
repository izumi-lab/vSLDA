from __future__ import annotations

from pathlib import Path

from src.evaluation.word_based import model_inputs


def test_resolve_preprocessed_corpus_path_uses_persisted_split_roots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    condition_dir = tmp_path / "condition"
    monkeypatch.setattr(
        model_inputs,
        "resolve_baseline_condition_dir",
        lambda **_kwargs: condition_dir,
    )

    train_path = model_inputs.resolve_preprocessed_corpus_path(
        model="sentlda",
        dataset="dataset",
        data_run="default",
        iteration=0,
        num_topics=10,
        category="all",
        split="train",
    )
    test_path = model_inputs.resolve_preprocessed_corpus_path(
        model="sentlda",
        dataset="dataset",
        data_run="default",
        iteration=0,
        num_topics=10,
        category="all",
        split="test",
    )

    assert train_path == condition_dir / "params" / "preprocessed_corpus.pkl"
    assert test_path == condition_dir / "infer" / "preprocessed_corpus.pkl"

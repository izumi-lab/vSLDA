from __future__ import annotations

import logging

from src.evaluation.classification.train import train


def test_train_logs_unknown_category_and_returns_none(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "src.evaluation.classification.train.get_dataset_targets",
        lambda *_args, **_kwargs: {"science": ["a", "b"]},
    )

    with caplog.at_level(logging.WARNING):
        result = train(
            category="missing",
            dataset="dummy",
            num_topics=10,
            it=0,
            classifiers_to_use=["svm"],
            vmf_assignment="hard",
        )

    assert result is None
    assert "unknown category 'missing' for dataset 'dummy'" in caplog.text

from __future__ import annotations

from src.baselines.models.ctm_helpers import WhiteSpacePreprocessingStopwords


def test_ctm_whitespace_preprocessing_normalizes_text_and_preserves_alignment() -> None:
    docs = [
        "CafE, 123 on THE hill!",
        "Numbers 999 stay when remove_numbers is off.",
        "",
    ]

    preprocessed, original, vocabulary, retained = WhiteSpacePreprocessingStopwords(
        docs,
        stopwords_list=["the", "on"],
        remove_numbers=False,
    ).preprocess()

    assert preprocessed == [
        "cafe 123 hill",
        "numbers 999 stay when remove numbers is off",
        "",
    ]
    assert original == docs
    assert retained == [0, 1, 2]
    assert set(vocabulary) == {
        "123",
        "999",
        "cafe",
        "hill",
        "is",
        "numbers",
        "off",
        "remove",
        "stay",
        "when",
    }


def test_ctm_whitespace_preprocessing_removes_numbers_when_enabled() -> None:
    preprocessed, _original, vocabulary, retained = WhiteSpacePreprocessingStopwords(
        ["Topic 101", "Topic 202"],
        remove_numbers=True,
    ).preprocess()

    assert preprocessed == ["topic", "topic"]
    assert vocabulary == ["topic"]
    assert retained == [0, 1]

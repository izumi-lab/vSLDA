"""Train / infer / persist round trip for the SAM baseline runner."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.baselines.models import sam
from src.baselines.models.sam import infer_sam, persist_sam_run, train_sam
from src.baselines.params import parse_sam_params
from src.data.preprocessing import PreprocessedDocument

_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def _document(tokens: list[str]) -> PreprocessedDocument:
    return PreprocessedDocument(
        raw_text=" ".join(tokens),
        sentences_raw=[" ".join(tokens)],
        sentences_tokenized=[list(tokens)],
        sentences_joined=[" ".join(tokens)],
        document_tokens=list(tokens),
    )


def _corpus(seed: int, count: int) -> list[PreprocessedDocument]:
    rng = np.random.default_rng(seed)
    documents = []
    for _ in range(count):
        size = int(rng.integers(3, 6))
        documents.append(_document(list(rng.choice(_VOCAB, size=size, replace=True))))
    return documents


_COMMON = dict(
    targets=None,
    text_column="data",
    target_column=None,
    delimiter=" / ",
    language="english",
    segmenter="delimiter",
    tokenizer="default",
    ja_replace_num=True,
    ja_stopwords_path=None,
    ja_dicdir=None,
    ja_require_unidic=False,
    num_topics=3,
    use_legacy=False,
)


@pytest.fixture()
def sam_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    train_docs = _corpus(seed=0, count=25)
    test_docs = _corpus(seed=1, count=10)

    def _fake_loader(*, csv_paths, **_kwargs):
        return train_docs if str(csv_paths[0]) == "train.csv" else test_docs

    monkeypatch.setattr(sam, "load_preprocessed_documents", _fake_loader)

    params = parse_sam_params(
        {"min_df": 1, "max_df": 1.0, "num_iterations": 5, "infer_num_iterations": 20}
    )
    train_result = train_sam(
        train_csvs=["train.csv"],
        params=params,
        train_dir=tmp_path / "unused",
        effective_random_state=3,
        **_COMMON,
    )
    infer_result = infer_sam(
        test_csvs=["test.csv"],
        params=params,
        train_result=train_result,
        **_COMMON,
    )
    train_dir = tmp_path / "params"
    infer_dir = tmp_path / "infer"
    artifacts = persist_sam_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=train_dir,
        infer_dir=infer_dir,
        category="all",
    )
    return train_result, infer_result, artifacts, train_dir, infer_dir


def test_train_produces_simplex_proportions_and_unit_topics(sam_run) -> None:
    train_result, _, _, _, _ = sam_run
    assert train_result.train_doc_topic.shape == (
        len(train_result.train_preprocessed),
        3,
    )
    assert np.allclose(train_result.train_doc_topic.sum(axis=1), 1.0)
    assert np.all(train_result.train_doc_topic >= 0.0)

    scores = train_result.topic_word_scores
    assert scores.shape == (3, train_result.vocabulary.size)
    assert np.allclose(np.linalg.norm(scores, axis=1), 1.0)


def test_infer_uses_the_training_vocabulary(sam_run) -> None:
    train_result, infer_result, _, _, _ = sam_run
    assert infer_result.test_doc_topic.shape[1] == 3
    assert np.allclose(infer_result.test_doc_topic.sum(axis=1), 1.0)
    assert len(infer_result.test_preprocessed) == len(
        infer_result.test_selection.documents
    )


def test_persist_writes_every_declared_artifact(sam_run) -> None:
    _, _, artifacts, train_dir, infer_dir = sam_run
    for path in artifacts.as_dict().values():
        assert Path(path).exists()

    assert (train_dir / "sam.pkl").exists()
    assert (train_dir / "topic_word_scores.pkl").exists()
    assert (train_dir / "vocabulary.json").exists()
    assert (train_dir / "idf.pkl").exists()
    assert (train_dir / "params.json").exists()
    assert (train_dir / "topic_term_weights.json").exists()
    assert (infer_dir / "all.pkl").exists()
    assert (infer_dir / "all_doc_topic_soft.pkl").exists()


def test_persisted_topic_word_matrix_is_topic_by_word(sam_run) -> None:
    train_result, _, _, train_dir, _ = sam_run
    scores = pickle.loads((train_dir / "topic_word_scores.pkl").read_bytes())
    vocabulary = json.loads((train_dir / "vocabulary.json").read_text())
    metadata = json.loads((train_dir / "params.json").read_text())

    assert metadata["topic_word_orientation"] == "topic_by_word"
    assert scores.shape == (metadata["num_topics"], len(vocabulary))
    assert len(vocabulary) == metadata["vocab_size"]
    # Internal layout is (V, T); the transpose must happen exactly once.
    assert np.allclose(scores.T, train_result.fit.state.mutilde)


def test_params_json_records_convergence_and_positive_mass(sam_run) -> None:
    _, _, _, train_dir, _ = sam_run
    metadata = json.loads((train_dir / "params.json").read_text())

    assert isinstance(metadata["converged"], bool)
    assert len(metadata["elbo_trace"]) == metadata["iterations"]
    assert np.all(np.diff(metadata["elbo_trace"]) >= -1e-9)
    assert metadata["feature_scheme"] == "tfidf"
    assert set(metadata["hyperparameters"]) == {"xi", "kappa", "kappa0", "alpha"}
    fractions = metadata["positive_mass_fraction_by_topic"]
    assert len(fractions) == metadata["num_topics"]
    assert all(0.0 <= value <= 1.0 for value in fractions)
    assert "resolved_mean_resultants" in metadata


def test_effective_seed_reaches_inference_and_params_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sam_run
) -> None:
    # SamParams.random_state defaults to None, so seeding only fit_sam left the
    # fold-in jitter unseeded and params.json reporting null.
    _, _, _, train_dir, _ = sam_run
    assert json.loads((train_dir / "params.json").read_text())["random_state"] == 3

    train_docs = _corpus(seed=0, count=25)
    test_docs = _corpus(seed=1, count=10)

    def _fake_loader(*, csv_paths, **_kwargs):
        return train_docs if str(csv_paths[0]) == "train.csv" else test_docs

    monkeypatch.setattr(sam, "load_preprocessed_documents", _fake_loader)
    params = parse_sam_params(
        {"min_df": 1, "max_df": 1.0, "num_iterations": 5, "infer_num_iterations": 3}
    )
    train_result = train_sam(
        train_csvs=["train.csv"],
        params=params,
        train_dir=tmp_path / "unused",
        effective_random_state=3,
        **_COMMON,
    )
    assert train_result.params.random_state == 3
    first = infer_sam(
        test_csvs=["test.csv"], params=params, train_result=train_result, **_COMMON
    )
    second = infer_sam(
        test_csvs=["test.csv"], params=params, train_result=train_result, **_COMMON
    )
    assert np.array_equal(first.test_doc_topic, second.test_doc_topic)


def test_topic_term_weights_expose_both_signs(sam_run) -> None:
    _, _, _, train_dir, _ = sam_run
    weights = json.loads((train_dir / "topic_term_weights.json").read_text())
    assert set(weights) == {"positive", "negative"}
    for topic_positive, topic_negative in zip(
        weights["positive"], weights["negative"], strict=True
    ):
        assert topic_positive[0][1] >= topic_positive[-1][1]
        assert topic_negative[0][1] <= topic_negative[-1][1]
        # The most positive term must outrank the most negative one.
        assert topic_positive[0][1] >= topic_negative[0][1]


def test_persisted_selection_keeps_source_raw_doc_indices(sam_run) -> None:
    train_result, infer_result, _, train_dir, infer_dir = sam_run
    train_selection = json.loads(
        (train_dir / "preprocessing_selection.json").read_text()
    )
    infer_selection = json.loads(
        (infer_dir / "preprocessing_selection.json").read_text()
    )
    assert train_selection["raw_doc_indices"] == list(
        train_result.train_selection.raw_doc_indices
    )
    assert infer_selection["raw_doc_indices"] == list(
        infer_result.test_selection.raw_doc_indices
    )
    assert len(train_selection["raw_doc_indices"]) == len(
        train_result.train_preprocessed
    )


def test_persist_rejects_misaligned_doc_topic_rows(sam_run) -> None:
    train_result, infer_result, _, train_dir, infer_dir = sam_run
    from dataclasses import replace

    broken = replace(train_result, train_doc_topic=train_result.train_doc_topic[:-1])
    with pytest.raises(ValueError, match="doc-topic rows"):
        persist_sam_run(
            train_result=broken,
            infer_result=infer_result,
            train_dir=train_dir,
            infer_dir=infer_dir,
            category="all",
        )

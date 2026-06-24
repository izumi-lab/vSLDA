from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines.models import bertopic_kmeans
from src.baselines.models.bertopic_kmeans import (
    _document_texts,
    _extract_topic_words,
    _load_preprocessed_documents_preserve_rows,
    _softmax_negative_distances,
    _validate_doc_topic,
    infer_bertopic_kmeans,
    persist_bertopic_kmeans_run,
    train_bertopic_kmeans,
)
from src.baselines.params import BertopicKMeansParams
from src.core.artifacts import load_pickle
from src.data.preprocessing import PreprocessedDocument


class _RoundTripEncoder:
    accepts_tokenized = False

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def encode(self, texts, show_progress_bar: bool = False) -> np.ndarray:
        _ = show_progress_bar
        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([0.0, 0.0])
            elif "gamma" in text:
                vectors.append([3.0, 0.0])
            else:
                vectors.append([0.0, 3.0])
        return np.asarray(vectors, dtype=np.float64)


class _RoundTripUmap:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding_ = np.asarray(embedding, dtype=np.float64)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(embeddings, dtype=np.float64)


class _RoundTripKMeans:
    cluster_centers_ = np.asarray([[0.0, 0.0], [3.0, 0.0]], dtype=np.float64)


class _RoundTripMapper:
    def get_mappings(self, *, original_topics: bool):
        assert original_topics is True
        return {0: 0, 1: 1}


class _RoundTripTopicModel:
    def __init__(self, embedding: np.ndarray) -> None:
        self.umap_model = _RoundTripUmap(embedding)
        self.hdbscan_model = _RoundTripKMeans()
        self.topic_mapper_ = _RoundTripMapper()

    def get_topic(self, topic_id: int):
        return [(f"topic-{topic_id}", 1.0)]


def test_softmax_negative_distances_returns_doc_topic_distribution() -> None:
    doc_topic = _softmax_negative_distances(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=float),
        temperature=1.0,
    )

    assert doc_topic.shape == (2, 2)
    np.testing.assert_allclose(doc_topic.sum(axis=1), [1.0, 1.0])
    assert doc_topic[0, 0] > doc_topic[0, 1]


def test_validate_doc_topic_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="doc-topic shape"):
        _validate_doc_topic(
            np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            num_docs=1,
            num_topics=2,
            name="train",
        )


def test_extract_topic_words_reorders_bertopic_ids_to_kmeans_label_order() -> None:
    class Mapper:
        def get_mappings(self, *, original_topics: bool):
            assert original_topics is True
            return {0: 2, 1: 0, 2: 1}

    class TopicModel:
        topic_mapper_ = Mapper()

        def get_topic(self, topic_id: int):
            return [(f"topic-{topic_id}", 1.0)]

    assert _extract_topic_words(TopicModel(), num_topics=3) == [
        [("topic-2", 1.0)],
        [("topic-0", 1.0)],
        [("topic-1", 1.0)],
    ]


def test_extract_topic_words_rejects_incomplete_topic_mapping() -> None:
    class Mapper:
        def get_mappings(self, *, original_topics: bool):
            assert original_topics is True
            return {0: 0}

    class TopicModel:
        topic_mapper_ = Mapper()

        def get_topic(self, topic_id: int):
            return [(f"topic-{topic_id}", 1.0)]

    with pytest.raises(ValueError, match="complete k-means label mapping"):
        _extract_topic_words(TopicModel(), num_topics=2)


def test_load_preprocessed_documents_preserves_filtered_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "data": ["alpha / beta", "skip", "gamma"],
            "target_str": ["a", "b", "a"],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    docs = _load_preprocessed_documents_preserve_rows(
        csv_paths=[str(csv_path)],
        text_column="data",
        target_column="target_str",
        targets=["a"],
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_dicdir=None,
        ja_require_unidic=True,
    )

    assert len(docs) == 2
    assert _document_texts(docs) == ["alpha beta", "gamma"]


def test_load_preprocessed_documents_fails_on_empty_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"data": ["   "], "target_str": ["a"]}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    with pytest.raises(ValueError, match="non-empty document text"):
        _load_preprocessed_documents_preserve_rows(
            csv_paths=[str(csv_path)],
            text_column="data",
            target_column="target_str",
            targets=["a"],
            delimiter=" / ",
            language="english",
            segmenter="delimiter",
            tokenizer="simple",
            ja_replace_num=True,
            ja_dicdir=None,
            ja_require_unidic=True,
        )


@pytest.mark.slow
def test_train_infer_and_persist_bertopic_kmeans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_docs = [
        PreprocessedDocument(
            raw_text="alpha beta",
            sentences_raw=["alpha beta"],
            sentences_tokenized=[["alpha", "beta"]],
            sentences_joined=["alpha beta"],
            document_tokens=["alpha", "beta"],
        ),
        PreprocessedDocument(
            raw_text="gamma delta",
            sentences_raw=["gamma delta"],
            sentences_tokenized=[["gamma", "delta"]],
            sentences_joined=["gamma delta"],
            document_tokens=["gamma", "delta"],
        ),
        PreprocessedDocument(
            raw_text="epsilon zeta",
            sentences_raw=["epsilon zeta"],
            sentences_tokenized=[["epsilon", "zeta"]],
            sentences_joined=["epsilon zeta"],
            document_tokens=["epsilon", "zeta"],
        ),
    ]
    test_docs = [
        PreprocessedDocument(
            raw_text="alpha",
            sentences_raw=["alpha"],
            sentences_tokenized=[["alpha"]],
            sentences_joined=["alpha"],
            document_tokens=["alpha"],
        ),
        PreprocessedDocument(
            raw_text="gamma",
            sentences_raw=["gamma"],
            sentences_tokenized=[["gamma"]],
            sentences_joined=["gamma"],
            document_tokens=["gamma"],
        ),
    ]

    def _fake_loader(*, csv_paths, **_kwargs):
        return train_docs if str(csv_paths[0]) == "train.csv" else test_docs

    def _fake_fit_bertopic_kmeans(**kwargs):
        return _RoundTripTopicModel(kwargs["embeddings"])

    monkeypatch.setattr(
        bertopic_kmeans,
        "_load_preprocessed_documents_preserve_rows",
        _fake_loader,
    )
    monkeypatch.setattr(bertopic_kmeans, "SentenceEncoder", _RoundTripEncoder)
    monkeypatch.setattr(
        bertopic_kmeans, "fit_encoder_on_documents", lambda *_args: None
    )
    monkeypatch.setattr(
        bertopic_kmeans,
        "_fit_bertopic_kmeans",
        _fake_fit_bertopic_kmeans,
    )

    train_result = train_bertopic_kmeans(
        train_csvs=["train.csv"],
        test_csvs=["test.csv"],
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
        ja_require_unidic=True,
        num_topics=2,
        encoder_device="cpu",
        effective_random_state=7,
        params=BertopicKMeansParams(encoder_model_name="fake-model"),
        train_dir=tmp_path / "train",
        use_legacy=False,
    )
    infer_result = infer_bertopic_kmeans(train_result=train_result)
    artifacts = persist_bertopic_kmeans_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "train",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    assert train_result.train_doc_topic.shape == (3, 2)
    assert infer_result.test_doc_topic.shape == (2, 2)
    assert np.allclose(train_result.train_doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(infer_result.test_doc_topic.sum(axis=1), 1.0, atol=1e-6)
    assert artifacts.train_path.name == "bertopic_kmeans.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert artifacts.extras["kmeans_model"].name == "kmeans.pkl"
    assert load_pickle(tmp_path / "train" / "bertopic_kmeans.pkl").shape == (3, 2)

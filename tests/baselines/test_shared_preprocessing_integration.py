from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from gensim.models import KeyedVectors

from src.baselines.dataset_adapters import load_preprocessed_documents
from src.baselines.models.ctm import infer_ctm, train_ctm
from src.baselines.models.gaussianlda import _prepare_docs
from src.baselines.params import CtmParams
from src.data.corpus import load_preprocessed_corpus
from src.data.preprocessing import PreprocessedDocument


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "data": rows,
            "target_str": ["all"] * len(rows),
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def test_shared_preprocessing_matches_between_vmf_and_baseline_english(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path, ["Alpha beta / Gamma delta!"])

    vmf_docs = load_preprocessed_corpus(
        csv_path,
        language="english",
        delimiter=" / ",
        segmenter="delimiter",
        tokenizer="simple",
        target_filter=["all"],
    ).documents
    baseline_docs = load_preprocessed_documents(
        csv_paths=[str(csv_path)],
        text_column="data",
        target_column="target_str",
        targets=["all"],
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
    )

    assert vmf_docs == baseline_docs
    assert len(vmf_docs) == 1
    assert vmf_docs[0].sentences_raw == ["Alpha beta", "Gamma delta!"]
    assert vmf_docs[0].sentences_tokenized == [["alpha", "beta"], ["gamma", "delta"]]
    assert vmf_docs[0].document_tokens == ["alpha", "beta", "gamma", "delta"]


def test_shared_preprocessing_matches_between_vmf_and_baseline_japanese(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = _write_csv(tmp_path, ["東京へ行く / 明日走る"])

    fake_tokens = {
        "東京へ行く": ["東京", "行く"],
        "明日走る": ["明日", "走る"],
    }

    monkeypatch.setattr(
        "src.data.text_processing.tokenize_japanese_text",
        lambda text, **_kwargs: list(fake_tokens.get(text, [])),
    )

    vmf_docs = load_preprocessed_corpus(
        csv_path,
        language="ja",
        delimiter=" / ",
        segmenter="delimiter",
        tokenizer="default",
        target_filter=["all"],
        ja_require_unidic=False,
    ).documents
    baseline_docs = load_preprocessed_documents(
        csv_paths=[str(csv_path)],
        text_column="data",
        target_column="target_str",
        targets=["all"],
        delimiter=" / ",
        language="ja",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=False,
    )

    assert vmf_docs == baseline_docs
    assert len(vmf_docs) == 1
    assert vmf_docs[0].sentences_raw == ["東京へ行く", "明日走る"]
    assert vmf_docs[0].sentences_tokenized == [["東京", "行く"], ["明日", "走る"]]
    assert vmf_docs[0].document_tokens == ["東京", "行く", "明日", "走る"]


def test_ctm_uses_shared_preprocessed_text_even_when_legacy_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [
        PreprocessedDocument(
            raw_text="Alpha beta / Gamma delta!",
            sentences_raw=["Alpha beta", "Gamma delta!"],
            sentences_tokenized=[["alpha", "beta"], ["gamma", "delta"]],
            sentences_joined=["alpha beta", "gamma delta"],
            document_tokens=["alpha", "beta", "gamma", "delta"],
        )
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.baselines.models.ctm._load_preprocessed_documents",
        lambda **_kwargs: docs,
    )

    class DummyTopicPreparation:
        def __init__(self, model_name: str) -> None:
            captured["model_name"] = model_name
            self.vocab = ["alpha", "beta", "gamma", "delta"]

        def fit(
            self,
            *,
            text_for_contextual: list[str],
            text_for_bow: list[str],
        ) -> SimpleNamespace:
            captured["text_for_contextual"] = text_for_contextual
            captured["text_for_bow"] = text_for_bow
            return SimpleNamespace(X_contextual=np.zeros((1, 4), dtype=np.float32))

    class DummyCombinedTM:
        def __init__(self, **kwargs) -> None:
            captured["ctm_init"] = kwargs

        def fit(self, dataset: SimpleNamespace) -> None:
            captured["fit_dataset"] = dataset

        def get_doc_topic_distribution(
            self,
            dataset: SimpleNamespace,
            _num_samples: int,
        ) -> np.ndarray:
            captured["distribution_dataset"] = dataset
            return np.asarray([[1.0, 0.0]], dtype=float)

    monkeypatch.setattr(
        "src.baselines.models.ctm.TopicModelDataPreparation",
        DummyTopicPreparation,
    )
    monkeypatch.setattr("src.baselines.models.ctm.CombinedTM", DummyCombinedTM)

    result = train_ctm(
        train_csvs=[str(tmp_path / "unused.csv")],
        targets=["all"],
        text_column="data",
        target_column="target_str",
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=False,
        num_topics=2,
        params=CtmParams(
            contextual_encode_prefix="query: ",
            num_epochs=1,
            num_samples=1,
            batch_size_cap=16,
        ),
        train_dir=tmp_path,
        use_legacy=True,
    )

    assert captured["model_name"] == "sentence-transformers/all-mpnet-base-v2"
    assert captured["text_for_bow"] == ["alpha beta gamma delta"]
    assert captured["text_for_contextual"] == ["query: Alpha beta Gamma delta!"]
    assert captured["ctm_init"]["contextual_size"] == 4
    assert captured["fit_dataset"] is captured["distribution_dataset"]
    assert result.train_preprocessed == docs


def test_ctm_custom_embeddings_keep_contextual_model_for_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    docs = [
        PreprocessedDocument(
            raw_text="Alpha beta",
            sentences_raw=["Alpha beta"],
            sentences_tokenized=[["alpha", "beta"]],
            sentences_joined=["alpha beta"],
            document_tokens=["alpha", "beta"],
        )
    ]

    monkeypatch.setattr(
        "src.baselines.models.ctm._load_preprocessed_documents",
        lambda **_kwargs: docs,
    )

    class DummyEncoder:
        accepts_tokenized = False

        def encode(self, texts: list[str]) -> np.ndarray:
            captured.setdefault("encoded_texts", []).append(list(texts))
            return np.zeros((len(texts), 4), dtype=np.float32)

    class DummyTopicPreparation:
        def __init__(self, model_name: str | None) -> None:
            captured["model_name"] = model_name
            self.contextualized_model = model_name
            self.vocab = ["alpha", "beta"]

        def fit(
            self,
            *,
            text_for_contextual: list[str],
            text_for_bow: list[str],
            custom_embeddings: np.ndarray | None = None,
        ) -> SimpleNamespace:
            assert custom_embeddings is not None
            return SimpleNamespace(X_contextual=custom_embeddings)

        def transform(
            self,
            *,
            text_for_contextual: list[str],
            text_for_bow: list[str],
            custom_embeddings: np.ndarray | None = None,
        ) -> SimpleNamespace:
            if self.contextualized_model is None:
                raise Exception("missing contextualized model")
            assert custom_embeddings is not None
            captured["transform_custom_embeddings_shape"] = custom_embeddings.shape
            return SimpleNamespace(X_contextual=custom_embeddings)

    class DummyCombinedTM:
        def __init__(self, **_kwargs) -> None:
            pass

        def fit(self, _dataset: SimpleNamespace) -> None:
            pass

        def get_doc_topic_distribution(
            self,
            dataset: SimpleNamespace,
            n_samples: int,
        ) -> np.ndarray:
            _ = n_samples
            return np.asarray([[1.0, 0.0]] * len(dataset.X_contextual), dtype=float)

    monkeypatch.setattr(
        "src.baselines.models.ctm._build_contextual_encoder",
        lambda **_kwargs: DummyEncoder(),
    )
    monkeypatch.setattr(
        "src.baselines.models.ctm.TopicModelDataPreparation",
        DummyTopicPreparation,
    )
    monkeypatch.setattr("src.baselines.models.ctm.CombinedTM", DummyCombinedTM)

    params = CtmParams(
        contextual_model_name="baai/bge-base-en-v1.5",
        use_custom_embeddings=True,
        num_epochs=1,
        num_samples=1,
    )
    train_result = train_ctm(
        train_csvs=[str(tmp_path / "unused_train.csv")],
        targets=["all"],
        text_column="data",
        target_column="target_str",
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=False,
        num_topics=2,
        params=params,
        train_dir=tmp_path,
        use_legacy=False,
    )

    infer_ctm(
        train_result=train_result,
        test_csvs=[str(tmp_path / "unused_test.csv")],
        targets=["all"],
        text_column="data",
        target_column="target_str",
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=False,
        num_topics=2,
        params=params,
        use_legacy=False,
    )

    assert captured["model_name"] == "baai/bge-base-en-v1.5"
    assert captured["transform_custom_embeddings_shape"] == (1, 4)


def test_gaussianlda_prepare_docs_uses_shared_tokens_even_when_legacy_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [
        PreprocessedDocument(
            raw_text="Alpha beta / Gamma delta!",
            sentences_raw=["Alpha beta", "Gamma delta!"],
            sentences_tokenized=[["alpha", "beta"], ["gamma", "delta"]],
            sentences_joined=["alpha beta", "gamma delta"],
            document_tokens=["alpha", "beta", "gamma", "delta"],
        )
    ]
    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(
        ["alpha", "beta", "gamma", "delta"],
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )

    monkeypatch.setattr(
        "src.baselines.models.gaussianlda.load_preprocessed_documents",
        lambda **_kwargs: docs,
    )

    corpus, embeddings, vocab, local_vectors, returned_docs = _prepare_docs(
        csv_paths=["unused.csv"],
        targets=["all"],
        text_column="data",
        target_column="target_str",
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="simple",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=False,
        use_legacy=True,
        word2vec=vectors,
        wikientvec_cache_dir=None,
    )

    assert corpus == [[0, 1, 2, 3]]
    assert embeddings.shape == (4, 2)
    assert vocab == ["alpha", "beta", "gamma", "delta"]
    assert local_vectors is None
    assert returned_docs == docs

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("torch")

from src.baselines.models.senclu import SenCluTrainResult, infer_senclu  # noqa: E402
from src.baselines.models.senclu_internal import SenClu  # noqa: E402


def test_infer_senclu_uses_precomputed_outputs_without_trainer_state() -> None:
    train_result = SenCluTrainResult(
        train_doc_topic=np.asarray([[0.7, 0.3]], dtype=np.float32),
        test_doc_topic=np.asarray([[0.2, 0.8]], dtype=np.float32),
        train_sentence_topic_soft=[
            np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        ],
        test_sentence_topic_soft=[
            np.asarray([[0.25, 0.75]], dtype=np.float32),
        ],
        train_preprocessed=[],
        test_preprocessed=[],
    )

    infer_result = infer_senclu(train_result=train_result)

    assert not hasattr(train_result, "trainer")
    assert infer_result.test_doc_topic.dtype == float
    assert infer_result.test_sentence_topic_soft[0].dtype == float
    np.testing.assert_allclose(infer_result.test_doc_topic, [[0.2, 0.8]])
    np.testing.assert_allclose(
        infer_result.test_sentence_topic_soft[0],
        [[0.25, 0.75]],
    )


def test_senclu_fit_transform_supports_explicit_cpu_device(
    monkeypatch,
) -> None:
    trainer = SenClu(device="cpu")

    def _fake_get_encoded_sentences(docs, device):
        assert device == "cpu"
        return docs, [np.asarray([[1.0, 0.0]], dtype=np.float32) for _doc in docs]

    def _fake_compute_topic_model(embdocs, device, ntopics, nepoch=20, alpha=1):
        assert device == "cpu"
        doc_count = len(embdocs)
        ptd = np.full((ntopics, doc_count), 1.0 / ntopics, dtype=np.float32)
        vec_t = np.zeros((embdocs[0].shape[1], ntopics), dtype=np.float32)
        assign = [np.zeros(len(doc), dtype=np.int64) for doc in embdocs]
        prob = [np.ones(len(doc), dtype=np.float32) for doc in embdocs]
        return ptd, vec_t, assign, prob

    monkeypatch.setattr(trainer, "getEncodedSentences", _fake_get_encoded_sentences)
    monkeypatch.setattr(
        trainer,
        "computeTopicModel_SenClu",
        _fake_compute_topic_model,
    )

    train_docs = [["train sentence"], ["train sentence 2"]]
    test_docs = [["test sentence"]]

    train_ptd, test_ptd = trainer.fit_transform(
        train_docs=train_docs,
        test_docs=test_docs,
        nTopics=2,
        verbose=False,
    )

    assert train_ptd.shape == (2, 2)
    assert test_ptd.shape == (1, 2)


def test_senclu_get_encoded_sentences_uses_configured_encoder_and_prefix(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def eval(self) -> None:
            captured["eval"] = True

        def encode(self, sentences):
            captured["sentences"] = list(sentences)
            return np.asarray([[1.0, 0.0] for _ in sentences], dtype=np.float32)

    monkeypatch.setattr(
        "src.baselines.models.senclu_internal.SentenceTransformer",
        FakeSentenceTransformer,
    )
    monkeypatch.setattr(
        "src.baselines.models.senclu_internal.multiprocessing.current_process",
        lambda: SimpleNamespace(daemon=True),
    )

    trainer = SenClu(
        device="cpu",
        encoder_model_name="cl-nagoya/ruri-v3-130m",
        encode_prefix="トピック: ",
    )
    trainer.verbose = False

    sendocs, embdocs = trainer.getEncodedSentences([["これは文です"]], "cpu")

    assert captured["model_name"] == "cl-nagoya/ruri-v3-130m"
    assert captured["device"] == "cpu"
    assert captured["sentences"] == ["トピック: これは文です"]
    assert sendocs == [["これは文です"]]
    assert embdocs[0].shape == (1, 2)


def test_senclu_fit_transform_reuses_existing_sentence_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trainer = SenClu(device="cpu")
    cache_dir = tmp_path / "senclu-cache"
    cache_dir.mkdir()
    sendocs = [["cached train"], ["cached test"]]
    embdocs = [
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    ]
    with (cache_dir / "topicDat.pic").open("wb") as handle:
        pickle.dump([sendocs, embdocs], handle)

    def _unexpected_encode(_docs, _device):
        raise AssertionError("existing sentence cache should be loaded")

    def _fake_compute_topic_model(loaded_embdocs, device, ntopics, nepoch=20, alpha=1):
        assert device == "cpu"
        assert len(loaded_embdocs) == 1
        doc_count = len(loaded_embdocs)
        ptd = np.full((ntopics, doc_count), 1.0 / ntopics, dtype=np.float32)
        vec_t = np.zeros((loaded_embdocs[0].shape[1], ntopics), dtype=np.float32)
        assign = [np.zeros(len(doc), dtype=np.int64) for doc in loaded_embdocs]
        prob = [np.ones(len(doc), dtype=np.float32) for doc in loaded_embdocs]
        return ptd, vec_t, assign, prob

    monkeypatch.setattr(trainer, "getEncodedSentences", _unexpected_encode)
    monkeypatch.setattr(
        trainer,
        "computeTopicModel_SenClu",
        _fake_compute_topic_model,
    )

    train_ptd, test_ptd = trainer.fit_transform(
        train_docs=[["raw train"]],
        test_docs=[["raw test"]],
        nTopics=2,
        loadAndStoreInFolder=str(cache_dir),
        verbose=False,
    )

    assert train_ptd.shape == (1, 2)
    assert test_ptd.shape == (1, 2)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.topic_pairs import inputs as inputs_module
from src.evaluation.topic_pairs.inputs import (
    SentenceAlignmentError,
    assert_same_sentences,
    embedding_cache_dir,
    encode_train_corpus,
    encoder_fingerprint,
    load_fine_labels,
    model_embedding_variant,
    normalize_model_name,
    reference_model,
    sentence_fingerprint,
    sentence_labels,
    unit_normalize,
)
from tests.evaluation.topic_pairs.conftest import make_corpus


def test_model_names_and_variants() -> None:
    assert normalize_model_name("vmf_sentence_lda") == "vmf"
    assert normalize_model_name("gaussian") == "sentence_gaussianlda"
    with pytest.raises(ValueError, match="Unsupported model"):
        normalize_model_name("bleilda")
    assert reference_model(["sentlda", "sentence_gaussianlda", "vmf"]) == "vmf"
    assert reference_model(["sentlda", "gaussian"]) == "sentence_gaussianlda"
    assert model_embedding_variant("vmf", "minilm") == "minilm"
    assert model_embedding_variant("sentence_gaussianlda", "minilm") == "minilm_norm"
    assert model_embedding_variant("sentlda", "minilm") is None


def test_sentence_fingerprint_is_order_sensitive() -> None:
    assert sentence_fingerprint(["a", "b"]) == sentence_fingerprint(["a", "b"])
    assert sentence_fingerprint(["a", "b"]) != sentence_fingerprint(["b", "a"])
    assert sentence_fingerprint(["ab", ""]) != sentence_fingerprint(["a", "b"])


def test_assert_same_sentences_detects_every_mismatch() -> None:
    reference = make_corpus([["s0", "s1"], ["s2"]])
    assert_same_sentences(
        reference, make_corpus([["s0", "s1"], ["s2"]], model="sentlda")
    )
    with pytest.raises(SentenceAlignmentError, match="documents"):
        assert_same_sentences(reference, make_corpus([["s0", "s1"]], model="sentlda"))
    with pytest.raises(SentenceAlignmentError, match="sentences in document 1"):
        assert_same_sentences(
            reference, make_corpus([["s0", "s1"], ["s2", "s3"]], model="sentlda")
        )
    with pytest.raises(SentenceAlignmentError, match="sha1"):
        assert_same_sentences(
            reference, make_corpus([["s0", "s1"], ["different"]], model="sentlda")
        )


def test_sentence_labels_repeat_document_labels() -> None:
    labels = sentence_labels(np.asarray([3, 1]), np.asarray([0, 2, 3]))
    assert labels.tolist() == [3, 3, 1]
    with pytest.raises(ValueError, match="aligned"):
        sentence_labels(np.asarray([3]), np.asarray([0, 2, 3]))


def test_unit_normalize_rejects_zero_rows() -> None:
    unit = unit_normalize(np.asarray([[3.0, 4.0]], dtype=np.float32))
    assert unit.tolist() == [[0.6, 0.8]]
    with pytest.raises(ValueError, match="zero"):
        unit_normalize(np.asarray([[0.0, 0.0]]))


class _FakeEncoder:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls = 0

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences, **kwargs):
        self.calls += 1
        return (
            np.arange(len(sentences) * self.dimension, dtype=np.float32).reshape(
                len(sentences), self.dimension
            )
            + 1.0
        )


def test_encode_train_corpus_round_trips_through_the_cache(tmp_path: Path) -> None:
    corpus = make_corpus([["s0", "s1"], ["s2"]])
    encoder = _FakeEncoder()
    factory_calls: list[dict] = []

    def _factory(config, device, batch_override):
        factory_calls.append({"config": dict(config), "device": device})
        return encoder, 8

    config = {"model_name": "fake-encoder", "backend": "sentence_transformers"}
    common = dict(
        encoder_config=config,
        cache_root=tmp_path / "cache",
        dataset="dummy",
        data_run="default",
        category="a",
        split="train",
        device="cpu",
        encoder_factory=_factory,
    )
    first = encode_train_corpus(corpus, **common)
    assert first.cache_hit is False
    assert first.embeddings.shape == (3, 3)
    assert first.embeddings.dtype == np.float32
    assert first.doc_offsets.tolist() == [0, 2, 3]
    assert first.manifest["sentence_sha1"] == corpus.sentence_sha1
    assert first.manifest["encoder_fingerprint"] == encoder_fingerprint(config)
    assert (first.cache_dir / "COMPLETE.json").exists()
    assert first.cache_dir == embedding_cache_dir(
        tmp_path / "cache",
        dataset="dummy",
        data_run="default",
        category="a",
        split="train",
        encoder_fp=encoder_fingerprint(config),
        sentence_sha1=corpus.sentence_sha1,
    )

    second = encode_train_corpus(corpus, **common)
    assert second.cache_hit is True
    assert encoder.calls == 1 and len(factory_calls) == 1
    assert np.array_equal(second.embeddings, first.embeddings)
    assert list(second.iter_documents())[1].shape == (1, 3)

    # A different sentence list or encoder gets its own cache entry.
    other = make_corpus([["s0", "s1"], ["changed"]])
    third = encode_train_corpus(other, **common)
    assert third.cache_hit is False and third.cache_dir != first.cache_dir
    fourth = encode_train_corpus(
        corpus, **{**common, "encoder_config": {"model_name": "other-encoder"}}
    )
    assert fourth.cache_hit is False and fourth.cache_dir != first.cache_dir


def test_encoder_fingerprint_ignores_runtime_only_fields() -> None:
    base = {"model_name": "m", "backend": "sentence_transformers", "device": "cuda"}
    assert encoder_fingerprint(base) == encoder_fingerprint({**base, "device": "cpu"})
    assert encoder_fingerprint(base) == encoder_fingerprint(
        {**base, "encode_batch_size": 128}
    )
    assert encoder_fingerprint(base) != encoder_fingerprint({**base, "pooling": "cls"})


def test_load_fine_labels_reads_rows_by_raw_index(monkeypatch) -> None:
    frame = pd.DataFrame(
        {"target_str": ["comp.a", "comp.b ", "other", "comp.a"], "text": list("wxyz")}
    )
    monkeypatch.setattr(
        inputs_module,
        "load_dataset_split",
        lambda dataset, split: (Path("x.csv"), frame),
    )
    monkeypatch.setattr(
        inputs_module,
        "get_dataset_targets",
        lambda dataset, **kwargs: {"computer": ["comp.a", "comp.b"], "misc": ["other"]},
    )
    labels, names = load_fine_labels(
        "dummy", split="train", raw_doc_indices=[3, 1, 0], category="computer"
    )
    assert names == ["comp.a", "comp.b"]
    assert labels.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="outside the category catalogue"):
        load_fine_labels(
            "dummy", split="train", raw_doc_indices=[2], category="computer"
        )
    with pytest.raises(ValueError, match="out of range"):
        load_fine_labels(
            "dummy", split="train", raw_doc_indices=[9], category="computer"
        )
    with pytest.raises(ValueError, match="not in the fine-label catalogue"):
        load_fine_labels("dummy", split="train", raw_doc_indices=[0], category="nope")
    all_labels, all_names = load_fine_labels(
        "dummy", split="train", raw_doc_indices=[2, 0], category="all"
    )
    assert all_names == ["comp.a", "comp.b", "other"]
    assert all_labels.tolist() == [2, 0]

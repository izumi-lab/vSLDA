from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.preprocessing import PreprocessedDocument
from src.evaluation.word_based.sentence_encoding import (
    encode_sentence_corpus,
    flatten_encoder_sentences,
    resolve_topic_word_encode_batch_size,
    resolve_topic_word_encoder_device,
)


def _document(*sentences: str) -> PreprocessedDocument:
    """Build a document whose raw and tokenized text are distinguishable.

    Training encodes ``sentences_raw`` for every backend except uSIF, so the
    fixture has to keep the two forms apart for the tests to mean anything.
    """

    tokenized = [[f"{sentence}-tok"] for sentence in sentences]
    return PreprocessedDocument(
        raw_text=" ".join(sentences),
        sentences_raw=list(sentences),
        sentences_tokenized=tokenized,
        sentences_joined=[" ".join(tokens) for tokens in tokenized],
        document_tokens=[token for sentence in tokenized for token in sentence],
    )


class _FakeEncoder:
    def __init__(
        self,
        *,
        dimension: int = 3,
        output: np.ndarray | None = None,
        accepts_tokenized: bool = False,
    ):
        self.dimension = dimension
        self.output = output
        self.accepts_tokenized = accepts_tokenized
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences, **kwargs):
        materialized = list(sentences)
        self.calls.append((materialized, dict(kwargs)))
        if self.output is not None:
            return self.output
        return np.arange(len(materialized) * self.dimension, dtype=np.float64).reshape(
            len(materialized), self.dimension
        )


def test_flatten_encoder_sentences_preserves_order_and_empty_documents() -> None:
    sentences, offsets = flatten_encoder_sentences(
        [_document("a", "b"), _document(), _document("c")],
        use_tokenized=False,
    )

    assert sentences == ["a", "b", "c"]
    np.testing.assert_array_equal(offsets, [0, 2, 2, 3])
    assert offsets.dtype == np.int64


def test_flatten_encoder_sentences_uses_tokenized_text_only_when_requested() -> None:
    documents = [_document("a", "b"), _document("c")]

    raw_sentences, raw_offsets = flatten_encoder_sentences(
        documents, use_tokenized=False
    )
    tokenized_sentences, tokenized_offsets = flatten_encoder_sentences(
        documents, use_tokenized=True
    )

    assert raw_sentences == ["a", "b", "c"]
    assert tokenized_sentences == ["a-tok", "b-tok", "c-tok"]
    np.testing.assert_array_equal(raw_offsets, tokenized_offsets)


def test_encode_sentence_corpus_encodes_raw_text_for_untokenized_backends() -> None:
    """Training encodes sentences_raw unless the backend accepts tokenized text."""

    encoder = _FakeEncoder()

    encode_sentence_corpus(
        encoder=encoder,
        documents=[_document("a", "b")],
        batch_size=4,
    )

    assert encoder.calls[0][0] == ["a", "b"]


def test_encode_sentence_corpus_encodes_tokenized_text_for_usif_like_backends() -> None:
    encoder = _FakeEncoder(accepts_tokenized=True)

    encode_sentence_corpus(
        encoder=encoder,
        documents=[_document("a", "b")],
        batch_size=4,
    )

    assert encoder.calls[0][0] == ["a-tok", "b-tok"]


def test_encode_sentence_corpus_calls_encoder_once_and_restores_documents() -> None:
    encoder = _FakeEncoder()
    encoded = encode_sentence_corpus(
        encoder=encoder,
        documents=[_document("a", "b"), _document(), _document("c")],
        batch_size=17,
    )

    assert encoder.calls == [
        (
            ["a", "b", "c"],
            {"batch_size": 17, "show_progress_bar": False},
        )
    ]
    assert encoded.embeddings.dtype == np.float32
    assert encoded.embeddings.shape == (3, 3)
    assert encoded.document(0).shape == (2, 3)
    assert encoded.document(1).shape == (0, 3)
    np.testing.assert_array_equal(encoded.document(2), encoded.embeddings[2:3])


def test_encode_sentence_corpus_does_not_call_encoder_for_empty_corpus() -> None:
    encoder = _FakeEncoder(dimension=5)

    encoded = encode_sentence_corpus(
        encoder=encoder,
        documents=[_document(), _document()],
        batch_size=32,
    )

    assert encoder.calls == []
    assert encoded.embeddings.shape == (0, 5)
    np.testing.assert_array_equal(encoded.doc_offsets, [0, 0, 0])


@pytest.mark.parametrize(
    "output,match",
    [
        (np.zeros((1, 3)), "row count mismatch"),
        (np.zeros((2, 4)), "dimension mismatch"),
        (np.asarray([[0.0, np.nan, 0.0], [0.0, 0.0, 0.0]]), "non-finite"),
    ],
)
def test_encode_sentence_corpus_rejects_invalid_encoder_output(
    output: np.ndarray, match: str
) -> None:
    encoder = _FakeEncoder(output=output)

    with pytest.raises(ValueError, match=match):
        encode_sentence_corpus(
            encoder=encoder,
            documents=[_document("a", "b")],
            batch_size=8,
        )


def test_encode_sentence_corpus_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        encode_sentence_corpus(
            encoder=_FakeEncoder(),
            documents=[_document("a")],
            batch_size=0,
        )


def test_batch_size_prefers_override_then_training_metadata() -> None:
    config = {
        "model_name": "cl-nagoya/ruri-v3-130m",
        "backend": "sentence_transformers",
        "encode_batch_size": 128,
    }

    training = resolve_topic_word_encode_batch_size(
        encoder_config=config, override=None
    )
    override = resolve_topic_word_encode_batch_size(encoder_config=config, override=64)

    assert (training.value, training.source, training.training_value) == (
        128,
        "training_metadata",
        128,
    )
    assert (override.value, override.source, override.training_value) == (
        64,
        "override",
        128,
    )


def test_batch_size_uses_stable_backend_default_for_vmf_metadata() -> None:
    resolved = resolve_topic_word_encode_batch_size(
        encoder_config={
            "model_name": "cl-nagoya/ruri-v3-130m",
            "backend": "sentence_transformers",
            "encode_batch_size": None,
        },
        override=None,
    )

    assert resolved.value == 32
    assert resolved.source == "backend_default"
    assert resolved.training_value is None


def test_device_resolver_handles_auto_and_explicit_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_topic_word_encoder_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_topic_word_encoder_device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_topic_word_encoder_device("auto") == "cuda"
    assert resolve_topic_word_encoder_device("cuda:1") == "cuda:1"
    with pytest.raises(RuntimeError, match="out of range"):
        resolve_topic_word_encoder_device("cuda:2")

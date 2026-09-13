from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.preprocessing import PreprocessedDocument
from src.evaluation.topic_pairs import metrics as metrics_module
from src.evaluation.topic_pairs.inputs import (
    CachedEmbeddings,
    TrainCorpus,
    sentence_fingerprint,
)
from src.evaluation.topic_pairs.metrics import CategoryContext, ModelPosterior

# A toy category: two documents, four sentences, three embedding dimensions,
# two fine labels. Sentences 0-1 (document 0, label 0) point along e1; sentence
# 2 (document 1, label 1) along e2; sentence 3 (document 1, label 1) along e3.
TOY_SENTENCES = [["s0", "s1"], ["s2", "s3"]]
TOY_EMBEDDINGS = np.asarray(
    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
TOY_DOC_LABELS = np.asarray([0, 1], dtype=np.int64)
TOY_LABEL_NAMES = ["label_x", "label_y"]


def make_document(sentences: list[str]) -> PreprocessedDocument:
    tokenized = [sentence.split() for sentence in sentences]
    return PreprocessedDocument(
        raw_text=" ".join(sentences),
        sentences_raw=list(sentences),
        sentences_tokenized=tokenized,
        sentences_joined=[" ".join(tokens) for tokens in tokenized],
        document_tokens=[token for tokens in tokenized for token in tokens],
    )


def make_corpus(
    sentences_by_doc: list[list[str]] = TOY_SENTENCES,
    *,
    model: str = "vmf",
    condition_dir: Path | None = None,
    raw_doc_indices: list[int] | None = None,
) -> TrainCorpus:
    documents = [make_document(sentences) for sentences in sentences_by_doc]
    flat = [sentence for sentences in sentences_by_doc for sentence in sentences]
    offsets = np.cumsum([0, *[len(sentences) for sentences in sentences_by_doc]])
    return TrainCorpus(
        model=model,
        condition_dir=condition_dir or Path(f"/fake/{model}"),
        documents=documents,
        raw_doc_indices=raw_doc_indices or list(range(len(documents))),
        sentences=flat,
        doc_offsets=np.asarray(offsets, dtype=np.int64),
        sentence_sha1=sentence_fingerprint(flat),
    )


def toy_posterior(model: str, iteration: int, num_topics: int) -> np.ndarray:
    """Deterministic, model- and iteration-dependent sentence posteriors."""

    num_sentences = TOY_EMBEDDINGS.shape[0]
    probs = np.full((num_sentences, num_topics), 0.02, dtype=np.float64)
    shift = 1 if model == "sentlda" else 0
    for sentence in range(num_sentences):
        # sentences 0 and 1 share a topic; the others get their own topic
        base = 0 if sentence < 2 else sentence - 1
        probs[sentence, (base + shift) % num_topics] += 1.0 + 0.1 * iteration
    return probs / probs.sum(axis=1, keepdims=True)


def make_context(tmp_path: Path, *, category: str = "a") -> CategoryContext:
    corpus = make_corpus(condition_dir=tmp_path / "vmf" / "ref")
    embeddings = CachedEmbeddings(
        embeddings=TOY_EMBEDDINGS,
        doc_offsets=corpus.doc_offsets,
        manifest={"sentence_sha1": corpus.sentence_sha1},
        cache_dir=tmp_path / "cache" / category,
        cache_hit=False,
    )
    return CategoryContext(
        dataset="dummy",
        data_run="default",
        category=category,
        split="train",
        reference_model="vmf",
        reference_condition_dir=corpus.condition_dir,
        corpus=corpus,
        embeddings=embeddings,
        embeddings_unit=TOY_EMBEDDINGS.astype(np.float64),
        encoder_config={"model_name": "fake-encoder"},
        encoder_fingerprint="fp0000",
        doc_labels=TOY_DOC_LABELS,
        labels=np.repeat(TOY_DOC_LABELS, np.diff(corpus.doc_offsets)),
        label_names=list(TOY_LABEL_NAMES),
    )


def fake_provenance(condition_dir: Path, *, model: str) -> dict[str, object]:
    return {
        "model_key": "vmf_sentence_lda" if model == "vmf" else model,
        "metadata_path": str(Path(condition_dir) / "metadata.json"),
    }


@pytest.fixture
def toy_runner(monkeypatch, tmp_path: Path):
    """Patch every disk-touching step of the runner with toy equivalents.

    Returns the list of ``compute_model_posterior`` calls, so tests can assert
    what was (not) computed.
    """

    calls: list[dict[str, object]] = []

    def _context(**kwargs) -> CategoryContext:
        return make_context(tmp_path, category=str(kwargs["category"]))

    def _resolve(**kwargs) -> Path:
        return (
            tmp_path
            / "runs"
            / str(kwargs["model"])
            / f"k{int(kwargs['num_topics'])}_it{int(kwargs['iteration'])}"
        )

    def _posterior(*, model, condition_dir, context, foldin_config) -> ModelPosterior:
        calls.append({"model": model, "condition_dir": Path(condition_dir)})
        iteration = int(str(condition_dir).rsplit("_it", 1)[1])
        num_topics = int(str(condition_dir).rsplit("k", 1)[1].split("_")[0])
        probs = toy_posterior(model, iteration, num_topics)
        return ModelPosterior(
            model=model,
            condition_dir=Path(condition_dir),
            probs=probs,
            alpha=np.full(num_topics, 0.1),
            posterior_metadata={"posterior_kind": "toy"},
        )

    def _reference(condition_dir: Path) -> dict[str, object]:
        num_topics = int(str(condition_dir).rsplit("k", 1)[1].split("_")[0])
        means = np.eye(num_topics, TOY_EMBEDDINGS.shape[1])
        means[num_topics - 1] = np.ones(TOY_EMBEDDINGS.shape[1])
        return {
            "topic_means": means,
            "kappa_model": np.linspace(100.0, 200.0, num_topics),
            "topic_counts": np.ones(num_topics),
            "kappa_default": 10.0,
            "max_kappa": None,
        }

    monkeypatch.setattr(metrics_module, "load_category_context", _context)
    monkeypatch.setattr(metrics_module, "resolve_condition_dir", _resolve)
    monkeypatch.setattr(metrics_module, "compute_model_posterior", _posterior)
    monkeypatch.setattr(metrics_module, "load_vmf_model_reference", _reference)
    monkeypatch.setattr(metrics_module, "provenance_for", fake_provenance)
    monkeypatch.setattr(
        metrics_module, "resolve_topic_word_encoder_device", lambda requested: "cpu"
    )
    return calls

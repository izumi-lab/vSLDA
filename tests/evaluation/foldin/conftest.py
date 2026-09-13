from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.artifacts import save_pickle
from src.data.preprocessing import PreprocessedDocument
from src.evaluation.topic_pairs.inputs import CachedEmbeddings

# Two documents of two and one sentence; document three is empty. Three
# embedding dimensions, two topics: sentences along e1 belong to topic 0,
# sentences along e2 to topic 1.
TOY_SENTENCES: list[list[str]] = [["s0 a", "s1 b"], ["s2 c"], []]
TOY_EMBEDDINGS = np.asarray(
    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
)
TOY_ALPHA = np.asarray([0.5, 0.25], dtype=np.float64)
ENCODER_CONFIG = {
    "model_name": "toy-encoder",
    "embedding_variant": "minilm",
    "strip_terminal_normalize": True,
    "pre_normalize_transform": "none",
}


def make_document(sentences: list[str]) -> PreprocessedDocument:
    tokenized = [sentence.split() for sentence in sentences]
    return PreprocessedDocument(
        raw_text=" ".join(sentences),
        sentences_raw=list(sentences),
        sentences_tokenized=tokenized,
        sentences_joined=[" ".join(tokens) for tokens in tokenized],
        document_tokens=[token for tokens in tokenized for token in tokens],
    )


def write_vmf_run(
    archive_dir: Path,
    *,
    condition_fingerprint: str = "cond-fp",
    sentences: list[list[str]] | None = None,
    num_topics: int = 2,
    with_metadata: bool = True,
) -> Path:
    """A minimal vMF run directory: metadata, both splits' corpora, doc_topic_test."""

    docs = sentences if sentences is not None else TOY_SENTENCES
    archive_dir.mkdir(parents=True, exist_ok=True)
    if with_metadata:
        (archive_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "condition_fingerprint": condition_fingerprint,
                    "encoder_config": ENCODER_CONFIG,
                }
            ),
            encoding="utf-8",
        )
    documents = [make_document(item) for item in docs]
    save_pickle(documents, archive_dir / "train_preprocessed.pkl")
    save_pickle(documents, archive_dir / "test_preprocessed.pkl")
    (archive_dir / "preprocessing_selection.json").write_text(
        json.dumps(
            {
                "train": {"raw_doc_indices": list(range(len(documents)))},
                "test": {"raw_doc_indices": list(range(len(documents)))},
            }
        ),
        encoding="utf-8",
    )
    save_pickle(
        np.full((len(documents), num_topics), 1.0 / num_topics),
        archive_dir / "doc_topic_test.pkl",
    )
    return archive_dir


def toy_embeddings(
    corpus, *, cache_dir: Path, encoder_fp: str = "fp"
) -> CachedEmbeddings:
    """Raw embeddings of the toy sentences, shaped like the corpus."""

    total = corpus.num_sentences
    values = np.zeros((total, 3), dtype=np.float32)
    for index in range(total):
        values[index] = TOY_EMBEDDINGS[index % TOY_EMBEDDINGS.shape[0]]
    return CachedEmbeddings(
        embeddings=values,
        doc_offsets=np.asarray(corpus.doc_offsets, dtype=np.int64),
        manifest={
            "sentence_sha1": corpus.sentence_sha1,
            "encoder_fingerprint": encoder_fp,
        },
        cache_dir=cache_dir,
        cache_hit=False,
    )


def toy_log_likelihoods(condition_dir: Path, encoded_documents):
    """Frozen-model stand-in: topic 0 likes e1, topic 1 likes e2, strongly."""

    blocks = []
    for block in encoded_documents:
        values = np.asarray(block, dtype=np.float64)
        scores = np.stack([values[:, 0], values[:, 1]], axis=1) * 8.0
        blocks.append(scores)
    return blocks, TOY_ALPHA.copy()

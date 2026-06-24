from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class VMFCorpusInferenceOutputs:
    counts: np.ndarray | None = None
    sentence_posteriors: list[np.ndarray] | None = None
    document_posteriors: np.ndarray | None = None


class VMFTopicInferencer:
    """Inference helpers for document- and sentence-level topic posteriors."""

    def __init__(
        self,
        *,
        num_topics: int,
        embedding_size: int,
        encode_document: Callable[[Sequence[str]], np.ndarray],
        log_vmf_density: Callable[[np.ndarray], np.ndarray],
        log_vmf_density_batch: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.num_topics = int(num_topics)
        self.embedding_size = int(embedding_size)
        self.encode_document = encode_document
        self.log_vmf_density = log_vmf_density
        self.log_vmf_density_batch = log_vmf_density_batch

    def build_document_topic_distribution(
        self, topic_counts_per_doc: np.ndarray
    ) -> np.ndarray:
        counts = np.asarray(topic_counts_per_doc, dtype=np.float64).T
        if counts.ndim != 2:
            raise ValueError(
                f"topic_counts_per_doc must be 2D, got shape {counts.shape}"
            )

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return counts / row_sums

    def infer_document_topic_counts(
        self, new_corpus: Sequence[Sequence[str]]
    ) -> np.ndarray:
        outputs = self.infer_corpus_topic_outputs(
            new_corpus,
            include_counts=True,
        )
        assert outputs.counts is not None
        return outputs.counts

    def infer_document_topic_distribution_soft(
        self, new_corpus: Sequence[Sequence[str]], temperature: float = 1.0
    ) -> np.ndarray:
        outputs = self.infer_corpus_topic_outputs(
            new_corpus,
            temperature=temperature,
            include_document_posteriors=True,
        )
        assert outputs.document_posteriors is not None
        return outputs.document_posteriors

    def infer_corpus_topic_outputs(
        self,
        new_corpus: Sequence[Sequence[str]],
        *,
        temperature: float = 1.0,
        include_counts: bool = False,
        include_sentence_posteriors: bool = False,
        include_document_posteriors: bool = False,
    ) -> VMFCorpusInferenceOutputs:
        encoded_corpus = [self.encode_document(doc) for doc in new_corpus]
        return self.infer_encoded_corpus_topic_outputs(
            encoded_corpus,
            temperature=temperature,
            include_counts=include_counts,
            include_sentence_posteriors=include_sentence_posteriors,
            include_document_posteriors=include_document_posteriors,
        )

    def infer_encoded_corpus_topic_outputs(
        self,
        encoded_corpus: Sequence[np.ndarray],
        *,
        temperature: float = 1.0,
        include_counts: bool = False,
        include_sentence_posteriors: bool = False,
        include_document_posteriors: bool = False,
    ) -> VMFCorpusInferenceOutputs:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (
            include_counts or include_sentence_posteriors or include_document_posteriors
        ):
            raise ValueError("At least one inference output must be requested.")

        counts = (
            np.zeros((len(encoded_corpus), self.num_topics), dtype=np.int32)
            if include_counts
            else None
        )
        sentence_posteriors = [] if include_sentence_posteriors else None
        document_posteriors = (
            np.zeros((len(encoded_corpus), self.num_topics), dtype=np.float64)
            if include_document_posteriors
            else None
        )

        for d, encoded_doc in enumerate(encoded_corpus):
            enc = np.asarray(encoded_doc, dtype=np.float64)
            if enc.size == 0:
                if include_sentence_posteriors and sentence_posteriors is not None:
                    sentence_posteriors.append(
                        np.zeros((0, self.num_topics), dtype=np.float64)
                    )
                continue
            if enc.ndim == 1:
                enc = enc.reshape(1, -1)

            log_lik_doc = self._log_likelihood_batch(enc)
            if include_counts and counts is not None:
                winners = np.argmax(log_lik_doc, axis=1)
                counts[d] = np.bincount(winners, minlength=self.num_topics).astype(
                    np.int32,
                    copy=False,
                )

            if include_sentence_posteriors or include_document_posteriors:
                probs = self._posterior_matrix_from_log_likelihood(
                    log_lik_doc,
                    temperature=temperature,
                )
                if include_sentence_posteriors and sentence_posteriors is not None:
                    sentence_posteriors.append(probs)
                if include_document_posteriors and document_posteriors is not None:
                    document_posteriors[d] = (
                        self._document_posterior_from_sentence_matrix(probs)
                    )

        return VMFCorpusInferenceOutputs(
            counts=counts,
            sentence_posteriors=sentence_posteriors,
            document_posteriors=document_posteriors,
        )

    def aggregate_document_topic_distribution_from_sentence_posteriors(
        self, sentence_posteriors: Sequence[np.ndarray]
    ) -> np.ndarray:
        dist = np.zeros((len(sentence_posteriors), self.num_topics), dtype=np.float64)
        for d, sent_probs in enumerate(sentence_posteriors):
            probs = np.asarray(sent_probs, dtype=np.float64)
            if probs.size == 0:
                continue
            if probs.ndim != 2 or probs.shape[1] != self.num_topics:
                raise ValueError(
                    "Invalid sentence posterior shape at "
                    f"doc {d}: {probs.shape}, expected (*, {self.num_topics})"
                )
            dist[d] = self._document_posterior_from_sentence_matrix(probs)
        return dist

    def infer_sentence_topic_distribution_soft(
        self, new_corpus: Sequence[Sequence[str]], temperature: float = 1.0
    ) -> list[np.ndarray]:
        outputs = self.infer_corpus_topic_outputs(
            new_corpus,
            temperature=temperature,
            include_sentence_posteriors=True,
        )
        assert outputs.sentence_posteriors is not None
        return outputs.sentence_posteriors

    def _log_likelihood_batch(self, enc: np.ndarray) -> np.ndarray:
        arr = np.asarray(enc, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, self.num_topics), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.embedding_size:
            raise ValueError(
                f"Expected embedding dim {self.embedding_size}, got {arr.shape[1]}"
            )

        if self.log_vmf_density_batch is not None:
            log_lik = np.asarray(self.log_vmf_density_batch(arr), dtype=np.float64)
        else:
            log_lik = np.vstack([self.log_vmf_density(x) for x in arr]).astype(
                np.float64,
                copy=False,
            )

        if log_lik.ndim != 2 or log_lik.shape[1] != self.num_topics:
            raise ValueError(
                "Invalid log-likelihood batch shape: "
                f"{log_lik.shape}, expected (*, {self.num_topics})"
            )
        return log_lik

    def _posterior_from_log_likelihood(
        self, log_lik: np.ndarray, *, temperature: float
    ) -> np.ndarray:
        probs = np.asarray(log_lik, dtype=np.float64).copy()
        if temperature != 1.0:
            probs = probs / temperature
        probs -= probs.max()
        probs = np.exp(probs)
        s = probs.sum()
        if not np.isfinite(s) or s <= 0.0:
            return np.full(self.num_topics, 1.0 / self.num_topics, dtype=np.float64)
        return probs / s

    def _posterior_matrix_from_log_likelihood(
        self, log_lik: np.ndarray, *, temperature: float
    ) -> np.ndarray:
        probs = np.asarray(log_lik, dtype=np.float64).copy()
        if probs.size == 0:
            return np.zeros((0, self.num_topics), dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != self.num_topics:
            raise ValueError(
                "Invalid log-likelihood batch shape: "
                f"{probs.shape}, expected (*, {self.num_topics})"
            )
        if temperature != 1.0:
            probs = probs / temperature
        probs -= probs.max(axis=1, keepdims=True)
        probs = np.exp(probs)
        row_sums = probs.sum(axis=1, keepdims=True)
        invalid = ~np.isfinite(row_sums) | (row_sums <= 0.0)
        row_sums[invalid] = 1.0
        probs /= row_sums
        if np.any(invalid):
            probs[invalid.ravel()] = 1.0 / self.num_topics
        return probs

    def _document_posterior_from_sentence_matrix(self, probs: np.ndarray) -> np.ndarray:
        doc_post = np.asarray(probs, dtype=np.float64).sum(axis=0)
        s = float(doc_post.sum())
        if s > 0.0:
            doc_post /= s
        return doc_post

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from src.baselines.models.gaussian_helpers import (
    load_gaussian_word_vectors,
    load_gaussianlda_model,
)
from src.core.artifacts import load_artifact_json, load_artifact_pickle
from src.core.paths import resolve_baseline_condition_dir

ModelType = Literal[
    "vmf",
    "gaussian",
    "sentence_gaussianlda",
    "sentlda",
    "bertopic_kmeans",
    "bleilda",
    "gaussianlda",
    "etm",
    "mvtm",
    "ctm",
    "senclu",
    "spherical_kmeans",
    "gaussian_kmeans",
    "movmf",
    "gaussian_mixture",
]
ScoreMode = Literal["npmi", "word_npmi"]
TopicWord = tuple[str, float]
TopicWords = list[list[TopicWord]]


@dataclass(frozen=True)
class TopicWordsResult:
    topic_words: TopicWords
    topic_word_source: str
    score_mode: str | None = None
    score_definition: str | None = None


def describe_proxy_word_score_mode(score_mode: ScoreMode) -> str:
    if score_mode == "npmi":
        return "PMI normalized by -log p(w,k)"
    if score_mode == "word_npmi":
        return "PMI normalized by -log p(w)"
    raise ValueError(f"Unsupported proxy word score mode '{score_mode}'.")


def build_baseline_param_dir(
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> Path:
    if model not in {
        "bleilda",
        "ctm",
        "etm",
        "gaussianlda",
        "mvtm",
        "sentence_gaussianlda",
        "sentlda",
        "bertopic_kmeans",
        "senclu",
        "spherical_kmeans",
        "gaussian_kmeans",
        "movmf",
        "gaussian_mixture",
    }:
        raise ValueError(f"Unsupported model in baseline path resolver: '{model}'.")
    return (
        resolve_baseline_condition_dir(
            model=model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
        / "params"
    )


def compute_topic_word_npmi(
    doc_topics: np.ndarray,
    corpus_bow: list[list[tuple[int, int]]],
    vocab_size: int,
    eps: float = 1e-12,
    score_mode: ScoreMode = "npmi",
) -> np.ndarray:
    if doc_topics.shape[0] != len(corpus_bow):
        raise ValueError(
            f"doc_topics size {doc_topics.shape[0]} does not match corpus size {len(corpus_bow)}"
        )
    num_topics = doc_topics.shape[1]
    joint_counts = np.zeros((num_topics, vocab_size), dtype=np.float64)
    word_counts = np.zeros(vocab_size, dtype=np.float64)
    topic_counts = np.zeros(num_topics, dtype=np.float64)
    total_tokens = 0.0

    for doc_idx, bow in enumerate(corpus_bow):
        if not bow:
            continue
        theta = doc_topics[doc_idx]
        doc_len = 0.0
        for word_id, count in bow:
            weight = float(count)
            doc_len += weight
            word_counts[word_id] += weight
            joint_counts[:, word_id] += theta * weight
        topic_counts += theta * doc_len
        total_tokens += doc_len

    if total_tokens == 0.0:
        return np.zeros_like(joint_counts)

    p_wk = joint_counts / total_tokens
    p_w = word_counts / total_tokens
    p_k = topic_counts / total_tokens

    denom = np.outer(p_k, p_w)
    p_wk_safe = np.maximum(p_wk, eps)
    denom_safe = np.maximum(denom, eps)
    pmi = np.log(p_wk_safe / denom_safe)
    if score_mode == "npmi":
        normalizer = -np.log(p_wk_safe)
    elif score_mode == "word_npmi":
        normalizer = -np.log(np.maximum(p_w, eps))[None, :]
    else:
        raise ValueError(f"Unsupported proxy word score mode '{score_mode}'.")
    scores = pmi / normalizer
    scores[p_wk == 0.0] = -1.0
    return scores


def compute_topic_word_npmi_from_sentence_topics(
    sentence_topics_by_doc: list[np.ndarray],
    sentence_bow_by_doc: list[list[list[tuple[int, int]]]],
    num_topics: int,
    vocab_size: int,
    eps: float = 1e-12,
    score_mode: ScoreMode = "npmi",
) -> np.ndarray:
    if len(sentence_topics_by_doc) != len(sentence_bow_by_doc):
        raise ValueError(
            f"sentence_topics docs={len(sentence_topics_by_doc)} != sentence_bow docs={len(sentence_bow_by_doc)}"
        )

    joint_counts = np.zeros((num_topics, vocab_size), dtype=np.float64)
    word_counts = np.zeros(vocab_size, dtype=np.float64)
    topic_counts = np.zeros(num_topics, dtype=np.float64)
    total_tokens = 0.0

    for doc_idx, (sentence_topics, sentence_bows) in enumerate(
        zip(sentence_topics_by_doc, sentence_bow_by_doc)
    ):
        if sentence_topics.shape[0] != len(sentence_bows):
            topic_len = int(sentence_topics.shape[0])
            bow_len = int(len(sentence_bows))
            if bow_len < topic_len:
                sentence_bows = list(sentence_bows) + [
                    [] for _ in range(topic_len - bow_len)
                ]
                warnings.warn(
                    "Sentence count mismatch resolved by padding empty sentence BoW: "
                    f"doc={doc_idx}, sentence_topics={topic_len}, sentence_bows={bow_len}"
                )
            else:
                sentence_bows = list(sentence_bows[:topic_len])
                warnings.warn(
                    "Sentence count mismatch resolved by truncating sentence BoW: "
                    f"doc={doc_idx}, sentence_topics={topic_len}, sentence_bows={bow_len}"
                )
        if sentence_topics.shape[1] != num_topics:
            raise ValueError(
                f"Topic count mismatch at doc {doc_idx}: "
                f"sentence_topics has {sentence_topics.shape[1]}, expected {num_topics}"
            )

        for sent_idx, bow in enumerate(sentence_bows):
            if not bow:
                continue
            theta = sentence_topics[sent_idx]
            sent_len = 0.0
            for word_id, count in bow:
                weight = float(count)
                sent_len += weight
                word_counts[word_id] += weight
                joint_counts[:, word_id] += theta * weight
            topic_counts += theta * sent_len
            total_tokens += sent_len

    if total_tokens == 0.0:
        return np.zeros_like(joint_counts)

    p_wk = joint_counts / total_tokens
    p_w = word_counts / total_tokens
    p_k = topic_counts / total_tokens

    denom = np.outer(p_k, p_w)
    p_wk_safe = np.maximum(p_wk, eps)
    denom_safe = np.maximum(denom, eps)
    pmi = np.log(p_wk_safe / denom_safe)
    if score_mode == "npmi":
        normalizer = -np.log(p_wk_safe)
    elif score_mode == "word_npmi":
        normalizer = -np.log(np.maximum(p_w, eps))[None, :]
    else:
        raise ValueError(f"Unsupported proxy word score mode '{score_mode}'.")
    scores = pmi / normalizer
    scores[p_wk == 0.0] = -1.0
    return scores


def select_top_words(
    scores: np.ndarray,
    dictionary: Dictionary,
    topn: int,
) -> TopicWords:
    if scores.size == 0 or len(dictionary) == 0:
        return []
    num_topics = scores.shape[0]
    topic_words: TopicWords = []
    for topic_idx in range(num_topics):
        row = scores[topic_idx]
        if not np.any(row):
            topic_words.append([])
            continue
        top_ids = np.argsort(-row)[:topn]
        topic_words.append(
            [(dictionary[word_id], float(row[word_id])) for word_id in top_ids]
        )
    return topic_words


def select_top_words_from_vocab_scores(
    scores: np.ndarray,
    vocab: list[str],
    topn: int,
) -> TopicWords:
    if scores.size == 0 or not vocab:
        return []
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D score matrix, got shape {scores.shape}")
    topic_words: TopicWords = []
    for topic_idx in range(scores.shape[0]):
        row = scores[topic_idx]
        if row.shape[0] != len(vocab):
            raise ValueError(
                f"Score row length {row.shape[0]} does not match vocab size {len(vocab)}"
            )
        if not np.any(row):
            topic_words.append([])
            continue
        top_ids = np.argsort(-row)[:topn]
        topic_words.append(
            [(vocab[word_id], float(row[word_id])) for word_id in top_ids]
        )
    return topic_words


def select_top_words_from_vocab_scores_restricted_to_dictionary(
    scores: np.ndarray,
    vocab: list[str],
    dictionary: Dictionary,
    topn: int,
) -> TopicWords:
    if scores.size == 0 or not vocab or len(dictionary) == 0:
        return []
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D score matrix, got shape {scores.shape}")
    if scores.shape[1] != len(vocab):
        raise ValueError(
            f"Score width {scores.shape[1]} does not match vocab size {len(vocab)}"
        )

    dict_vocab = [dictionary[word_id] for word_id in range(len(dictionary))]
    if not dict_vocab:
        return []
    limit = min(topn, len(dict_vocab))
    model_id_by_word = {word: idx for idx, word in enumerate(vocab)}
    model_ids = np.array([model_id_by_word.get(word, -1) for word in dict_vocab])

    topic_words: TopicWords = []
    for topic_idx in range(scores.shape[0]):
        row = np.asarray(scores[topic_idx], dtype=np.float64)
        dict_scores = np.full(len(dict_vocab), -np.inf, dtype=np.float64)
        present_mask = model_ids >= 0
        if np.any(present_mask):
            dict_scores[present_mask] = row[model_ids[present_mask]]
        dict_scores[~np.isfinite(dict_scores)] = -np.inf
        top_ids = np.argsort(-dict_scores, kind="stable")[:limit]
        topic_words.append(
            [(dict_vocab[word_id], float(dict_scores[word_id])) for word_id in top_ids]
        )
    return topic_words


def serialize_topic_words(topic_words: TopicWords) -> list[dict[str, object]]:
    return [
        {
            "topic_id": int(topic_id),
            "words": [
                {"word": str(word), "score": float(score)} for word, score in topic
            ],
        }
        for topic_id, topic in enumerate(topic_words)
    ]


def _patch_torch_for_ctm_loading() -> None:
    import numpy as _np
    import torch

    if getattr(torch, "_coherence_load_patch_applied", False):
        return
    torch.serialization.add_safe_globals([_np.core.multiarray._reconstruct])
    original_torch_load = torch.load

    def _torch_load_weights_only_false(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_weights_only_false
    torch._coherence_load_patch_applied = True


def _find_ctm_model_dir(param_dir: Path) -> Path:
    candidates = [
        path
        for path in param_dir.iterdir()
        if path.is_dir() and list(path.glob("epoch_*.pth"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"CTM checkpoint dir not found under: {param_dir} (expected epoch_*.pth)"
        )
    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda path: path.name)
        warnings.warn(
            f"Multiple CTM checkpoint dirs found under {param_dir}; using {candidates[-1].name}"
        )
    return candidates[-1]


def load_bleilda_topic_words(
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    topn: int,
    dictionary: Dictionary | None = None,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> TopicWords:
    model_dir = build_baseline_param_dir(
        model="bleilda",
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
    )
    model_path = model_dir / "model.gensim"
    if not model_path.exists():
        raise FileNotFoundError(f"BleiLDA model not found: {model_path}")
    lda = LdaModel.load(model_path.as_posix())
    scores = np.asarray(lda.get_topics(), dtype=float)
    if scores.ndim != 2 or scores.shape[0] != num_topics:
        raise ValueError(f"Unexpected BleiLDA topic matrix shape: {scores.shape}")
    vocab = [str(lda.id2word[word_id]) for word_id in range(scores.shape[1])]
    if dictionary is not None:
        return select_top_words_from_vocab_scores_restricted_to_dictionary(
            scores=scores,
            vocab=vocab,
            dictionary=dictionary,
            topn=topn,
        )
    return select_top_words_from_vocab_scores(scores=scores, vocab=vocab, topn=topn)


def load_ctm_topic_words(
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    topn: int,
    dictionary: Dictionary | None = None,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> TopicWords:
    _patch_torch_for_ctm_loading()
    from contextualized_topic_models.models.ctm import CombinedTM

    param_dir = build_baseline_param_dir(
        model="ctm",
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
    )
    tp_path = param_dir / "tp.pkl"
    tp = load_artifact_pickle(tp_path)
    model_dir = _find_ctm_model_dir(param_dir)
    epoch_files = sorted(model_dir.glob("epoch_*.pth"))
    if not epoch_files:
        raise FileNotFoundError(f"No CTM checkpoints found in: {model_dir}")
    epoch = int(epoch_files[-1].stem.split("_", 1)[1])
    ctm = CombinedTM(
        bow_size=len(tp.vocab),
        contextual_size=768,
        n_components=num_topics,
        num_epochs=20,
        num_data_loader_workers=0,
    )
    ctm.load(model_dir=model_dir.as_posix(), epoch=epoch)
    scores = np.asarray(ctm.get_topic_word_distribution(), dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"Unexpected CTM topic-word shape: {scores.shape}")
    if scores.shape[0] != num_topics and scores.shape[1] == num_topics:
        scores = scores.T
    if scores.shape[0] != num_topics:
        raise ValueError(f"Unexpected CTM topic-word shape: {scores.shape}")
    vocab = [str(word) for word in tp.vocab]
    if dictionary is not None:
        return select_top_words_from_vocab_scores_restricted_to_dictionary(
            scores=scores,
            vocab=vocab,
            dictionary=dictionary,
            topn=topn,
        )
    return select_top_words_from_vocab_scores(scores=scores, vocab=vocab, topn=topn)


def load_bertopic_kmeans_topic_words(
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    topn: int,
    dictionary: Dictionary | None = None,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> TopicWords:
    _ = dictionary
    param_dir = build_baseline_param_dir(
        model="bertopic_kmeans",
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
    )
    topic_words = load_artifact_pickle(param_dir / "topic_words.pkl")
    if not isinstance(topic_words, (list, tuple)):
        raise ValueError("Invalid BERTopic-KMeans topic words artifact.")
    normalized: TopicWords = []
    for topic_idx, topic in enumerate(topic_words):
        if not isinstance(topic, (list, tuple)):
            raise ValueError(
                f"Invalid BERTopic-KMeans topic words for topic {topic_idx}."
            )
        normalized.append(
            [(str(word), float(score)) for word, score in list(topic)[:topn]]
        )
    if len(normalized) != num_topics:
        raise ValueError(
            f"BERTopic-KMeans topic word count {len(normalized)} != {num_topics}."
        )
    return normalized


def load_gaussianlda_topic_words(
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    topn: int,
    word2vec: str,
    dictionary: Dictionary | None = None,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> TopicWords:
    param_dir = build_baseline_param_dir(
        model="gaussianlda",
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
    )
    persisted = load_gaussianlda_model(
        param_dir=param_dir,
        word2vec=word2vec,
    )
    vocab = persisted.vocab
    embeddings = persisted.embeddings
    model = persisted.model
    dict_vocab: list[str] = []
    dict_model_ids: np.ndarray | None = None
    if dictionary is not None:
        dict_vocab = [dictionary[word_id] for word_id in range(len(dictionary))]
        model_id_by_word = {word: idx for idx, word in enumerate(vocab)}
        dict_model_ids = np.array(
            [model_id_by_word.get(word, -1) for word in dict_vocab],
            dtype=np.int64,
        )

    topic_words: TopicWords = []
    for topic_idx in range(num_topics):
        row = np.asarray(
            model.log_multivariate_tdensity(embeddings, topic_idx),
            dtype=float,
        )
        if row.ndim != 1 or row.shape[0] != len(vocab):
            raise ValueError(
                f"Unexpected GaussianLDA topic-word score shape: {row.shape}, vocab={len(vocab)}"
            )
        if dictionary is not None:
            if not dict_vocab:
                topic_words.append([])
                continue
            limit = min(topn, len(dict_vocab))
            dict_scores = np.full(len(dict_vocab), -np.inf, dtype=np.float64)
            present_mask = dict_model_ids >= 0
            if np.any(present_mask):
                dict_scores[present_mask] = row[dict_model_ids[present_mask]]
            dict_scores[~np.isfinite(dict_scores)] = -np.inf
            top_ids = np.argsort(-dict_scores, kind="stable")[:limit]
            topic_words.append(
                [
                    (dict_vocab[word_id], float(dict_scores[word_id]))
                    for word_id in top_ids
                ]
            )
        else:
            top_ids = np.argsort(-row)[:topn]
            topic_words.append(
                [(vocab[word_id], float(row[word_id])) for word_id in top_ids]
            )
    return topic_words


def _score_mvtm_vocab(
    *,
    vocab_vectors: np.ndarray,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    kappa_per_topic: np.ndarray,
) -> np.ndarray:
    x = np.asarray(vocab_vectors, dtype=np.float64)
    if x.size == 0:
        return np.zeros((component_means.shape[0], 0), dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    weights = np.asarray(mixture_weights, dtype=np.float64)
    means = np.asarray(component_means, dtype=np.float64)
    kappa = np.asarray(kappa_per_topic, dtype=np.float64)
    scores = np.einsum("vd,kcd->vkc", x, kappa[:, None, None] * means, optimize=True)
    log_comp = scores + np.log(weights + 1e-12)[None, :, :]
    max_log = log_comp.max(axis=2, keepdims=True)
    topic_scores = max_log + np.log(
        np.exp(log_comp - max_log).sum(axis=2, keepdims=True) + 1e-12
    )
    return topic_scores[..., 0].T


def load_mvtm_topic_words(
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    topn: int,
    word2vec: str,
    dictionary: Dictionary | None = None,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> TopicWords:
    param_dir = build_baseline_param_dir(
        model="mvtm",
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
    )
    word_vectors = load_gaussian_word_vectors(word2vec, param_dir=param_dir)
    vocab = [str(word) for word in word_vectors.key_to_index.keys()]
    scores = _score_mvtm_vocab(
        vocab_vectors=np.asarray(word_vectors.vectors, dtype=np.float64),
        mixture_weights=load_artifact_pickle(param_dir / "mixture_weights.pkl"),
        component_means=load_artifact_pickle(param_dir / "component_means.pkl"),
        kappa_per_topic=load_artifact_pickle(param_dir / "kappa_per_topic.pkl"),
    )
    if scores.ndim != 2 or scores.shape[0] != num_topics:
        raise ValueError(f"Unexpected MvTM topic-word score shape: {scores.shape}")
    if dictionary is not None:
        return select_top_words_from_vocab_scores_restricted_to_dictionary(
            scores=scores,
            vocab=vocab,
            dictionary=dictionary,
            topn=topn,
        )
    return select_top_words_from_vocab_scores(scores=scores, vocab=vocab, topn=topn)


def load_etm_topic_words(
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    topn: int,
    dictionary: Dictionary | None = None,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> TopicWords:
    param_dir = build_baseline_param_dir(
        model="etm",
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
    )
    scores = np.asarray(
        load_artifact_pickle(param_dir / "topic_word_scores.pkl"),
        dtype=np.float64,
    )
    vocabulary_payload = load_artifact_json(param_dir / "vocabulary.json")
    if isinstance(vocabulary_payload, dict):
        vocab = [
            word
            for word, _word_id in sorted(
                (
                    (str(word), int(word_id))
                    for word, word_id in vocabulary_payload.items()
                ),
                key=lambda item: item[1],
            )
        ]
    elif isinstance(vocabulary_payload, list):
        vocab = [str(word) for word in vocabulary_payload]
    else:
        raise ValueError(
            f"Invalid ETM vocabulary artifact: {param_dir / 'vocabulary.json'}"
        )
    if scores.ndim != 2 or scores.shape != (num_topics, len(vocab)):
        raise ValueError(
            f"Unexpected ETM topic-word score shape: {scores.shape}; "
            f"expected ({num_topics}, {len(vocab)})"
        )
    if dictionary is not None:
        return select_top_words_from_vocab_scores_restricted_to_dictionary(
            scores=scores,
            vocab=vocab,
            dictionary=dictionary,
            topn=topn,
        )
    return select_top_words_from_vocab_scores(scores=scores, vocab=vocab, topn=topn)


def load_learned_topic_words(
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    topn: int,
    gaussian_word2vec: str,
    dictionary: Dictionary | None = None,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> TopicWords:
    if model == "bleilda":
        return load_bleilda_topic_words(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            topn=topn,
            dictionary=dictionary,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
    if model == "ctm":
        return load_ctm_topic_words(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            topn=topn,
            dictionary=dictionary,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
    if model == "bertopic_kmeans":
        return load_bertopic_kmeans_topic_words(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            topn=topn,
            dictionary=dictionary,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
    if model == "gaussianlda":
        return load_gaussianlda_topic_words(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            topn=topn,
            word2vec=gaussian_word2vec,
            dictionary=dictionary,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
    if model == "mvtm":
        return load_mvtm_topic_words(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            topn=topn,
            word2vec=gaussian_word2vec,
            dictionary=dictionary,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
    if model == "etm":
        return load_etm_topic_words(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            topn=topn,
            dictionary=dictionary,
            data_run=data_run,
            embedding_variant=embedding_variant,
        )
    raise ValueError(f"No learned topic-word loader for model '{model}'.")


def extract_topic_words_from_learned_model(
    *,
    model: ModelType,
    dataset: str,
    iteration: int,
    num_topics: int,
    category: str,
    topn: int,
    gaussian_word2vec: str,
    dictionary: Dictionary,
    data_run: str = "default",
    embedding_variant: str | None = None,
) -> TopicWordsResult:
    return TopicWordsResult(
        topic_words=load_learned_topic_words(
            model=model,
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            topn=topn,
            gaussian_word2vec=gaussian_word2vec,
            dictionary=dictionary,
            data_run=data_run,
            embedding_variant=embedding_variant,
        ),
        topic_word_source="learned_topic_word_distribution",
    )


def extract_topic_words_from_doc_topic_npmi(
    *,
    doc_topics: np.ndarray,
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    topn: int,
    score_mode: ScoreMode = "npmi",
) -> TopicWordsResult:
    scores = compute_topic_word_npmi(
        doc_topics=doc_topics,
        corpus_bow=corpus_bow,
        vocab_size=len(dictionary),
        score_mode=score_mode,
    )
    return TopicWordsResult(
        topic_words=select_top_words(scores=scores, dictionary=dictionary, topn=topn),
        topic_word_source="document_topic_proxy_npmi",
        score_mode=score_mode,
        score_definition=describe_proxy_word_score_mode(score_mode),
    )


def extract_topic_words_from_sentence_topic_npmi(
    *,
    sentence_topics_by_doc: list[np.ndarray],
    sentence_bow_by_doc: list[list[list[tuple[int, int]]]],
    num_topics: int,
    dictionary: Dictionary,
    topn: int,
    score_mode: ScoreMode = "npmi",
) -> TopicWordsResult:
    scores = compute_topic_word_npmi_from_sentence_topics(
        sentence_topics_by_doc=sentence_topics_by_doc,
        sentence_bow_by_doc=sentence_bow_by_doc,
        num_topics=num_topics,
        vocab_size=len(dictionary),
        score_mode=score_mode,
    )
    return TopicWordsResult(
        topic_words=select_top_words(scores=scores, dictionary=dictionary, topn=topn),
        topic_word_source="sentence_topic_proxy_npmi",
        score_mode=score_mode,
        score_definition=describe_proxy_word_score_mode(score_mode),
    )

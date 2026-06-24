"""
Implementation reference:
  https://github.com/adjidieng/ETM

Original implementation licensed under the MIT License.
Copyright (c) 2019 Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import gensim
import numpy as np
import torch
from torch import nn

from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import load_preprocessed_documents
from src.baselines.models.gaussian_helpers import (
    build_local_word2vec,
    load_word_vectors,
    should_use_external_vectors,
)
from src.baselines.params import EtmParams
from src.core.artifacts import (
    PREPROCESSING_SELECTION_FILENAME,
    PickleArtifactSpec,
    save_json,
    save_split_jsons,
    save_split_pickles,
)
from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    filter_selected_corpus_by_vocabulary,
    select_modelable_documents,
)


@dataclass(frozen=True)
class EtmTrainResult:
    model: "EtmModel"
    params: EtmParams
    train_doc_topic: np.ndarray
    topic_word_scores: np.ndarray
    vocabulary: list[str]
    embeddings: np.ndarray
    local_word_vectors: gensim.models.KeyedVectors | None
    train_preprocessed: list[PreprocessedDocument]
    average_loss: list[float]
    device: str
    train_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class EtmInferResult:
    test_doc_topic: np.ndarray
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class _EtmCorpus:
    bow: np.ndarray
    token_docs: list[list[str]]
    preprocessed: list[PreprocessedDocument]
    selection: SelectedCorpus


class EtmModel(nn.Module):
    def __init__(
        self,
        *,
        embeddings: np.ndarray,
        num_topics: int,
        hidden_size: int,
        theta_act: str,
        enc_drop: float,
    ) -> None:
        super().__init__()
        embedding_tensor = torch.as_tensor(embeddings, dtype=torch.float32)
        self.register_buffer("rho", embedding_tensor)
        vocab_size, embedding_dim = embedding_tensor.shape
        self.num_topics = int(num_topics)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            _build_activation(theta_act),
            nn.Linear(hidden_size, hidden_size),
            _build_activation(theta_act),
            nn.Dropout(enc_drop),
        )
        self.mu_q_theta = nn.Linear(hidden_size, num_topics)
        self.logsigma_q_theta = nn.Linear(hidden_size, num_topics)
        self.alphas = nn.Parameter(torch.empty(embedding_dim, num_topics))
        nn.init.xavier_uniform_(self.alphas)

    def encode(self, bows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.q_theta(bows)
        return self.mu_q_theta(hidden), self.logsigma_q_theta(hidden)

    def get_theta(self, normalized_bows: torch.Tensor, *, sample: bool) -> torch.Tensor:
        mu, logsigma = self.encode(normalized_bows)
        if sample:
            std = torch.exp(0.5 * logsigma)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return torch.softmax(z, dim=-1)

    def get_beta(self) -> torch.Tensor:
        logits = torch.matmul(self.rho, self.alphas)
        return torch.softmax(logits, dim=0).transpose(0, 1)

    def forward(
        self,
        bows: torch.Tensor,
        normalized_bows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logsigma = self.encode(normalized_bows)
        std = torch.exp(0.5 * logsigma)
        theta = torch.softmax(mu + std * torch.randn_like(std), dim=-1)
        beta = self.get_beta()
        preds = torch.matmul(theta, beta)
        recon = -(torch.log(preds + 1e-10) * bows).sum(dim=1)
        kld = -0.5 * torch.sum(1.0 + logsigma - mu.pow(2) - logsigma.exp(), dim=1)
        return recon, kld


def _build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    if name == "rrelu":
        return nn.RReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported ETM activation: {name}")


def _resolve_device(encoder_device: str) -> torch.device:
    requested = str(encoder_device or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _set_random_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_documents(
    *,
    csv_paths: Sequence[str],
    targets: Sequence[str] | None,
    text_column: str,
    target_column: str | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
) -> list[PreprocessedDocument]:
    return load_preprocessed_documents(
        csv_paths=csv_paths,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
        delimiter=delimiter,
        language=language,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )


def _load_or_build_vectors(
    *,
    token_docs: Sequence[Sequence[str]],
    word2vec: str,
    wikientvec_cache_dir: str | None,
    local_word_vectors: gensim.models.KeyedVectors | None = None,
) -> tuple[gensim.models.KeyedVectors, gensim.models.KeyedVectors | None]:
    if local_word_vectors is not None:
        return local_word_vectors, local_word_vectors
    if not isinstance(word2vec, str) or should_use_external_vectors(word2vec):
        return (
            load_word_vectors(
                word2vec,
                wikientvec_cache_dir=wikientvec_cache_dir,
            ),
            None,
        )
    if word2vec.strip().lower() not in {"local", "local-word2vec"}:
        return (
            load_word_vectors(
                word2vec,
                wikientvec_cache_dir=wikientvec_cache_dir,
            ),
            None,
        )
    built = build_local_word2vec(token_docs)
    return built, built


def _select_training_vocabulary(
    token_docs: Sequence[Sequence[str]],
    vectors: gensim.models.KeyedVectors,
) -> list[str]:
    observed = {token for doc in token_docs for token in doc if token in vectors}
    return [str(word) for word in vectors.key_to_index.keys() if str(word) in observed]


def _filter_document(
    document: PreprocessedDocument,
    vocabulary: set[str],
) -> PreprocessedDocument | None:
    kept_raw: list[str] = []
    kept_tokens: list[list[str]] = []
    for raw_sentence, sentence_tokens in zip(
        document.sentences_raw,
        document.sentences_tokenized,
    ):
        tokens = [token for token in sentence_tokens if token in vocabulary]
        if tokens:
            kept_raw.append(raw_sentence)
            kept_tokens.append(tokens)
    document_tokens = [
        token for sentence_tokens in kept_tokens for token in sentence_tokens
    ]
    if not document_tokens:
        return None
    return PreprocessedDocument(
        raw_text=document.raw_text,
        sentences_raw=kept_raw,
        sentences_tokenized=kept_tokens,
        sentences_joined=[" ".join(tokens) for tokens in kept_tokens],
        document_tokens=document_tokens,
    )


def _build_bow(
    token_docs: Sequence[Sequence[str]], vocabulary: list[str]
) -> np.ndarray:
    word_ids = {word: idx for idx, word in enumerate(vocabulary)}
    bow = np.zeros((len(token_docs), len(vocabulary)), dtype=np.float32)
    for doc_idx, tokens in enumerate(token_docs):
        for token in tokens:
            word_id = word_ids.get(token)
            if word_id is not None:
                bow[doc_idx, word_id] += 1.0
    return bow


def _prepare_corpus(
    *,
    documents: Sequence[PreprocessedDocument],
    vocabulary: list[str],
    empty_error_message: str | None,
    base_selection: SelectedCorpus | None = None,
) -> _EtmCorpus:
    vocabulary_set = set(vocabulary)
    selection = filter_selected_corpus_by_vocabulary(
        base_selection or select_modelable_documents(documents),
        vocabulary_set,
    )
    if not selection.documents:
        if empty_error_message is not None:
            raise ValueError(empty_error_message)
        empty_selection = selection
        return _EtmCorpus(
            bow=np.zeros((0, len(vocabulary)), dtype=np.float32),
            token_docs=[],
            preprocessed=[],
            selection=empty_selection,
        )
    token_docs = [list(doc.document_tokens) for doc in selection.documents]
    return _EtmCorpus(
        bow=_build_bow(token_docs, vocabulary),
        token_docs=token_docs,
        preprocessed=selection.documents,
        selection=selection,
    )


def _normalize_bow(bow: np.ndarray, *, bow_norm: bool) -> np.ndarray:
    if not bow_norm:
        return bow.astype(np.float32, copy=True)
    totals = bow.sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0
    return (bow / totals).astype(np.float32, copy=False)


def _build_optimizer(params: EtmParams, model: EtmModel) -> torch.optim.Optimizer:
    optimizer_kwargs = {
        "lr": params.lr,
        "weight_decay": params.weight_decay,
    }
    if params.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    if params.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), **optimizer_kwargs)
    if params.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), **optimizer_kwargs)
    if params.optimizer == "adadelta":
        return torch.optim.Adadelta(model.parameters(), **optimizer_kwargs)
    if params.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), **optimizer_kwargs)
    raise ValueError(f"Unsupported ETM optimizer: {params.optimizer}")


def _infer_doc_topics(
    *,
    model: EtmModel,
    bow: np.ndarray,
    batch_size: int,
    bow_norm: bool,
    device: torch.device,
) -> np.ndarray:
    if bow.shape[0] == 0:
        return np.zeros((0, model.num_topics), dtype=np.float64)
    model.eval()
    normalized = _normalize_bow(bow, bow_norm=bow_norm)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, bow.shape[0], batch_size):
            batch = torch.as_tensor(
                normalized[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            theta = model.get_theta(batch, sample=False)
            outputs.append(theta.detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float64, copy=False)


def train_etm(
    *,
    train_csvs: Sequence[str],
    targets: Sequence[str] | None,
    text_column: str,
    target_column: str | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
    num_topics: int,
    params: EtmParams,
    train_dir: Path,
    use_legacy: bool,
    encoder_device: str,
    effective_random_state: int,
) -> EtmTrainResult:
    _ = train_dir, use_legacy
    seed = int(effective_random_state)
    _set_random_seeds(seed)
    documents = _load_documents(
        csv_paths=train_csvs,
        targets=targets,
        text_column=text_column,
        target_column=target_column,
        delimiter=delimiter,
        language=language,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )
    base_selection = select_modelable_documents(documents)
    raw_token_docs = [
        doc.document_tokens for doc in base_selection.documents if doc.document_tokens
    ]
    if not raw_token_docs:
        raise ValueError("No valid tokenized docs available for ETM.")
    vectors, local_vectors = _load_or_build_vectors(
        token_docs=raw_token_docs,
        word2vec=params.word2vec,
        wikientvec_cache_dir=params.wikientvec_cache_dir,
    )
    vocabulary = _select_training_vocabulary(raw_token_docs, vectors)
    if not vocabulary:
        raise ValueError("No ETM training tokens remain after word-vector filtering.")
    corpus = _prepare_corpus(
        documents=documents,
        vocabulary=vocabulary,
        empty_error_message="No ETM training docs remain after OOV filtering.",
        base_selection=base_selection,
    )
    embeddings = np.vstack(
        [np.asarray(vectors[word], dtype=np.float32) for word in vocabulary]
    )
    device = _resolve_device(encoder_device)
    model = EtmModel(
        embeddings=embeddings,
        num_topics=num_topics,
        hidden_size=params.t_hidden_size,
        theta_act=params.theta_act,
        enc_drop=params.enc_drop,
    ).to(device)
    optimizer = _build_optimizer(params, model)
    rng = np.random.default_rng(seed)
    average_loss: list[float] = []
    normalized = _normalize_bow(corpus.bow, bow_norm=params.bow_norm)

    for _epoch in range(params.num_epochs):
        model.train()
        order = rng.permutation(corpus.bow.shape[0])
        epoch_loss = 0.0
        epoch_docs = 0
        for start in range(0, len(order), params.batch_size):
            batch_ids = order[start : start + params.batch_size]
            bows = torch.as_tensor(
                corpus.bow[batch_ids],
                dtype=torch.float32,
                device=device,
            )
            normalized_bows = torch.as_tensor(
                normalized[batch_ids],
                dtype=torch.float32,
                device=device,
            )
            recon, kld = model(bows, normalized_bows)
            loss = (recon + kld).mean()
            optimizer.zero_grad()
            loss.backward()
            if params.clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), params.clip)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * int(len(batch_ids))
            epoch_docs += int(len(batch_ids))
        average_loss.append(epoch_loss / max(epoch_docs, 1))

    train_doc_topic = _infer_doc_topics(
        model=model,
        bow=corpus.bow,
        batch_size=params.eval_batch_size,
        bow_norm=params.bow_norm,
        device=device,
    )
    topic_word_scores = model.get_beta().detach().cpu().numpy().astype(np.float64)
    return EtmTrainResult(
        model=model,
        params=params,
        train_doc_topic=train_doc_topic,
        topic_word_scores=topic_word_scores,
        vocabulary=vocabulary,
        embeddings=embeddings.astype(np.float64, copy=False),
        local_word_vectors=local_vectors,
        train_preprocessed=corpus.preprocessed,
        average_loss=average_loss,
        device=str(device),
        train_selection=corpus.selection,
    )


def infer_etm(
    *,
    train_result: EtmTrainResult,
    test_csvs: Sequence[str],
    targets: Sequence[str] | None,
    text_column: str,
    target_column: str | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_stopwords_path: str | None,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
    num_topics: int,
    params: EtmParams,
    use_legacy: bool,
) -> EtmInferResult:
    _ = num_topics, use_legacy
    documents = _load_documents(
        csv_paths=test_csvs,
        targets=targets,
        text_column=text_column,
        target_column=target_column,
        delimiter=delimiter,
        language=language,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )
    base_selection = select_modelable_documents(documents)
    corpus = _prepare_corpus(
        documents=documents,
        vocabulary=train_result.vocabulary,
        empty_error_message=None,
        base_selection=base_selection,
    )
    device = next(train_result.model.parameters()).device
    test_doc_topic = _infer_doc_topics(
        model=train_result.model,
        bow=corpus.bow,
        batch_size=params.eval_batch_size,
        bow_norm=params.bow_norm,
        device=device,
    )
    return EtmInferResult(
        test_doc_topic=test_doc_topic,
        test_preprocessed=corpus.preprocessed,
        test_selection=corpus.selection,
    )


def persist_etm_run(
    *,
    train_result: EtmTrainResult,
    infer_result: EtmInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    params_path = train_dir / "params.json"
    vocabulary_path = train_dir / "vocabulary.json"
    model_state_path = train_dir / "model_state.pt"
    baseline_params = asdict(train_result.params)
    baseline_params["word2vec"] = str(baseline_params["word2vec"])
    save_json(
        {
            "baseline_params": baseline_params,
            "num_topics": int(train_result.train_doc_topic.shape[1]),
            "vocab_size": int(len(train_result.vocabulary)),
            "embedding_dim": int(train_result.embeddings.shape[1]),
            "average_loss": [float(value) for value in train_result.average_loss],
            "device": train_result.device,
        },
        params_path,
    )
    save_json(train_result.vocabulary, vocabulary_path)
    torch.save(train_result.model.state_dict(), model_state_path)
    saved = save_split_pickles(
        [
            PickleArtifactSpec(
                name="train_path",
                filename="etm.pkl",
                payload=train_result.train_doc_topic,
                split="train",
            ),
            PickleArtifactSpec(
                name="infer_path",
                filename=f"{category}.pkl",
                payload=infer_result.test_doc_topic,
                split="infer",
            ),
            PickleArtifactSpec(
                name="test_doc_topic_soft",
                filename=f"{category}_doc_topic_soft.pkl",
                payload=infer_result.test_doc_topic,
                split="infer",
            ),
            PickleArtifactSpec(
                name="topic_word_scores",
                filename="topic_word_scores.pkl",
                payload=train_result.topic_word_scores,
                split="train",
            ),
            PickleArtifactSpec(
                name="embeddings",
                filename="embeddings.pkl",
                payload=train_result.embeddings,
                split="train",
            ),
            PickleArtifactSpec(
                name="train_preprocessed",
                filename="preprocessed_corpus.pkl",
                payload=train_result.train_preprocessed,
                split="train",
            ),
            PickleArtifactSpec(
                name="infer_preprocessed",
                filename="preprocessed_corpus.pkl",
                payload=infer_result.test_preprocessed,
                split="infer",
            ),
        ],
        train_dir=train_dir,
        infer_dir=infer_dir,
    )
    train_selection = train_result.train_selection or select_modelable_documents(
        train_result.train_preprocessed
    )
    test_selection = infer_result.test_selection or select_modelable_documents(
        infer_result.test_preprocessed
    )
    selection_saved = save_split_jsons(
        {
            "train_preprocessing_selection": (
                train_selection.to_json_dict(),
                PREPROCESSING_SELECTION_FILENAME,
                "train",
            ),
            "infer_preprocessing_selection": (
                test_selection.to_json_dict(),
                PREPROCESSING_SELECTION_FILENAME,
                "infer",
            ),
        },
        train_dir=train_dir,
        infer_dir=infer_dir,
    )
    extras: dict[str, Path] = {
        "params_json": params_path,
        "model_state": model_state_path,
        "vocabulary": vocabulary_path,
        "topic_word_scores": saved["topic_word_scores"],
        "train_preprocessed": saved["train_preprocessed"],
        "infer_preprocessed": saved["infer_preprocessed"],
        "train_preprocessing_selection": selection_saved[
            "train_preprocessing_selection"
        ],
        "infer_preprocessing_selection": selection_saved[
            "infer_preprocessing_selection"
        ],
        "test_doc_topic_soft": saved["test_doc_topic_soft"],
    }
    if train_result.local_word_vectors is not None:
        kv_path = train_dir / "local_word2vec.kv"
        train_result.local_word_vectors.save(kv_path.as_posix())
        extras["local_word2vec"] = kv_path
    return BaselineArtifacts(
        train_path=saved["train_path"],
        infer_path=saved["infer_path"],
        extras=extras,
    )

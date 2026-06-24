from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy.special import ive, logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.baselines.contracts import BaselineArtifacts
from src.baselines.dataset_adapters import load_preprocessed_documents
from src.baselines.params import (
    GaussianKMeansParams,
    GaussianMixtureParams,
    MovMFParams,
    SphericalKMeansParams,
)
from src.core.artifacts import (
    PREPROCESSING_SELECTION_FILENAME,
    PickleArtifactSpec,
    save_split_jsons,
    save_split_pickles,
)
from src.data.preprocessing import (
    PreprocessedDocument,
    SelectedCorpus,
    select_modelable_documents,
)
from src.utils.encoder import SentenceEncoder
from src.utils.encoder_inputs import (
    encode_sentences,
    fit_encoder_on_sentences,
    sentence_flat_inputs_for_encoder,
)

ClusterMethod = Literal[
    "spherical_kmeans",
    "gaussian_kmeans",
    "movmf",
    "gaussian_mixture",
]
ClusteringParams = (
    SphericalKMeansParams | GaussianKMeansParams | MovMFParams | GaussianMixtureParams
)


@dataclass(frozen=True)
class SentenceEmbeddingCorpus:
    preprocessed: list[PreprocessedDocument]
    sentences: list[str]
    sentences_tokenized: list[list[str]]
    doc_offsets: np.ndarray
    selection: SelectedCorpus | None = None

    @property
    def num_docs(self) -> int:
        return int(self.doc_offsets.size) - 1

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)


@dataclass(frozen=True)
class SentenceEmbeddingClusteringModelState:
    method: str
    params: dict[str, object]
    encoder_model_name: str
    encode_prefix: str | None
    normalize_embeddings: bool
    centers: np.ndarray | None
    mixture_weights: np.ndarray | None = None
    kappa: np.ndarray | None = None
    covariances: object | None = None
    converged: bool | None = None
    n_iter: int | None = None
    lower_bound: float | None = None


@dataclass(frozen=True)
class SentenceEmbeddingClusteringTrainResult:
    method: str
    model: object
    encoder: SentenceEncoder
    model_state: SentenceEmbeddingClusteringModelState
    train_doc_topic: np.ndarray
    train_sentence_topic_soft: list[np.ndarray]
    train_sentence_topic_assignments: list[np.ndarray]
    train_preprocessed: list[PreprocessedDocument]
    train_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class SentenceEmbeddingClusteringInferResult:
    test_doc_topic: np.ndarray
    test_sentence_topic_soft: list[np.ndarray]
    test_sentence_topic_assignments: list[np.ndarray]
    test_preprocessed: list[PreprocessedDocument]
    test_selection: SelectedCorpus | None = None


@dataclass(frozen=True)
class _SphericalKMeansModel:
    centers: np.ndarray
    inertia: float
    n_iter: int
    converged: bool

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        x = _normalize_rows(embeddings)
        labels = np.argmax(x @ self.centers.T, axis=1)
        return _one_hot(labels, self.centers.shape[0])

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(embeddings), axis=1).astype(np.int32)


@dataclass(frozen=True)
class _GaussianKMeansModel:
    estimator: KMeans

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        labels = self.estimator.predict(np.asarray(embeddings, dtype=np.float64))
        return _one_hot(labels, self.estimator.n_clusters)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.estimator.predict(np.asarray(embeddings, dtype=np.float64)).astype(
            np.int32
        )


@dataclass(frozen=True)
class _MovMFModel:
    mu: np.ndarray
    kappa: np.ndarray
    weights: np.ndarray
    lower_bound: float
    n_iter: int
    converged: bool

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        x = _normalize_rows(embeddings)
        log_prob = _movmf_log_prob_matrix(x, self.mu, self.kappa, self.weights)
        return np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(embeddings), axis=1).astype(np.int32)


@dataclass(frozen=True)
class _GaussianMixtureModel:
    estimator: GaussianMixture

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(np.asarray(embeddings, dtype=np.float64))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.estimator.predict(np.asarray(embeddings, dtype=np.float64)).astype(
            np.int32
        )


def _load_sentence_corpus(
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
    use_legacy: bool,
) -> SentenceEmbeddingCorpus:
    documents = load_preprocessed_documents(
        csv_paths=csv_paths,
        text_column=text_column,
        target_column=target_column,
        targets=targets,
        delimiter=" / " if use_legacy else delimiter,
        language=language,
        segmenter="delimiter" if use_legacy else segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=ja_stopwords_path,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )

    selection = select_modelable_documents(documents)
    filtered: list[PreprocessedDocument] = []
    sentences: list[str] = []
    sentences_tokenized: list[list[str]] = []
    doc_offsets = [0]
    for document in selection.documents:
        doc_sentences = [sentence for sentence in document.sentences_raw if sentence]
        if not doc_sentences:
            continue
        filtered.append(document)
        sentences.extend(doc_sentences)
        sentences_tokenized.extend(
            [list(tokens) for tokens in document.sentences_tokenized]
        )
        doc_offsets.append(len(sentences))

    if not filtered:
        raise ValueError("sentence embedding clustering requires non-empty sentences.")
    return SentenceEmbeddingCorpus(
        preprocessed=filtered,
        sentences=sentences,
        sentences_tokenized=sentences_tokenized,
        doc_offsets=np.asarray(doc_offsets, dtype=np.int32),
        selection=selection,
    )


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms <= 0.0)] = 1.0
    return x / norms


def _one_hot(labels: np.ndarray, num_topics: int) -> np.ndarray:
    normalized = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.zeros((normalized.size, num_topics), dtype=np.float64)
    if normalized.size:
        out[np.arange(normalized.size), normalized] = 1.0
    return out


def _group_rows_by_doc(rows: np.ndarray, doc_offsets: np.ndarray) -> list[np.ndarray]:
    grouped: list[np.ndarray] = []
    for doc_index in range(int(doc_offsets.size) - 1):
        start = int(doc_offsets[doc_index])
        end = int(doc_offsets[doc_index + 1])
        grouped.append(np.asarray(rows[start:end], dtype=np.float64).copy())
    return grouped


def _aggregate_doc_topic(rows: np.ndarray, doc_offsets: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    doc_topic = np.zeros((int(doc_offsets.size) - 1, rows.shape[1]), dtype=np.float64)
    for doc_index in range(doc_topic.shape[0]):
        start = int(doc_offsets[doc_index])
        end = int(doc_offsets[doc_index + 1])
        if end > start:
            doc_topic[doc_index] = rows[start:end].mean(axis=0)
        else:
            doc_topic[doc_index] = 1.0 / float(rows.shape[1])
    return _row_normalize(doc_topic)


def _row_normalize(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    row_sums = out.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(row_sums)) | (row_sums <= 0.0)
    if np.any(bad):
        out[bad[:, 0]] = 1.0 / float(out.shape[1])
        row_sums = out.sum(axis=1, keepdims=True)
    return out / row_sums


def _init_centers(
    x: np.ndarray,
    *,
    num_topics: int,
    init: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if x.shape[0] < num_topics:
        raise ValueError(
            "num_topics cannot exceed the number of training sentences: "
            f"{num_topics} > {x.shape[0]}."
        )
    if init == "random":
        indices = rng.choice(x.shape[0], size=num_topics, replace=False)
        return _normalize_rows(x[indices])

    centers = [x[int(rng.integers(x.shape[0]))]]
    closest_dist = 1.0 - np.clip(x @ centers[0], -1.0, 1.0)
    for _ in range(1, num_topics):
        weights = np.maximum(closest_dist, 0.0) ** 2
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            candidates = np.setdiff1d(np.arange(x.shape[0]), np.asarray([], dtype=int))
            next_index = int(rng.choice(candidates))
        else:
            next_index = int(rng.choice(x.shape[0], p=weights / total))
        centers.append(x[next_index])
        sim = x @ centers[-1]
        closest_dist = np.minimum(closest_dist, 1.0 - np.clip(sim, -1.0, 1.0))
    return _normalize_rows(np.asarray(centers, dtype=np.float64))


def _fit_spherical_once(
    x: np.ndarray,
    *,
    num_topics: int,
    init: str,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> _SphericalKMeansModel:
    centers = _init_centers(x, num_topics=num_topics, init=init, rng=rng)
    previous_objective: float | None = None
    converged = False
    labels = np.zeros(x.shape[0], dtype=np.int32)
    objective = -np.inf

    for iteration in range(1, max_iter + 1):
        sims = x @ centers.T
        labels = np.argmax(sims, axis=1).astype(np.int32)
        chosen = sims[np.arange(x.shape[0]), labels]
        objective = float(chosen.sum())

        new_centers = np.zeros_like(centers)
        for topic in range(num_topics):
            members = x[labels == topic]
            if members.size:
                new_centers[topic] = members.mean(axis=0)
            else:
                worst_index = int(np.argmin(chosen))
                new_centers[topic] = x[worst_index]
        new_centers = _normalize_rows(new_centers)

        if previous_objective is not None:
            improvement = abs(objective - previous_objective)
            threshold = tol * max(1.0, abs(previous_objective))
            if improvement <= threshold:
                centers = new_centers
                converged = True
                break
        centers = new_centers
        previous_objective = objective

    inertia = float(np.sum(1.0 - np.max(x @ centers.T, axis=1)))
    return _SphericalKMeansModel(
        centers=centers,
        inertia=inertia,
        n_iter=iteration,
        converged=converged,
    )


def _fit_spherical_kmeans(
    embeddings: np.ndarray,
    *,
    num_topics: int,
    params: SphericalKMeansParams,
) -> _SphericalKMeansModel:
    x = _normalize_rows(embeddings)
    best: _SphericalKMeansModel | None = None
    for init_index in range(params.n_init):
        rng = np.random.default_rng(params.random_state + init_index)
        candidate = _fit_spherical_once(
            x,
            num_topics=num_topics,
            init=params.init,
            max_iter=params.max_iter,
            tol=params.tol,
            rng=rng,
        )
        if best is None or candidate.inertia < best.inertia:
            best = candidate
    if best is None:
        raise RuntimeError("spherical k-means did not produce a fitted model.")
    return best


def _fit_gaussian_kmeans(
    embeddings: np.ndarray,
    *,
    num_topics: int,
    params: GaussianKMeansParams,
) -> _GaussianKMeansModel:
    estimator = KMeans(
        n_clusters=num_topics,
        init="k-means++" if params.init == "kmeans++" else params.init,
        n_init=params.n_init,
        max_iter=params.max_iter,
        tol=params.tol,
        random_state=params.random_state,
        verbose=1 if params.verbose else 0,
    )
    estimator.fit(np.asarray(embeddings, dtype=np.float64))
    return _GaussianKMeansModel(estimator=estimator)


def _log_vmf_normalizer(kappa: np.ndarray, dim: int) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=np.float64)
    order = float(dim) / 2.0 - 1.0
    safe_kappa = np.maximum(kappa, 1e-12)
    log_bessel = np.log(np.maximum(ive(order, safe_kappa), 1e-300)) + safe_kappa
    fallback = safe_kappa - 0.5 * np.log(2.0 * np.pi * safe_kappa)
    log_bessel = np.where(np.isfinite(log_bessel), log_bessel, fallback)
    return (
        order * np.log(safe_kappa)
        - (float(dim) / 2.0) * np.log(2.0 * np.pi)
        - log_bessel
    )


def _estimate_kappa(
    resultant_norm: np.ndarray,
    counts: np.ndarray,
    *,
    dim: int,
    min_kappa: float,
    max_kappa: float,
) -> np.ndarray:
    rbar = resultant_norm / np.maximum(counts, 1e-12)
    rbar = np.clip(rbar, 1e-8, 1.0 - 1e-8)
    kappa = rbar * (float(dim) - rbar**2) / np.maximum(1.0 - rbar**2, 1e-12)
    return np.clip(kappa, min_kappa, max_kappa)


def _movmf_log_prob_matrix(
    x: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    dim = x.shape[1]
    log_c = _log_vmf_normalizer(kappa, dim)
    log_weights = np.log(np.maximum(weights, 1e-300))
    return x @ (mu * kappa[:, None]).T + log_c[None, :] + log_weights[None, :]


def _initialize_movmf(
    x: np.ndarray,
    *,
    num_topics: int,
    params: MovMFParams,
    init_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if params.init == "spherical_kmeans":
        spherical_params = SphericalKMeansParams(
            encoder_model_name=params.encoder_model_name,
            encode_prefix=params.encode_prefix,
            encode_batch_size=params.encode_batch_size,
            random_state=params.random_state + init_index,
            n_init=1,
            max_iter=min(50, params.max_iter),
            tol=params.tol,
            init="kmeans++",
        )
        spherical = _fit_spherical_kmeans(
            x,
            num_topics=num_topics,
            params=spherical_params,
        )
        labels = spherical.predict(x)
    else:
        rng = np.random.default_rng(params.random_state + init_index)
        labels = rng.integers(num_topics, size=x.shape[0], dtype=np.int32)

    resp = _one_hot(labels, num_topics)
    counts = resp.sum(axis=0) + 1e-12
    resultant = resp.T @ x
    mu = _normalize_rows(resultant)
    weights = counts / counts.sum()
    kappa = _estimate_kappa(
        np.linalg.norm(resultant, axis=1),
        counts,
        dim=x.shape[1],
        min_kappa=params.min_kappa,
        max_kappa=params.max_kappa,
    )
    return mu, kappa, weights


def _fit_movmf_once(
    x: np.ndarray,
    *,
    num_topics: int,
    params: MovMFParams,
    init_index: int,
) -> _MovMFModel:
    mu, kappa, weights = _initialize_movmf(
        x,
        num_topics=num_topics,
        params=params,
        init_index=init_index,
    )
    previous_lower_bound: float | None = None
    converged = False
    lower_bound = -np.inf
    resp = np.zeros((x.shape[0], num_topics), dtype=np.float64)

    for iteration in range(1, params.max_iter + 1):
        log_prob = _movmf_log_prob_matrix(x, mu, kappa, weights)
        log_norm = logsumexp(log_prob, axis=1, keepdims=True)
        resp = np.exp(log_prob - log_norm)
        lower_bound = float(np.mean(log_norm))

        counts = resp.sum(axis=0) + 1e-12
        resultant = resp.T @ x
        empty = counts <= 1e-8
        if np.any(empty):
            rng = np.random.default_rng(params.random_state + init_index + iteration)
            replacement = x[rng.choice(x.shape[0], size=int(empty.sum()), replace=True)]
            resultant[empty] = replacement
            counts[empty] = 1.0
        mu = _normalize_rows(resultant)
        weights = counts / counts.sum()
        kappa = _estimate_kappa(
            np.linalg.norm(resultant, axis=1),
            counts,
            dim=x.shape[1],
            min_kappa=params.min_kappa,
            max_kappa=params.max_kappa,
        )

        if previous_lower_bound is not None:
            improvement = abs(lower_bound - previous_lower_bound)
            threshold = params.tol * max(1.0, abs(previous_lower_bound))
            if improvement <= threshold:
                converged = True
                break
        previous_lower_bound = lower_bound

    return _MovMFModel(
        mu=mu,
        kappa=kappa,
        weights=weights,
        lower_bound=lower_bound,
        n_iter=iteration,
        converged=converged,
    )


def _fit_movmf(
    embeddings: np.ndarray,
    *,
    num_topics: int,
    params: MovMFParams,
) -> _MovMFModel:
    x = _normalize_rows(embeddings)
    best: _MovMFModel | None = None
    for init_index in range(params.n_init):
        candidate = _fit_movmf_once(
            x,
            num_topics=num_topics,
            params=params,
            init_index=init_index,
        )
        if best is None or candidate.lower_bound > best.lower_bound:
            best = candidate
    if best is None:
        raise RuntimeError("movMF did not produce a fitted model.")
    return best


def _fit_gaussian_mixture(
    embeddings: np.ndarray,
    *,
    num_topics: int,
    params: GaussianMixtureParams,
) -> _GaussianMixtureModel:
    estimator = GaussianMixture(
        n_components=num_topics,
        covariance_type=params.covariance_type,
        tol=params.tol,
        reg_covar=params.reg_covar,
        max_iter=params.max_iter,
        n_init=params.n_init,
        init_params=params.init,
        random_state=params.random_state,
        verbose=1 if params.verbose else 0,
    )
    estimator.fit(np.asarray(embeddings, dtype=np.float64))
    return _GaussianMixtureModel(estimator=estimator)


def _fit_model(
    *,
    method: ClusterMethod,
    embeddings: np.ndarray,
    num_topics: int,
    params: ClusteringParams,
) -> object:
    if embeddings.shape[0] < num_topics:
        raise ValueError(
            f"{method} requires at least num_topics training sentences: "
            f"sentences={embeddings.shape[0]}, num_topics={num_topics}."
        )
    if method == "spherical_kmeans":
        return _fit_spherical_kmeans(
            embeddings,
            num_topics=num_topics,
            params=_require_params(params, SphericalKMeansParams),
        )
    if method == "gaussian_kmeans":
        return _fit_gaussian_kmeans(
            embeddings,
            num_topics=num_topics,
            params=_require_params(params, GaussianKMeansParams),
        )
    if method == "movmf":
        return _fit_movmf(
            embeddings,
            num_topics=num_topics,
            params=_require_params(params, MovMFParams),
        )
    if method == "gaussian_mixture":
        return _fit_gaussian_mixture(
            embeddings,
            num_topics=num_topics,
            params=_require_params(params, GaussianMixtureParams),
        )
    raise ValueError(f"Unknown sentence embedding clustering method: {method}")


def _require_params(params: object, expected_type: type) -> object:
    if not isinstance(params, expected_type):
        raise TypeError(
            f"Expected params {expected_type.__name__}, got {type(params).__name__}."
        )
    return params


def _predict_grouped(
    *,
    model: object,
    embeddings: np.ndarray,
    doc_offsets: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    probs = np.asarray(model.predict_proba(embeddings), dtype=np.float64)
    probs = _row_normalize(probs)
    assignments_flat = np.asarray(model.predict(embeddings), dtype=np.int32)
    doc_topic = _aggregate_doc_topic(probs, doc_offsets)
    sentence_topic = _group_rows_by_doc(probs, doc_offsets)
    assignments = _group_rows_by_doc(assignments_flat[:, None], doc_offsets)
    assignments = [item.reshape(-1).astype(np.int32) for item in assignments]
    return doc_topic, sentence_topic, assignments


def _model_state(
    *,
    method: ClusterMethod,
    model: object,
    params: ClusteringParams,
) -> SentenceEmbeddingClusteringModelState:
    centers: np.ndarray | None = None
    mixture_weights: np.ndarray | None = None
    kappa: np.ndarray | None = None
    covariances: object | None = None
    converged: bool | None = None
    n_iter: int | None = None
    lower_bound: float | None = None

    if isinstance(model, _SphericalKMeansModel):
        centers = model.centers
        converged = model.converged
        n_iter = model.n_iter
        lower_bound = -model.inertia
    elif isinstance(model, _GaussianKMeansModel):
        centers = np.asarray(model.estimator.cluster_centers_, dtype=np.float64)
        converged = True
        n_iter = int(model.estimator.n_iter_)
        lower_bound = -float(model.estimator.inertia_)
    elif isinstance(model, _MovMFModel):
        centers = model.mu
        mixture_weights = model.weights
        kappa = model.kappa
        converged = model.converged
        n_iter = model.n_iter
        lower_bound = model.lower_bound
    elif isinstance(model, _GaussianMixtureModel):
        centers = np.asarray(model.estimator.means_, dtype=np.float64)
        mixture_weights = np.asarray(model.estimator.weights_, dtype=np.float64)
        covariances = model.estimator.covariances_
        converged = bool(model.estimator.converged_)
        n_iter = int(model.estimator.n_iter_)
        lower_bound = float(model.estimator.lower_bound_)

    return SentenceEmbeddingClusteringModelState(
        method=method,
        params=asdict(params),
        encoder_model_name=params.encoder_model_name,
        encode_prefix=params.encode_prefix,
        normalize_embeddings=method in {"spherical_kmeans", "movmf"},
        centers=centers,
        mixture_weights=mixture_weights,
        kappa=kappa,
        covariances=covariances,
        converged=converged,
        n_iter=n_iter,
        lower_bound=lower_bound,
    )


def train_sentence_embedding_clustering(
    *,
    method: ClusterMethod,
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
    encoder_device: str,
    params: ClusteringParams,
    train_dir: Path,
    use_legacy: bool,
) -> SentenceEmbeddingClusteringTrainResult:
    _ = train_dir
    train_corpus = _load_sentence_corpus(
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
        use_legacy=use_legacy,
    )
    encoder = SentenceEncoder(
        params.encoder_model_name,
        device=encoder_device,
        encode_prefix=params.encode_prefix,
        backend=params.encoder_backend,
        pooling=params.pooling,
        encode_prompt=params.encode_prompt,
        encode_prompt_name=params.encode_prompt_name,
        encode_batch_size=params.encode_batch_size,
        model_kwargs=params.model_kwargs,
        tokenizer_kwargs=params.tokenizer_kwargs,
        normalize_embeddings=params.normalize_embeddings,
        truncate_dim=params.truncate_dim,
        strip_terminal_normalize=bool(
            getattr(params, "strip_terminal_normalize", True)
        ),
        normalize=False,
    )
    fit_encoder_on_sentences(encoder, train_corpus.preprocessed)
    train_sentences, train_tokens, train_offsets, train_preprocessed = (
        sentence_flat_inputs_for_encoder(train_corpus.preprocessed, encoder)
    )
    train_corpus = SentenceEmbeddingCorpus(
        preprocessed=train_preprocessed,
        sentences=train_sentences,
        sentences_tokenized=train_tokens,
        doc_offsets=train_offsets,
        selection=train_corpus.selection,
    )
    train_embeddings = encode_sentences(
        encoder,
        train_corpus.sentences,
        train_corpus.sentences_tokenized,
        show_progress_bar=params.verbose,
    )
    model = _fit_model(
        method=method,
        embeddings=np.asarray(train_embeddings, dtype=np.float64),
        num_topics=num_topics,
        params=params,
    )
    train_doc_topic, train_sentence_topic, train_assignments = _predict_grouped(
        model=model,
        embeddings=np.asarray(train_embeddings, dtype=np.float64),
        doc_offsets=train_corpus.doc_offsets,
    )
    return SentenceEmbeddingClusteringTrainResult(
        method=method,
        model=model,
        encoder=encoder,
        model_state=_model_state(method=method, model=model, params=params),
        train_doc_topic=train_doc_topic,
        train_sentence_topic_soft=train_sentence_topic,
        train_sentence_topic_assignments=train_assignments,
        train_preprocessed=train_corpus.preprocessed,
        train_selection=train_corpus.selection,
    )


def infer_sentence_embedding_clustering(
    *,
    train_result: SentenceEmbeddingClusteringTrainResult,
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
    params: ClusteringParams,
    use_legacy: bool,
) -> SentenceEmbeddingClusteringInferResult:
    _ = (num_topics, params)
    test_corpus = _load_sentence_corpus(
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
        use_legacy=use_legacy,
    )
    test_sentences, test_tokens, test_offsets, test_preprocessed = (
        sentence_flat_inputs_for_encoder(test_corpus.preprocessed, train_result.encoder)
    )
    test_corpus = SentenceEmbeddingCorpus(
        preprocessed=test_preprocessed,
        sentences=test_sentences,
        sentences_tokenized=test_tokens,
        doc_offsets=test_offsets,
        selection=test_corpus.selection,
    )
    test_embeddings = encode_sentences(
        train_result.encoder,
        test_corpus.sentences,
        test_corpus.sentences_tokenized,
        show_progress_bar=getattr(params, "verbose", False),
    )
    test_doc_topic, test_sentence_topic, test_assignments = _predict_grouped(
        model=train_result.model,
        embeddings=np.asarray(test_embeddings, dtype=np.float64),
        doc_offsets=test_corpus.doc_offsets,
    )
    return SentenceEmbeddingClusteringInferResult(
        test_doc_topic=test_doc_topic,
        test_sentence_topic_soft=test_sentence_topic,
        test_sentence_topic_assignments=test_assignments,
        test_preprocessed=test_corpus.preprocessed,
        test_selection=test_corpus.selection,
    )


def persist_sentence_embedding_clustering_run(
    *,
    train_result: SentenceEmbeddingClusteringTrainResult,
    infer_result: SentenceEmbeddingClusteringInferResult,
    train_dir: Path,
    infer_dir: Path,
    category: str,
) -> BaselineArtifacts:
    saved = save_split_pickles(
        [
            PickleArtifactSpec(
                name="train_path",
                filename=f"{category}.pkl",
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
                name="model_state",
                filename="model_state.pkl",
                payload=train_result.model_state,
                split="train",
            ),
            PickleArtifactSpec(
                name="train_sentence_topic_soft",
                filename=f"{category}_sentence_topic_soft.pkl",
                payload=train_result.train_sentence_topic_soft,
                split="train",
            ),
            PickleArtifactSpec(
                name="test_sentence_topic_soft",
                filename=f"{category}_sentence_topic_soft.pkl",
                payload=infer_result.test_sentence_topic_soft,
                split="infer",
            ),
            PickleArtifactSpec(
                name="train_sentence_topic_assignments",
                filename=f"{category}_sentence_topic_assignments.pkl",
                payload=train_result.train_sentence_topic_assignments,
                split="train",
            ),
            PickleArtifactSpec(
                name="test_sentence_topic_assignments",
                filename=f"{category}_sentence_topic_assignments.pkl",
                payload=infer_result.test_sentence_topic_assignments,
                split="infer",
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
    return BaselineArtifacts(
        train_path=saved["train_path"],
        infer_path=saved["infer_path"],
        extras={
            "model_state": saved["model_state"],
            "train_sentence_topic_soft": saved["train_sentence_topic_soft"],
            "test_sentence_topic_soft": saved["test_sentence_topic_soft"],
            "train_sentence_topic_assignments": saved[
                "train_sentence_topic_assignments"
            ],
            "test_sentence_topic_assignments": saved["test_sentence_topic_assignments"],
            "train_preprocessed": saved["train_preprocessed"],
            "infer_preprocessed": saved["infer_preprocessed"],
            "train_preprocessing_selection": selection_saved[
                "train_preprocessing_selection"
            ],
            "infer_preprocessing_selection": selection_saved[
                "infer_preprocessing_selection"
            ],
        },
    )


def train_spherical_kmeans(**kwargs) -> SentenceEmbeddingClusteringTrainResult:
    return train_sentence_embedding_clustering(method="spherical_kmeans", **kwargs)


def infer_spherical_kmeans(**kwargs) -> SentenceEmbeddingClusteringInferResult:
    return infer_sentence_embedding_clustering(**kwargs)


def persist_spherical_kmeans_run(**kwargs) -> BaselineArtifacts:
    return persist_sentence_embedding_clustering_run(**kwargs)


def train_gaussian_kmeans(**kwargs) -> SentenceEmbeddingClusteringTrainResult:
    return train_sentence_embedding_clustering(method="gaussian_kmeans", **kwargs)


def infer_gaussian_kmeans(**kwargs) -> SentenceEmbeddingClusteringInferResult:
    return infer_sentence_embedding_clustering(**kwargs)


def persist_gaussian_kmeans_run(**kwargs) -> BaselineArtifacts:
    return persist_sentence_embedding_clustering_run(**kwargs)


def train_movmf(**kwargs) -> SentenceEmbeddingClusteringTrainResult:
    return train_sentence_embedding_clustering(method="movmf", **kwargs)


def infer_movmf(**kwargs) -> SentenceEmbeddingClusteringInferResult:
    return infer_sentence_embedding_clustering(**kwargs)


def persist_movmf_run(**kwargs) -> BaselineArtifacts:
    return persist_sentence_embedding_clustering_run(**kwargs)


def train_gaussian_mixture(**kwargs) -> SentenceEmbeddingClusteringTrainResult:
    return train_sentence_embedding_clustering(method="gaussian_mixture", **kwargs)


def infer_gaussian_mixture(**kwargs) -> SentenceEmbeddingClusteringInferResult:
    return infer_sentence_embedding_clustering(**kwargs)


def persist_gaussian_mixture_run(**kwargs) -> BaselineArtifacts:
    return persist_sentence_embedding_clustering_run(**kwargs)

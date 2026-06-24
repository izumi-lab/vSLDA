from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.special import digamma as _digamma
from scipy.special import ive as _ive

from src.core.contracts import TopicModelOutput
from src.core.progress import ProgressReporter, TqdmProgressReporter
from src.models.vmf_artifacts import (
    build_vmf_model_artifact_payload,
    save_vmf_model_artifacts,
)
from src.models.vmf_encoding import VMFDocumentEncoder
from src.models.vmf_inference import VMFCorpusInferenceOutputs, VMFTopicInferencer
from src.models.vmf_numba import (
    ACCUMULATE_DOC_ASSIGNMENT_STATISTICS_BACKEND,
    ACCUMULATE_DOC_AVG_LL_BACKEND,
    SAMPLE_DOC_TOPIC_ASSIGNMENTS_BACKEND,
    accumulate_doc_assignment_statistics,
    sample_doc_topic_assignments,
)
from src.utils.encoder import SentenceEncoder
from src.utils.evaluation import calculate_avg_ll_vmf_from_encoded
from src.utils.logging import get_logger


@dataclass(frozen=True)
class VMFIterationDiagnostics:
    iteration: int
    num_sweeps: int
    num_samples: int
    active_topics: int
    empty_topics: list[int]
    min_topic_count: int
    max_topic_count: int
    alpha_min: float
    alpha_max: float
    alpha_mean: float
    alpha_floor_count: int
    repair_enabled: bool
    repair_num_targets: int
    repair_num_repaired: int
    repair_num_failed: int
    repair_failed_topics: list[int]
    alpha_updated: bool
    alpha_converged: bool | None
    avg_log_likelihood: float
    e_step_sec: float
    repair_sec: float
    m_step_sec: float
    alpha_update_sec: float
    avg_log_likelihood_sec: float
    iteration_elapsed_sec: float


@dataclass(frozen=True)
class VMFEmbeddingCacheReport:
    strategy: str
    num_documents: int
    total_sentences: int
    embedding_size: int
    pre_normalize_transform: str
    reused_for_training_iterations: bool
    reused_for_avg_log_likelihood: bool


@dataclass(frozen=True)
class VMFIterationResult:
    avg_log_likelihood: float
    alpha_updated: bool
    alpha_converged: bool | None
    e_step_sec: float
    repair_sec: float
    m_step_sec: float
    alpha_update_sec: float
    avg_log_likelihood_sec: float
    iteration_elapsed_sec: float
    repair_enabled: bool
    repair_report: dict[str, object] | None
    alpha_min_value: float


@dataclass(frozen=True)
class VMFInvariantReport:
    total_sentences: int
    assigned_sentences: int
    topic_count_sum: int
    doc_topic_count_sum: int
    active_topics: int
    alpha_positive: bool
    alpha_finite: bool
    topic_counts_match_assignments: bool
    doc_topic_counts_match_assignments: bool
    topic_means_unit_norm: bool
    component_means_unit_norm: bool
    mixture_weights_normalized: bool

    @property
    def is_valid(self) -> bool:
        return all(
            [
                self.alpha_positive,
                self.alpha_finite,
                self.topic_counts_match_assignments,
                self.doc_topic_counts_match_assignments,
                self.topic_means_unit_norm,
                self.component_means_unit_norm,
                self.mixture_weights_normalized,
            ]
        )


class VMFLDATrainer:
    """Trainer for vMF-Sentence-LDA.

    - If num_components == 1: single vMF per topic.
    - If num_components  > 1: mixture of vMFs per topic.
    """

    EMBEDDING_STORAGE_DTYPE = np.float32

    @staticmethod
    def _should_run_periodic_step(
        *,
        iteration: int,
        total_iterations: int,
        every: int,
    ) -> bool:
        return ((iteration + 1) % every == 0) or (iteration == total_iterations - 1)

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        encoder: SentenceEncoder,
        num_topics: int,
        alpha: float | Sequence[float] | None,
        kappa: float,
        num_components: int = 1,
        pre_normalize_transform: str = "none",
        whitening_eps: float = 1e-5,
        algorithm_variant: str | None = None,
        log=None,
        save_path: Optional[os.PathLike | str] = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        """
        Args:
            corpus:
                Each document is a list of sentences (or segments).
            encoder:
                Sentence encoder (e.g. sentence-transformers).
                Must implement: encode(list[str]) -> np.ndarray[num_sentences, embedding_dim].
            num_topics:
                Number of topics K.
            alpha:
                Dirichlet hyperparameter.
                - None: initialize with standard symmetric 50/K.
                - float: symmetric Dirichlet with that value.
                - sequence length K: asymmetric Dirichlet.
            kappa:
                Default κ value used as a fallback when estimation is unstable.
            num_components:
                Number of vMF components per topic C.
                C=1: single vMF per topic.
                C>1: mixture of vMFs per topic.
            pre_normalize_transform:
                Embedding transform applied after encode and before L2 normalization.
                One of: {"none", "mean_center", "whitening"}.
            whitening_eps:
                Positive stabilizer used only when pre_normalize_transform="whitening".
            log:
                Logger instance (if None, get_logger("vMF-LDA") is used).
            save_path:
                Directory to save parameters. If None, parameters are not saved.
        """
        if log is None:
            log = get_logger("vMF-LDA")
        self.log = log
        self.progress = progress or TqdmProgressReporter()

        self.corpus: list[list[str]] = [list(doc) for doc in corpus]
        self.encoder = encoder
        self.embedding_size = encoder.get_sentence_embedding_dimension()

        self.num_topics = int(num_topics)
        self.num_documents = len(self.corpus)

        self.alpha = self._init_alpha(alpha)
        self.kappa_default = float(kappa)
        self.document_encoder = VMFDocumentEncoder(
            encoder=encoder,
            embedding_size=self.embedding_size,
            pre_normalize_transform=pre_normalize_transform,
            whitening_eps=whitening_eps,
            log=self.log,
            progress=self.progress,
        )
        self.pre_normalize_transform = self.document_encoder.pre_normalize_transform
        self.whitening_eps = self.document_encoder.whitening_eps
        self.embedding_preprocessor = self.document_encoder.embedding_preprocessor
        self.algorithm_variant = (
            None if algorithm_variant is None else str(algorithm_variant)
        )

        # Number of mixture components per topic (C)
        self.num_components = int(num_components)
        if self.num_components <= 0:
            raise ValueError("num_components must be >= 1")

        # Topic counts n_k (for compatibility / inspection)
        self.topic_counts = np.zeros(self.num_topics, dtype=np.int32)
        self.e_step_kernel_backend = SAMPLE_DOC_TOPIC_ASSIGNMENTS_BACKEND
        self.m_step_statistics_kernel_backend = (
            ACCUMULATE_DOC_ASSIGNMENT_STATISTICS_BACKEND
        )
        self.avg_ll_kernel_backend = ACCUMULATE_DOC_AVG_LL_BACKEND

        # Topic-by-document counts n_{k,d}
        self.topic_counts_per_doc = np.zeros(
            (self.num_topics, self.num_documents), dtype=np.int32
        )

        # Topic assignment per sentence: topic_assignments[d][i] = topic id of i-th sentence in document d
        self.topic_assignments: list[np.ndarray] = []

        # Sum of sentence vectors per topic s_k = Σ x_{d,i}
        # (for compatibility; in mixture case we also maintain per-component sums separately in M-step)
        self.sum_topic_vectors = np.zeros(
            (self.num_topics, self.embedding_size), dtype=self.EMBEDDING_STORAGE_DTYPE
        )

        # Effective topic mean directions μ_k (unit vectors, "average" over mixture components)
        self.topic_means = np.zeros(
            (self.num_topics, self.embedding_size), dtype=self.EMBEDDING_STORAGE_DTYPE
        )

        # Mixture parameters (C components per topic)
        # mixture_weights[k, c] = π_{k|c}
        self.mixture_weights = np.full(
            (self.num_topics, self.num_components),
            1.0 / self.num_components,
            dtype=np.float64,
        )
        # component_means[k, c, :] = μ_{k|c}
        self.component_means = np.zeros(
            (self.num_topics, self.num_components, self.embedding_size),
            dtype=self.EMBEDDING_STORAGE_DTYPE,
        )

        # Topic concentration parameters κ_k (shared across components of topic k)
        self.kappa_per_topic = np.full(
            self.num_topics, self.kappa_default, dtype=np.float64
        )
        # Cache for log C_M(κ_k) to avoid recomputation in the E-step
        self._log_c_per_topic = np.zeros(self.num_topics, dtype=np.float64)
        self._log_mixture_weights = np.zeros(
            (self.num_topics, self.num_components), dtype=np.float64
        )
        self._scaled_topic_means = np.zeros(
            (self.num_topics, self.embedding_size), dtype=np.float64
        )
        self._scaled_component_means = np.zeros(
            (self.num_topics, self.num_components, self.embedding_size),
            dtype=np.float64,
        )

        # History of average log-likelihood
        self.average_ll: list[float] = []
        self.iteration_diagnostics: list[VMFIterationDiagnostics] = []

        self.save_path = Path(save_path) if save_path is not None else None

        encode_start = time.perf_counter()
        self.document_encoder.fit_on_corpus(self.corpus)
        self.encoded_corpus = self.document_encoder.encode_corpus(
            self.corpus,
            desc="Encoding corpus",
        )
        self.training_corpus_encoding_sec = time.perf_counter() - encode_start

        self.log.info("Initializing assignments (vMF-LDA / MvTM)")
        # First initialize as single-vMF per topic (to get reasonable μ_k, κ_k, z)
        self._initialize_single_vmf()
        # Then, if C>1, initialize mixture bases around μ_k
        if self.num_components > 1:
            self._init_mixture_from_topic_means()
        self._refresh_density_caches()
        self.inferencer = VMFTopicInferencer(
            num_topics=self.num_topics,
            embedding_size=self.embedding_size,
            encode_document=self.document_encoder.encode_and_normalize,
            log_vmf_density=self.log_vmf_density_tables,
            log_vmf_density_batch=self.log_vmf_density_matrix,
        )
        self.log.info("Initialization done (vMF-LDA / MvTM)")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _init_alpha(self, alpha: float | Sequence[float] | None) -> np.ndarray:
        """
        Initialize asymmetric Dirichlet alpha vector.
        Standard default is alpha_k = 50 / K when alpha is None.
        """
        if alpha is None:
            val = 50.0 / float(self.num_topics)
            return np.full(self.num_topics, val, dtype=np.float64)
        if isinstance(alpha, (int, float, np.floating, np.integer)):
            return np.full(self.num_topics, float(alpha), dtype=np.float64)

        arr = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if arr.size != self.num_topics:
            raise ValueError(f"alpha must be length {self.num_topics}, got {arr.size}")
        if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
            raise ValueError("alpha must be positive and finite")
        return arr.copy()

    def _log_alpha_stats(self, prefix: str = "Alpha") -> None:
        alpha = np.asarray(self.alpha, dtype=np.float64)
        self.log.info(
            "%s stats: min=%.3e max=%.3e mean=%.3e sum=%.3e",
            prefix,
            float(alpha.min()),
            float(alpha.max()),
            float(alpha.mean()),
            float(alpha.sum()),
        )

    def _update_alpha_fixed_point(
        self, max_iter: int = 100, tol: float = 1e-5, min_alpha: float = 1e-3
    ) -> bool:
        """
        Update asymmetric alpha via Minka's fixed-point iteration.
        Returns True if converged within tol, False otherwise.
        """
        counts = np.asarray(self.topic_counts_per_doc, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[0] != self.num_topics:
            return False
        doc_lengths = counts.sum(axis=0)
        if doc_lengths.size == 0:
            return False

        eps = float(min_alpha)
        alpha = np.asarray(self.alpha, dtype=np.float64).copy()
        alpha = np.clip(alpha, eps, None)

        for _ in range(max_iter):
            alpha0 = float(alpha.sum())
            if not np.isfinite(alpha0) or alpha0 <= 0.0:
                break

            denom = np.sum(_digamma(doc_lengths + alpha0) - _digamma(alpha0))
            if not np.isfinite(denom) or denom <= 0.0:
                break

            numer = np.sum(
                _digamma(counts + alpha[:, None]) - _digamma(alpha)[:, None], axis=1
            )
            if not np.all(np.isfinite(numer)):
                break

            alpha_new = alpha * (numer / denom)
            alpha_new = np.clip(alpha_new, eps, None)

            rel_change = np.max(np.abs(alpha_new - alpha) / (alpha + eps))
            alpha = alpha_new
            if rel_change < tol:
                self.alpha = alpha
                return True

        self.alpha = alpha
        return False

    def _log_vmf_normalization_const(self, kappa: np.ndarray) -> np.ndarray:
        """
        Compute log C_M(kappa) for the vMF distribution.

        log C_M(kappa) = (M/2 - 1) * log kappa - (M/2) * log(2π) - log I_{M/2-1}(kappa)
        """
        kappa = np.asarray(kappa, dtype=np.float64)
        kappa_safe = np.clip(kappa, 1e-12, None)
        d_dim = float(self.embedding_size)
        v = d_dim / 2.0 - 1.0

        ive_val = np.maximum(_ive(v, kappa_safe), 1e-300)
        log_iv = np.log(ive_val) + kappa_safe
        return v * np.log(kappa_safe) - (d_dim / 2.0) * math.log(2.0 * math.pi) - log_iv

    def _refresh_log_c_cache(self) -> None:
        """Refresh cached log C_M(κ_k) values after κ updates."""
        self._log_c_per_topic = self._log_vmf_normalization_const(self.kappa_per_topic)

    def _refresh_density_caches(self) -> None:
        """Refresh topic-side caches used by E-step and inference."""
        self._refresh_log_c_cache()
        self._scaled_topic_means = self.kappa_per_topic[:, None] * self.topic_means
        self._scaled_component_means = (
            self.kappa_per_topic[:, None, None] * self.component_means
        )
        self._log_mixture_weights = np.log(self.mixture_weights + 1e-12)

    def _random_topic_vector(self, n_sentences: int) -> np.ndarray:
        return np.random.randint(
            self.num_topics,
            size=int(n_sentences),
            dtype=np.int32,
        )

    def _update_topic_mean_and_kappa_single(self, k: int) -> None:
        """
        Update mean direction μ_k and concentration κ_k for topic k
        in the single-vMF (C=1) case.

        Uses:
            - s_k (sum of vectors) to compute μ_k
            - Banerjee-style approximation to estimate κ_k
        """
        s_k = self.sum_topic_vectors[k]
        n_k = int(self.topic_counts[k])

        norm_s = np.linalg.norm(s_k)
        if norm_s == 0.0 or n_k == 0:
            v = np.random.randn(self.embedding_size)
            v /= np.linalg.norm(v) + 1e-12
            self.topic_means[k] = np.asarray(v, dtype=self.EMBEDDING_STORAGE_DTYPE)
            self.kappa_per_topic[k] = self.kappa_default
            return

        # Mean direction μ_k
        self.topic_means[k] = np.asarray(
            s_k / (norm_s + 1e-12),
            dtype=self.EMBEDDING_STORAGE_DTYPE,
        )

        # If only one point is assigned, kappa estimation is unstable; use default
        if n_k <= 1:
            self.kappa_per_topic[k] = self.kappa_default
            return

        # Mean resultant length l_k = ||s_k|| / n_k
        l_k = norm_s / (n_k + 1e-12)
        l_k = float(np.clip(l_k, 1e-6, 1.0 - 1e-6))

        d = self.embedding_size
        # Approximate κ from l_k (single vMF; standard approximation)
        numerator = l_k * (d - l_k**2)
        denominator = 1.0 - l_k**2
        kappa_est = numerator / (denominator + 1e-12)

        if not np.isfinite(kappa_est) or kappa_est <= 0:
            self.kappa_per_topic[k] = self.kappa_default
        else:
            self.kappa_per_topic[k] = kappa_est

    def _initialize_single_vmf(self) -> None:
        """
        Initialize vMF-LDA with a single vMF per topic:
            - Randomly assign each sentence in each document to a topic.
            - Compute s_k, μ_k, κ_k from the assignments.
        This serves as a good starting point even when num_components > 1.
        """
        self.topic_assignments = []
        self.topic_counts[:] = 0
        self.topic_counts_per_doc[:] = 0
        self.sum_topic_vectors[:] = 0.0
        self.topic_means[:] = 0.0

        pbar = self.progress.wrap(
            enumerate(self.encoded_corpus),
            desc="Initializing (single vMF)",
            total=self.num_documents,
        )

        for d, enc in pbar:
            n_sent = enc.shape[0]

            topics = self._random_topic_vector(n_sent)
            self.topic_assignments.append(topics)
            if topics.size > 0:
                self.topic_counts_per_doc[:, d] = np.bincount(
                    topics,
                    minlength=self.num_topics,
                ).astype(np.int32, copy=False)

            for x, k in zip(enc, topics):
                if x.shape[0] != self.embedding_size:
                    continue
                self.topic_counts[k] += 1
                self.sum_topic_vectors[k] += x

        for k in range(self.num_topics):
            self._update_topic_mean_and_kappa_single(k)
            if self.num_components == 1:
                self.component_means[k, 0] = self.topic_means[k]
                self.mixture_weights[k, 0] = 1.0
        self._refresh_density_caches()

    def _init_mixture_from_topic_means(self) -> None:
        """
        Initialize mixture components (C > 1) around the single-vMF topic means.

        - π_{k|c} is set to 1/C.
        - μ_{k|c} is a small random perturbation around μ_k.
        """
        eps = 1e-6
        for k in range(self.num_topics):
            base = self.topic_means[k]
            if np.linalg.norm(base) == 0.0:
                base = np.random.randn(self.embedding_size)
                base /= np.linalg.norm(base) + eps

            for c in range(self.num_components):
                v = base + 0.01 * np.random.randn(self.embedding_size)
                v /= np.linalg.norm(v) + eps
                self.component_means[k, c] = np.asarray(
                    v,
                    dtype=self.EMBEDDING_STORAGE_DTYPE,
                )

            self.mixture_weights[k] = 1.0 / self.num_components

    def _random_init_topic(self, k: int) -> None:
        """
        Randomly initialize mixture parameters for topic k.
        Used when no data is assigned to topic k in an EM iteration.
        """
        eps = 1e-6
        self.mixture_weights[k] = 1.0 / self.num_components
        for c in range(self.num_components):
            v = np.random.randn(self.embedding_size)
            v /= np.linalg.norm(v) + eps
            self.component_means[k, c] = np.asarray(
                v,
                dtype=self.EMBEDDING_STORAGE_DTYPE,
            )
        self.kappa_per_topic[k] = self.kappa_default

        eff = self.component_means[k].mean(axis=0)
        eff_norm = np.linalg.norm(eff)
        if eff_norm > 0:
            eff /= eff_norm
        self.topic_means[k] = np.asarray(eff, dtype=self.EMBEDDING_STORAGE_DTYPE)
        self.topic_counts[k] = 0
        self.sum_topic_vectors[k] = 0.0

    def _init_topic_from_vector(self, k: int, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if not np.isfinite(norm) or norm <= 0.0:
            self._random_init_topic(k)
            return

        direction = arr / (norm + 1e-12)
        self.mixture_weights[k] = 1.0 / self.num_components
        for c in range(self.num_components):
            if c == 0:
                v = direction
            else:
                v = direction + 0.01 * np.random.randn(self.embedding_size)
                v /= np.linalg.norm(v) + 1e-12
            self.component_means[k, c] = np.asarray(
                v,
                dtype=self.EMBEDDING_STORAGE_DTYPE,
            )

        eff = self.component_means[k].mean(axis=0)
        eff_norm = np.linalg.norm(eff)
        if eff_norm > 0.0:
            eff /= eff_norm
        self.topic_means[k] = np.asarray(eff, dtype=self.EMBEDDING_STORAGE_DTYPE)
        self.kappa_per_topic[k] = self.kappa_default

    def _recompute_doc_topic_counts(self) -> None:
        """
        Recompute document–topic counts n_{k,d} from current topic_assignments.

        This ensures that in the E-step we really use N_{dk}^{(-n)} from the
        previous state of z.
        """
        self.topic_counts_per_doc[:] = 0
        for d, doc_topics in enumerate(self.topic_assignments):
            topics = np.asarray(doc_topics, dtype=np.int64)
            if topics.size == 0:
                continue
            valid = topics[(topics >= 0) & (topics < self.num_topics)]
            if valid.size == 0:
                continue
            self.topic_counts_per_doc[:, d] = np.bincount(
                valid,
                minlength=self.num_topics,
            ).astype(np.int32, copy=False)

    def _recompute_topic_counts_from_doc_counts(self) -> None:
        self.topic_counts[:] = self.topic_counts_per_doc.sum(axis=1).astype(
            np.int32,
            copy=False,
        )

    def _get_empty_topics(self, *, min_count: int = 1) -> np.ndarray:
        counts = np.asarray(self.topic_counts_per_doc.sum(axis=1), dtype=np.int64)
        return np.where(counts < int(min_count))[0]

    def _find_repair_donor_sentence(
        self,
        *,
        protected_topics: set[int],
        min_topic_count_for_repair: int,
    ) -> tuple[int, int, int] | None:
        counts = self.topic_counts_per_doc.sum(axis=1).astype(np.int64)
        candidate_topics = np.argsort(-counts)

        for donor_topic_value in candidate_topics:
            donor_topic = int(donor_topic_value)
            if donor_topic in protected_topics:
                continue
            if counts[donor_topic] <= min_topic_count_for_repair:
                continue

            locations: list[tuple[int, int]] = []
            for d, assignments in enumerate(self.topic_assignments):
                arr = np.asarray(assignments)
                if arr.size == 0:
                    continue
                sentence_indices = np.where(arr == donor_topic)[0]
                locations.extend((d, int(i)) for i in sentence_indices)

            if locations:
                selected = locations[np.random.randint(len(locations))]
                return selected[0], selected[1], donor_topic

        return None

    def _repair_empty_topics(
        self,
        *,
        min_topic_count_for_repair: int = 1,
    ) -> dict[str, object]:
        self._recompute_doc_topic_counts()
        self._recompute_topic_counts_from_doc_counts()

        target_topics = [
            int(topic)
            for topic in self._get_empty_topics(min_count=min_topic_count_for_repair)
        ]
        repaired: list[int] = []
        failed: list[int] = []
        moves: list[dict[str, int]] = []
        protected_topics: set[int] = set()

        for target_topic in target_topics:
            donor = self._find_repair_donor_sentence(
                protected_topics=protected_topics,
                min_topic_count_for_repair=min_topic_count_for_repair,
            )
            if donor is None:
                failed.append(target_topic)
                continue

            doc_index, sentence_index, old_topic = donor
            self.topic_assignments[doc_index][sentence_index] = target_topic
            self.topic_counts_per_doc[old_topic, doc_index] -= 1
            self.topic_counts_per_doc[target_topic, doc_index] += 1
            self._init_topic_from_vector(
                target_topic,
                self.encoded_corpus[doc_index][sentence_index],
            )

            repaired.append(target_topic)
            protected_topics.add(target_topic)
            moves.append(
                {
                    "target_topic": int(target_topic),
                    "old_topic": int(old_topic),
                    "doc_index": int(doc_index),
                    "sentence_index": int(sentence_index),
                }
            )

        self._recompute_topic_counts_from_doc_counts()
        if repaired:
            self._refresh_density_caches()

        return {
            "num_targets": len(target_topics),
            "num_repaired": len(repaired),
            "num_failed": len(failed),
            "target_topics": target_topics,
            "repaired_topics": repaired,
            "failed_topics": failed,
            "moves": moves,
        }

    # -------------------------------------------------------------------------
    # vMF log density (for Gibbs sampling over topics)
    # -------------------------------------------------------------------------
    def log_vmf_density_tables(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log of unnormalized vMF mixture density for all topics for a single unit vector x.

        - If num_components == 1:
            log p_k ∝ log C_M(κ_k) + κ_k μ_k^T x
        - If num_components  > 1:
            log p_k ∝ log C_M(κ_k) + log ∑_c π_{k|c} exp(κ_k μ_{k|c}^T x)

        The normalization constant C_M(κ_k) penalizes large κ_k and is
        required for fair topic comparisons.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"x must be 1D, got shape {arr.shape}")

        if self.num_components == 1:
            return self._log_c_per_topic + self._scaled_topic_means @ arr

        # Mixture of vMFs per topic
        # component_means: (K, C, D)
        # x: (D,)
        # dots[k, c] = μ_{k|c}^T x
        scores = np.tensordot(self._scaled_component_means, arr, axes=([2], [0]))
        log_comp = self._log_mixture_weights + scores

        # log-sum-exp over mixture components
        m = log_comp.max(axis=1, keepdims=True)  # (K, 1)
        mix_log = m + np.log(
            np.exp(log_comp - m).sum(axis=1, keepdims=True) + 1e-12
        )  # (K, 1)

        return self._log_c_per_topic + mix_log.ravel()  # (K,)

    def log_vmf_density_matrix(self, x: np.ndarray) -> np.ndarray:
        """Compute topic log-likelihoods for a batch of unit vectors."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, self.num_topics), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.embedding_size:
            raise ValueError(
                f"Expected embedding dim {self.embedding_size}, got {arr.shape[1]}"
            )

        if self.num_components == 1:
            return arr @ self._scaled_topic_means.T + self._log_c_per_topic[None, :]

        scores = np.einsum(
            "nd,kcd->nkc",
            arr,
            self._scaled_component_means,
            optimize=True,
        )
        log_comp = scores + self._log_mixture_weights[None, :, :]
        m = log_comp.max(axis=2, keepdims=True)
        mix_log = m + np.log(np.exp(log_comp - m).sum(axis=2, keepdims=True) + 1e-12)
        return self._log_c_per_topic[None, :] + mix_log[..., 0]

    def _component_responsibilities(self, k: int, x: np.ndarray) -> np.ndarray:
        """
        Compute responsibilities weight_{ic} for a given topic k and vector x.

            weight_ic ∝ π_{k|c} * vMF(x | μ_{k|c}, κ_k)
                      ∝ π_{k|c} * exp(κ_k μ_{k|c}^T x)

        Because κ_k is shared across components of topic k, c_M(κ_k) cancels
        between numerator and denominator.
        """
        if self.num_components == 1:
            return np.array([1.0], dtype=np.float64)

        log_comp = self._log_mixture_weights[k] + self._scaled_component_means[k] @ x
        m = log_comp.max()
        comp = np.exp(log_comp - m)
        s = comp.sum()
        if not np.isfinite(s) or s <= 0.0:
            return np.full(
                self.num_components, 1.0 / self.num_components, dtype=np.float64
            )
        return comp / s

    def _ensure_document_assignments(self, doc_index: int, n_sentences: int) -> None:
        if doc_index >= len(self.topic_assignments):
            topics = self._random_topic_vector(n_sentences)
            self.topic_assignments.append(topics)
            self.topic_counts_per_doc[:, doc_index] = 0
            if topics.size > 0:
                self.topic_counts_per_doc[:, doc_index] = np.bincount(
                    topics,
                    minlength=self.num_topics,
                ).astype(np.int32, copy=False)
            return

        if len(self.topic_assignments[doc_index]) != n_sentences:
            topics = self._random_topic_vector(n_sentences)
            self.topic_assignments[doc_index] = topics
            self.topic_counts_per_doc[:, doc_index] = 0
            if topics.size > 0:
                self.topic_counts_per_doc[:, doc_index] = np.bincount(
                    topics,
                    minlength=self.num_topics,
                ).astype(np.int32, copy=False)

    def _sample_topic_assignment(
        self,
        doc_index: int,
        sentence_index: int,
        log_lik: np.ndarray,
    ) -> None:
        old_k = self.topic_assignments[doc_index][sentence_index]
        if 0 <= old_k < self.num_topics:
            self.topic_counts_per_doc[old_k, doc_index] -= 1

        counts = self.topic_counts_per_doc[:, doc_index].astype(float) + self.alpha
        log_post = np.log(counts) + log_lik
        log_post -= log_post.max()
        post = np.exp(log_post)
        post_sum = post.sum()
        if not np.isfinite(post_sum) or post_sum <= 0.0:
            post[:] = 1.0 / self.num_topics
        else:
            post /= post_sum

        new_k = int(np.random.choice(self.num_topics, p=post))
        self.topic_assignments[doc_index][sentence_index] = new_k
        self.topic_counts_per_doc[new_k, doc_index] += 1

    def _accumulate_assignment_statistics(
        self,
        *,
        encoded_docs: Sequence[np.ndarray],
        sample_assignments: Sequence[Sequence[int]],
        nk: np.ndarray,
        nk_comp: np.ndarray,
        r: np.ndarray,
    ) -> None:
        for d in range(self.num_documents):
            enc = encoded_docs[d]
            if enc.size == 0 or d >= len(sample_assignments):
                continue
            assignments = np.asarray(sample_assignments[d], dtype=np.int32)
            if assignments.size == 0:
                continue
            limit = min(enc.shape[0], assignments.shape[0])
            if limit <= 0:
                continue
            accumulate_doc_assignment_statistics(
                encoded_doc=enc[:limit],
                assignments=assignments[:limit],
                log_mixture_weights=self._log_mixture_weights,
                scaled_component_means=self._scaled_component_means,
                nk=nk,
                nk_comp=nk_comp,
                r=r,
            )

    def _compute_assignment_statistics(
        self,
        *,
        encoded_docs: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_topics = self.num_topics
        c_components = self.num_components
        d_dim = self.embedding_size
        nk = np.zeros(k_topics, dtype=np.float64)
        nk_comp = np.zeros((k_topics, c_components), dtype=np.float64)
        r = np.zeros((k_topics, c_components, d_dim), dtype=np.float64)
        self._accumulate_assignment_statistics(
            encoded_docs=encoded_docs,
            sample_assignments=self.topic_assignments,
            nk=nk,
            nk_comp=nk_comp,
            r=r,
        )
        return nk, nk_comp, r

    def _run_e_step(
        self,
        *,
        encoded_docs: Sequence[np.ndarray],
        num_sweeps: int,
        num_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_topics = self.num_topics
        c_components = self.num_components
        d_dim = self.embedding_size
        nk = np.zeros(k_topics, dtype=np.float64)
        r = np.zeros((k_topics, c_components, d_dim), dtype=np.float64)
        nk_comp = np.zeros((k_topics, c_components), dtype=np.float64)
        kept_sample_count = 0
        for sweep in range(num_sweeps):
            desc = (
                "E-step (vMF-Sentence-LDA)"
                if num_sweeps == 1
                else f"E-step sweep {sweep + 1}/{num_sweeps}"
            )
            pbar = self.progress.wrap(range(self.num_documents), desc=desc)
            for d in pbar:
                enc = encoded_docs[d]
                if enc.size == 0:
                    continue
                self._ensure_document_assignments(d, enc.shape[0])
                log_lik_doc = self.log_vmf_density_matrix(enc)
                counts_doc = np.array(
                    self.topic_counts_per_doc[:, d],
                    dtype=np.int32,
                    copy=True,
                )
                uniforms = np.random.random(enc.shape[0]).astype(np.float64, copy=False)
                sample_doc_topic_assignments(
                    assignments=self.topic_assignments[d],
                    counts=counts_doc,
                    log_lik_doc=log_lik_doc,
                    alpha=self.alpha,
                    uniforms=uniforms,
                )
                self.topic_counts_per_doc[:, d] = counts_doc

            if sweep >= num_sweeps - num_samples:
                self._accumulate_assignment_statistics(
                    encoded_docs=encoded_docs,
                    sample_assignments=self.topic_assignments,
                    nk=nk,
                    nk_comp=nk_comp,
                    r=r,
                )
                kept_sample_count += 1

        sample_count = float(max(kept_sample_count, 1))
        if sample_count > 1.0:
            nk /= sample_count
            nk_comp /= sample_count
            r /= sample_count
        return nk, nk_comp, r

    def _apply_m_step_updates(
        self,
        *,
        nk: np.ndarray,
        nk_comp: np.ndarray,
        r: np.ndarray,
    ) -> None:
        self.log.info("M-step: updating mixture parameters (μ_k|c, π_k|c, κ_k)")
        c_components = self.num_components
        d_dim = self.embedding_size
        eps = 1e-12

        for k in range(self.num_topics):
            n_k = nk[k]
            if n_k <= 0:
                self._random_init_topic(k)
                continue

            pi_k = nk_comp[k] / (n_k + eps)
            if not np.isfinite(pi_k).all() or pi_k.sum() <= 0.0:
                pi_k = np.full(c_components, 1.0 / c_components, dtype=np.float64)
            else:
                pi_k /= pi_k.sum()
            self.mixture_weights[k] = pi_k

            lengths: list[float] = []
            for c in range(c_components):
                rkc = r[k, c]
                norm_rkc = np.linalg.norm(rkc)
                if norm_rkc == 0.0:
                    v = np.random.randn(d_dim)
                    v /= np.linalg.norm(v) + eps
                    self.component_means[k, c] = np.asarray(
                        v,
                        dtype=self.EMBEDDING_STORAGE_DTYPE,
                    )
                else:
                    self.component_means[k, c] = np.asarray(
                        rkc / (norm_rkc + eps),
                        dtype=self.EMBEDDING_STORAGE_DTYPE,
                    )
                    lengths.append(float(norm_rkc))

            r_k = 0.0
            if lengths:
                r_k = float(sum(lengths) / (n_k + eps))
                r_k = float(np.clip(r_k, 1e-6, 1.0 - 1e-6))

            if n_k <= 1 or r_k <= 0.0:
                self.kappa_per_topic[k] = self.kappa_default
            else:
                numerator = r_k * d_dim - r_k**3
                denominator = 1.0 - r_k**2
                kappa_est = numerator / (denominator + eps)
                if not np.isfinite(kappa_est) or kappa_est <= 0:
                    self.kappa_per_topic[k] = self.kappa_default
                else:
                    self.kappa_per_topic[k] = kappa_est

            eff = (self.mixture_weights[k][:, None] * self.component_means[k]).sum(
                axis=0
            )
            norm_eff = np.linalg.norm(eff)
            if norm_eff > 0.0:
                eff /= norm_eff
            self.topic_means[k] = np.asarray(eff, dtype=self.EMBEDDING_STORAGE_DTYPE)
            self.sum_topic_vectors[k] = np.asarray(
                r[k].sum(axis=0),
                dtype=self.EMBEDDING_STORAGE_DTYPE,
            )

        self.topic_counts[:] = self.topic_counts_per_doc.sum(axis=1).astype(
            np.int32,
            copy=False,
        )
        self._refresh_density_caches()

    def _update_alpha_if_needed(
        self,
        *,
        iteration: int,
        estimate_alpha: bool,
        alpha_update_every: int,
        alpha_max_iter: int,
        alpha_tol: float,
        alpha_min_value: float,
    ) -> tuple[bool, bool | None]:
        if not estimate_alpha or ((iteration + 1) % alpha_update_every != 0):
            return False, None

        self.log.info("Updating alpha (fixed-point)")
        converged = self._update_alpha_fixed_point(
            max_iter=alpha_max_iter,
            tol=alpha_tol,
            min_alpha=alpha_min_value,
        )
        self._log_alpha_stats(prefix="Alpha (updated)")
        alpha = np.asarray(self.alpha, dtype=np.float64)
        num_at_floor = int(np.sum(alpha <= alpha_min_value + 1e-12))
        self.log.info(
            "Alpha floor stats: alpha_min_value=%.3e num_at_floor=%d/%d",
            alpha_min_value,
            num_at_floor,
            self.num_topics,
        )
        if not converged:
            self.log.warning(
                "Alpha update did not converge within %s iterations.",
                alpha_max_iter,
            )
        return True, converged

    def _compute_average_log_likelihood(
        self, *, encoded_docs: Sequence[np.ndarray]
    ) -> float:
        self.log.info("Computing average log-likelihood (vMF-Sentence-LDA)")
        if self.num_components > 1:
            ave_ll = self._compute_average_mixture_log_likelihood(
                encoded_docs=encoded_docs
            )
        else:
            ave_ll = calculate_avg_ll_vmf_from_encoded(
                encoded_corpus=encoded_docs,
                topic_assignments=self.topic_assignments,
                topic_means=self.topic_means,
                kappa_per_topic=self.kappa_per_topic,
            )
        self.average_ll.append(ave_ll)
        self.log.info("Average LL (vMF-Sentence-LDA): {:.3e}".format(ave_ll))
        return ave_ll

    def _compute_average_mixture_log_likelihood(
        self, *, encoded_docs: Sequence[np.ndarray]
    ) -> float:
        total_ll = 0.0
        total_count = 0
        for enc, doc_topics in zip(encoded_docs, self.topic_assignments, strict=False):
            arr = np.asarray(enc, dtype=np.float64)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            topics = np.asarray(doc_topics, dtype=np.int64)
            if topics.size == 0:
                continue
            limit = min(arr.shape[0], topics.shape[0])
            if limit <= 0:
                continue
            scores = self.log_vmf_density_matrix(arr[:limit])
            for row_index, topic_index in enumerate(topics[:limit]):
                if 0 <= topic_index < self.num_topics:
                    total_ll += float(scores[row_index, topic_index])
                    total_count += 1
        if total_count == 0:
            return float("nan")
        return total_ll / total_count

    def _record_iteration_diagnostics(
        self,
        *,
        iteration: int,
        num_sweeps: int,
        num_samples: int,
        result: VMFIterationResult,
    ) -> None:
        topic_counts = np.asarray(self.topic_counts, dtype=np.int64)
        empty_topics = np.where(topic_counts == 0)[0].astype(int).tolist()
        alpha = np.asarray(self.alpha, dtype=np.float64)
        repair_report = result.repair_report or {}
        diagnostics = VMFIterationDiagnostics(
            iteration=int(iteration),
            num_sweeps=int(num_sweeps),
            num_samples=int(num_samples),
            active_topics=int(np.count_nonzero(topic_counts)),
            empty_topics=empty_topics,
            min_topic_count=int(topic_counts.min()) if topic_counts.size else 0,
            max_topic_count=int(topic_counts.max()) if topic_counts.size else 0,
            alpha_min=float(alpha.min()) if alpha.size else float("nan"),
            alpha_max=float(alpha.max()) if alpha.size else float("nan"),
            alpha_mean=float(alpha.mean()) if alpha.size else float("nan"),
            alpha_floor_count=int(np.sum(alpha <= result.alpha_min_value + 1e-12)),
            repair_enabled=bool(result.repair_enabled),
            repair_num_targets=int(repair_report.get("num_targets", 0)),
            repair_num_repaired=int(repair_report.get("num_repaired", 0)),
            repair_num_failed=int(repair_report.get("num_failed", 0)),
            repair_failed_topics=[
                int(topic) for topic in repair_report.get("failed_topics", [])
            ],
            alpha_updated=bool(result.alpha_updated),
            alpha_converged=result.alpha_converged,
            avg_log_likelihood=float(result.avg_log_likelihood),
            e_step_sec=float(result.e_step_sec),
            repair_sec=float(result.repair_sec),
            m_step_sec=float(result.m_step_sec),
            alpha_update_sec=float(result.alpha_update_sec),
            avg_log_likelihood_sec=float(result.avg_log_likelihood_sec),
            iteration_elapsed_sec=float(result.iteration_elapsed_sec),
        )
        self.iteration_diagnostics.append(diagnostics)

    def build_embedding_cache_report(self) -> VMFEmbeddingCacheReport:
        return VMFEmbeddingCacheReport(
            strategy="preencoded_training_corpus",
            num_documents=int(len(self.encoded_corpus)),
            total_sentences=int(sum(enc.shape[0] for enc in self.encoded_corpus)),
            embedding_size=int(self.embedding_size),
            pre_normalize_transform=str(self.pre_normalize_transform),
            reused_for_training_iterations=True,
            reused_for_avg_log_likelihood=True,
        )

    def _run_training_iteration(
        self,
        *,
        iteration: int,
        num_sweeps: int,
        num_samples: int,
        estimate_alpha: bool,
        alpha_update_every: int,
        alpha_max_iter: int,
        alpha_tol: float,
        alpha_min_value: float,
        repair_empty_topics: bool,
        min_topic_count_for_repair: int,
        compute_avg_log_likelihood: bool,
    ) -> VMFIterationResult:
        encoded_docs = self.encoded_corpus
        iteration_start = time.perf_counter()

        e_step_start = time.perf_counter()
        nk, nk_comp, r = self._run_e_step(
            encoded_docs=encoded_docs,
            num_sweeps=num_sweeps,
            num_samples=num_samples,
        )
        e_step_sec = time.perf_counter() - e_step_start

        repair_start = time.perf_counter()
        repair_report = None
        if repair_empty_topics:
            repair_report = self._repair_empty_topics(
                min_topic_count_for_repair=min_topic_count_for_repair,
            )
            if repair_report["num_targets"] > 0:
                self.log.info(
                    "Empty-topic repair: targets=%s repaired=%s failed=%s failed_topics=%s",
                    repair_report["num_targets"],
                    repair_report["num_repaired"],
                    repair_report["num_failed"],
                    repair_report["failed_topics"],
                )
            if repair_report["num_repaired"] > 0:
                nk, nk_comp, r = self._compute_assignment_statistics(
                    encoded_docs=encoded_docs,
                )
        repair_sec = time.perf_counter() - repair_start

        m_step_start = time.perf_counter()
        self._apply_m_step_updates(nk=nk, nk_comp=nk_comp, r=r)
        m_step_sec = time.perf_counter() - m_step_start

        alpha_update_start = time.perf_counter()
        alpha_updated, alpha_converged = self._update_alpha_if_needed(
            iteration=iteration,
            estimate_alpha=estimate_alpha,
            alpha_update_every=alpha_update_every,
            alpha_max_iter=alpha_max_iter,
            alpha_tol=alpha_tol,
            alpha_min_value=alpha_min_value,
        )
        alpha_update_sec = time.perf_counter() - alpha_update_start

        if compute_avg_log_likelihood:
            avg_ll_start = time.perf_counter()
            avg_log_likelihood = self._compute_average_log_likelihood(
                encoded_docs=encoded_docs
            )
            avg_log_likelihood_sec = time.perf_counter() - avg_ll_start
        else:
            avg_log_likelihood = (
                float(self.average_ll[-1]) if self.average_ll else float("nan")
            )
            avg_log_likelihood_sec = 0.0
        iteration_elapsed_sec = time.perf_counter() - iteration_start
        return VMFIterationResult(
            avg_log_likelihood=float(avg_log_likelihood),
            alpha_updated=bool(alpha_updated),
            alpha_converged=alpha_converged,
            e_step_sec=float(e_step_sec),
            repair_sec=float(repair_sec),
            m_step_sec=float(m_step_sec),
            alpha_update_sec=float(alpha_update_sec),
            avg_log_likelihood_sec=float(avg_log_likelihood_sec),
            iteration_elapsed_sec=float(iteration_elapsed_sec),
            repair_enabled=bool(repair_empty_topics),
            repair_report=repair_report,
            alpha_min_value=float(alpha_min_value),
        )

    def build_invariant_report(self) -> VMFInvariantReport:
        total_sentences = int(sum(enc.shape[0] for enc in self.encoded_corpus))
        assigned_sentences = int(sum(len(doc) for doc in self.topic_assignments))
        topic_count_sum = int(np.asarray(self.topic_counts, dtype=np.int64).sum())
        doc_topic_count_sum = int(
            np.asarray(self.topic_counts_per_doc, dtype=np.int64).sum()
        )

        assignment_topic_counts = np.zeros(self.num_topics, dtype=np.int64)
        assignment_doc_counts = np.zeros_like(self.topic_counts_per_doc, dtype=np.int64)
        for d, (enc, doc_topics) in enumerate(
            zip(self.encoded_corpus, self.topic_assignments, strict=False)
        ):
            expected_length = int(enc.shape[0])
            topics = np.asarray(doc_topics, dtype=np.int64)
            if topics.shape[0] != expected_length:
                continue
            valid = topics[(topics >= 0) & (topics < self.num_topics)]
            if valid.size == 0:
                continue
            bincounts = np.bincount(valid, minlength=self.num_topics)
            assignment_topic_counts += bincounts
            assignment_doc_counts[:, d] = bincounts

        topic_mean_norms = np.linalg.norm(self.topic_means, axis=1)
        component_mean_norms = np.linalg.norm(self.component_means, axis=2)
        mixture_weight_sums = self.mixture_weights.sum(axis=1)

        return VMFInvariantReport(
            total_sentences=total_sentences,
            assigned_sentences=assigned_sentences,
            topic_count_sum=topic_count_sum,
            doc_topic_count_sum=doc_topic_count_sum,
            active_topics=int(np.count_nonzero(self.topic_counts)),
            alpha_positive=bool(np.all(np.asarray(self.alpha) > 0.0)),
            alpha_finite=bool(np.all(np.isfinite(np.asarray(self.alpha)))),
            topic_counts_match_assignments=bool(
                np.array_equal(
                    np.asarray(self.topic_counts, dtype=np.int64),
                    assignment_topic_counts,
                )
                and topic_count_sum == assigned_sentences == total_sentences
            ),
            doc_topic_counts_match_assignments=bool(
                np.array_equal(
                    np.asarray(self.topic_counts_per_doc, dtype=np.int64),
                    assignment_doc_counts,
                )
                and doc_topic_count_sum == assigned_sentences
            ),
            topic_means_unit_norm=bool(
                np.allclose(topic_mean_norms, np.ones_like(topic_mean_norms), atol=1e-6)
            ),
            component_means_unit_norm=bool(
                np.allclose(
                    component_mean_norms,
                    np.ones_like(component_mean_norms),
                    atol=1e-6,
                )
            ),
            mixture_weights_normalized=bool(
                np.all(self.mixture_weights >= 0.0)
                and np.allclose(
                    mixture_weight_sums,
                    np.ones_like(mixture_weight_sums),
                    atol=1e-6,
                )
            ),
        )

    def assert_valid_state(self) -> VMFInvariantReport:
        report = self.build_invariant_report()
        if not report.is_valid:
            raise ValueError(f"Invalid vMF-LDA state: {report}")
        return report

    # -------------------------------------------------------------------------
    # EM-style learning (E-step: Gibbs for z, M-step: update mixture parameters)
    # -------------------------------------------------------------------------
    def sample(
        self,
        num_iterations: int,
        num_sweeps: int = 1,
        num_samples: int = 1,
        *,
        estimate_alpha: bool = True,
        alpha_update_every: int = 1,
        alpha_max_iter: int = 100,
        alpha_tol: float = 1e-5,
        alpha_min_value: float = 1e-3,
        repair_empty_topics: bool = True,
        min_topic_count_for_repair: int = 1,
        avg_log_likelihood_every: int = 1,
        invariant_check_every: int = 1,
    ) -> None:
        """
        Run EM-style learning for vMF-Sentence-LDA.

        Each EM iteration t:

        E-step:
            - Keep topic parameters (μ_{k|c}, κ_k, π_{k|c}) fixed.
            - Recompute n_{k,d} from current z.
            - For each document d, run one sweep of Gibbs sampling over sentences:
                  p(z_{d,i} = k | z_{d,-i}, α, Δ)
                  ∝ (N_{dk}^{(-i)} + α_k) * vMF_mixture(x_{d,i} | Δ_k)

        M-step:
            - Given new topic assignments z, recompute mixture parameters:
                R_{k|c} = Σ_i weight_{ic} x_i
                μ_{k|c} = R_{k|c} / ||R_{k|c}||
                π_{k|c} = (1 / N_k) Σ_i weight_{ic}
                r_k = (1 / N_k) Σ_c ||R_{k|c}||
                κ_k ≈ (r_k M - r_k^3) / (1 - r_k^2)

        Args:
            num_iterations: Number of EM iterations.
            num_sweeps: Number of Gibbs sweeps per EM iteration (ζ).
            num_samples: Number of samples to keep from the last sweeps (B).

        At the end of each iteration, compute the average log-likelihood.
        """
        if num_sweeps < 1:
            raise ValueError("num_sweeps must be >= 1")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if alpha_update_every < 1:
            raise ValueError("alpha_update_every must be >= 1")
        if not np.isfinite(alpha_min_value) or alpha_min_value <= 0.0:
            raise ValueError("alpha_min_value must be positive and finite")
        if min_topic_count_for_repair < 1:
            raise ValueError("min_topic_count_for_repair must be >= 1")
        if avg_log_likelihood_every < 1:
            raise ValueError("avg_log_likelihood_every must be >= 1")
        if invariant_check_every < 1:
            raise ValueError("invariant_check_every must be >= 1")
        if num_samples > num_sweeps:
            self.log.warning(
                "num_samples (%s) > num_sweeps (%s); using num_sweeps instead.",
                num_samples,
                num_sweeps,
            )
            num_samples = num_sweeps
        for it in range(num_iterations):
            self.log.info(f"Iteration {it} (E-step)")
            compute_avg_log_likelihood = self._should_run_periodic_step(
                iteration=it,
                total_iterations=num_iterations,
                every=avg_log_likelihood_every,
            )
            result = self._run_training_iteration(
                iteration=it,
                num_sweeps=num_sweeps,
                num_samples=num_samples,
                estimate_alpha=estimate_alpha,
                alpha_update_every=alpha_update_every,
                alpha_max_iter=alpha_max_iter,
                alpha_tol=alpha_tol,
                alpha_min_value=alpha_min_value,
                repair_empty_topics=repair_empty_topics,
                min_topic_count_for_repair=min_topic_count_for_repair,
                compute_avg_log_likelihood=compute_avg_log_likelihood,
            )
            self._record_iteration_diagnostics(
                iteration=it,
                num_sweeps=num_sweeps,
                num_samples=num_samples,
                result=result,
            )
            if self._should_run_periodic_step(
                iteration=it,
                total_iterations=num_iterations,
                every=invariant_check_every,
            ):
                self.assert_valid_state()

            if self.save_path is not None:
                self.log.info("Saving model")
                self.save()

    # -------------------------------------------------------------------------
    # Save parameters
    # -------------------------------------------------------------------------
    def save(self) -> None:
        """Save model parameters to the configured save_path."""
        if self.save_path is None:
            return

        payload = build_vmf_model_artifact_payload(
            average_ll=self.average_ll,
            iteration_diagnostics=[asdict(item) for item in self.iteration_diagnostics],
            embedding_cache=asdict(self.build_embedding_cache_report()),
            alpha=self.alpha,
            num_topics=self.num_topics,
            kappa_default=self.kappa_default,
            num_components=self.num_components,
            pre_normalize_transform=self.pre_normalize_transform,
            whitening_eps=self.whitening_eps,
            algorithm_variant=self.algorithm_variant,
            topic_counts=self.topic_counts,
            topic_counts_per_doc=self.topic_counts_per_doc,
            topic_means=self.topic_means,
            sum_topic_vectors=self.sum_topic_vectors,
            kappa_per_topic=self.kappa_per_topic,
            mixture_weights=self.mixture_weights,
            component_means=self.component_means,
            embedding_preprocessor=self.embedding_preprocessor,
        )
        save_vmf_model_artifacts(payload, self.save_path)

    def to_output(self) -> TopicModelOutput:
        """Convert the current trained state into the common model output."""

        return TopicModelOutput(
            doc_topic=self.get_document_topic_distribution(),
            sentence_topic=list(self.topic_assignments),
            topic_embeddings=self.topic_means,
            metadata={
                "model_name": "vmf_sentence_lda",
                "num_topics": self.num_topics,
                "num_components": self.num_components,
                "algorithm_variant": self.algorithm_variant,
            },
        )

    # -------------------------------------------------------------------------
    # Document-topic distribution (θ_dk)
    # -------------------------------------------------------------------------
    def get_document_topic_distribution(self) -> np.ndarray:
        """
        Return document–topic distribution theta with shape (n_docs, num_topics).

        Each row is normalized so that it sums to 1.
        Internally, topic_counts_per_doc has shape (num_topics, num_documents).

        Note: You must run `sample(...)` at least once before calling this.
        """
        if not hasattr(self, "topic_counts_per_doc"):
            raise AttributeError(
                "topic_counts_per_doc is not available. "
                "Make sure you have run `sample(...)` before calling this method."
            )

        return self.inferencer.build_document_topic_distribution(
            self.topic_counts_per_doc
        )

    def get_num_documents(self) -> int:
        """Return the number of documents used to train the model."""
        if not hasattr(self, "topic_counts_per_doc"):
            raise AttributeError(
                "topic_counts_per_doc is not available. "
                "Make sure you have run `sample(...)` before calling this method."
            )
        return int(self.topic_counts_per_doc.shape[1])

    def infer_document_topic_counts(
        self, new_corpus: Sequence[Sequence[str]]
    ) -> np.ndarray:
        """
        Infer document–topic counts for new documents using a greedy argmax assignment.

        Each sentence is encoded and assigned to the topic with the highest vMF
        mixture density under the current parameters. Returns an integer matrix with
        shape (n_docs, num_topics).
        """
        return self.inferencer.infer_document_topic_counts(new_corpus)

    def infer_document_topic_distribution_soft(
        self, new_corpus: Sequence[Sequence[str]], temperature: float = 1.0
    ) -> np.ndarray:
        """
        Infer document–topic distributions using soft per-sentence assignments.

        For each sentence vector x:
            p(z=k|x) ∝ exp(log_vmf_density_tables(x) / temperature)
        Document distribution is the normalized sum of per-sentence posteriors.
        """
        return self.inferencer.infer_document_topic_distribution_soft(
            new_corpus,
            temperature=temperature,
        )

    def infer_corpus_topic_outputs(
        self,
        new_corpus: Sequence[Sequence[str]],
        *,
        temperature: float = 1.0,
        include_counts: bool = False,
        include_sentence_posteriors: bool = False,
        include_document_posteriors: bool = False,
    ) -> VMFCorpusInferenceOutputs:
        return self.inferencer.infer_corpus_topic_outputs(
            new_corpus,
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
        return self.inferencer.infer_encoded_corpus_topic_outputs(
            encoded_corpus,
            temperature=temperature,
            include_counts=include_counts,
            include_sentence_posteriors=include_sentence_posteriors,
            include_document_posteriors=include_document_posteriors,
        )

    def aggregate_document_topic_distribution_from_sentence_posteriors(
        self, sentence_posteriors: Sequence[np.ndarray]
    ) -> np.ndarray:
        """
        Aggregate per-sentence posteriors into document-topic distributions.

        Args:
            sentence_posteriors:
                Sequence of arrays, each shaped (n_sentences_in_doc, num_topics).

        Returns:
            2D array of shape (n_docs, num_topics), row-normalized.
        """
        return self.inferencer.aggregate_document_topic_distribution_from_sentence_posteriors(
            sentence_posteriors
        )

    def infer_sentence_topic_distribution_soft(
        self, new_corpus: Sequence[Sequence[str]], temperature: float = 1.0
    ) -> list[np.ndarray]:
        """
        Infer per-sentence topic posteriors for each document.

        Returns:
            A list with length n_docs. Each element is a 2D array of shape
            (n_sentences_in_doc, num_topics), where each row sums to 1.
        """
        return self.inferencer.infer_sentence_topic_distribution_soft(
            new_corpus,
            temperature=temperature,
        )

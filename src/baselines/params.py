from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal
from typing import Any, Mapping

import numpy as np

from src.utils.encoder_profiles import resolve_encoder_settings

DEFAULT_PRETRAINED_WORD2VEC = "word2vec-google-news-300"


COVARIANCE_TYPES: tuple[str, ...] = ("full", "diag", "spherical")
_COVARIANCE_TYPE_ALIASES = {
    "iso": "spherical",
    "isotropic": "spherical",
    "diagonal": "diag",
}
_COVARIANCE_VARIANT_LABELS = {"diag": "cov-diag", "spherical": "cov-iso"}


def normalize_covariance_type(value: object, *, default: str = "full") -> str:
    """Canonical covariance type of the sentence Gaussian LDA family.

    ``None``/empty means the default (``full``); ``iso``/``isotropic`` map to ``spherical`` and
    ``diagonal`` to ``diag``. Any other value raises.
    """
    if value is None:
        return default
    key = str(value).strip().lower()
    if not key:
        return default
    key = _COVARIANCE_TYPE_ALIASES.get(key, key)
    if key not in COVARIANCE_TYPES:
        raise ValueError(
            "covariance_type must be one of 'full', 'diag', or 'spherical' "
            f"(aliases: 'iso', 'isotropic', 'diagonal'); got {value!r}."
        )
    return key


def is_reduced_covariance(covariance_type: object) -> bool:
    return normalize_covariance_type(covariance_type) != "full"


def format_covariance_variant(covariance_type: object) -> str | None:
    """Result-path label of a covariance type: ``None`` for ``full`` (paths unchanged), else
    ``cov-diag`` / ``cov-iso``. The single definition of the label used in display keys.
    """
    key = normalize_covariance_type(covariance_type)
    return _COVARIANCE_VARIANT_LABELS.get(key)


def format_prior_scale_variant(value: float) -> str:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("prior_scale must be finite and > 0.")
    normalized = format(Decimal(str(scale)).normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return f"psi0-{normalized.replace('.', 'p')}"


@dataclass(frozen=True)
class BleiLdaParams:
    passes: int = 20
    num_iterations: int = 50


@dataclass(frozen=True)
class SamParams:
    r"""Spherical Admixture Model (Reisinger et al., ICML 2010).

    The three vMF concentrations are dimension dependent: ``A_V(kappa)`` is what
    actually controls the model, and the paper's ``kappa = 1500`` corresponds to
    ``A_V ~ 0.09`` at its ``V = 16552``.  Each concentration may therefore be
    given either as a raw value or, by leaving it ``None``, as a target mean
    resultant length that is resolved against the fitted vocabulary size.

    The reported ``SAM`` is the **tf** condition (runner ``sam_tf``); the tf-idf
    condition is reported as ``SAM (tf-idf)`` (runner ``sam``).  Each runner pins
    its own ``feature_scheme``, so this is a labelling choice, not a change to
    either result tree.  The reason is comparability: LDA and sentLDA are fitted on
    raw counts, and idf down-weights exactly the frequent words that NPMI also
    penalizes, so tf-idf aligns SAM's input with the coherence metric in a way the
    other baselines do not enjoy.  Measured across the full grid, moving SAM from
    tf-idf to tf drops it from rank 1 to ranks 4-6 on NYT coherence at K=20..100.
    Note this is the conservative direction for SAM: the paper's own best result is
    tf-idf (93.3 vs 88.6 on its news-20 "different" task).

    ``min_df`` / ``max_df`` default to no pruning at all so that SAM sees exactly
    the vocabulary the other baselines see.  ``bleilda`` builds its dictionary as
    ``gensim.corpora.Dictionary(training_data)`` with no ``filter_extremes`` call,
    and the rest follow the shared preprocessing likewise; pruning only SAM to
    ``min_df=5, max_df=0.5`` cut 20 Newsgroups from ``V = 44007`` to ``V = 10676``
    and, because coherence is scored over each model's own top words, handed SAM a
    vocabulary that no longer contains any rare term.  The resulting coherence and
    accuracy were not comparable with the other models.  The paper's own setup does
    prune (``V = 16552``); reproduce it by passing ``min_df`` / ``max_df``
    explicitly rather than by changing these defaults.

    ``tol`` is a relative change of the *variable* part of the bound, i.e. with
    ``D * log c_V(kappa)`` subtracted.  That term is a constant while ``kappa`` is
    fixed and dwarfs everything else (``3.82e8`` against a ``1.3e6`` span at 20
    Newsgroups), so judging convergence on the raw bound measures the constant
    rather than the fit.

    The default was picked from a planted-topic recovery sweep, where topic
    directions and document proportions saturate at ``1e-5``:

    ==========  ======  ============  ==========
    tol         iters   mean \|cos\|   Spearman
    ==========  ======  ============  ==========
    1e-3             6        0.9806      0.8866
    1e-4            16        0.9960      0.9408
    **1e-5**        35        0.9998      0.9822
    1e-6            50        0.9999      0.9897
    ==========  ======  ============  ==========

    ``xi`` is the one that must not be guessed.  ``E_q[phi_t] = A_V(xi) mutilde_t``,
    so a small ``A_V(xi)`` shrinks every topic direction out of the likelihood and
    the bound is then maximized by collapsing all topics onto the corpus mean and
    spreading ``alphatilde`` to shrink ``S_d``.  Measured on a planted-topic
    recovery task, ``A_V(xi) = 0.05`` recovers nothing (mean ``|cos| = 0.50``,
    exactly the collapsed value) while ``A_V(xi) >= 0.9`` recovers essentially
    perfectly (``>= 0.993``).  Hence the default below, and hence ``kappa0`` stays
    small so the sharp topic prior does not also pull the topics together.
    """

    feature_scheme: str = "tfidf"
    min_df: int = 1
    max_df: float = 1.0
    max_vocabulary_size: int | None = None
    sublinear_tf: bool = False
    kappa: float | None = None
    kappa_mean_resultant: float = 0.1
    xi: float | None = None
    xi_mean_resultant: float = 0.99
    kappa0: float | None = None
    kappa0_mean_resultant: float = 0.01
    alpha: float = 1.0
    num_iterations: int = 200
    alpha_steps: int = 20
    mu_steps: int = 5
    alpha_step_size: float = 0.1
    mu_step_size: float = 0.1
    tol: float = 1e-5
    infer_num_iterations: int = 100
    optimize_kappa: bool = False
    require_convergence: bool = False
    init: str = "corpus_random"
    random_state: int | None = None


@dataclass(frozen=True)
class CtmParams:
    contextual_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    contextual_encode_prefix: str | None = None
    encoder_backend: str = "sentence_transformers"
    pooling: str | None = None
    encode_prompt: str | None = None
    encode_prompt_name: str | None = None
    encode_batch_size: int = 128
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    normalize_embeddings: bool | None = None
    truncate_dim: int | None = None
    use_custom_embeddings: bool = False
    num_epochs: int = 50
    num_samples: int = 20
    batch_size_cap: int = 64


@dataclass(frozen=True)
class SenCluParams:
    encoder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    encode_prefix: str | None = None
    encoder_backend: str = "sentence_transformers"
    pooling: str | None = None
    encode_prompt: str | None = None
    encode_prompt_name: str | None = None
    encode_batch_size: int = 128
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    normalize_embeddings: bool | None = None
    truncate_dim: int | None = None
    alpha: float | None = None
    num_epochs: int = 40
    soft_temperature: float = 1.0
    verbose: bool = False
    embedding_cache_dir: str | None = None


@dataclass(frozen=True)
class GaussianLdaParams:
    word2vec: str = DEFAULT_PRETRAINED_WORD2VEC
    wikientvec_cache_dir: str | None = None
    num_iterations: int = 20
    prior_scale: float = 0.1


@dataclass(frozen=True)
class MvTMParams:
    word2vec: str = DEFAULT_PRETRAINED_WORD2VEC
    wikientvec_cache_dir: str | None = None
    num_iterations: int = 20
    num_components: int = 1
    alpha: float | None = None
    estimate_alpha: bool = False
    kappa_default: float = 10.0
    max_kappa: float = 10_000.0
    gibbs_sweeps: int = 1
    num_samples: int = 1
    alpha_update_every: int = 1
    alpha_max_iter: int = 100
    alpha_tol: float = 1e-5
    avg_log_likelihood_every: int = 1
    invariant_check_every: int = 1
    soft_temperature: float = 1.0


@dataclass(frozen=True)
class EtmParams:
    word2vec: str = DEFAULT_PRETRAINED_WORD2VEC
    wikientvec_cache_dir: str | None = None
    num_epochs: int = 100
    batch_size: int = 128
    eval_batch_size: int = 128
    t_hidden_size: int = 800
    theta_act: str = "relu"
    lr: float = 0.002
    weight_decay: float = 1.2e-6
    enc_drop: float = 0.0
    clip: float = 0.0
    bow_norm: bool = True
    optimizer: str = "adam"
    random_state: int | None = None
    reference_profile: str = "repo_default"


@dataclass(frozen=True)
class SentenceGaussianLdaParams:
    encoder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    encode_prefix: str | None = None
    encoder_backend: str = "sentence_transformers"
    pooling: str | None = None
    encode_prompt: str | None = None
    encode_prompt_name: str | None = None
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    normalize_embeddings: bool | None = None
    truncate_dim: int | None = None
    strip_terminal_normalize: bool = True
    num_iterations: int = 10
    num_gibbs_iters: int = 20
    encode_batch_size: int = 128
    preencode_corpus: bool = True
    soft_temperature: float = 1.0
    prior_scale: float = 0.1
    covariance_type: str = "full"


@dataclass(frozen=True)
class SentLdaParams:
    num_iterations: int = 20
    alpha: float | None = None
    beta: float | None = None
    random_state: int | None = None
    infer_num_iterations: int = 50
    save_phi: bool = True
    backend: str = "auto"


@dataclass(frozen=True)
class BertopicKMeansParams:
    encoder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    encode_prefix: str | None = None
    encoder_backend: str = "sentence_transformers"
    pooling: str | None = None
    encode_prompt: str | None = None
    encode_prompt_name: str | None = None
    encode_batch_size: int = 128
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    normalize_embeddings: bool | None = None
    truncate_dim: int | None = None
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    kmeans_n_init: int = 10
    soft_temperature: float = 1.0
    random_state: int | None = None
    verbose: bool = False


@dataclass(frozen=True)
class SphericalKMeansParams:
    encoder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    encode_prefix: str | None = None
    encoder_backend: str = "sentence_transformers"
    pooling: str | None = None
    encode_prompt: str | None = None
    encode_prompt_name: str | None = None
    encode_batch_size: int = 128
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    normalize_embeddings: bool | None = None
    truncate_dim: int | None = None
    random_state: int = 0
    soft_temperature: float = 1.0
    verbose: bool = False
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    init: str = "kmeans++"


@dataclass(frozen=True)
class GaussianKMeansParams:
    encoder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    encode_prefix: str | None = None
    encoder_backend: str = "sentence_transformers"
    pooling: str | None = None
    encode_prompt: str | None = None
    encode_prompt_name: str | None = None
    encode_batch_size: int = 128
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    normalize_embeddings: bool | None = None
    truncate_dim: int | None = None
    strip_terminal_normalize: bool = True
    random_state: int = 0
    soft_temperature: float = 1.0
    verbose: bool = False
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    init: str = "kmeans++"


@dataclass(frozen=True)
class MovMFParams:
    encoder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    encode_prefix: str | None = None
    encoder_backend: str = "sentence_transformers"
    pooling: str | None = None
    encode_prompt: str | None = None
    encode_prompt_name: str | None = None
    encode_batch_size: int = 128
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    normalize_embeddings: bool | None = None
    truncate_dim: int | None = None
    random_state: int = 0
    soft_temperature: float = 1.0
    verbose: bool = False
    n_init: int = 5
    max_iter: int = 100
    tol: float = 1e-4
    init: str = "spherical_kmeans"
    min_kappa: float = 1e-3
    max_kappa: float = 1e4


@dataclass(frozen=True)
class GaussianMixtureParams:
    encoder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    encode_prefix: str | None = None
    encoder_backend: str = "sentence_transformers"
    pooling: str | None = None
    encode_prompt: str | None = None
    encode_prompt_name: str | None = None
    encode_batch_size: int = 128
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    normalize_embeddings: bool | None = None
    truncate_dim: int | None = None
    strip_terminal_normalize: bool = True
    random_state: int = 0
    soft_temperature: float = 1.0
    verbose: bool = False
    n_init: int = 5
    max_iter: int = 100
    tol: float = 1e-4
    init: str = "kmeans"
    covariance_type: str = "diag"
    reg_covar: float = 1e-6


BaselineParams = (
    BleiLdaParams
    | CtmParams
    | SenCluParams
    | GaussianLdaParams
    | SentenceGaussianLdaParams
    | SentLdaParams
    | BertopicKMeansParams
    | SphericalKMeansParams
    | GaussianKMeansParams
    | MvTMParams
    | SamParams
    | EtmParams
    | MovMFParams
    | GaussianMixtureParams
    | dict[str, Any]
)


def parse_bleilda_params(options: dict[str, Any]) -> BleiLdaParams:
    return BleiLdaParams(
        passes=int(options.get("passes", 20)),
        num_iterations=int(options.get("num_iterations", 50)),
    )


def _positive_float(options: dict[str, Any], key: str, *, default: float) -> float:
    value = float(options.get(key, default))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"sam params.{key} must be finite and > 0.")
    return value


def _optional_positive_float(options: dict[str, Any], key: str) -> float | None:
    if options.get(key) is None:
        return None
    value = float(options[key])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"sam params.{key} must be finite and > 0.")
    return value


def _unit_interval(options: dict[str, Any], key: str, *, default: float) -> float:
    value = float(options.get(key, default))
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"sam params.{key} must lie strictly between 0 and 1.")
    return value


def parse_sam_params(options: dict[str, Any]) -> SamParams:
    feature_scheme = str(options.get("feature_scheme", "tfidf")).strip().lower()
    if feature_scheme not in {"tfidf", "tf"}:
        raise ValueError("sam params.feature_scheme must be 'tfidf' or 'tf'.")

    min_df = int(options.get("min_df", 1))
    if min_df < 1:
        raise ValueError("sam params.min_df must be >= 1.")
    max_df = float(options.get("max_df", 1.0))
    if not np.isfinite(max_df) or not 0.0 < max_df <= 1.0:
        raise ValueError("sam params.max_df must lie in (0, 1].")

    num_iterations = int(options.get("num_iterations", 200))
    if num_iterations <= 0:
        raise ValueError("sam params.num_iterations must be > 0.")
    alpha_steps = int(options.get("alpha_steps", 20))
    if alpha_steps <= 0:
        raise ValueError("sam params.alpha_steps must be > 0.")
    mu_steps = int(options.get("mu_steps", 5))
    if mu_steps <= 0:
        raise ValueError("sam params.mu_steps must be > 0.")
    infer_num_iterations = int(options.get("infer_num_iterations", 100))
    if infer_num_iterations <= 0:
        raise ValueError("sam params.infer_num_iterations must be > 0.")

    init = str(options.get("init", "corpus_random")).strip().lower()
    if init not in {"corpus_random", "random"}:
        raise ValueError("sam params.init must be 'corpus_random' or 'random'.")

    return SamParams(
        feature_scheme=feature_scheme,
        min_df=min_df,
        max_df=max_df,
        max_vocabulary_size=_optional_positive_int(options, "max_vocabulary_size"),
        sublinear_tf=_bool_option(options, "sublinear_tf", default=False),
        kappa=(
            _optional_positive_float(options, "kappa") if "kappa" in options else None
        ),
        kappa_mean_resultant=_unit_interval(
            options, "kappa_mean_resultant", default=0.1
        ),
        xi=_optional_positive_float(options, "xi"),
        xi_mean_resultant=_unit_interval(options, "xi_mean_resultant", default=0.99),
        kappa0=_optional_positive_float(options, "kappa0"),
        kappa0_mean_resultant=_unit_interval(
            options, "kappa0_mean_resultant", default=0.01
        ),
        alpha=_positive_float(options, "alpha", default=1.0),
        num_iterations=num_iterations,
        alpha_steps=alpha_steps,
        mu_steps=mu_steps,
        alpha_step_size=_positive_float(options, "alpha_step_size", default=0.1),
        mu_step_size=_positive_float(options, "mu_step_size", default=0.1),
        tol=_positive_float(options, "tol", default=1e-5),
        infer_num_iterations=infer_num_iterations,
        optimize_kappa=_bool_option(options, "optimize_kappa", default=False),
        require_convergence=_bool_option(options, "require_convergence", default=False),
        init=init,
        random_state=_optional_int(options.get("random_state")),
    )


def parse_sam_tf_params(options: dict[str, Any]) -> SamParams:
    """``sam_tf``: SAM on L2-normalized term frequencies instead of tf-idf.

    The paper reports tf and tf-idf as two conditions ("l2-normalized tf or
    tf-idf document representations") and the gap between them is large, so the
    tf variant is a separate runner rather than a parameter: two runners give two
    result trees, whereas one runner with a parameter would collide on the same
    display key.  Everything except the default ``feature_scheme`` is shared with
    :func:`parse_sam_params`; an explicit ``feature_scheme`` is still honoured.
    """

    merged = dict(options)
    merged.setdefault("feature_scheme", "tf")
    return parse_sam_params(merged)


def _optional_str(options: dict[str, Any], key: str) -> str | None:
    return None if options.get(key) is None else str(options.get(key))


def _optional_bool(options: dict[str, Any], key: str) -> bool | None:
    return None if options.get(key) is None else bool(options.get(key))


def _bool_option(options: dict[str, Any], key: str, *, default: bool) -> bool:
    value = options.get(key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"{key} must be a boolean value.")
    return bool(value)


def _optional_positive_int(options: dict[str, Any], key: str) -> int | None:
    if options.get(key) is None:
        return None
    value = int(options[key])
    if value <= 0:
        raise ValueError(f"{key} must be > 0.")
    return value


def _parse_encoder_common(
    options: dict[str, Any],
    *,
    model_key: str = "encoder_model_name",
    prefix_key: str = "encode_prefix",
) -> dict[str, Any]:
    model_name = str(options.get(model_key, "sentence-transformers/all-mpnet-base-v2"))
    resolved = resolve_encoder_settings(
        model_name=model_name,
        backend=str(options.get("encoder_backend", options.get("backend", "auto"))),
        pooling=_optional_str(options, "pooling"),
        encode_prefix=_optional_str(options, prefix_key),
        encode_prompt=_optional_str(options, "encode_prompt"),
        encode_prompt_name=_optional_str(options, "encode_prompt_name"),
        encode_batch_size=_optional_positive_int(options, "encode_batch_size"),
        model_kwargs=dict(options.get("model_kwargs") or {}),
        tokenizer_kwargs=dict(options.get("tokenizer_kwargs") or {}),
        normalize_embeddings=_optional_bool(options, "normalize_embeddings"),
        truncate_dim=_optional_positive_int(options, "truncate_dim"),
    )
    encode_batch_size = resolved.encode_batch_size or 128
    return {
        "encoder_model_name": resolved.model_name,
        "encode_prefix": resolved.encode_prefix,
        "encoder_backend": resolved.backend,
        "pooling": resolved.pooling,
        "encode_prompt": resolved.encode_prompt,
        "encode_prompt_name": resolved.encode_prompt_name,
        "encode_batch_size": encode_batch_size,
        "model_kwargs": resolved.model_kwargs,
        "tokenizer_kwargs": resolved.tokenizer_kwargs,
        "normalize_embeddings": resolved.normalize_embeddings,
        "truncate_dim": resolved.truncate_dim,
    }


def parse_ctm_params(options: dict[str, Any]) -> CtmParams:
    common = _parse_encoder_common(
        options,
        model_key="contextual_model_name",
        prefix_key="contextual_encode_prefix",
    )
    return CtmParams(
        contextual_model_name=common["encoder_model_name"],
        contextual_encode_prefix=common["encode_prefix"],
        encoder_backend=common["encoder_backend"],
        pooling=common["pooling"],
        encode_prompt=common["encode_prompt"],
        encode_prompt_name=common["encode_prompt_name"],
        encode_batch_size=common["encode_batch_size"],
        model_kwargs=common["model_kwargs"],
        tokenizer_kwargs=common["tokenizer_kwargs"],
        normalize_embeddings=common["normalize_embeddings"],
        truncate_dim=common["truncate_dim"],
        use_custom_embeddings=bool(options.get("use_custom_embeddings", True)),
        num_epochs=int(options.get("num_epochs", 50)),
        num_samples=int(options.get("num_samples", 20)),
        batch_size_cap=int(options.get("batch_size_cap", 64)),
    )


def parse_senclu_params(options: dict[str, Any]) -> SenCluParams:
    alpha = options.get("alpha")
    common = _parse_encoder_common(options)
    return SenCluParams(
        encoder_model_name=common["encoder_model_name"],
        encode_prefix=common["encode_prefix"],
        encoder_backend=common["encoder_backend"],
        pooling=common["pooling"],
        encode_prompt=common["encode_prompt"],
        encode_prompt_name=common["encode_prompt_name"],
        encode_batch_size=common["encode_batch_size"],
        model_kwargs=common["model_kwargs"],
        tokenizer_kwargs=common["tokenizer_kwargs"],
        normalize_embeddings=common["normalize_embeddings"],
        truncate_dim=common["truncate_dim"],
        alpha=None if alpha is None else float(alpha),
        num_epochs=int(options.get("num_epochs", 40)),
        soft_temperature=float(options.get("soft_temperature", 1.0)),
        verbose=bool(options.get("verbose", False)),
        embedding_cache_dir=(
            None
            if options.get("embedding_cache_dir") is None
            else str(options.get("embedding_cache_dir"))
        ),
    )


def parse_gaussianlda_params(options: dict[str, Any]) -> GaussianLdaParams:
    params = GaussianLdaParams(
        word2vec=str(options.get("word2vec", DEFAULT_PRETRAINED_WORD2VEC)),
        wikientvec_cache_dir=(
            None
            if options.get("wikientvec_cache_dir") is None
            else str(options.get("wikientvec_cache_dir"))
        ),
        num_iterations=int(options.get("num_iterations", 20)),
        prior_scale=float(options.get("prior_scale", 0.1)),
    )
    if not np.isfinite(params.prior_scale) or params.prior_scale <= 0.0:
        raise ValueError("gaussianlda params.prior_scale must be finite and > 0.")
    return params


def parse_mvtm_params(options: dict[str, Any]) -> MvTMParams:
    alpha = options.get("alpha")
    params = MvTMParams(
        word2vec=str(options.get("word2vec", DEFAULT_PRETRAINED_WORD2VEC)),
        wikientvec_cache_dir=(
            None
            if options.get("wikientvec_cache_dir") is None
            else str(options.get("wikientvec_cache_dir"))
        ),
        num_iterations=int(options.get("num_iterations", 20)),
        num_components=int(options.get("num_components", 1)),
        alpha=None if alpha is None else float(alpha),
        estimate_alpha=bool(options.get("estimate_alpha", False)),
        kappa_default=float(options.get("kappa_default", 10.0)),
        max_kappa=float(options.get("max_kappa", 10_000.0)),
        gibbs_sweeps=int(options.get("gibbs_sweeps", 1)),
        num_samples=int(options.get("num_samples", 1)),
        alpha_update_every=int(options.get("alpha_update_every", 1)),
        alpha_max_iter=int(options.get("alpha_max_iter", 100)),
        alpha_tol=float(options.get("alpha_tol", 1e-5)),
        avg_log_likelihood_every=int(options.get("avg_log_likelihood_every", 1)),
        invariant_check_every=int(options.get("invariant_check_every", 1)),
        soft_temperature=float(options.get("soft_temperature", 1.0)),
    )
    if params.num_iterations < 1:
        raise ValueError("mvtm params.num_iterations must be >= 1.")
    if params.num_components < 1:
        raise ValueError("mvtm params.num_components must be >= 1.")
    if params.alpha is not None and params.alpha <= 0.0:
        raise ValueError("mvtm params.alpha must be > 0 when provided.")
    if params.kappa_default <= 0.0:
        raise ValueError("mvtm params.kappa_default must be > 0.")
    if not np.isfinite(params.max_kappa) or params.max_kappa <= 0.0:
        raise ValueError("mvtm params.max_kappa must be finite and > 0.")
    if not np.isfinite(params.kappa_default):
        raise ValueError("mvtm params.kappa_default must be finite.")
    if params.kappa_default > params.max_kappa:
        raise ValueError("mvtm params.kappa_default must be <= max_kappa.")
    if params.gibbs_sweeps < 1:
        raise ValueError("mvtm params.gibbs_sweeps must be >= 1.")
    if params.num_samples < 1:
        raise ValueError("mvtm params.num_samples must be >= 1.")
    if params.alpha_update_every < 1:
        raise ValueError("mvtm params.alpha_update_every must be >= 1.")
    if params.alpha_max_iter < 1:
        raise ValueError("mvtm params.alpha_max_iter must be >= 1.")
    if params.alpha_tol <= 0.0:
        raise ValueError("mvtm params.alpha_tol must be > 0.")
    if params.avg_log_likelihood_every < 1:
        raise ValueError("mvtm params.avg_log_likelihood_every must be >= 1.")
    if params.invariant_check_every < 1:
        raise ValueError("mvtm params.invariant_check_every must be >= 1.")
    if params.soft_temperature <= 0.0:
        raise ValueError("mvtm params.soft_temperature must be > 0.")
    return params


def parse_etm_params(options: dict[str, Any]) -> EtmParams:
    theta_act = str(options.get("theta_act", "relu")).strip().lower()
    optimizer = str(options.get("optimizer", "adam")).strip().lower()
    params = EtmParams(
        word2vec=str(options.get("word2vec", DEFAULT_PRETRAINED_WORD2VEC)),
        wikientvec_cache_dir=(
            None
            if options.get("wikientvec_cache_dir") is None
            else str(options.get("wikientvec_cache_dir"))
        ),
        num_epochs=int(options.get("num_epochs", 100)),
        batch_size=int(options.get("batch_size", 128)),
        eval_batch_size=int(options.get("eval_batch_size", 128)),
        t_hidden_size=int(options.get("t_hidden_size", 800)),
        theta_act=theta_act,
        lr=float(options.get("lr", 0.002)),
        weight_decay=float(options.get("weight_decay", 1.2e-6)),
        enc_drop=float(options.get("enc_drop", 0.0)),
        clip=float(options.get("clip", 0.0)),
        bow_norm=bool(options.get("bow_norm", True)),
        optimizer=optimizer,
        random_state=_optional_int(options.get("random_state")),
        reference_profile=str(options.get("reference_profile", "repo_default")),
    )
    if params.num_epochs < 1:
        raise ValueError("etm params.num_epochs must be >= 1.")
    if params.batch_size < 1:
        raise ValueError("etm params.batch_size must be >= 1.")
    if params.eval_batch_size < 1:
        raise ValueError("etm params.eval_batch_size must be >= 1.")
    if params.t_hidden_size < 1:
        raise ValueError("etm params.t_hidden_size must be >= 1.")
    if params.lr <= 0.0:
        raise ValueError("etm params.lr must be > 0.")
    if params.weight_decay < 0.0:
        raise ValueError("etm params.weight_decay must be >= 0.")
    if not (0.0 <= params.enc_drop < 1.0):
        raise ValueError("etm params.enc_drop must satisfy 0 <= enc_drop < 1.")
    if params.clip < 0.0:
        raise ValueError("etm params.clip must be >= 0.")
    if params.theta_act not in {"relu", "tanh", "softplus", "rrelu", "leakyrelu"}:
        raise ValueError(
            "etm params.theta_act must be one of 'relu', 'tanh', 'softplus', "
            "'rrelu', or 'leakyrelu'."
        )
    if params.optimizer not in {"adam", "sgd", "adagrad", "adadelta", "rmsprop"}:
        raise ValueError(
            "etm params.optimizer must be one of 'adam', 'sgd', 'adagrad', "
            "'adadelta', or 'rmsprop'."
        )
    return params


def parse_sentence_gaussianlda_params(
    options: dict[str, Any],
) -> SentenceGaussianLdaParams:
    common = _parse_encoder_common(options)
    params = SentenceGaussianLdaParams(
        encoder_model_name=common["encoder_model_name"],
        encode_prefix=common["encode_prefix"],
        encoder_backend=common["encoder_backend"],
        pooling=common["pooling"],
        encode_prompt=common["encode_prompt"],
        encode_prompt_name=common["encode_prompt_name"],
        model_kwargs=common["model_kwargs"],
        tokenizer_kwargs=common["tokenizer_kwargs"],
        normalize_embeddings=common["normalize_embeddings"],
        truncate_dim=common["truncate_dim"],
        strip_terminal_normalize=_bool_option(
            options, "strip_terminal_normalize", default=True
        ),
        num_iterations=int(options.get("num_iterations", 10)),
        num_gibbs_iters=int(options.get("num_gibbs_iters", 20)),
        encode_batch_size=common["encode_batch_size"],
        preencode_corpus=bool(options.get("preencode_corpus", True)),
        soft_temperature=float(options.get("soft_temperature", 1.0)),
        prior_scale=float(options.get("prior_scale", 0.1)),
        covariance_type=_parse_covariance_type(options.get("covariance_type")),
    )
    if not np.isfinite(params.prior_scale) or params.prior_scale <= 0.0:
        raise ValueError(
            "sentence_gaussianlda params.prior_scale must be finite and > 0."
        )
    return params


def _parse_covariance_type(value: object) -> str:
    try:
        return normalize_covariance_type(value)
    except ValueError as exc:
        raise ValueError(f"sentence_gaussianlda params.{exc}") from exc


def parse_sentlda_params(options: dict[str, Any]) -> SentLdaParams:
    alpha = options.get("alpha")
    beta = options.get("beta")
    backend = str(options.get("backend", "auto")).strip().lower()
    if backend not in {"auto", "python", "numba"}:
        raise ValueError(
            "sentlda params.backend must be one of 'auto', 'python', or 'numba'."
        )
    return SentLdaParams(
        num_iterations=int(options.get("num_iterations", 20)),
        alpha=None if alpha is None else float(alpha),
        beta=None if beta is None else float(beta),
        random_state=_optional_int(options.get("random_state")),
        infer_num_iterations=int(options.get("infer_num_iterations", 50)),
        save_phi=bool(options.get("save_phi", True)),
        backend=backend,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def parse_bertopic_kmeans_params(options: dict[str, Any]) -> BertopicKMeansParams:
    common = _parse_encoder_common(options)
    params = BertopicKMeansParams(
        encoder_model_name=common["encoder_model_name"],
        encode_prefix=common["encode_prefix"],
        encoder_backend=common["encoder_backend"],
        pooling=common["pooling"],
        encode_prompt=common["encode_prompt"],
        encode_prompt_name=common["encode_prompt_name"],
        encode_batch_size=common["encode_batch_size"],
        model_kwargs=common["model_kwargs"],
        tokenizer_kwargs=common["tokenizer_kwargs"],
        normalize_embeddings=common["normalize_embeddings"],
        truncate_dim=common["truncate_dim"],
        umap_n_neighbors=int(options.get("umap_n_neighbors", 15)),
        umap_n_components=int(options.get("umap_n_components", 5)),
        umap_min_dist=float(options.get("umap_min_dist", 0.0)),
        umap_metric=str(options.get("umap_metric", "cosine")),
        kmeans_n_init=int(options.get("kmeans_n_init", 10)),
        soft_temperature=float(options.get("soft_temperature", 1.0)),
        random_state=_optional_int(options.get("random_state")),
        verbose=bool(options.get("verbose", False)),
    )
    if params.encode_batch_size <= 0:
        raise ValueError("bertopic_kmeans params.encode_batch_size must be > 0.")
    if params.umap_n_neighbors < 2:
        raise ValueError("bertopic_kmeans params.umap_n_neighbors must be >= 2.")
    if params.umap_n_components < 1:
        raise ValueError("bertopic_kmeans params.umap_n_components must be >= 1.")
    if params.kmeans_n_init < 1:
        raise ValueError("bertopic_kmeans params.kmeans_n_init must be >= 1.")
    if params.soft_temperature <= 0.0:
        raise ValueError("bertopic_kmeans params.soft_temperature must be > 0.")
    return params


def _parse_sentence_embedding_common(options: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_encoder_common(options)
    parsed.update(
        {
            "random_state": int(options.get("random_state", 0)),
            "soft_temperature": float(options.get("soft_temperature", 1.0)),
            "verbose": bool(options.get("verbose", False)),
        }
    )
    if parsed["encode_batch_size"] <= 0:
        raise ValueError("params.encode_batch_size must be > 0.")
    if parsed["soft_temperature"] <= 0.0:
        raise ValueError("params.soft_temperature must be > 0.")
    return parsed


def _validate_kmeans_init(value: str, *, param_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"kmeans++", "random"}:
        raise ValueError(f"{param_name} must be one of 'kmeans++' or 'random'.")
    return normalized


def parse_spherical_kmeans_params(options: dict[str, Any]) -> SphericalKMeansParams:
    params = SphericalKMeansParams(
        **_parse_sentence_embedding_common(options),
        n_init=int(options.get("n_init", 10)),
        max_iter=int(options.get("max_iter", 300)),
        tol=float(options.get("tol", 1e-4)),
        init=_validate_kmeans_init(options.get("init", "kmeans++"), param_name="init"),
    )
    _validate_iterative_params(
        "spherical_kmeans", params.n_init, params.max_iter, params.tol
    )
    return params


def parse_gaussian_kmeans_params(options: dict[str, Any]) -> GaussianKMeansParams:
    params = GaussianKMeansParams(
        **_parse_sentence_embedding_common(options),
        strip_terminal_normalize=_bool_option(
            options, "strip_terminal_normalize", default=True
        ),
        n_init=int(options.get("n_init", 10)),
        max_iter=int(options.get("max_iter", 300)),
        tol=float(options.get("tol", 1e-4)),
        init=_validate_kmeans_init(options.get("init", "kmeans++"), param_name="init"),
    )
    _validate_iterative_params(
        "gaussian_kmeans", params.n_init, params.max_iter, params.tol
    )
    return params


def parse_movmf_params(options: dict[str, Any]) -> MovMFParams:
    init = str(options.get("init", "spherical_kmeans")).strip().lower()
    if init not in {"spherical_kmeans", "random"}:
        raise ValueError(
            "movmf params.init must be one of 'spherical_kmeans' or 'random'."
        )
    params = MovMFParams(
        **_parse_sentence_embedding_common(options),
        n_init=int(options.get("n_init", 5)),
        max_iter=int(options.get("max_iter", 100)),
        tol=float(options.get("tol", 1e-4)),
        init=init,
        min_kappa=float(options.get("min_kappa", 1e-3)),
        max_kappa=float(options.get("max_kappa", 1e4)),
    )
    _validate_iterative_params("movmf", params.n_init, params.max_iter, params.tol)
    if params.min_kappa <= 0.0:
        raise ValueError("movmf params.min_kappa must be > 0.")
    if params.max_kappa < params.min_kappa:
        raise ValueError("movmf params.max_kappa must be >= min_kappa.")
    return params


def parse_gaussian_mixture_params(options: dict[str, Any]) -> GaussianMixtureParams:
    init = str(options.get("init", "kmeans")).strip().lower()
    if init not in {"kmeans", "k-means++", "random", "random_from_data"}:
        raise ValueError(
            "gaussian_mixture params.init must be one of 'kmeans', "
            "'k-means++', 'random', or 'random_from_data'."
        )
    covariance_type = str(options.get("covariance_type", "diag")).strip().lower()
    if covariance_type not in {"full", "tied", "diag", "spherical"}:
        raise ValueError(
            "gaussian_mixture params.covariance_type must be one of "
            "'full', 'tied', 'diag', or 'spherical'."
        )
    params = GaussianMixtureParams(
        **_parse_sentence_embedding_common(options),
        strip_terminal_normalize=_bool_option(
            options, "strip_terminal_normalize", default=True
        ),
        n_init=int(options.get("n_init", 5)),
        max_iter=int(options.get("max_iter", 100)),
        tol=float(options.get("tol", 1e-4)),
        init=init,
        covariance_type=covariance_type,
        reg_covar=float(options.get("reg_covar", 1e-6)),
    )
    _validate_iterative_params(
        "gaussian_mixture", params.n_init, params.max_iter, params.tol
    )
    if params.reg_covar < 0.0:
        raise ValueError("gaussian_mixture params.reg_covar must be >= 0.")
    return params


def _validate_iterative_params(
    runner: str,
    n_init: int,
    max_iter: int,
    tol: float,
) -> None:
    if n_init < 1:
        raise ValueError(f"{runner} params.n_init must be >= 1.")
    if max_iter < 1:
        raise ValueError(f"{runner} params.max_iter must be >= 1.")
    if tol < 0.0:
        raise ValueError(f"{runner} params.tol must be >= 0.")


def normalize_baseline_params(
    runner: str,
    params: Mapping[str, Any] | None,
) -> BaselineParams:
    raw = {} if params is None else dict(params)
    runner_key = str(runner).strip().lower()

    if runner_key == "bleilda":
        return parse_bleilda_params(raw)
    if runner_key == "sam":
        return parse_sam_params(raw)
    if runner_key == "sam_tf":
        return parse_sam_tf_params(raw)
    if runner_key == "ctm":
        return parse_ctm_params(raw)
    if runner_key == "senclu":
        return parse_senclu_params(raw)
    if runner_key == "gaussianlda":
        return parse_gaussianlda_params(raw)
    if runner_key == "mvtm":
        return parse_mvtm_params(raw)
    if runner_key == "etm":
        return parse_etm_params(raw)
    if runner_key == "sentence_gaussianlda":
        return parse_sentence_gaussianlda_params(raw)
    if runner_key == "sentlda":
        return parse_sentlda_params(raw)
    if runner_key == "bertopic_kmeans":
        return parse_bertopic_kmeans_params(raw)
    if runner_key == "spherical_kmeans":
        return parse_spherical_kmeans_params(raw)
    if runner_key == "gaussian_kmeans":
        return parse_gaussian_kmeans_params(raw)
    if runner_key == "movmf":
        return parse_movmf_params(raw)
    if runner_key == "gaussian_mixture":
        return parse_gaussian_mixture_params(raw)
    return raw


def baseline_params_to_options(params: BaselineParams | None) -> dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    if is_dataclass(params):
        return dict(asdict(params))
    raise TypeError(f"Unsupported baseline params type: {type(params).__name__}")


def baseline_params_to_variant(params: BaselineParams | None) -> str:
    options = baseline_params_to_options(params)
    if not options:
        return "default"

    omit_defaults = {
        "encoder_backend": "sentence_transformers",
        "pooling": None,
        "encode_prompt": None,
        "encode_prompt_name": None,
        "model_kwargs": {},
        "tokenizer_kwargs": {},
        "normalize_embeddings": None,
        "truncate_dim": None,
        "covariance_type": "full",
    }
    if isinstance(params, (CtmParams, SenCluParams)):
        omit_defaults["encode_batch_size"] = 128
    if isinstance(params, CtmParams):
        options.pop("use_custom_embeddings", None)
    options = {
        key: value
        for key, value in options.items()
        if key not in omit_defaults or value != omit_defaults[key]
    }
    if not options:
        return "default"

    def _stringify(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if value is None:
            return "none"
        return str(value)

    items = sorted((str(key), _stringify(value)) for key, value in options.items())
    return "__".join(f"{key}={value}" for key, value in items)

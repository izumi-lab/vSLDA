from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

_SLUG_RE = re.compile(r"[^a-z0-9]+")

MODEL_ALIASES: dict[str, str] = {
    "sentence-transformers/all-minilm-l6-v2": "minilm",
    "all-minilm-l6-v2": "minilm",
    "sentence-transformers/all-mpnet-base-v2": "mpnet",
    "all-mpnet-base-v2": "mpnet",
    "baai/bge-base-en-v1.5": "bge",
    "bge-base-en-v1.5": "bge",
    "cl-nagoya/ruri-v3-130m": "ruri",
    "ruri-v3-130m": "ruri",
    "usif": "usif",
}

DEFAULT_MODEL_BY_EMBEDDING_VARIANT: dict[str, str] = {
    "minilm": "sentence-transformers/all-minilm-l6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "baai/bge-base-en-v1.5",
    "ruri": "cl-nagoya/ruri-v3-130m",
    "usif": "usif",
}

USIF_MODELS = {"usif"}
MODEL_DEFAULT_ENCODE_BATCH_SIZES: dict[str, int] = {
    "usif": 128,
}
MODEL_DEFAULT_KWARGS: dict[str, dict[str, Any]] = {
    "usif": {
        "word2vec": "glove-wiki-gigaword-100",
        "word_probability_source": "train",
        "component_policy": "auto",
    },
}


@dataclass(frozen=True)
class ResolvedEncoderSettings:
    model_name: str
    backend: str
    pooling: str | None
    encode_prefix: str | None
    encode_prompt: str | None
    encode_prompt_name: str | None
    encode_batch_size: int | None
    model_kwargs: dict[str, Any]
    tokenizer_kwargs: dict[str, Any]
    normalize_embeddings: bool | None
    truncate_dim: int | None
    embedding_variant: str

    def payload(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "pooling": self.pooling,
            "encode_prefix": self.encode_prefix,
            "encode_prompt": self.encode_prompt,
            "encode_prompt_name": self.encode_prompt_name,
            "encode_batch_size": self.encode_batch_size,
            "model_kwargs": dict(self.model_kwargs),
            "tokenizer_kwargs": dict(self.tokenizer_kwargs),
            "normalize_embeddings": self.normalize_embeddings,
            "truncate_dim": self.truncate_dim,
            "embedding_variant": self.embedding_variant,
        }


def _model_key(model_name: str) -> str:
    return str(model_name).strip().lower()


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "encoder"


def encoder_model_alias(model_name: str) -> str:
    key = _model_key(model_name)
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    tail = str(model_name).rstrip("/").split("/")[-1]
    return _slugify(tail)


def embedding_variant_base(embedding_variant: str) -> str:
    variant = str(embedding_variant).strip().lower()
    for suffix in ("_raw", "_norm"):
        if variant.endswith(suffix):
            return variant[: -len(suffix)]
    return variant


def default_encoder_model_for_embedding_variant(
    embedding_variant: str,
) -> str | None:
    return DEFAULT_MODEL_BY_EMBEDDING_VARIANT.get(
        embedding_variant_base(embedding_variant)
    )


def normalize_encoder_backend(model_name: str, backend: str | None) -> str:
    normalized = str(backend or "auto").strip().lower().replace("-", "_")
    key = _model_key(model_name)
    if normalized == "auto":
        if key in USIF_MODELS:
            return "usif"
        return "sentence_transformers"
    if normalized in {"sentence_transformer", "sentence_transformers", "st"}:
        return "sentence_transformers"
    if normalized in {"simcse", "transformers_pooler"}:
        return "simcse"
    if normalized in {"usif", "u_sif"}:
        return "usif"
    raise ValueError(
        "encoder.backend must be one of auto, sentence_transformers, simcse, or usif."
    )


def resolve_encoder_settings(
    *,
    model_name: str,
    backend: str | None = "auto",
    pooling: str | None = None,
    encode_prefix: str | None = None,
    encode_prompt: str | None = None,
    encode_prompt_name: str | None = None,
    encode_batch_size: int | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    normalize_embeddings: bool | None = None,
    truncate_dim: int | None = None,
) -> ResolvedEncoderSettings:
    resolved_backend = normalize_encoder_backend(model_name, backend)
    resolved_pooling = None if pooling is None else str(pooling).strip().lower()
    resolved_prefix = None if encode_prefix in {None, ""} else str(encode_prefix)
    resolved_prompt = None if encode_prompt in {None, ""} else str(encode_prompt)
    resolved_prompt_name = (
        None if encode_prompt_name in {None, ""} else str(encode_prompt_name)
    )
    if resolved_prefix is not None and resolved_prompt is not None:
        raise ValueError(
            "encoder.encode_prefix and encoder.encode_prompt are mutually exclusive."
        )
    if resolved_prompt is not None and resolved_prompt_name is not None:
        raise ValueError(
            "encoder.encode_prompt and encoder.encode_prompt_name are mutually exclusive."
        )

    key = _model_key(model_name)
    resolved_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    if key in MODEL_DEFAULT_KWARGS:
        resolved_model_kwargs = {
            **MODEL_DEFAULT_KWARGS[key],
            **resolved_model_kwargs,
        }
    if resolved_backend == "simcse":
        if resolved_pooling == "cls_before_pooler":
            resolved_pooling = "cls"
        if resolved_pooling is None:
            resolved_pooling = "pooler"
        if resolved_pooling not in {"cls", "pooler"}:
            raise ValueError("simcse pooling must be 'cls' or 'pooler'.")

    resolved_encode_batch_size = (
        None if encode_batch_size is None else int(encode_batch_size)
    )
    if resolved_encode_batch_size is None:
        resolved_encode_batch_size = MODEL_DEFAULT_ENCODE_BATCH_SIZES.get(key)
    if resolved_encode_batch_size is None and resolved_backend == "simcse":
        resolved_encode_batch_size = 128
    if resolved_encode_batch_size is not None and resolved_encode_batch_size <= 0:
        raise ValueError("encoder.encode_batch_size must be > 0.")

    resolved_truncate_dim = None if truncate_dim is None else int(truncate_dim)
    if resolved_truncate_dim is not None and resolved_truncate_dim <= 0:
        raise ValueError("encoder.truncate_dim must be > 0.")

    return ResolvedEncoderSettings(
        model_name=str(model_name),
        backend=resolved_backend,
        pooling=resolved_pooling,
        encode_prefix=resolved_prefix,
        encode_prompt=resolved_prompt,
        encode_prompt_name=resolved_prompt_name,
        encode_batch_size=resolved_encode_batch_size,
        model_kwargs=resolved_model_kwargs,
        tokenizer_kwargs={} if tokenizer_kwargs is None else dict(tokenizer_kwargs),
        normalize_embeddings=normalize_embeddings,
        truncate_dim=resolved_truncate_dim,
        embedding_variant=encoder_model_alias(model_name),
    )


def encoder_config_payload(
    *,
    model_name: str,
    backend: str | None = "auto",
    pooling: str | None = None,
    encode_prefix: str | None = None,
    encode_prompt: str | None = None,
    encode_prompt_name: str | None = None,
    encode_batch_size: int | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    normalize_embeddings: bool | None = None,
    truncate_dim: int | None = None,
) -> dict[str, Any]:
    return resolve_encoder_settings(
        model_name=model_name,
        backend=backend,
        pooling=pooling,
        encode_prefix=encode_prefix,
        encode_prompt=encode_prompt,
        encode_prompt_name=encode_prompt_name,
        encode_batch_size=encode_batch_size,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        normalize_embeddings=normalize_embeddings,
        truncate_dim=truncate_dim,
    ).payload()

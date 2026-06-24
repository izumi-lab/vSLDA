from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from src.utils.encoder_profiles import resolve_encoder_settings

SentenceTransformer: Any | None = None
models: Any | None = None


def _load_sentence_transformers() -> tuple[Any, Any]:
    global SentenceTransformer, models
    if SentenceTransformer is not None and models is not None:
        return SentenceTransformer, models
    try:
        from sentence_transformers import SentenceTransformer as _SentenceTransformer
        from sentence_transformers import models as _models
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformer backends require ML dependencies. "
            "Install them with: poetry install --with ml"
        ) from exc
    if SentenceTransformer is None:
        SentenceTransformer = _SentenceTransformer
    models = _models
    return SentenceTransformer, models


def _model_kwargs_for_sentence_transformer(
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(model_kwargs)
    if "torch_dtype" in resolved:
        torch_dtype = resolved.pop("torch_dtype")
        resolved.setdefault("dtype", torch_dtype)
    for key in ("dtype",):
        value = resolved.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().removeprefix("torch.")
        dtype_names = {
            "float16": "float16",
            "fp16": "float16",
            "half": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
        }
        dtype_name = dtype_names.get(normalized)
        if dtype_name is None:
            continue
        import torch

        resolved[key] = getattr(torch, dtype_name)
    return resolved


class SentenceEncoder:
    """Thin wrapper around SentenceTransformer for sentence embeddings.

    This class centralizes model loading and optional normalization.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        normalize: bool = False,
        encode_prefix: Optional[str] = None,
        backend: str = "auto",
        pooling: Optional[str] = None,
        encode_prompt: Optional[str] = None,
        encode_prompt_name: Optional[str] = None,
        encode_batch_size: Optional[int] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        normalize_embeddings: Optional[bool] = None,
        truncate_dim: Optional[int] = None,
        strip_terminal_normalize: bool = True,
        cache_folder: Optional[str] = None,
    ) -> None:
        """Initialize the sentence encoder.

        Args:
            model_name: Name or path of the sentence-transformers model.
            device: Device identifier (e.g., "cuda", "cuda:0", "cpu").
            normalize: If True, L2-normalize embeddings on encode().
            encode_prefix:
                Optional text prefix added to each input sentence at encode time.
                When empty/None, inputs are passed through unchanged.
            backend:
                Encoder backend. "auto" selects a known profile, otherwise use
                "sentence_transformers", "simcse", or "usif".
            pooling: Optional pooling profile for non-SentenceTransformers backends.
            encode_prompt:
                Optional SentenceTransformers prompt passed to encode().
            encode_prompt_name:
                Optional SentenceTransformers prompt name passed to encode().
            encode_batch_size:
                Optional default batch size passed to SentenceTransformer.encode
                when callers do not specify batch_size explicitly.
            model_kwargs: Optional keyword arguments for model construction.
            tokenizer_kwargs: Optional tokenizer keyword arguments.
            normalize_embeddings:
                Optional backend-level normalization setting. Defaults to normalize.
            truncate_dim: Optional embedding truncation dimension.
            strip_terminal_normalize:
                If True (default) and the last SentenceTransformer module is Normalize,
                rebuild the model without that terminal module.
            cache_folder: Optional local directory to store downloaded models.
        """
        settings = resolve_encoder_settings(
            model_name=model_name,
            backend=backend,
            pooling=pooling,
            encode_prefix=encode_prefix,
            encode_prompt=encode_prompt,
            encode_prompt_name=encode_prompt_name,
            encode_batch_size=encode_batch_size,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            normalize_embeddings=(
                normalize if normalize_embeddings is None else normalize_embeddings
            ),
            truncate_dim=truncate_dim,
        )
        self._settings = settings
        self._backend = settings.backend
        self._pooling = settings.pooling
        self._device = device
        self._stripped_terminal_normalize = False
        if self._backend == "sentence_transformers":
            sentence_transformer_cls, sentence_transformer_models = (
                _load_sentence_transformers()
            )
            init_kwargs: dict[str, Any] = {
                "device": device,
                "cache_folder": cache_folder,
            }
            if settings.model_kwargs:
                init_kwargs["model_kwargs"] = _model_kwargs_for_sentence_transformer(
                    settings.model_kwargs
                )
            if settings.tokenizer_kwargs:
                init_kwargs["tokenizer_kwargs"] = settings.tokenizer_kwargs
            if settings.truncate_dim is not None:
                init_kwargs["truncate_dim"] = settings.truncate_dim
            model = sentence_transformer_cls(model_name, **init_kwargs)
            if strip_terminal_normalize:
                module_list = list(model._modules.values())
                if module_list and isinstance(
                    module_list[-1], sentence_transformer_models.Normalize
                ):
                    module_list.pop()
                    model = sentence_transformer_cls(
                        modules=module_list,
                        device=device,
                    )
                    self._stripped_terminal_normalize = True
            self._model = model
            self._tokenizer = None
        elif self._backend == "simcse":
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_folder,
                **settings.tokenizer_kwargs,
            )
            self._model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_folder,
                **settings.model_kwargs,
            )
            import torch

            resolved_device = device
            if resolved_device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = resolved_device
            self._model.to(resolved_device)
            self._model.eval()
        elif self._backend == "usif":
            from src.utils.usif_encoder import UsifSentenceEncoder

            usif_kwargs = dict(settings.model_kwargs)
            usif_kwargs["normalize_embeddings"] = bool(settings.normalize_embeddings)
            self._model = UsifSentenceEncoder(**usif_kwargs)
            self._tokenizer = None
        else:
            raise ValueError(f"Unsupported encoder backend: {self._backend}")

        self._normalize = (
            bool(settings.normalize_embeddings) and self._backend != "usif"
        )
        self._encode_prefix = settings.encode_prefix
        self._encode_prompt = settings.encode_prompt
        self._encode_prompt_name = settings.encode_prompt_name
        self._truncate_dim = settings.truncate_dim
        self._encode_batch_size = settings.encode_batch_size

    def encode(self, sentences: Sequence[str], **encode_kwargs) -> np.ndarray:
        """Encode a batch of sentences into embeddings.

        Args:
            sentences: Iterable of sentence strings.
            **encode_kwargs: Extra keyword arguments passed to
                SentenceTransformer.encode (e.g., batch_size, show_progress_bar).

        Returns:
            2D numpy array with shape (n_sentences, embedding_dim).
        """
        encode_inputs: Sequence[str]
        if self._encode_prefix:
            prefix = self._encode_prefix
            encode_inputs = [
                text if text.startswith(prefix) else f"{prefix}{text}"
                for text in sentences
            ]
        else:
            encode_inputs = sentences

        if "batch_size" not in encode_kwargs and self._encode_batch_size is not None:
            encode_kwargs["batch_size"] = self._encode_batch_size

        if self._backend == "sentence_transformers":
            if self._encode_prompt is not None and "prompt" not in encode_kwargs:
                encode_kwargs["prompt"] = self._encode_prompt
            if (
                self._encode_prompt_name is not None
                and "prompt_name" not in encode_kwargs
            ):
                encode_kwargs["prompt_name"] = self._encode_prompt_name
            if self._truncate_dim is not None and "truncate_dim" not in encode_kwargs:
                encode_kwargs["truncate_dim"] = self._truncate_dim
            embeddings: np.ndarray = self._model.encode(
                encode_inputs,
                convert_to_numpy=True,
                **encode_kwargs,
            )
        elif self._backend == "simcse":
            embeddings = self._encode_simcse(encode_inputs, **encode_kwargs)
        else:
            embeddings = self._model.encode(encode_inputs, **encode_kwargs)

        if self._backend == "usif" and self._truncate_dim is not None:
            embeddings = embeddings[:, : self._truncate_dim]

        if self._normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0.0] = 1.0
            embeddings = embeddings / norms

        return embeddings

    @property
    def requires_fit(self) -> bool:
        return self._backend == "usif"

    @property
    def accepts_tokenized(self) -> bool:
        return self._backend == "usif"

    def fit_tokenized(self, tokenized_sentences: Sequence[Sequence[str]]) -> None:
        if self._backend == "usif":
            self._model.fit_tokenized(tokenized_sentences)

    def encode_tokenized(
        self,
        tokenized_sentences: Sequence[Sequence[str]],
        **encode_kwargs: Any,
    ) -> np.ndarray:
        if self._backend == "usif":
            embeddings = self._model.encode_tokenized(
                tokenized_sentences, **encode_kwargs
            )
            if self._truncate_dim is not None:
                embeddings = embeddings[:, : self._truncate_dim]
            return embeddings
        raw = [
            " ".join(str(token) for token in tokens) for tokens in tokenized_sentences
        ]
        return self.encode(raw, **encode_kwargs)

    def _encode_simcse(self, sentences: Sequence[str], **encode_kwargs) -> np.ndarray:
        import torch

        if self._encode_prompt is not None:
            inputs = [f"{self._encode_prompt}{text}" for text in sentences]
        else:
            inputs = list(sentences)
        batch_size = int(encode_kwargs.pop("batch_size", self._encode_batch_size or 32))
        show_progress_bar = bool(encode_kwargs.pop("show_progress_bar", False))
        if encode_kwargs:
            unsupported = ", ".join(sorted(encode_kwargs))
            raise TypeError(f"Unsupported simcse encode kwargs: {unsupported}")
        if not inputs:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)

        iterator: Sequence[list[str]]
        batches = [
            inputs[start : start + batch_size]
            for start in range(0, len(inputs), batch_size)
        ]
        if show_progress_bar:
            from tqdm.auto import tqdm

            iterator = tqdm(batches, desc="Encoding")
        else:
            iterator = batches

        rows: list[np.ndarray] = []
        for batch in iterator:
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self._model(**encoded)
            if self._pooling not in {None, "cls", "pooler"}:
                raise ValueError("simcse pooling must be 'cls' or 'pooler'.")
            if self._pooling == "cls":
                pooled = output.last_hidden_state[:, 0]
            else:
                pooled = getattr(output, "pooler_output", None)
                if pooled is None:
                    raise ValueError(
                        "simcse pooling='pooler' requires model output pooler_output."
                    )
            rows.append(pooled.detach().cpu().numpy())
        embeddings = np.vstack(rows).astype(np.float32, copy=False)
        if self._truncate_dim is not None:
            embeddings = embeddings[:, : self._truncate_dim]
        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of sentence embeddings."""
        if self._backend == "sentence_transformers":
            return self._model.get_sentence_embedding_dimension()
        if self._backend == "usif":
            dim = int(self._model.get_sentence_embedding_dimension())
            if self._truncate_dim is not None:
                return min(dim, self._truncate_dim)
            return dim
        dim = int(getattr(self._model.config, "hidden_size"))
        if self._truncate_dim is not None:
            return min(dim, self._truncate_dim)
        return dim

    def get_sentence_embedding_dimension(self) -> int:
        """Backward-compatible alias for embedding_dimension."""
        return self.embedding_dimension

    @property
    def encoder_config(self) -> dict[str, Any]:
        """Return the resolved encoder configuration used by this instance."""
        return self._settings.payload()

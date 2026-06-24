from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.integration
sentence_transformers = pytest.importorskip("sentence_transformers")
models = sentence_transformers.models

from src.utils.encoder import SentenceEncoder  # noqa: E402
from src.utils.encoder_profiles import (  # noqa: E402
    encoder_model_alias,
    resolve_encoder_settings,
)


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        module_list = kwargs.get("modules")
        if module_list is None:
            self._modules: dict[str, object] = {}
        else:
            self._modules = {str(idx): module for idx, module in enumerate(module_list)}

    def encode(self, inputs, **kwargs) -> np.ndarray:
        self.calls.append((list(inputs), dict(kwargs)))
        return np.ones((len(inputs), 2), dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return 2


class _FakeSimcseTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "_FakeSimcseTokenizer":
        return cls()

    def __call__(self, batch, **kwargs):
        import torch

        size = len(batch)
        return {
            "input_ids": torch.zeros((size, 2), dtype=torch.long),
            "attention_mask": torch.ones((size, 2), dtype=torch.long),
        }


class _FakeSimcseModel:
    def __init__(self, *, include_pooler: bool = True) -> None:
        self.include_pooler = include_pooler
        self.config = SimpleNamespace(hidden_size=2)

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "_FakeSimcseModel":
        return cls()

    def to(self, device: str) -> "_FakeSimcseModel":
        self.device = device
        return self

    def eval(self) -> "_FakeSimcseModel":
        return self

    def __call__(self, **encoded):
        import torch

        size = int(encoded["input_ids"].shape[0])
        last_hidden = torch.zeros((size, 2, 2), dtype=torch.float32)
        pooler = torch.zeros((size, 2), dtype=torch.float32)
        for idx in range(size):
            last_hidden[idx, 0, :] = torch.tensor(
                [float(idx + 3), float(idx + 4)], dtype=torch.float32
            )
            pooler[idx, :] = torch.tensor(
                [float(idx + 30), float(idx + 40)], dtype=torch.float32
            )
        if self.include_pooler:
            return SimpleNamespace(
                last_hidden_state=last_hidden,
                pooler_output=pooler,
            )
        return SimpleNamespace(last_hidden_state=last_hidden)


def _patch_simcse_backend(
    monkeypatch: pytest.MonkeyPatch, *, include_pooler: bool = True
) -> _FakeSimcseModel:
    model = _FakeSimcseModel(include_pooler=include_pooler)

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _FakeSimcseTokenizer(),
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda *args, **kwargs: model,
    )
    return model


def test_sentence_encoder_uses_default_encode_batch_size(monkeypatch) -> None:
    created_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        model = _FakeSentenceTransformer(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder(
        "fake-model",
        device="cpu",
        encode_batch_size=16,
        strip_terminal_normalize=False,
    )

    out = encoder.encode(["alpha", "beta"])

    assert out.shape == (2, 2)
    assert created_models[0].calls == [
        (
            ["alpha", "beta"],
            {"convert_to_numpy": True, "batch_size": 16},
        )
    ]


def test_encoder_model_aliases_match_supported_profiles() -> None:
    assert encoder_model_alias("sentence-transformers/all-minilm-l6-v2") == "minilm"
    assert encoder_model_alias("sentence-transformers/all-mpnet-base-v2") == "mpnet"
    assert encoder_model_alias("baai/bge-base-en-v1.5") == "bge"
    assert encoder_model_alias("cl-nagoya/ruri-v3-130m") == "ruri"
    assert encoder_model_alias("usif") == "usif"
    assert encoder_model_alias("all-minilm-l6-v2") == "minilm"
    assert encoder_model_alias("all-mpnet-base-v2") == "mpnet"
    assert encoder_model_alias("bge-base-en-v1.5") == "bge"
    assert encoder_model_alias("ruri-v3-130m") == "ruri"


def test_sentence_encoder_explicit_batch_size_overrides_default(monkeypatch) -> None:
    created_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        model = _FakeSentenceTransformer(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder(
        "fake-model",
        device="cpu",
        encode_batch_size=16,
        strip_terminal_normalize=False,
    )

    encoder.encode(["alpha"], batch_size=4)

    assert created_models[0].calls == [
        (
            ["alpha"],
            {"convert_to_numpy": True, "batch_size": 4},
        )
    ]


def test_sentence_encoder_converts_legacy_torch_dtype_to_dtype(monkeypatch) -> None:
    import torch

    created_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        model = _FakeSentenceTransformer(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    SentenceEncoder(
        "fake-model",
        device="cpu",
        model_kwargs={"torch_dtype": "float16"},
        strip_terminal_normalize=False,
    )

    assert created_models[0].init_kwargs["model_kwargs"] == {"dtype": torch.float16}


def test_sentence_encoder_unprofiled_e5_name_does_not_add_prefix(
    monkeypatch,
) -> None:
    created_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        model = _FakeSentenceTransformer(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder(
        "intfloat/e5-base-v2",
        device="cpu",
        strip_terminal_normalize=False,
    )
    encoder.encode(["alpha"])

    assert created_models[0].calls[0][0] == ["alpha"]
    assert encoder.encoder_config["embedding_variant"] == "e5-base-v2"


def test_sentence_encoder_unprofiled_e5_prompt_name_is_passed_through(
    monkeypatch,
) -> None:
    created_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        model = _FakeSentenceTransformer(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder(
        "intfloat/e5-base-v2",
        device="cpu",
        encode_prompt_name="query",
        strip_terminal_normalize=False,
    )
    encoder.encode(["alpha"])

    assert created_models[0].calls[0][0] == ["alpha"]
    assert created_models[0].calls[0][1]["prompt_name"] == "query"
    assert encoder.encoder_config["encode_prefix"] is None


def test_sentence_encoder_passes_prompt_name_and_model_kwargs(monkeypatch) -> None:
    import torch

    created_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        model = _FakeSentenceTransformer(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder(
        "sentence-transformers/all-mpnet-base-v2",
        device="cpu",
        encode_prompt_name="query",
        encode_batch_size=4,
        model_kwargs={"attn_implementation": "flash_attention_2", "dtype": "float16"},
        tokenizer_kwargs={"padding_side": "left"},
        strip_terminal_normalize=False,
    )
    encoder.encode(["alpha"])

    assert created_models[0].init_kwargs["model_kwargs"] == {
        "attn_implementation": "flash_attention_2",
        "dtype": torch.float16,
    }
    assert created_models[0].init_kwargs["tokenizer_kwargs"] == {"padding_side": "left"}
    assert created_models[0].calls[0][1]["prompt_name"] == "query"
    assert created_models[0].calls[0][1]["batch_size"] == 4
    assert encoder.encoder_config["embedding_variant"] == "mpnet"
    assert encoder.encoder_config["model_kwargs"]["dtype"] == "float16"


def test_sentence_encoder_does_not_add_implicit_prompt(monkeypatch) -> None:
    created_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        model = _FakeSentenceTransformer(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder(
        "sentence-transformers/all-mpnet-base-v2",
        device="cpu",
        strip_terminal_normalize=False,
    )
    encoder.encode(["alpha"])

    assert "prompt" not in created_models[0].calls[0][1]
    assert "prompt_name" not in created_models[0].calls[0][1]


def test_sentence_encoder_passes_custom_prompt(monkeypatch) -> None:
    created_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        model = _FakeSentenceTransformer(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder(
        "sentence-transformers/all-mpnet-base-v2",
        device="cpu",
        encode_prompt="Instruct: Given a document, identify its topic\nQuery: ",
        strip_terminal_normalize=False,
    )
    encoder.encode(["alpha"])

    assert created_models[0].calls[0][1]["prompt"] == (
        "Instruct: Given a document, identify its topic\nQuery: "
    )


def test_sentence_encoder_rejects_prefix_and_prompt() -> None:
    with pytest.raises(ValueError, match="encode_prefix"):
        SentenceEncoder(
            "fake-model",
            device="cpu",
            encode_prefix="query: ",
            encode_prompt="prompt: ",
            strip_terminal_normalize=False,
        )


def test_sentence_encoder_strips_terminal_normalize_module(monkeypatch) -> None:
    first = _FakeSentenceTransformer("fake-model")
    first._modules = {
        "0": object(),
        "1": models.Normalize(),
    }
    rebuilt_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        if "modules" not in kwargs:
            return first
        model = _FakeSentenceTransformer(*args, **kwargs)
        rebuilt_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder("fake-model", device="cpu")

    assert encoder._stripped_terminal_normalize is True
    assert len(rebuilt_models) == 1
    assert list(rebuilt_models[0]._modules.values()) == [first._modules["0"]]


def test_sentence_encoder_keeps_terminal_normalize_when_configured(
    monkeypatch,
) -> None:
    first = _FakeSentenceTransformer("fake-model")
    first._modules = {
        "0": object(),
        "1": models.Normalize(),
    }
    rebuilt_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        if "modules" not in kwargs:
            return first
        model = _FakeSentenceTransformer(*args, **kwargs)
        rebuilt_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder(
        "fake-model",
        device="cpu",
        strip_terminal_normalize=False,
    )

    assert encoder._stripped_terminal_normalize is False
    assert rebuilt_models == []
    assert list(encoder._model._modules.values()) == list(first._modules.values())


def test_sentence_encoder_does_not_strip_non_terminal_normalize(monkeypatch) -> None:
    first = _FakeSentenceTransformer("fake-model")
    terminal = object()
    first._modules = {
        "0": models.Normalize(),
        "1": terminal,
    }
    rebuilt_models: list[_FakeSentenceTransformer] = []

    def _fake_sentence_transformer(*args, **kwargs) -> _FakeSentenceTransformer:
        if "modules" not in kwargs:
            return first
        model = _FakeSentenceTransformer(*args, **kwargs)
        rebuilt_models.append(model)
        return model

    monkeypatch.setattr(
        "src.utils.encoder.SentenceTransformer",
        _fake_sentence_transformer,
    )

    encoder = SentenceEncoder("fake-model", device="cpu")

    assert encoder._stripped_terminal_normalize is False
    assert rebuilt_models == []
    assert list(encoder._model._modules.values()) == [first._modules["0"], terminal]


def test_encoder_profiles_use_usif_default_batch_size() -> None:
    usif = resolve_encoder_settings(model_name="usif")

    assert usif.encode_batch_size == 128


def test_encoder_profiles_normalize_simcse_pooling_alias() -> None:
    resolved = resolve_encoder_settings(
        model_name="princeton-nlp/sup-simcse-roberta-base",
        backend="simcse",
        pooling="cls_before_pooler",
    )

    assert resolved.pooling == "cls"


def test_encoder_profiles_keep_explicit_simcse_pooler() -> None:
    resolved = resolve_encoder_settings(
        model_name="princeton-nlp/sup-simcse-roberta-base",
        backend="simcse",
        pooling="pooler",
    )

    assert resolved.pooling == "pooler"


def test_encoder_profiles_reject_invalid_simcse_pooling() -> None:
    with pytest.raises(ValueError, match="simcse pooling"):
        resolve_encoder_settings(
            model_name="princeton-nlp/sup-simcse-roberta-base",
            backend="simcse",
            pooling="mean",
        )


def test_encoder_profiles_keep_explicit_batch_size() -> None:
    resolved = resolve_encoder_settings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_batch_size=4,
    )

    assert resolved.encode_batch_size == 4


def test_simcse_pooling_cls_uses_last_hidden_state(monkeypatch) -> None:
    _patch_simcse_backend(monkeypatch)
    encoder = SentenceEncoder(
        "princeton-nlp/sup-simcse-roberta-base",
        backend="simcse",
        pooling="cls",
        device="cpu",
    )

    embeddings = encoder.encode(["alpha", "beta"])

    np.testing.assert_allclose(
        embeddings,
        np.asarray([[3.0, 4.0], [4.0, 5.0]], dtype=np.float32),
    )


def test_simcse_default_pooling_uses_pooler_output(monkeypatch) -> None:
    _patch_simcse_backend(monkeypatch)
    encoder = SentenceEncoder(
        "princeton-nlp/sup-simcse-roberta-base",
        backend="simcse",
        device="cpu",
    )

    embeddings = encoder.encode(["alpha", "beta"])

    np.testing.assert_allclose(
        embeddings,
        np.asarray([[30.0, 40.0], [31.0, 41.0]], dtype=np.float32),
    )
    assert encoder.encoder_config["pooling"] == "pooler"


def test_simcse_pooling_alias_uses_last_hidden_state(monkeypatch) -> None:
    _patch_simcse_backend(monkeypatch)
    encoder = SentenceEncoder(
        "princeton-nlp/sup-simcse-roberta-base",
        backend="simcse",
        pooling="cls_before_pooler",
        device="cpu",
    )

    embeddings = encoder.encode(["alpha"])

    np.testing.assert_allclose(
        embeddings,
        np.asarray([[3.0, 4.0]], dtype=np.float32),
    )
    assert encoder.encoder_config["pooling"] == "cls"


def test_simcse_pooling_pooler_uses_pooler_output(monkeypatch) -> None:
    _patch_simcse_backend(monkeypatch)
    encoder = SentenceEncoder(
        "princeton-nlp/sup-simcse-roberta-base",
        backend="simcse",
        pooling="pooler",
        device="cpu",
    )

    embeddings = encoder.encode(["alpha", "beta"])

    np.testing.assert_allclose(
        embeddings,
        np.asarray([[30.0, 40.0], [31.0, 41.0]], dtype=np.float32),
    )


def test_simcse_pooling_pooler_requires_pooler_output(monkeypatch) -> None:
    _patch_simcse_backend(monkeypatch, include_pooler=False)
    encoder = SentenceEncoder(
        "princeton-nlp/sup-simcse-roberta-base",
        backend="simcse",
        pooling="pooler",
        device="cpu",
    )

    with pytest.raises(ValueError, match="pooler_output"):
        encoder.encode(["alpha"])


def test_simcse_normalize_embeddings_applies_after_pooling(monkeypatch) -> None:
    _patch_simcse_backend(monkeypatch)
    encoder = SentenceEncoder(
        "princeton-nlp/sup-simcse-roberta-base",
        backend="simcse",
        pooling="cls",
        normalize_embeddings=True,
        device="cpu",
    )

    embeddings = encoder.encode(["alpha"])

    np.testing.assert_allclose(
        embeddings,
        np.asarray([[0.6, 0.8]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

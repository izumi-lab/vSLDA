from __future__ import annotations

from .contracts import ModelArtifacts, ModelRunnerSpec, ModelRunRequest
from .vmf_sentence_lda import VMFLDATrainer


def __getattr__(name: str):
    if name in {"VMF_RUNNER", "get_model_runner_spec", "run_model_request"}:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_model_runner_spec(name: str) -> ModelRunnerSpec:
    from .registry import get_model_runner_spec as _get_model_runner_spec

    return _get_model_runner_spec(name)


def run_model_request(request: ModelRunRequest) -> ModelArtifacts:
    from .registry import run_model_request as _run_model_request

    return _run_model_request(request)


def get_trainer_class(name: str):
    name = name.lower()
    if name in {"vmf_sentence_lda"}:
        return VMFLDATrainer
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "ModelArtifacts",
    "ModelRunRequest",
    "ModelRunnerSpec",
    "VMF_RUNNER",
    "get_model_runner_spec",
    "get_trainer_class",
    "run_model_request",
    "VMFLDATrainer",
]

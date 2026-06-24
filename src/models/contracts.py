from __future__ import annotations

from src.core.contracts import RunSpec, TopicModelOutput
from src.core.runner_contracts import RunArtifacts as ModelArtifacts
from src.core.runner_contracts import RunCallable as ModelRunnerCallable
from src.core.runner_contracts import RunnerSpec as ModelRunnerSpec
from src.core.runner_contracts import RunRequest as ModelRunRequest

__all__ = [
    "ModelArtifacts",
    "ModelRunRequest",
    "ModelRunnerCallable",
    "ModelRunnerSpec",
    "RunSpec",
    "TopicModelOutput",
]

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MODEL_KIND_TOPIC_MODEL = "topic_model"
MODEL_KIND_CLUSTERING = "clustering"


@dataclass(frozen=True)
class RunRequest:
    name: str
    category: str
    dataset: str
    num_topics: int
    iteration: int
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunArtifacts:
    train_path: Path
    infer_path: Path
    extras: dict[str, Path] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Path]:
        return {
            "train_path": self.train_path,
            "infer_path": self.infer_path,
            **self.extras,
        }

    @property
    def train_dir(self) -> Path:
        return self.extras.get("train_dir", self.train_path.parent)

    @property
    def infer_dir(self) -> Path:
        return self.extras.get("infer_dir", self.infer_path.parent)

    def get_path(self, name: str) -> Path:
        if name == "train_path":
            return self.train_path
        if name == "infer_path":
            return self.infer_path
        if name not in self.extras:
            raise KeyError(f"Unknown artifact name: {name}")
        return self.extras[name]


RunCallable = Callable[[RunRequest], RunArtifacts]


@dataclass(frozen=True)
class RunnerSpec:
    key: str
    display_name: str
    family: str
    runner: RunCallable
    method_kind: str = MODEL_KIND_TOPIC_MODEL

    def __post_init__(self) -> None:
        if self.method_kind not in {MODEL_KIND_TOPIC_MODEL, MODEL_KIND_CLUSTERING}:
            raise ValueError(f"Unknown runner method_kind: {self.method_kind}")

    @property
    def is_topic_model(self) -> bool:
        return self.method_kind == MODEL_KIND_TOPIC_MODEL

    @property
    def is_clustering(self) -> bool:
        return self.method_kind == MODEL_KIND_CLUSTERING

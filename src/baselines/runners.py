from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.baselines.contracts import BaselineArtifacts, BaselineRunRequest
from src.baselines.registry import RUNNERS, get_runner_spec


def run_baseline(
    name: str,
    category: str,
    dataset: str,
    num_topics: int,
    iteration: int,
    **kwargs: Any,
) -> Dict[str, Path]:
    request = BaselineRunRequest(
        name=name,
        category=category,
        dataset=dataset,
        num_topics=num_topics,
        iteration=iteration,
        options=dict(kwargs),
    )
    return run_baseline_request(request).as_dict()


def run_baseline_request(request: BaselineRunRequest) -> BaselineArtifacts:
    """
    Run a borrowed baseline model end-to-end (train + infer).

    Args:
        request: Structured baseline execution request.
    """
    spec = get_runner_spec(request.name)
    artifacts = spec.runner(request)
    if not isinstance(artifacts, BaselineArtifacts):
        raise TypeError(
            f"Baseline runner '{spec.key}' must return BaselineArtifacts, "
            f"got {type(artifacts).__name__}."
        )
    return artifacts

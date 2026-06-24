from __future__ import annotations

from src.baselines.adapters import (
    run_bertopic_kmeans,
    run_bleilda,
    run_ctm,
    run_etm,
    run_gaussian_kmeans,
    run_gaussian_mixture,
    run_gaussianlda,
    run_movmf,
    run_mvtm,
    run_senclu,
    run_sentence_gaussianlda,
    run_sentlda,
    run_spherical_kmeans,
)
from src.baselines.contracts import BaselineRunnerCallable, BaselineRunnerSpec
from src.baselines.model_kinds import baseline_method_kind


def _runner_spec(
    *,
    key: str,
    display_name: str,
    family: str,
    runner: BaselineRunnerCallable,
) -> BaselineRunnerSpec:
    return BaselineRunnerSpec(
        key=key,
        display_name=display_name,
        family=family,
        runner=runner,
        method_kind=baseline_method_kind(key),
    )


RUNNERS: dict[str, BaselineRunnerSpec] = {
    "ctm": _runner_spec(
        key="ctm",
        display_name="Contextual TM",
        family="ctm",
        runner=run_ctm,
    ),
    "bleilda": _runner_spec(
        key="bleilda",
        display_name="Blei LDA",
        family="bleilda",
        runner=run_bleilda,
    ),
    "bertopic_kmeans": _runner_spec(
        key="bertopic_kmeans",
        display_name="BERTopic (UMAP + k-means)",
        family="bertopic_kmeans",
        runner=run_bertopic_kmeans,
    ),
    "gaussianlda": _runner_spec(
        key="gaussianlda",
        display_name="Gaussian LDA",
        family="gaussianlda",
        runner=run_gaussianlda,
    ),
    "etm": _runner_spec(
        key="etm",
        display_name="ETM",
        family="etm",
        runner=run_etm,
    ),
    "mvtm": _runner_spec(
        key="mvtm",
        display_name="MvTM",
        family="mvtm",
        runner=run_mvtm,
    ),
    "spherical_kmeans": _runner_spec(
        key="spherical_kmeans",
        display_name="Spherical k-means",
        family="spherical_kmeans",
        runner=run_spherical_kmeans,
    ),
    "gaussian_kmeans": _runner_spec(
        key="gaussian_kmeans",
        display_name="Gaussian k-means",
        family="gaussian_kmeans",
        runner=run_gaussian_kmeans,
    ),
    "movmf": _runner_spec(
        key="movmf",
        display_name="movMF",
        family="movmf",
        runner=run_movmf,
    ),
    "gaussian_mixture": _runner_spec(
        key="gaussian_mixture",
        display_name="Gaussian mixture",
        family="gaussian_mixture",
        runner=run_gaussian_mixture,
    ),
    "senclu": _runner_spec(
        key="senclu",
        display_name="SenClu",
        family="senclu",
        runner=run_senclu,
    ),
    "sentlda": _runner_spec(
        key="sentlda",
        display_name="sentLDA",
        family="sentlda",
        runner=run_sentlda,
    ),
    "sentence_gaussianlda": _runner_spec(
        key="sentence_gaussianlda",
        display_name="Sentence LDA",
        family="sentence_gaussianlda",
        runner=run_sentence_gaussianlda,
    ),
}


def get_runner_spec(name: str) -> BaselineRunnerSpec:
    if name not in RUNNERS:
        raise ValueError(f"Unknown baseline runner: {name}")
    return RUNNERS[name]

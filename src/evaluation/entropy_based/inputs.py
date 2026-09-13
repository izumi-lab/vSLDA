"""Document-topic matrix loading for the entropy-based evaluation task.

The persisted document-topic artifacts differ in "hardness" across models:
vMF stores the mean of per-sentence posteriors, sentLDA / Sentence Gaussian LDA /
SenClu store hard sentence-assignment counts, and MvTM / ETM additionally store a
soft ``<category>_doc_topic_soft.pkl``. ``doc_topic_source="soft_proxy"`` resolves
the softest representation available for every model so the entropy metrics are
comparable:

1. a soft document-topic file (``doc_topic_<split>_soft.pkl`` for vMF,
   ``infer/<category>_doc_topic_soft.pkl`` for baselines),
2. otherwise the per-document mean of persisted sentence posteriors
   (``infer/<category>_sentence_topic_soft.pkl``),
3. otherwise the hard document-topic file (``infer/<category>.pkl``).

What each model actually persists, and what ``soft_proxy`` resolves to:

===================================  ==============================  ==========================
model                                hard file content               ``soft_proxy`` resolves to
===================================  ==============================  ==========================
vmf                                  mean of sentence posteriors     soft file (identical)
sentlda, sentence_gaussianlda,       hard sentence-assignment counts sentence posterior mean
senclu
mvtm, etm                            word counts / soft theta        soft file
ctm                                  soft theta                      hard file (already soft)
gaussianlda                          word-level counts               normalized counts
bleilda                              gensim probs truncated <0.01    normalized probabilities
===================================  ==============================  ==========================

``doc_topic_source="hard"`` or ``"soft"`` forces one file and fails when absent.
``"auto"`` (the default) resolves per model: the manuscript's fold-in estimator
(``foldincounts``) for the vMF family (vMF Sentence LDA and MvTM / vLDA) and
``soft_proxy`` for the other baselines; the resolved value is what the condition
records as its ``doc_topic_source``.
``doc_topic_source="foldin"`` reads ``doc_topic_<split>_foldin.pkl`` (vMF) or
``<split dir>/<category>_doc_topic_foldin.pkl`` (MvTM), the collapsed fold-in
estimate ``(E[n_dk] + alpha_k) / (N_d + sum alpha)`` that training and
``evaluation vmf-foldin-theta`` write for the training and held-out documents
alike; ``soft_proxy`` never falls back to it, and no other baseline provides it.
The resolved source per iteration is recorded in the task metadata as
``doc_topic_source_resolved``.

Caveats: baselines only persist test-split matrices, so ``split="test"`` is the
only option for them (except the fold-in sources of MvTM, written for both splits). Document counts differ by one between word-embedding
baselines and the rest (documents without vocabulary tokens are dropped by the
former); per-document and per-topic metrics are unaffected and ``num_documents``
is recorded per iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from src.baselines.adapter_runtime import compose_gaussian_parameter_variant
from src.baselines.params import format_prior_scale_variant
from src.core.artifacts import load_artifact_pickle, load_json
from src.core.paths import resolve_baseline_condition_dir, resolve_vmf_experiment_dir
from src.core.vmf_assignment import DEFAULT_DOC_TOPIC_SOURCE
from src.data.preprocessing_selection import (
    parse_raw_doc_indices,
    resolve_preprocessing_selection,
)
from src.evaluation.model_provenance import load_model_provenance
from src.evaluation.word_based.model_inputs import (
    aggregate_doc_topics_from_sentence_topics,
    normalize_sentence_topic_payload,
)

DocTopicSource = Literal["auto", "soft_proxy", "soft", "hard", "foldin", "foldincounts"]
DOC_TOPIC_SOURCES: tuple[str, ...] = (
    "auto",
    "soft_proxy",
    "soft",
    "hard",
    "foldin",
    "foldincounts",
)

SUPPORTED_MODELS: tuple[str, ...] = (
    "vmf",
    "bleilda",
    "sam",
    "sam_tf",
    "sentlda",
    "sentence_gaussianlda",
    "gaussianlda",
    "mvtm",
    "etm",
    "ctm",
    "senclu",
)
MODEL_ALIASES: dict[str, str] = {
    "vmf_sentence_lda": "vmf",
    "gaussian": "sentence_gaussianlda",
}
SENTENCE_ENCODER_MODELS = frozenset({"vmf", "sentence_gaussianlda", "ctm", "senclu"})
WORD_EMBEDDING_MODELS = frozenset({"etm", "gaussianlda", "mvtm"})
GAUSSIAN_PRIOR_SCALE_MODELS = frozenset({"gaussianlda", "sentence_gaussianlda"})
DEFAULT_WORD_EMBEDDING_VARIANT = "googlenews300"

SOURCE_SOFT_FILE = "doc_topic_soft_file"
SOURCE_SENTENCE_AGGREGATE = "sentence_topic_soft_aggregate"
SOURCE_HARD_FILE = "doc_topic_hard_file"
SOURCE_FOLDIN_FILE = "doc_topic_foldin_file"
SOURCE_FOLDIN_COUNTS_FILE = "doc_topic_foldin_counts_file"


def normalize_model_name(model: str) -> str:
    key = str(model).strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model for entropy-based metrics: '{model}'. "
            f"Use one of {list(SUPPORTED_MODELS)}."
        )
    return key


def normalize_model_names(models: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for model in models:
        key = normalize_model_name(model)
        if key not in resolved:
            resolved.append(key)
    return resolved


def effective_embedding_variant(
    model: str,
    embedding_variant: str | None,
    word_embedding_variant: str | None = DEFAULT_WORD_EMBEDDING_VARIANT,
) -> str | None:
    """Map the requested sentence-encoder variant to the model's result-path suffix."""
    key = normalize_model_name(model)
    if key in WORD_EMBEDDING_MODELS:
        if word_embedding_variant in {None, ""}:
            return None
        return str(word_embedding_variant)
    if key not in SENTENCE_ENCODER_MODELS:
        return None
    if embedding_variant in {None, ""}:
        return None
    variant = str(embedding_variant).strip()
    if key == "sentence_gaussianlda" and not variant.endswith(("_raw", "_norm")):
        return f"{variant}_norm"
    return variant


def parameter_variant_for(
    model: str,
    prior_scale: float | None,
    covariance_type: str | None = None,
) -> str | None:
    key = normalize_model_name(model)
    if prior_scale is None or key not in GAUSSIAN_PRIOR_SCALE_MODELS:
        return None
    return compose_gaussian_parameter_variant(
        runner=key,
        prior_scale=prior_scale,
        covariance_type=covariance_type,
    )


def resolve_condition_dir(
    *,
    model: str,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    embedding_variant: str | None,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
) -> Path:
    """Resolve the trained condition directory holding the doc-topic artifacts.

    ``embedding_variant`` must already be the model-effective suffix (see
    :func:`effective_embedding_variant`); ``vmf_variant`` is the hyperparameter label of
    the vMF runs (None = the default runs).
    """
    key = normalize_model_name(model)
    if key == "vmf":
        return resolve_vmf_experiment_dir(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            run_name=data_run,
            embedding_variant=embedding_variant,
            parameter_variant=vmf_variant,
        )
    return resolve_baseline_condition_dir(
        model=key,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        data_run=data_run,
        embedding_variant=embedding_variant,
        parameter_variant=parameter_variant_for(key, prior_scale, covariance_type),
    )


@dataclass(frozen=True)
class DocTopicLoad:
    theta: np.ndarray
    source: str
    path: Path
    condition_dir: Path
    raw_doc_indices: list[int] | None = None


FOLDIN_SOURCES: tuple[str, ...] = ("foldin", "foldincounts")
# Models whose runs carry the fold-in artifacts (the vMF family).
FOLDIN_MODELS: frozenset[str] = frozenset({"vmf", "mvtm"})


def resolve_doc_topic_source(model: str, doc_topic_source: str) -> str:
    """The concrete source ``auto`` stands for: the manuscript's fold-in estimator for a
    vMF-family run (DEFAULT_DOC_TOPIC_SOURCE; vMF Sentence LDA and MvTM) and
    ``soft_proxy`` for every other baseline, which has no fold-in artifact. Any other
    value is returned unchanged after validation."""

    if doc_topic_source not in DOC_TOPIC_SOURCES:
        raise ValueError(
            f"Unsupported doc_topic_source '{doc_topic_source}'. Use one of {DOC_TOPIC_SOURCES}."
        )
    if doc_topic_source != "auto":
        return doc_topic_source
    return (
        DEFAULT_DOC_TOPIC_SOURCE
        if normalize_model_name(model) in FOLDIN_MODELS
        else "soft_proxy"
    )


def _candidate_paths(
    *,
    model: str,
    condition_dir: Path,
    category: str,
    split: str,
) -> tuple[Path, Path, Path, Path, Path]:
    """Return ``(soft_doc, sentence_soft, hard_doc, foldin_doc, foldin_counts_doc)`` paths.

    Only the vMF family (vMF, MvTM) holds the fold-in files; the other baseline
    paths are named for uniformity and never exist.
    """
    if model == "vmf":
        return (
            condition_dir / f"doc_topic_{split}_soft.pkl",
            condition_dir / f"sentence_topic_{split}_soft.pkl",
            condition_dir / f"doc_topic_{split}.pkl",
            condition_dir / f"doc_topic_{split}_foldin.pkl",
            condition_dir / f"doc_topic_{split}_foldin_counts.pkl",
        )
    split_dir = condition_dir / ("infer" if split == "test" else "params")
    return (
        split_dir / f"{category}_doc_topic_soft.pkl",
        split_dir / f"{category}_sentence_topic_soft.pkl",
        split_dir / f"{category}.pkl",
        split_dir / f"{category}_doc_topic_foldin.pkl",
        split_dir / f"{category}_doc_topic_foldin_counts.pkl",
    )


def _row_normalize(arr: np.ndarray, *, path: Path) -> np.ndarray:
    matrix = np.asarray(arr, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected 2D doc-topic array, got shape {matrix.shape} at {path}"
        )
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return matrix / row_sums


def _load_raw_doc_indices(condition_dir: Path, *, split: str) -> list[int] | None:
    candidates = [
        condition_dir
        / ("infer" if split == "test" else "params")
        / "preprocessing_selection.json",
        condition_dir / "preprocessing_selection.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = load_json(candidate)
            # Baselines write the flat per-split form, the vMF runner writes the
            # combined {"train": ..., "test": ...} form; the shared resolver
            # accepts both.
            selection = resolve_preprocessing_selection(
                payload, split=split, selection_path=candidate
            )
            return parse_raw_doc_indices(
                selection, selection_path=candidate, split=split
            )
        except Exception:  # noqa: BLE001 - best-effort provenance only
            return None
    return None


def load_doc_topic_matrix(
    *,
    model: str,
    dataset: str,
    data_run: str,
    iteration: int,
    num_topics: int,
    category: str,
    split: str = "test",
    doc_topic_source: str = "auto",
    embedding_variant: str | None = None,
    word_embedding_variant: str | None = DEFAULT_WORD_EMBEDDING_VARIANT,
    prior_scale: float | None = None,
    covariance_type: str | None = None,
    vmf_variant: str | None = None,
) -> DocTopicLoad:
    """Load a row-normalized ``(D, K)`` document-topic matrix for one condition.

    ``doc_topic_source="auto"`` resolves per model (:func:`resolve_doc_topic_source`).
    """
    key = normalize_model_name(model)
    doc_topic_source = resolve_doc_topic_source(key, doc_topic_source)
    if key != "vmf" and split != "test":
        # The baselines persist hard/soft matrices for the test split only; the
        # fold-in files of MvTM exist for both splits (``params/`` for train).
        if not (key in FOLDIN_MODELS and doc_topic_source in FOLDIN_SOURCES):
            raise ValueError(
                f"Model '{key}' only provides doc-topic distributions on the test split."
            )
    variant = effective_embedding_variant(
        key,
        embedding_variant,
        word_embedding_variant=word_embedding_variant,
    )
    condition_dir = resolve_condition_dir(
        model=key,
        dataset=dataset,
        data_run=data_run,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        embedding_variant=variant,
        prior_scale=prior_scale,
        covariance_type=covariance_type,
        vmf_variant=vmf_variant,
    )
    soft_path, sentence_path, hard_path, foldin_path, foldin_counts_path = (
        _candidate_paths(
            model=key,
            condition_dir=condition_dir,
            category=category,
            split=split,
        )
    )

    if doc_topic_source == "foldincounts":
        if not foldin_counts_path.exists():
            raise FileNotFoundError(
                "Fold-in expected-count artifact not found (vMF-family runs only; "
                f"write it with `evaluation vmf-foldin-theta`): {foldin_counts_path}"
            )
        theta = _row_normalize(
            load_artifact_pickle(foldin_counts_path), path=foldin_counts_path
        )
        chosen, source = foldin_counts_path, SOURCE_FOLDIN_COUNTS_FILE
    elif doc_topic_source == "foldin":
        if not foldin_path.exists():
            raise FileNotFoundError(
                "Fold-in doc-topic artifact not found (vMF-family runs only; write it "
                f"with `evaluation vmf-foldin-theta`): {foldin_path}"
            )
        theta = _row_normalize(load_artifact_pickle(foldin_path), path=foldin_path)
        chosen, source = foldin_path, SOURCE_FOLDIN_FILE
    elif doc_topic_source == "hard":
        if not hard_path.exists():
            raise FileNotFoundError(f"Hard doc-topic artifact not found: {hard_path}")
        theta = _row_normalize(load_artifact_pickle(hard_path), path=hard_path)
        chosen, source = hard_path, SOURCE_HARD_FILE
    elif doc_topic_source == "soft":
        if not soft_path.exists():
            raise FileNotFoundError(f"Soft doc-topic artifact not found: {soft_path}")
        theta = _row_normalize(load_artifact_pickle(soft_path), path=soft_path)
        chosen, source = soft_path, SOURCE_SOFT_FILE
    elif soft_path.exists():
        theta = _row_normalize(load_artifact_pickle(soft_path), path=soft_path)
        chosen, source = soft_path, SOURCE_SOFT_FILE
    elif sentence_path.exists():
        raw = load_artifact_pickle(sentence_path)
        sentence_topics = normalize_sentence_topic_payload(
            raw, num_topics=num_topics, path=sentence_path
        )
        theta = aggregate_doc_topics_from_sentence_topics(
            sentence_topics_by_doc=sentence_topics,
            num_topics=num_topics,
        )
        chosen, source = sentence_path, SOURCE_SENTENCE_AGGREGATE
    else:
        if not hard_path.exists():
            raise FileNotFoundError(
                "No doc-topic artifact found for "
                f"model={key} dataset={dataset} category={category} "
                f"iteration={iteration} num_topics={num_topics}: tried {soft_path}, "
                f"{sentence_path}, {hard_path}"
            )
        theta = _row_normalize(load_artifact_pickle(hard_path), path=hard_path)
        chosen, source = hard_path, SOURCE_HARD_FILE

    if theta.shape[1] != int(num_topics):
        raise ValueError(
            f"Doc-topic width {theta.shape[1]} != num_topics {num_topics} at {chosen}"
        )
    return DocTopicLoad(
        theta=theta,
        source=source,
        path=chosen,
        condition_dir=condition_dir,
        raw_doc_indices=_load_raw_doc_indices(condition_dir, split=split),
    )


def provenance_for(condition_dir: Path, *, model: str) -> dict[str, object]:
    key = normalize_model_name(model)
    return load_model_provenance(
        condition_dir,
        model_key="vmf_sentence_lda" if key == "vmf" else key,
    )

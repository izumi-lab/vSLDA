"""Representative words of the reference runs, as a ``*.scores.json`` sidecar.

The manuscript's appendix lists every topic of a few reference runs with its
representative words next to the pair quantities of the topic-pair sidecars.
The words are the NPMI display ranking of the coherence run of the same
condition (the ranking every table of the paper quotes), so this module
resolves that run through the coherence ``latest`` pointers and writes the
words, with the coherence run's identity, into
``summaries/<dataset>/<data_run>/<encoder>/reference_words_<dataset>_<category>_it<i>_<K>topic_<encoder>.scores.json``.
Nothing is aggregated; the paper side joins the words with the topic-pair
sidecar of the same run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.core.artifacts import load_json
from src.core.paths import RESULTS_ROOT, resolve_project_path
from src.core.vmf_variant import vmf_variant_matches
from src.evaluation.topic_pairs.inputs import (
    TASK_MODELS,
    model_embedding_variant,
    normalize_model_name,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_COHERENCE_ROOT = RESULTS_ROOT / "topic_analysis" / "coherence"
WORDS_TASK_NAME = "topic_pair_reference_words"
# The reference runs the manuscript lists topic by topic: (dataset, category, iteration).
PAPER_REFERENCE_RUNS: tuple[tuple[str, str, int], ...] = (
    ("20newsgroup", "computer", 0),
    ("nyt", "arts", 0),
)
PAPER_REFERENCE_TOPICS = 20
PAPER_REFERENCE_ENCODER = "minilm"
PAPER_REFERENCE_MODELS: tuple[str, ...] = TASK_MODELS
# The Gaussian sentence LDA runs the manuscript reports record their NIW prior
# scale; only that variant carries a prior scale, so the lookup matches on it.
PAPER_GAUSSIAN_PRIOR_SCALE = 0.1
DEFAULT_TOP_N = 10


class ReferenceWordsError(RuntimeError):
    """The coherence run carrying the words could not be resolved unambiguously."""


@dataclass(frozen=True)
class DisplayWords:
    words: list[list[str]]
    archive_dir: Path
    condition_id: str | None
    condition_fingerprint: str | None
    score_mode: str | None
    effective_embedding_variant: str | None


def resolve_display_words(
    coherence_root: Path,
    *,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    model: str,
    embedding_variant: str | None,
    top_n: int,
    prior_scale: float | None = None,
) -> DisplayWords:
    """Top-``top_n`` NPMI display words per topic of the matching coherence run.

    The run is found through ``latest/<dataset>/<data_run>/<category>`` of the
    coherence root: same model, topic count, single iteration, sentence-encoder
    variant, Gaussian prior-scale variant, and no vMF hyperparameter-sweep
    variant (``parameter_variant``; the sweep of tmp_hyper_sweep.sh writes its
    own pointers beside the main ones). When the main condition was evaluated
    more than once, the newest execution is taken, as the coherence summary
    does; two executions started at the same instant are an error.
    """

    latest = Path(coherence_root) / "latest" / dataset / data_run / category
    wanted_variant = model_embedding_variant(model, embedding_variant)
    matches: list[tuple[Path, dict[str, Any]]] = []
    for pointer_path in sorted(
        latest.glob(f"it{int(iteration)}__k{int(num_topics)}__*/CURRENT.json")
    ):
        pointer = load_json(pointer_path)
        if not isinstance(pointer, dict) or not pointer.get("archive_dir"):
            continue
        archive_dir = resolve_project_path(str(pointer["archive_dir"]))
        metadata_path = archive_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        meta = load_json(metadata_path)
        if not isinstance(meta, dict) or str(meta.get("model")) != model:
            continue
        if int(meta.get("num_topics", -1)) != int(num_topics):
            continue
        if [int(value) for value in meta.get("iterations") or []] != [int(iteration)]:
            continue
        # The effective variant is the one the run was read from; SentLDA records
        # the requested variant but reads no embeddings, so its effective value is None.
        variant = (
            meta.get("effective_embedding_variant")
            if "effective_embedding_variant" in meta
            else meta.get("embedding_variant")
        )
        if (None if variant in {None, "", "None"} else str(variant)) != wanted_variant:
            continue
        recorded_scale = meta.get("prior_scale")
        if prior_scale is None:
            if recorded_scale is not None:
                continue
        elif recorded_scale is None or float(recorded_scale) != float(prior_scale):
            continue
        # The vMF hyperparameter sweep (src/core/vmf_variant.py) records its point as
        # ``parameter_variant``; the reference words are those of the main setting.
        if model == "vmf" and not vmf_variant_matches(
            meta.get("parameter_variant"), None
        ):
            continue
        matches.append((archive_dir, meta))
    where = f"{dataset}/{data_run}/{category} it{iteration} K={num_topics} {model}"
    if not matches:
        raise ReferenceWordsError(
            f"no coherence run with display words for {where} under {latest}"
        )
    if len(matches) > 1:
        matches.sort(key=lambda item: str(item[1].get("started_at", "")))
        if str(matches[-1][1].get("started_at", "")) == str(
            matches[-2][1].get("started_at", "")
        ):
            raise ReferenceWordsError(
                f"{len(matches)} coherence runs match {where} and the newest two started "
                f"at the same time: {[str(m[0]) for m in matches]}"
            )
        logger.warning(
            "%s: %d coherence runs match; keeping the newest execution %s",
            where,
            len(matches),
            matches[-1][0],
        )
    archive_dir, meta = matches[-1]
    display = load_json(archive_dir / "topic_words_display_topk.json")
    results = display.get("results", display) if isinstance(display, dict) else {}
    entries = results.get("per_iteration") if isinstance(results, dict) else None
    if not entries:
        raise ReferenceWordsError(f"no per-iteration display words in {archive_dir}")
    entry = next(
        (item for item in entries if int(item.get("iteration", 0)) == int(iteration)),
        None,
    )
    if entry is None:
        raise ReferenceWordsError(f"iteration {iteration} missing from {archive_dir}")
    words: list[list[str]] = [[] for _ in range(int(num_topics))]
    for topic in entry.get("topics") or []:
        topic_id = int(topic.get("topic_id"))
        listed = topic.get("words") or []
        words[topic_id] = [
            str(item.get("word") if isinstance(item, dict) else item)
            for item in listed[:top_n]
        ]
    return DisplayWords(
        words=words,
        archive_dir=archive_dir,
        condition_id=(
            None if meta.get("condition_id") is None else str(meta.get("condition_id"))
        ),
        condition_fingerprint=(
            None
            if meta.get("condition_fingerprint") is None
            else str(meta.get("condition_fingerprint"))
        ),
        score_mode=(
            None
            if meta.get("topic_word_score_mode") is None
            else str(meta.get("topic_word_score_mode"))
        ),
        effective_embedding_variant=(
            None if wanted_variant is None else str(wanted_variant)
        ),
    )


def _slug(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def reference_words_sidecar_path(
    output_dir: Path,
    *,
    dataset: str,
    data_run: str,
    encoder: str,
    category: str,
    iteration: int,
    num_topics: int,
) -> Path:
    return (
        Path(output_dir)
        / _slug(dataset)
        / _slug(data_run)
        / _slug(encoder)
        / (
            f"reference_words_{_slug(dataset)}_{_slug(category)}_it{int(iteration)}_"
            f"{int(num_topics)}topic_{_slug(encoder)}.scores.json"
        )
    )


def build_reference_words_payload(
    *,
    coherence_root: Path,
    dataset: str,
    data_run: str,
    category: str,
    iteration: int,
    num_topics: int,
    encoder_variant: str,
    models: Sequence[str],
    top_n: int,
    gaussian_prior_scale: float | None = PAPER_GAUSSIAN_PRIOR_SCALE,
) -> dict[str, Any]:
    words: dict[str, list[list[str]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for model in models:
        model_scale = (
            gaussian_prior_scale
            if normalize_model_name(model) == "sentence_gaussianlda"
            else None
        )
        resolved = resolve_display_words(
            coherence_root,
            dataset=dataset,
            data_run=data_run,
            category=category,
            iteration=iteration,
            num_topics=num_topics,
            model=model,
            embedding_variant=encoder_variant,
            top_n=top_n,
            prior_scale=model_scale,
        )
        words[model] = resolved.words
        provenance[model] = {
            "coherence_archive_dir": str(resolved.archive_dir),
            "condition_id": resolved.condition_id,
            "condition_fingerprint": resolved.condition_fingerprint,
            "topic_word_score_mode": resolved.score_mode,
            "topic_word_source": "posthoc_word_topic_npmi",
            "encoder_model": resolved.effective_embedding_variant,
            "embedding_variant": resolved.effective_embedding_variant,
            "prior_scale": model_scale,
        }
    return {
        "task": WORDS_TASK_NAME,
        "dataset": dataset,
        "data_run": data_run,
        "category": category,
        "iteration": int(iteration),
        "topics": int(num_topics),
        "encoder_variant": encoder_variant,
        "top_n": int(top_n),
        "models": list(models),
        "words": words,
        "provenance": provenance,
    }


def write_reference_words(
    *,
    output_dir: Path,
    coherence_root: Path = DEFAULT_COHERENCE_ROOT,
    runs: Sequence[tuple[str, str, int]] = PAPER_REFERENCE_RUNS,
    num_topics: int = PAPER_REFERENCE_TOPICS,
    encoder_variant: str = PAPER_REFERENCE_ENCODER,
    data_run: str = "default",
    models: Sequence[str] = PAPER_REFERENCE_MODELS,
    top_n: int = DEFAULT_TOP_N,
    gaussian_prior_scale: float | None = PAPER_GAUSSIAN_PRIOR_SCALE,
) -> list[Path]:
    """One sidecar per reference run with every model's display words."""

    written: list[Path] = []
    for dataset, category, iteration in runs:
        payload = build_reference_words_payload(
            coherence_root=coherence_root,
            dataset=dataset,
            data_run=data_run,
            category=category,
            iteration=iteration,
            num_topics=num_topics,
            encoder_variant=encoder_variant,
            models=models,
            top_n=top_n,
            gaussian_prior_scale=gaussian_prior_scale,
        )
        path = reference_words_sidecar_path(
            output_dir,
            dataset=dataset,
            data_run=data_run,
            encoder=encoder_variant,
            category=category,
            iteration=iteration,
            num_topics=num_topics,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        logger.info("[write] %s", path)
        written.append(path)
    return written


__all__ = [
    "DEFAULT_COHERENCE_ROOT",
    "DEFAULT_TOP_N",
    "PAPER_REFERENCE_ENCODER",
    "PAPER_REFERENCE_MODELS",
    "PAPER_REFERENCE_RUNS",
    "PAPER_REFERENCE_TOPICS",
    "WORDS_TASK_NAME",
    "DisplayWords",
    "ReferenceWordsError",
    "build_reference_words_payload",
    "reference_words_sidecar_path",
    "resolve_display_words",
    "write_reference_words",
]

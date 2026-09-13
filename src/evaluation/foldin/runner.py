"""Runner of the ``vmf_foldin_theta`` task.

Enumerates the vMF Sentence LDA runs under ``results/experiments/<dataset>/
<data_run>/vmf_sentence_lda/latest/<category>/<display_key>/CURRENT.json``,
groups them by (dataset, data run, category, split, sentence encoder) so that
each unit's sentences are loaded and encoded once (the split-keyed embedding
cache of the topic-pair analysis is shared), and writes the collapsed fold-in
document-topic distributions of every selected run into its archive directory
(:mod:`.theta`). A summary CSV lists every run/split with its status and timing.

With ``model="mvtm"`` the same is done for the MvTM (vLDA) runs under
``results/baselines/<dataset>/<data_run>/mvtm/latest/...`` (:mod:`.mvtm`): the
word vectors named in a run's metadata are loaded once per process and shared,
and the token-unit fold-in is written under ``params/`` and ``infer/``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.core.artifacts import load_json
from src.core.errors import MissingArtifactError
from src.core.paths import (
    BASELINE_RESULTS_ROOT,
    EXPERIMENT_RESULTS_ROOT,
    RESULTS_ROOT,
    resolve_project_path,
)
from src.evaluation.foldin.artifacts import (
    DEFAULT_CHUNK_DOCS,
    FOLDIN_MODELS,
    FoldInLayout,
)
from src.evaluation.foldin.theta import (
    SPLITS,
    FoldInRunResult,
    _atomic_write_json,
    compute_foldin_for_run,
    condition_fingerprint_of,
    foldin_fingerprint,
    foldin_is_current,
    write_foldin_artifacts,
)
from src.evaluation.reporting import write_csv_rows
from src.evaluation.topic_pairs.inputs import (
    CachedEmbeddings,
    SentenceAlignmentError,
    TrainCorpus,
    encode_train_corpus,
    encoder_config_of,
    encoder_fingerprint,
    load_train_corpus,
)
from src.evaluation.topic_pairs.metrics import DEFAULT_CACHE_ROOT
from src.evaluation.word_based.sentence_encoding import (
    resolve_topic_word_encoder_device,
)
from src.evaluation.word_based.topic_assignment import CollapsedFoldInConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

TASK_NAME = "vmf_foldin_theta"
VMF_MODEL_DIRNAME = "vmf_sentence_lda"
DEFAULT_MODEL = "vmf_sentence_lda"
CURRENT_POINTER_FILENAME = "CURRENT.json"
DEFAULT_DATASETS: tuple[str, ...] = ("20newsgroup", "nyt")
DEFAULT_SUMMARY_ROOT = RESULTS_ROOT / "topic_analysis" / "foldin" / "summaries"
DEFAULT_SUMMARY_FILENAME = "vmf_foldin_theta.csv"
SUMMARY_FILENAME_BY_MODEL: dict[str, str] = {
    "vmf_sentence_lda": DEFAULT_SUMMARY_FILENAME,
    "mvtm": "mvtm_foldin_theta.csv",
}
CONDITION_FAILURE_POLICIES: tuple[str, ...] = ("fail-fast", "isolate")
SUMMARY_FIELDS: tuple[str, ...] = (
    "model",
    "dataset",
    "data_run",
    "category",
    "display_key",
    "num_topics",
    "iteration",
    "embedding_variant",
    "parameter_variant",
    "split",
    "status",
    "archive_dir",
    "num_documents",
    "total_sentences",
    "num_empty_documents",
    "embedding_cache_hit",
    "encode_sec",
    "log_likelihood_sec",
    "collapsed_foldin_sec",
    "total_sec",
    "fingerprint",
    "error",
)
_DISPLAY_KEY_RE = re.compile(r"^k(?P<k>\d+)_it(?P<it>\d+)(?:_|$)")


def normalize_foldin_model(model: str) -> str:
    key = str(model).strip().lower()
    key = {"vmf": "vmf_sentence_lda", "vlda": "mvtm"}.get(key, key)
    if key not in FOLDIN_MODELS:
        raise ValueError(
            f"Unsupported fold-in model {model!r}; use one of {', '.join(FOLDIN_MODELS)}"
        )
    return key


def _latest_root(
    model: str, *, dataset: str, data_run: str, results_root: Path | None
) -> Path:
    if model == "mvtm":
        root = (
            Path(results_root)
            if results_root is not None
            else Path(BASELINE_RESULTS_ROOT)
        )
        return root / str(dataset) / str(data_run) / "mvtm" / "latest"
    root = (
        Path(results_root)
        if results_root is not None
        else Path(EXPERIMENT_RESULTS_ROOT)
    )
    return root / str(dataset) / str(data_run) / VMF_MODEL_DIRNAME / "latest"


@dataclass(frozen=True)
class VmfRunPointer:
    """One ``CURRENT.json`` of a vMF Sentence LDA (or MvTM) run."""

    dataset: str
    data_run: str
    category: str
    display_key: str
    num_topics: int
    iteration: int
    embedding_variant: str | None
    parameter_variant: str | None
    archive_dir: Path
    pointer_path: Path
    payload: dict[str, Any]
    model: str = DEFAULT_MODEL

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "data_run": self.data_run,
            "category": self.category,
            "display_key": self.display_key,
            "num_topics": self.num_topics,
            "iteration": self.iteration,
            "embedding_variant": self.embedding_variant,
            "parameter_variant": self.parameter_variant,
            "archive_dir": str(self.archive_dir),
        }


def parse_display_key(display_key: str) -> tuple[int, int]:
    match = _DISPLAY_KEY_RE.match(str(display_key))
    if match is None:
        raise ValueError(f"display_key without k<K>_it<i> prefix: {display_key!r}")
    return int(match.group("k")), int(match.group("it"))


def _pointer_from_payload(
    pointer_path: Path, payload: Mapping[str, Any], *, model: str = DEFAULT_MODEL
) -> VmfRunPointer:
    display_key = str(payload.get("display_key") or pointer_path.parent.name)
    num_topics, iteration = parse_display_key(display_key)
    archive_raw = payload.get("archive_dir")
    if not archive_raw:
        raise ValueError(f"pointer without archive_dir: {pointer_path}")
    variant = payload.get("embedding_variant")
    parameter_variant = payload.get("parameter_variant")
    return VmfRunPointer(
        dataset=str(payload.get("dataset") or pointer_path.parents[4].name),
        data_run=str(payload.get("data_run") or pointer_path.parents[3].name),
        category=str(payload.get("category") or pointer_path.parents[1].name),
        display_key=display_key,
        num_topics=num_topics,
        iteration=iteration,
        embedding_variant=None if variant in {None, ""} else str(variant),
        parameter_variant=(
            None if parameter_variant in {None, ""} else str(parameter_variant)
        ),
        archive_dir=resolve_project_path(str(archive_raw)),
        pointer_path=pointer_path,
        payload=dict(payload),
        model=model,
    )


def iter_vmf_run_pointers(
    *,
    datasets: Sequence[str],
    data_runs: Sequence[str] = ("default",),
    results_root: Path | None = None,
    model: str = DEFAULT_MODEL,
) -> list[VmfRunPointer]:
    """Every ``CURRENT.json`` of the selected datasets and data runs, sorted.

    ``results_root`` overrides ``results/experiments`` (vMF Sentence LDA) or
    ``results/baselines`` (MvTM).
    """

    model = normalize_foldin_model(model)
    pointers: list[VmfRunPointer] = []
    for dataset in datasets:
        for data_run in data_runs:
            latest_root = _latest_root(
                model, dataset=dataset, data_run=data_run, results_root=results_root
            )
            if not latest_root.is_dir():
                logger.warning("no %s runs under %s", model, latest_root)
                continue
            for pointer_path in sorted(
                latest_root.glob(f"*/*/{CURRENT_POINTER_FILENAME}")
            ):
                payload = load_json(pointer_path)
                if not isinstance(payload, dict):
                    continue
                try:
                    pointers.append(
                        _pointer_from_payload(pointer_path, payload, model=model)
                    )
                except ValueError as exc:
                    logger.warning("skipping pointer %s: %s", pointer_path, exc)
    pointers.sort(
        key=lambda item: (
            item.dataset,
            item.data_run,
            item.category,
            item.embedding_variant or "",
            item.num_topics,
            item.iteration,
            item.parameter_variant or "",
        )
    )
    return pointers


def select_runs(
    pointers: Iterable[VmfRunPointer],
    *,
    categories: Sequence[str] | None = None,
    iterations: Sequence[int] | None = None,
    num_topics: Sequence[int] | None = None,
    embedding_variants: Sequence[str] | None = None,
    vmf_variants: Sequence[str] | None = None,
    include_vmf_variants: bool = False,
    all_vmf_runs: bool = False,
) -> list[VmfRunPointer]:
    """Apply the CLI filters; ``all_vmf_runs`` keeps everything.

    Without ``vmf_variants`` and ``include_vmf_variants`` only the runs of the
    main hyperparameter setting (no ``parameter_variant``) are selected.
    """

    if all_vmf_runs:
        return list(pointers)
    wanted_categories = {str(value) for value in categories} if categories else None
    wanted_iterations = {int(value) for value in iterations} if iterations else None
    wanted_topics = {int(value) for value in num_topics} if num_topics else None
    wanted_variants = (
        {str(value) for value in embedding_variants} if embedding_variants else None
    )
    wanted_vmf_variants = (
        {str(value) for value in vmf_variants} if vmf_variants else None
    )
    selected: list[VmfRunPointer] = []
    for pointer in pointers:
        if wanted_categories is not None and pointer.category not in wanted_categories:
            continue
        if wanted_iterations is not None and pointer.iteration not in wanted_iterations:
            continue
        if wanted_topics is not None and pointer.num_topics not in wanted_topics:
            continue
        if (
            wanted_variants is not None
            and (pointer.embedding_variant or "") not in wanted_variants
        ):
            continue
        if wanted_vmf_variants is not None:
            if (pointer.parameter_variant or "") not in wanted_vmf_variants:
                continue
        elif pointer.parameter_variant is not None and not include_vmf_variants:
            continue
        selected.append(pointer)
    return selected


@dataclass
class _UnitContext:
    """Sentences and embeddings of one (unit, split, encoder) group."""

    corpus: TrainCorpus
    embeddings: CachedEmbeddings
    encoder_fp: str
    reference: VmfRunPointer
    encode_sec: float
    runs_served: int = 0


@dataclass
class _Outcome:
    rows: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    computed: int = 0
    skipped: int = 0


def _update_pointer(pointer: VmfRunPointer, artifacts: Mapping[str, str]) -> None:
    """Add the fold-in artifact keys to ``CURRENT.json`` and nothing else."""

    payload = load_json(pointer.pointer_path)
    if not isinstance(payload, dict):
        raise ValueError(f"pointer is not a JSON object: {pointer.pointer_path}")
    existing = payload.get("artifacts")
    merged = dict(existing) if isinstance(existing, Mapping) else {}
    merged.update({str(key): str(value) for key, value in artifacts.items()})
    payload["artifacts"] = merged
    _atomic_write_json(pointer.pointer_path, payload)


def _summary_row(
    pointer: VmfRunPointer,
    *,
    split: str,
    status: str,
    result: FoldInRunResult | None = None,
    fingerprint: str | None = None,
    total_sec: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **pointer.describe(),
        "split": split,
        "status": status,
        "num_documents": None,
        "total_sentences": None,
        "num_empty_documents": None,
        "embedding_cache_hit": None,
        "encode_sec": None,
        "log_likelihood_sec": None,
        "collapsed_foldin_sec": None,
        "total_sec": total_sec,
        "fingerprint": fingerprint,
        "error": error,
    }
    if result is not None:
        row.update(
            {
                "num_documents": result.num_documents,
                "total_sentences": result.total_sentences,
                "num_empty_documents": result.num_empty_documents,
                "embedding_cache_hit": result.metadata.get("embedding_cache_hit"),
                "encode_sec": result.timing.get("encode_sec"),
                "log_likelihood_sec": result.timing.get("log_likelihood_sec"),
                "collapsed_foldin_sec": result.timing.get("collapsed_foldin_sec"),
                "fingerprint": result.fingerprint,
            }
        )
    return row


def run_vmf_foldin_theta(
    *,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    data_runs: Sequence[str] = ("default",),
    categories: Sequence[str] | None = None,
    iterations: Sequence[int] | None = None,
    num_topics: Sequence[int] | None = None,
    splits: Sequence[str] = SPLITS,
    embedding_variants: Sequence[str] | None = None,
    vmf_variants: Sequence[str] | None = None,
    include_vmf_variants: bool = False,
    all_vmf_runs: bool = False,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    encoder_device: str = "auto",
    encode_batch_size: int | None = None,
    foldin_config: CollapsedFoldInConfig | None = None,
    skip_existing: bool = True,
    write_sentence_posteriors: bool = False,
    update_pointer: bool = True,
    condition_failure_policy: str = "fail-fast",
    summary_path: Path | None = None,
    results_root: Path | None = None,
    model: str = DEFAULT_MODEL,
    chunk_docs: int = DEFAULT_CHUNK_DOCS,
) -> Path:
    """Write the fold-in θ of every selected run of ``model``; return the summary CSV.

    vMF Sentence LDA runs get ``doc_topic_{split}_foldin*.pkl``; MvTM runs get
    ``params/<category>_doc_topic_foldin*.pkl`` and the same under ``infer/``.
    ``chunk_docs`` (MvTM only) is the number of documents per sampler call.
    """

    model = normalize_foldin_model(model)
    if condition_failure_policy not in CONDITION_FAILURE_POLICIES:
        raise ValueError(
            f"Unsupported condition_failure_policy '{condition_failure_policy}'. "
            f"Use one of {CONDITION_FAILURE_POLICIES}."
        )
    split_values = [str(value) for value in splits]
    for split in split_values:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
    if not split_values:
        raise ValueError("splits must contain at least one value.")
    config = foldin_config or CollapsedFoldInConfig()
    config.validate()
    if int(chunk_docs) <= 0:
        raise ValueError("chunk_docs must be positive")
    cache_root = Path(cache_root)
    summary_path = (
        Path(summary_path)
        if summary_path is not None
        else DEFAULT_SUMMARY_ROOT / SUMMARY_FILENAME_BY_MODEL[model]
    )
    failure_exceptions = (
        MissingArtifactError,
        FileNotFoundError,
        ValueError,
        SentenceAlignmentError,
    )

    pointers = select_runs(
        iter_vmf_run_pointers(
            datasets=datasets,
            data_runs=data_runs,
            results_root=results_root,
            model=model,
        ),
        categories=categories,
        iterations=iterations,
        num_topics=num_topics,
        embedding_variants=embedding_variants,
        vmf_variants=vmf_variants,
        include_vmf_variants=include_vmf_variants,
        all_vmf_runs=all_vmf_runs,
    )
    logger.info(
        "%d %s run(s) selected for fold-in on split(s) %s",
        len(pointers),
        model,
        split_values,
    )
    outcome = _Outcome()

    def _record_failure(exc: BaseException, pointer: VmfRunPointer, split: str) -> None:
        if condition_failure_policy == "fail-fast":
            raise exc
        failure = {
            **pointer.describe(),
            "split": split,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        outcome.failures.append(failure)
        outcome.rows.append(
            _summary_row(
                pointer,
                split=split,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        logger.warning(
            "skipping %s/%s/%s (%s) after %s: %s",
            pointer.dataset,
            pointer.category,
            pointer.display_key,
            split,
            type(exc).__name__,
            exc,
        )

    if model == "mvtm":
        _run_mvtm(
            pointers,
            split_values=split_values,
            config=config,
            chunk_docs=int(chunk_docs),
            skip_existing=skip_existing,
            update_pointer=update_pointer,
            outcome=outcome,
            record_failure=_record_failure,
            failure_exceptions=failure_exceptions,
        )
        return _write_summary(summary_path, outcome)

    resolved_device = resolve_topic_word_encoder_device(encoder_device)
    # Group by unit and encoder so the sentences are read and encoded once.
    contexts: dict[tuple[str, str, str, str, str], _UnitContext] = {}
    for split in split_values:
        for pointer in pointers:
            started = time.perf_counter()
            try:
                encoder_config = encoder_config_of(pointer.archive_dir)
                encoder_fp = encoder_fingerprint(encoder_config)
                key = (
                    pointer.dataset,
                    pointer.data_run,
                    pointer.category,
                    split,
                    encoder_fp,
                )
                if key not in contexts:
                    encode_started = time.perf_counter()
                    corpus = load_train_corpus(
                        pointer.archive_dir, model="vmf", split=split
                    )
                    embeddings = encode_train_corpus(
                        corpus,
                        encoder_config=encoder_config,
                        cache_root=cache_root,
                        dataset=pointer.dataset,
                        data_run=pointer.data_run,
                        category=pointer.category,
                        split=split,
                        device=resolved_device,
                        encode_batch_size=encode_batch_size,
                    )
                    logger.info(
                        "[%s/%s/%s] %s sentence embeddings %s (%d x %d) from %s",
                        pointer.dataset,
                        pointer.category,
                        split,
                        pointer.embedding_variant or encoder_fp,
                        (
                            "read from cache"
                            if embeddings.cache_hit
                            else "encoded and cached"
                        ),
                        embeddings.embeddings.shape[0],
                        embeddings.embedding_dim,
                        embeddings.cache_dir,
                    )
                    contexts[key] = _UnitContext(
                        corpus=corpus,
                        embeddings=embeddings,
                        encoder_fp=encoder_fp,
                        reference=pointer,
                        encode_sec=time.perf_counter() - encode_started,
                    )
                context = contexts[key]
                run_corpus = (
                    context.corpus
                    if pointer.archive_dir == context.reference.archive_dir
                    else load_train_corpus(
                        pointer.archive_dir, model="vmf", split=split
                    )
                )
                fingerprint = foldin_fingerprint(
                    config=config,
                    encoder_fp=encoder_fp,
                    corpus_sha1=run_corpus.sentence_sha1,
                    condition_fp=condition_fingerprint_of(pointer.archive_dir),
                )
                if skip_existing and foldin_is_current(
                    pointer.archive_dir, split=split, fingerprint=fingerprint
                ):
                    outcome.skipped += 1
                    outcome.rows.append(
                        _summary_row(
                            pointer,
                            split=split,
                            status="skipped",
                            fingerprint=fingerprint,
                            total_sec=time.perf_counter() - started,
                        )
                    )
                    continue
                result = compute_foldin_for_run(
                    pointer.archive_dir,
                    split=split,
                    dataset=pointer.dataset,
                    data_run=pointer.data_run,
                    category=pointer.category,
                    cache_root=cache_root,
                    encoder_device=resolved_device,
                    encode_batch_size=encode_batch_size,
                    foldin_config=config,
                    corpus=run_corpus,
                    reference_corpus=context.corpus,
                    embeddings=context.embeddings,
                )
                # The unit's sentences were read and encoded once, before the
                # first run of the group; charge that time to the first row.
                if context.runs_served == 0:
                    result.timing["encode_sec"] = context.encode_sec
                artifacts = write_foldin_artifacts(
                    pointer.archive_dir,
                    result=result,
                    write_sentence_posteriors=write_sentence_posteriors,
                )
                if update_pointer:
                    _update_pointer(pointer, artifacts)
                context.runs_served += 1
                outcome.computed += 1
                outcome.rows.append(
                    _summary_row(
                        pointer,
                        split=split,
                        status="computed",
                        result=result,
                        total_sec=time.perf_counter() - started,
                    )
                )
                logger.info(
                    "[%s/%s/%s] %s: fold-in theta (%d x %d) in %.1fs",
                    pointer.dataset,
                    pointer.category,
                    split,
                    pointer.display_key,
                    result.num_documents,
                    result.num_topics,
                    time.perf_counter() - started,
                )
            except failure_exceptions as exc:
                _record_failure(exc, pointer, split)
        # The split's embeddings are not needed by the next split.
        for key in [item for item in contexts if item[3] == split]:
            contexts.pop(key)

    return _write_summary(summary_path, outcome)


@dataclass
class _WordVectorContext:
    """Word vectors of one source, shared by every MvTM run trained with them."""

    vectors: Any
    encoder_fp: str
    load_sec: float
    runs_served: int = 0


def _run_mvtm(
    pointers: Sequence[VmfRunPointer],
    *,
    split_values: Sequence[str],
    config: CollapsedFoldInConfig,
    chunk_docs: int,
    skip_existing: bool,
    update_pointer: bool,
    outcome: _Outcome,
    record_failure: Any,
    failure_exceptions: tuple[type[BaseException], ...],
) -> None:
    from src.evaluation.foldin.mvtm import (
        compute_mvtm_foldin_for_run,
        load_mvtm_word_vectors,
        load_token_corpus,
        mvtm_encoder_fingerprint,
        mvtm_foldin_fingerprint,
    )

    vector_contexts: dict[str, _WordVectorContext] = {}
    for split in split_values:
        for pointer in pointers:
            started = time.perf_counter()
            try:
                layout = FoldInLayout.for_model("mvtm", category=pointer.category)
                corpus = load_token_corpus(pointer.archive_dir, split=split)
                fingerprint = mvtm_foldin_fingerprint(
                    pointer.archive_dir,
                    split=split,
                    config=config,
                    chunk_docs=chunk_docs,
                    corpus=corpus,
                )
                if skip_existing and foldin_is_current(
                    pointer.archive_dir,
                    split=split,
                    fingerprint=fingerprint,
                    layout=layout,
                ):
                    outcome.skipped += 1
                    outcome.rows.append(
                        _summary_row(
                            pointer,
                            split=split,
                            status="skipped",
                            fingerprint=fingerprint,
                            total_sec=time.perf_counter() - started,
                        )
                    )
                    continue
                encoder_fp = mvtm_encoder_fingerprint(pointer.archive_dir)
                if encoder_fp not in vector_contexts:
                    load_started = time.perf_counter()
                    vectors = load_mvtm_word_vectors(pointer.archive_dir)
                    vector_contexts[encoder_fp] = _WordVectorContext(
                        vectors=vectors,
                        encoder_fp=encoder_fp,
                        load_sec=time.perf_counter() - load_started,
                    )
                    logger.info(
                        "[%s/%s] word vectors %s loaded (%d x %d) in %.1fs",
                        pointer.dataset,
                        pointer.category,
                        pointer.embedding_variant or encoder_fp,
                        len(vectors.key_to_index),
                        int(vectors.vector_size),
                        vector_contexts[encoder_fp].load_sec,
                    )
                context = vector_contexts[encoder_fp]
                result = compute_mvtm_foldin_for_run(
                    pointer.archive_dir,
                    split=split,
                    dataset=pointer.dataset,
                    data_run=pointer.data_run,
                    category=pointer.category,
                    foldin_config=config,
                    chunk_docs=chunk_docs,
                    corpus=corpus,
                    vectors=context.vectors,
                )
                if context.runs_served == 0:
                    result.timing["encode_sec"] = context.load_sec
                artifacts = write_foldin_artifacts(pointer.archive_dir, result=result)
                if update_pointer:
                    _update_pointer(pointer, artifacts)
                context.runs_served += 1
                outcome.computed += 1
                outcome.rows.append(
                    _summary_row(
                        pointer,
                        split=split,
                        status="computed",
                        result=result,
                        total_sec=time.perf_counter() - started,
                    )
                )
                logger.info(
                    "[%s/%s/%s] %s: MvTM fold-in theta (%d x %d, %d tokens) in %.1fs",
                    pointer.dataset,
                    pointer.category,
                    split,
                    pointer.display_key,
                    result.num_documents,
                    result.num_topics,
                    result.total_sentences,
                    time.perf_counter() - started,
                )
            except failure_exceptions as exc:
                record_failure(exc, pointer, split)


def _write_summary(summary_path: Path, outcome: _Outcome) -> Path:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(fieldnames=SUMMARY_FIELDS, rows=outcome.rows, path=summary_path)
    if outcome.failures:
        _atomic_write_json(
            summary_path.with_suffix(".failures.json"),
            {
                "task": TASK_NAME,
                "written_at": datetime.now(UTC).isoformat(),
                "failures": outcome.failures,
            },
        )
    logger.info(
        "%s: %d computed, %d skipped, %d failed; summary at %s",
        TASK_NAME,
        outcome.computed,
        outcome.skipped,
        len(outcome.failures),
        summary_path,
    )
    return summary_path


__all__ = [
    "CONDITION_FAILURE_POLICIES",
    "DEFAULT_DATASETS",
    "DEFAULT_MODEL",
    "DEFAULT_SUMMARY_ROOT",
    "SUMMARY_FIELDS",
    "SUMMARY_FILENAME_BY_MODEL",
    "TASK_NAME",
    "VmfRunPointer",
    "iter_vmf_run_pointers",
    "normalize_foldin_model",
    "parse_display_key",
    "run_vmf_foldin_theta",
    "select_runs",
]

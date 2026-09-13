"""Read-only artifact and alignment audit for topic-word evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.core.artifacts import PREPROCESSING_SELECTION_FILENAME, load_artifact_json
from src.data.preprocessing_selection import (
    parse_raw_doc_indices,
    resolve_preprocessing_selection,
)
from src.evaluation.word_based.topic_assignment import (
    fingerprint_jsonable,
    resolve_topic_word_protocol,
)

REQUIRED_ARTIFACT_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "bleilda": (("params/model.gensim", "model.gensim"),),
    "vmf_sentence_lda": (
        ("params.json",),
        ("kappa_per_topic.pkl",),
        ("mixture_weights.pkl",),
        ("component_means.pkl",),
    ),
    "sentlda": (("params/model_state.pkl", "model_state.pkl"),),
    "sentence_gaussianlda": (
        ("params/params.json", "params.json"),
        ("params/prior_mu.pkl", "prior_mu.pkl"),
        ("params/table_counts.pkl", "table_counts.pkl"),
        ("params/table_means.pkl", "table_means.pkl"),
    ),
    "mvtm": (
        ("params/params.json", "params.json"),
        ("params/kappa_per_topic.pkl", "kappa_per_topic.pkl"),
        ("params/mixture_weights.pkl", "mixture_weights.pkl"),
        ("params/component_means.pkl", "component_means.pkl"),
    ),
    "gaussianlda": (
        ("params/params.json", "params.json"),
        ("params/table_counts.pkl", "table_counts.pkl"),
        ("params/table_means.pkl", "table_means.pkl"),
        ("params/log_determinants.pkl", "log_determinants.pkl"),
        (
            "params/table_cholesky_ltriangular_mat.pkl",
            "table_cholesky_ltriangular_mat.pkl",
        ),
    ),
    "etm": (
        ("params/params.json", "params.json"),
        ("params/model_state.pt", "model_state.pt"),
        ("params/topic_word_scores.pkl", "topic_word_scores.pkl"),
        ("params/vocabulary.json", "vocabulary.json"),
        ("params/embeddings.pkl", "embeddings.pkl"),
    ),
    "ctm": (("params/tp.pkl", "tp.pkl"),),
    "sam": (
        ("params/params.json", "params.json"),
        ("params/topic_word_scores.pkl", "topic_word_scores.pkl"),
        ("params/vocabulary.json", "vocabulary.json"),
        ("params/sam.pkl", "sam.pkl"),
        ("params/idf.pkl", "idf.pkl"),
    ),
}
REQUIRED_ARTIFACT_GROUPS["sam_tf"] = REQUIRED_ARTIFACT_GROUPS["sam"]


@dataclass(frozen=True)
class ArtifactAuditResult:
    model: str
    protocol: str
    condition_dir: Path
    found: tuple[Path, ...]
    missing_groups: tuple[tuple[str, ...], ...]
    warnings: tuple[str, ...]
    inconsistencies: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing_groups and not self.inconsistencies


def audit_model_artifacts(*, model: str, condition_dir: Path) -> ArtifactAuditResult:
    normalized = "vmf_sentence_lda" if model == "vmf" else model
    groups = REQUIRED_ARTIFACT_GROUPS.get(normalized)
    if groups is None:
        raise ValueError(f"no artifact audit specification for model {model!r}")
    found: list[Path] = []
    missing: list[tuple[str, ...]] = []
    warnings: list[str] = []
    inconsistencies: list[str] = []
    for alternatives in groups:
        matches = [condition_dir / relative for relative in alternatives]
        match = next((path for path in matches if path.exists()), None)
        if match is None:
            missing.append(alternatives)
        else:
            found.append(match)

    if normalized == "ctm":
        checkpoints = sorted(condition_dir.rglob("epoch_*.pth"))
        if checkpoints:
            found.append(checkpoints[-1])
        else:
            missing.append(("params/contextualized_topic_model_*/epoch_*.pth",))

    if normalized in {"mvtm", "gaussianlda"}:
        metadata_path = condition_dir / "metadata.json"
        if not metadata_path.exists():
            missing.append(("metadata.json",))
        else:
            metadata = load_artifact_json(metadata_path)
            baseline_params = metadata.get("baseline_params")
            if not isinstance(baseline_params, dict) or not baseline_params.get(
                "word2vec"
            ):
                missing.append(("metadata.json:baseline_params.word2vec",))
            else:
                word2vec = str(baseline_params["word2vec"])
                # Filesystem-looking sources must exist; registry names (for
                # example gensim download identifiers) are resolved at load time.
                looks_like_path = "/" in word2vec or Path(word2vec).suffix in {
                    ".kv",
                    ".bin",
                    ".txt",
                    ".vec",
                }
                if looks_like_path:
                    candidates = (
                        Path(word2vec),
                        condition_dir / word2vec,
                        condition_dir / "params" / word2vec,
                    )
                    if not any(candidate.exists() for candidate in candidates):
                        missing.append((f"word2vec source: {word2vec}",))
                cache_dir = baseline_params.get("wikientvec_cache_dir")
                if cache_dir and not Path(str(cache_dir)).exists():
                    inconsistencies.append(
                        f"wikientvec_cache_dir does not exist: {cache_dir}"
                    )

    # vMF transform files are conditional on the saved transform mode
    # (params.json key "pre_normalize_transform": none | mean_center | whitening).
    if normalized == "vmf_sentence_lda":
        params_path = condition_dir / "params.json"
        if params_path.exists():
            params = load_artifact_json(params_path)
            transform = str(params.get("pre_normalize_transform", "none")).lower()
            mean_path = condition_dir / "embedding_transform_mean.pkl"
            whitening_path = condition_dir / "embedding_transform_whitening_matrix.pkl"
            if transform == "none":
                required: tuple[Path, ...] = ()
            elif transform == "mean_center":
                required = (mean_path,)
            elif transform == "whitening":
                required = (mean_path, whitening_path)
            else:
                required = ()
                inconsistencies.append(
                    f"unknown pre_normalize_transform {transform!r} in {params_path}"
                )
            for path in required:
                if path.exists():
                    found.append(path)
                else:
                    missing.append((path.name,))
            for path in (mean_path, whitening_path):
                if path not in required and path.exists():
                    inconsistencies.append(
                        f"transform artifact {path.name} exists but "
                        f"pre_normalize_transform={transform!r} does not use it"
                    )

    has_preprocessed = bool(
        list(condition_dir.rglob("train_preprocessed.pkl"))
        or list(condition_dir.rglob("preprocessed_corpus.pkl"))
    )
    if not has_preprocessed:
        warnings.append("no saved preprocessed training corpus found")
    inconsistencies.extend(
        _selection_inconsistencies(normalized=normalized, condition_dir=condition_dir)
    )
    return ArtifactAuditResult(
        model=normalized,
        protocol=resolve_topic_word_protocol(normalized),
        condition_dir=condition_dir,
        found=tuple(found),
        missing_groups=tuple(missing),
        warnings=tuple(warnings),
        inconsistencies=tuple(inconsistencies),
    )


def _selection_inconsistencies(*, normalized: str, condition_dir: Path) -> list[str]:
    """Lightweight structural audit of preprocessing_selection.json.

    Only JSON is inspected here; the strict document-count comparison against
    the preprocessed pickle happens when the corpus is actually loaded.
    """
    if normalized == "vmf_sentence_lda":
        layout = {
            "train": (
                condition_dir / "train_preprocessed.pkl",
                condition_dir / PREPROCESSING_SELECTION_FILENAME,
            ),
            "test": (
                condition_dir / "test_preprocessed.pkl",
                condition_dir / PREPROCESSING_SELECTION_FILENAME,
            ),
        }
    else:
        layout = {
            "train": (
                condition_dir / "params" / "preprocessed_corpus.pkl",
                condition_dir / "params" / PREPROCESSING_SELECTION_FILENAME,
            ),
            "test": (
                condition_dir / "infer" / "preprocessed_corpus.pkl",
                condition_dir / "infer" / PREPROCESSING_SELECTION_FILENAME,
            ),
        }
    problems: list[str] = []
    payload_cache: dict[Path, object] = {}
    for split, (preprocessed_path, selection_path) in layout.items():
        if not preprocessed_path.exists():
            continue
        if not selection_path.exists():
            problems.append(
                f"missing preprocessing selection for split {split!r}: "
                f"{selection_path}"
            )
            continue
        try:
            if selection_path not in payload_cache:
                payload_cache[selection_path] = load_artifact_json(selection_path)
            selection = resolve_preprocessing_selection(
                payload_cache[selection_path],
                split=split,
                selection_path=selection_path,
            )
            parse_raw_doc_indices(selection, selection_path=selection_path, split=split)
        except ValueError as error:
            problems.append(str(error))
    return problems


def validate_document_alignment(
    selections_by_model: Mapping[str, Sequence[str | int]],
) -> tuple[str | int, ...]:
    if not selections_by_model:
        raise ValueError("at least one model selection is required")
    iterator = iter(selections_by_model.items())
    first_model, first_ids = next(iterator)
    expected = tuple(first_ids)
    if len(expected) != len(set(expected)):
        raise ValueError(f"duplicate raw document IDs for model {first_model}")
    for model, raw_ids in iterator:
        actual = tuple(raw_ids)
        if actual != expected:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(expected, actual))
                    if pair[0] != pair[1]
                ),
                min(len(expected), len(actual)),
            )
            raise ValueError(
                f"raw document ID alignment differs for model {model} at index {mismatch}"
            )
    return expected


def evaluation_vocabulary_fingerprint(vocabulary: Sequence[str]) -> str:
    words = [str(word) for word in vocabulary]
    if len(words) != len(set(words)):
        raise ValueError("evaluation vocabulary contains duplicate words")
    return fingerprint_jsonable({"ordered_vocabulary": words})

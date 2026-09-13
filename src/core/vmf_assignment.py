"""The document-topic estimators a vMF Sentence LDA run can supply as features.

``hard``   training documents: normalized topic counts of the final Gibbs sweep;
           held-out documents: mean of the independent per-sentence posteriors
           (``doc_topic_{split}.pkl``).
``soft``   training documents: mean of the per-sentence posteriors under the fitted
           parameters; held-out documents: identical to ``hard``
           (``doc_topic_{split}_soft.pkl``).
``foldin`` both splits: collapsed fold-in with the fitted parameters frozen,
           ``theta_dk = (E[n_dk] + alpha_k) / (N_d + sum_j alpha_j)``, written by
           ``evaluation vmf-foldin-theta`` (``doc_topic_{split}_foldin.pkl``).
``foldincounts``
           both splits: the fold-in expected sentence-assignment proportions
           ``theta_dk = E[n_dk] / N_d`` without the Dirichlet smoothing of ``foldin``,
           read from the expected counts the same command writes
           (``doc_topic_{split}_foldin_counts.pkl``; the loaders row-normalize).
           The manuscript's vSLDA estimator since 2026-09-05.

The same estimators exist for MvTM (vLDA, the sampler applied to word vectors;
``VMF_FAMILY_MODELS``): ``hard`` is its ``params/table_counts_per_doc.pkl`` /
``infer/<category>.pkl`` (final-sweep counts / per-token argmax counts), ``soft``
its ``doc_topic_train_soft.pkl`` / ``<category>_doc_topic_soft.pkl``, and the
fold-in files are ``params/<category>_doc_topic_foldin[_counts].pkl`` and
``infer/<category>_doc_topic_foldin[_counts].pkl`` (token units).
"""

from __future__ import annotations

VMF_ASSIGNMENTS: tuple[str, ...] = ("hard", "soft", "foldin", "foldincounts")
# The models whose document-topic features depend on ``vmf_assignment``.
VMF_FAMILY_MODELS: tuple[str, ...] = ("vmf_sentence_lda", "mvtm")
# The estimator the evaluation commands use when none is named: the manuscript's
# (``foldincounts``) for the classification features and the entropy diagnostics of
# vMF runs. Results written before ``vmf_assignment`` was recorded in their metadata
# were produced with the hard estimator, which LEGACY_VMF_ASSIGNMENT names.
DEFAULT_VMF_ASSIGNMENT = "foldincounts"
DEFAULT_DOC_TOPIC_SOURCE = "foldincounts"
LEGACY_VMF_ASSIGNMENT = "hard"
VMF_DOC_TOPIC_SUFFIX: dict[str, str] = {
    "hard": "",
    "soft": "_soft",
    "foldin": "_foldin",
    "foldincounts": "_foldin_counts",
}
VMF_DISPLAY_SUFFIX: dict[str, str] = {
    "hard": "",
    "soft": " (soft)",
    "foldin": " (fold-in)",
    "foldincounts": " (fold-in counts)",
}


def normalize_vmf_assignment(value: object) -> str:
    """Return the canonical assignment name or raise ``ValueError``."""

    text = str(value).strip().lower().replace("-", "").replace("_", "")
    if text in VMF_ASSIGNMENTS:
        return text
    raise ValueError(
        f"Unsupported vmf_assignment {value!r}; use one of {', '.join(VMF_ASSIGNMENTS)}"
    )


def vmf_doc_topic_filename(split: str, assignment: str) -> str:
    """``doc_topic_<split><suffix>.pkl`` for the given estimator."""

    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    suffix = VMF_DOC_TOPIC_SUFFIX[normalize_vmf_assignment(assignment)]
    return f"doc_topic_{split}{suffix}.pkl"


def mvtm_doc_topic_relpath(split: str, assignment: str, *, category: str) -> str:
    """The MvTM document-topic file of ``assignment``, relative to the run directory."""

    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    key = normalize_vmf_assignment(assignment)
    split_dir = "params" if split == "train" else "infer"
    if key == "hard":
        name = "table_counts_per_doc.pkl" if split == "train" else f"{category}.pkl"
    elif key == "soft":
        name = (
            "doc_topic_train_soft.pkl"
            if split == "train"
            else f"{category}_doc_topic_soft.pkl"
        )
    else:
        name = f"{category}_doc_topic{VMF_DOC_TOPIC_SUFFIX[key]}.pkl"
    return f"{split_dir}/{name}"


def vmf_family_doc_topic_relpath(
    split: str, assignment: str, *, model: str, category: str | None = None
) -> str:
    """Run-relative document-topic file of a vMF-family model under ``assignment``."""

    if model == "vmf_sentence_lda":
        return vmf_doc_topic_filename(split, assignment)
    if model == "mvtm":
        if not category:
            raise ValueError("the MvTM document-topic path needs the category")
        return mvtm_doc_topic_relpath(split, assignment, category=category)
    raise ValueError(f"{model!r} is not a vMF-family model ({VMF_FAMILY_MODELS})")


__all__ = [
    "VMF_ASSIGNMENTS",
    "VMF_FAMILY_MODELS",
    "mvtm_doc_topic_relpath",
    "vmf_family_doc_topic_relpath",
    "VMF_DISPLAY_SUFFIX",
    "VMF_DOC_TOPIC_SUFFIX",
    "normalize_vmf_assignment",
    "vmf_doc_topic_filename",
]

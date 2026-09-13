"""Model identity and series appearance, shared by every figure and table.

Two things live here because both the sample-efficiency figures
(:mod:`src.evaluation.classification.plot_limited`) and the topic-count sweep
(:mod:`src.evaluation.reports.topic_sweep_plot`) need them and must not drift:

*identity*
    A model shows up under three different spellings depending on where it was
    written. Classification ``*.scores.json`` uses the display name with the
    classifier and the encoder appended (``"Blei LDA [SVM]"``); the coherence
    ``summary.csv`` uses the runner key (``"bleilda"``), except for the proposed
    method, which it writes as ``"vmf"`` with an empty ``runner_family``.
    :func:`canonical_model_key` folds all three onto the selector key used on
    the command line (``bleilda``, ``vmf_sentence_lda``, ...).

*appearance*
    Line style comes from whether topics are assigned per sentence (solid) or
    not (dashed); colour comes from the model family, i.e. what a topic emits.
    Marker shape is per model so series stay separable in grayscale.

This module deliberately does not import matplotlib at module scope: the
aggregation path (:mod:`src.evaluation.reports.topic_sweep`) needs the identity
half without pulling in a plotting backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "CANONICAL_MODEL_KEYS",
    "DASHED",
    "EXCLUDED_MODELS",
    "FAMILY_COLORS",
    "LINEWIDTH",
    "MARKER",
    "MARKEREDGEWIDTH",
    "MARKERSIZE",
    "MARKER_SIZE_SCALE",
    "MODEL_COLOR_OVERRIDES",
    "MODEL_LABELS",
    "MODEL_MARKERS",
    "MODEL_ORDER",
    "MODEL_TAXONOMY",
    "ModelStyle",
    "PROPOSED_KEY",
    "PROPOSED_LABEL",
    "PROPOSED_LINEWIDTH",
    "PROPOSED_MARKERSIZE",
    "UNIT_LINESTYLES",
    "base_model_name",
    "canonical_model_key",
    "display_model_name",
    "fallback_color",
    "label_for_key",
    "model_sort_key",
    "model_style",
]

PROPOSED_LABEL = "vSLDA (proposed)"
PROPOSED_KEY = "vmf_sentence_lda"
PROPOSED_DISPLAY_NAME = "vMF Sentence LDA"

MODEL_LABELS = {
    "Blei LDA": "LDA",
    "SAM (tf-idf)": "SAM (tf-idf)",
    "SAM": "SAM",
    "sentLDA": "SentLDA",
    "Gaussian LDA": "GLDA",
    "MvTM": "vLDA",
    "ETM": "ETM",
    "Contextual TM": "ConTM",
    "SenClu": "SenClu",
    "Sentence LDA": "GSLDA",
    PROPOSED_DISPLAY_NAME: PROPOSED_LABEL,
}
MODEL_ORDER = [
    "LDA",
    "SentLDA",
    "GLDA",
    "SAM",
    "SAM (tf-idf)",
    "vLDA",
    "ETM",
    "ConTM",
    "GSLDA",
    PROPOSED_LABEL,
]
# Display labels that are never drawn nor listed in the legend.
EXCLUDED_MODELS = frozenset({"SenClu"})

# ``MODEL_TAXONOMY`` maps display label -> (sentence_level, family).
MODEL_TAXONOMY: Dict[str, Tuple[bool, str]] = {
    "LDA": (False, "categorical"),
    "SentLDA": (True, "categorical"),
    "GLDA": (False, "gaussian"),
    "SAM": (False, "vmf_bow"),
    "SAM (tf-idf)": (False, "vmf_bow_tf"),
    "vLDA": (False, "vmf_word"),
    "ETM": (False, "neural_word"),
    "ConTM": (False, "neural_contextual"),
    "GSLDA": (True, "gaussian"),
    PROPOSED_LABEL: (True, "vmf"),
}
# Okabe-Ito palette (colour-vision-deficiency safe) for baselines. Red is
# reserved for the proposed method alone; the other two vMF-family baselines get
# the neighbouring warm hues so the family reads as a group without any of them
# competing with vSLDA. The classic bag-of-words models are black.
#
# ``vmf_word`` (MvTM, vMF over word embeddings) and ``vmf_bow`` ("SAM": the
# manuscript's SAM, vMF over the L2-normalized term-frequency vector of a
# document) were split out of a single ``vmf`` family: sharing it would have
# given all three the same colour *and* -- for the two document-level ones --
# the same dashed linestyle, leaving only the marker to tell them apart.
FAMILY_COLORS = {
    "vmf": "#D62728",
    "vmf_word": "#D55E00",
    "vmf_bow": "#E69F00",
    # "SAM (tf-idf)" (the tf-idf weighting of the original study, not in the
    # manuscript) shares SAM's dashed linestyle, so it needs its own hue or the
    # two would be one (linestyle, colour) cell; Okabe-Ito blue is the last
    # CVD-safe hue not yet taken and sits far from the warm vMF trio.
    "vmf_bow_tf": "#0072B2",
    "categorical": "#000000",
    "gaussian": "#56B4E9",
    "neural_word": "#009E73",
    "neural_contextual": "#CC79A7",
}
# Explicit dash pattern so the period survives shrinking to the page width.
DASHED = (0, (4.0, 1.5))
UNIT_LINESTYLES: Dict[bool, object] = {True: "-", False: DASHED}
MODEL_COLOR_OVERRIDES = {
    label: FAMILY_COLORS[family] for label, (_, family) in MODEL_TAXONOMY.items()
}
MARKER = "o"  # fallback for labels outside ``MODEL_ORDER``
# One marker shape per model so that series stay separable even where colour and
# line style coincide or the figure is printed in grayscale.
#
# Written out explicitly rather than zipped against ``MODEL_ORDER``: a positional
# zip silently truncates when a model is added (the extra label falls back to the
# shared ``MARKER``), and inserting a model anywhere but the end reassigns every
# downstream marker -- including the proposed method's -- which would change
# already-published figures.
MODEL_MARKERS: Dict[str, str] = {
    "LDA": "o",
    "SentLDA": "s",
    "GLDA": "^",
    "SAM": "P",
    "SAM (tf-idf)": "X",
    "vLDA": "D",
    "ETM": "v",
    "ConTM": "x",
    "GSLDA": "+",
    PROPOSED_LABEL: "*",
}
# Stroke-only and star markers look smaller than filled ones at equal size.
MARKER_SIZE_SCALE = {"x": 1.3, "+": 1.45, "*": 1.35}
LINEWIDTH = 1.0
PROPOSED_LINEWIDTH = 1.6
MARKERSIZE = 3.0
PROPOSED_MARKERSIZE = 3.5
MARKEREDGEWIDTH = 0.8


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

_CLASSIFIER_SUFFIX_RE = re.compile(r"\s*\[(?:SVM|LogReg)\]\s*$")
_VARIANT_SUFFIX_RE = re.compile(r"\s*\[[^\]]+\]\s*$")
# ``(soft)`` / ``(fold-in)`` / ``(fold-in counts)`` name the vSLDA document-topic estimator,
# not a model.
_SOFT_SUFFIX_RE = re.compile(r"\s*\((?:soft|fold-in|fold-in counts)\)\s*$")


def base_model_name(name: str) -> str:
    """``"vMF Sentence LDA [c1_bge] [SVM]"`` -> ``"vMF Sentence LDA"``."""
    normalized = _CLASSIFIER_SUFFIX_RE.sub("", str(name).strip())
    while True:
        without_variant = _VARIANT_SUFFIX_RE.sub("", normalized)
        if without_variant == normalized:
            break
        normalized = without_variant
    return _SOFT_SUFFIX_RE.sub("", normalized).strip()


def display_model_name(name: str) -> str:
    """Short label used in legends, e.g. ``"Blei LDA [SVM]"`` -> ``"LDA"``."""
    base = base_model_name(name)
    return MODEL_LABELS.get(base, base)


def model_sort_key(name: str) -> tuple[int, str]:
    display = display_model_name(name)
    try:
        return (MODEL_ORDER.index(display), display)
    except ValueError:
        return (len(MODEL_ORDER), display)


def _runner_display_names() -> dict[str, str]:
    """Selector key -> display name, read from the baseline runner registry.

    Imported lazily so that reading a CSV never depends on the baseline
    adapters being importable.
    """
    from src.baselines.registry import RUNNERS

    names = {key: spec.display_name for key, spec in RUNNERS.items()}
    names[PROPOSED_KEY] = PROPOSED_DISPLAY_NAME
    return names


# Spellings of the proposed method that are not derivable from the registry.
# ``vmf`` is what the coherence summary writes in its ``model`` column.
_MODEL_KEY_ALIASES = {
    "vmf": PROPOSED_KEY,
    "vslda": PROPOSED_KEY,
    "gslda": "sentence_gaussianlda",
    "bertopic": "bertopic_kmeans",
}


def canonical_model_key(value: str) -> str:
    """Selector key for any spelling of a model.

    Accepts a selector key (``"bleilda"``), the coherence ``model`` column
    (``"vmf"``), or a classification display name with its variant and
    classifier suffixes (``"ETM [googlenews300] [SVM]"``). Unknown values are
    returned lowercased and underscored rather than raising, so an aggregation
    over old artifacts degrades to an unmapped row instead of failing.
    """
    raw = str(value).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in _MODEL_KEY_ALIASES:
        return _MODEL_KEY_ALIASES[lowered]
    display_names = _runner_display_names()
    # Selector keys are matched case-sensitively because "SAM" is now the display
    # name of the tf runner while "sam" remains the key of the tf-idf runner.
    # Lowercasing first would resolve every classification row labelled
    # "SAM [LogReg]" to the tf-idf tree.  The coherence ``model`` column is written
    # in lowercase keys, so it still takes this branch.
    if raw in display_names:
        return raw
    base = base_model_name(raw)
    for key, display in display_names.items():
        if base.lower() == display.lower():
            return key
    # A short label ("LDA", "vSLDA (proposed)") round-trips through MODEL_LABELS.
    for display, label in MODEL_LABELS.items():
        if base.lower() in {label.lower(), display.lower()}:
            for key, registry_display in display_names.items():
                if registry_display.lower() == display.lower():
                    return key
    return lowered.replace(" ", "_").replace("-", "_")


def label_for_key(key: str) -> str:
    """Selector key -> short legend label (``"bleilda"`` -> ``"LDA"``)."""
    display_names = _runner_display_names()
    display = display_names.get(str(key).strip().lower())
    if display is None:
        return str(key)
    return MODEL_LABELS.get(display, display)


CANONICAL_MODEL_KEYS = (
    PROPOSED_KEY,
    "bleilda",
    "sam",
    "sam_tf",
    "sentlda",
    "etm",
    "ctm",
    "gaussianlda",
    "sentence_gaussianlda",
    "mvtm",
)


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelStyle:
    color: object
    linestyle: object
    linewidth: float
    markersize: float
    zorder: int
    marker: str = MARKER

    def line_kwargs(self) -> Dict[str, object]:
        return {
            "color": self.color,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "marker": self.marker,
            "markersize": self.markersize * MARKER_SIZE_SCALE.get(self.marker, 1.0),
            "markeredgewidth": MARKEREDGEWIDTH,
        }


def _stable_label_index(label: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(label))


def fallback_color(label: str, colormap: str) -> object:
    """Mechanical colour for labels outside ``MODEL_TAXONOMY``."""
    import matplotlib.pyplot as plt

    try:
        color_index = MODEL_ORDER.index(label)
    except ValueError:
        color_index = len(MODEL_ORDER) + _stable_label_index(label)

    cmap = plt.get_cmap(colormap)
    color_count = getattr(cmap, "N", None)
    if isinstance(color_count, int) and color_count <= 20:
        return cmap(color_index % color_count)

    scale_count = max(len(MODEL_ORDER), 2)
    return cmap((color_index % scale_count) / (scale_count - 1))


def model_style(label: str, colormap: str) -> ModelStyle:
    """Single source of truth for series appearance (panels and legend)."""
    taxonomy = MODEL_TAXONOMY.get(label)
    if taxonomy is None:
        color = MODEL_COLOR_OVERRIDES.get(label, fallback_color(label, colormap))
        return ModelStyle(
            color=color,
            linestyle=UNIT_LINESTYLES[True],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            zorder=2,
            marker=MODEL_MARKERS.get(label, MARKER),
        )

    sentence_level, family = taxonomy
    proposed = label == PROPOSED_LABEL
    return ModelStyle(
        color=FAMILY_COLORS[family],
        linestyle=UNIT_LINESTYLES[sentence_level],
        linewidth=PROPOSED_LINEWIDTH if proposed else LINEWIDTH,
        markersize=PROPOSED_MARKERSIZE if proposed else MARKERSIZE,
        zorder=3 if proposed else 2,
        marker=MODEL_MARKERS.get(label, MARKER),
    )

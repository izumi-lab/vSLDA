"""Guards for the shared figure identity/appearance table."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from src.evaluation.reports import model_style  # noqa: E402


def test_every_ordered_model_has_its_own_marker() -> None:
    """Regression guard for a positional ``zip`` that silently truncated.

    ``MODEL_MARKERS`` used to be ``dict(zip(MODEL_ORDER, <8-tuple>))``.  Adding a
    ninth model dropped it from the mapping without any error -- it just fell
    back to the shared default marker and became indistinguishable from LDA in
    grayscale -- and inserting a model anywhere but the end reassigned every
    downstream marker, including the proposed method's.
    """

    assert set(model_style.MODEL_MARKERS) == set(model_style.MODEL_ORDER)
    assert len(model_style.MODEL_MARKERS) == len(model_style.MODEL_ORDER)
    assert len(set(model_style.MODEL_MARKERS.values())) == len(
        model_style.MODEL_MARKERS
    )


def test_every_ordered_model_has_a_taxonomy_entry() -> None:
    assert set(model_style.MODEL_TAXONOMY) == set(model_style.MODEL_ORDER)
    for label, (_, family) in model_style.MODEL_TAXONOMY.items():
        assert family in model_style.FAMILY_COLORS, label


def test_red_is_reserved_for_the_proposed_method() -> None:
    """The most salient colour must identify exactly one series.

    SAM and MvTM are both vMF-family document-level models, so before the family
    split they shared the proposed method's red *and* its dashed linestyle.
    """

    red = model_style.FAMILY_COLORS["vmf"]
    wearing_red = [
        label
        for label, colour in model_style.MODEL_COLOR_OVERRIDES.items()
        if colour == red
    ]
    assert wearing_red == [model_style.PROPOSED_LABEL]


def test_vmf_family_baselines_are_visually_distinct() -> None:
    colours = {
        label: model_style.MODEL_COLOR_OVERRIDES[label]
        for label in ("SAM", "vLDA", model_style.PROPOSED_LABEL)
    }
    assert len(set(colours.values())) == 3


def test_sam_is_registered_as_a_document_level_model() -> None:
    """ "SAM" is the tf condition (runner ``sam_tf``); tf-idf is the named variant.

    tf is what the other bag-of-words baselines see -- LDA and sentLDA are fitted on
    raw counts -- whereas idf down-weights precisely the frequent words that NPMI
    also penalizes, which aligns tf-idf with the coherence metric in a way the other
    models do not get.  The primary styling follows the reported condition.
    """

    sentence_level, family = model_style.MODEL_TAXONOMY["SAM"]
    assert sentence_level is False  # dashed linestyle, like LDA and ETM
    assert family == "vmf_bow"
    assert "sam_tf" in model_style.CANONICAL_MODEL_KEYS
    assert model_style.canonical_model_key("SAM") == "sam_tf"
    assert model_style.label_for_key("sam_tf") == "SAM"
    assert "SAM" not in model_style.EXCLUDED_MODELS


def test_sam_tfidf_variant_has_its_own_colour_and_marker() -> None:
    """Distinct hue and marker, so that the two SAM input representations can be
    told apart on one axis even though both are dashed document-level series."""

    idf_level, idf_family = model_style.MODEL_TAXONOMY["SAM (tf-idf)"]
    sam_level, sam_family = model_style.MODEL_TAXONOMY["SAM"]
    assert idf_level is sam_level is False  # same dashed (document-level) linestyle
    # ... which is exactly why the colour must differ: (linestyle, colour) is
    # the cell the figures rely on to keep series apart.
    assert idf_family != sam_family
    assert (
        model_style.FAMILY_COLORS[idf_family] != model_style.FAMILY_COLORS[sam_family]
    )
    assert model_style.MODEL_MARKERS["SAM (tf-idf)"] != model_style.MODEL_MARKERS["SAM"]
    assert "sam" in model_style.CANONICAL_MODEL_KEYS
    assert model_style.canonical_model_key("SAM (tf-idf)") == "sam"
    assert model_style.label_for_key("sam") == "SAM (tf-idf)"
    assert "SAM (tf-idf)" in model_style.MODEL_ORDER


def test_base_model_name_strips_the_estimator_suffixes() -> None:
    """``(soft)`` and ``(fold-in)`` name the vSLDA document-topic estimator."""

    assert (
        model_style.base_model_name("vMF Sentence LDA [c1_bge] [SVM]")
        == "vMF Sentence LDA"
    )
    assert (
        model_style.base_model_name("vMF Sentence LDA (soft) [c1_minilm] [LogReg]")
        == "vMF Sentence LDA"
    )
    assert (
        model_style.base_model_name("vMF Sentence LDA (fold-in) [c1_minilm] [LogReg]")
        == "vMF Sentence LDA"
    )
    assert (
        model_style.base_model_name(
            "vMF Sentence LDA (fold-in counts) [c1_minilm] [LogReg]"
        )
        == "vMF Sentence LDA"
    )

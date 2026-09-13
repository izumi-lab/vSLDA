"""Result-path label of vMF hyperparameter overrides (src/core/vmf_variant.py)."""

from __future__ import annotations

import pytest

from src.core.vmf_variant import (
    format_vmf_parameter_variant,
    is_vmf_parameter_variant,
    normalize_vmf_parameter_variant,
    parse_vmf_parameter_variant,
    vmf_variant_matches,
)

DEFAULTS = {
    "kappa0": 10.0,
    "alpha0": 2.5,
    "gibbs_sweeps": 20,
    "num_samples": 8,
    "num_iterations": 10,
}


def test_no_override_means_no_label() -> None:
    assert format_vmf_parameter_variant(DEFAULTS, ()) is None
    assert format_vmf_parameter_variant(DEFAULTS, None) is None


def test_labels_follow_the_prior_scale_convention() -> None:
    assert (
        format_vmf_parameter_variant({**DEFAULTS, "kappa0": 100.0}, ("kappa0",))
        == "kappa0-100"
    )
    assert (
        format_vmf_parameter_variant({**DEFAULTS, "alpha0": 0.1}, ("alpha0",))
        == "alpha0-0p1"
    )
    assert (
        format_vmf_parameter_variant(
            {**DEFAULTS, "gibbs_sweeps": 40}, ("gibbs_sweeps",)
        )
        == "zeta-40"
    )
    assert (
        format_vmf_parameter_variant({**DEFAULTS, "num_samples": 16}, ("num_samples",))
        == "b-16"
    )
    assert (
        format_vmf_parameter_variant(
            {**DEFAULTS, "num_iterations": 5}, ("num_iterations",)
        )
        == "t-5"
    )


def test_several_overrides_join_in_canonical_order() -> None:
    label = format_vmf_parameter_variant(
        {**DEFAULTS, "kappa0": 1.0, "num_iterations": 30},
        ("num_iterations", "kappa0"),
    )
    assert label == "kappa0-1_t-30"
    assert parse_vmf_parameter_variant(label) == {"kappa0": 1.0, "num_iterations": 30.0}


def test_unknown_or_vector_overrides_are_rejected() -> None:
    with pytest.raises(ValueError):
        format_vmf_parameter_variant(DEFAULTS, ("max_kappa",))
    with pytest.raises(ValueError):
        format_vmf_parameter_variant({**DEFAULTS, "alpha0": [1.0, 2.0]}, ("alpha0",))


def test_is_vmf_parameter_variant_recognizes_labels_only() -> None:
    assert is_vmf_parameter_variant("kappa0-100")
    assert is_vmf_parameter_variant("alpha0-0p1_zeta-40_b-16_t-5")
    assert not is_vmf_parameter_variant("")
    assert not is_vmf_parameter_variant("minilm")
    assert not is_vmf_parameter_variant("psi0-0p1")
    # out of canonical order or repeated keys are not labels this module writes
    assert not is_vmf_parameter_variant("t-5_kappa0-100")
    assert not is_vmf_parameter_variant("t-5_t-10")


def test_normalize_and_match() -> None:
    assert normalize_vmf_parameter_variant(None) is None
    assert normalize_vmf_parameter_variant("  ") is None
    assert normalize_vmf_parameter_variant("zeta-40") == "zeta-40"
    with pytest.raises(ValueError):
        normalize_vmf_parameter_variant("kappa-100")
    assert vmf_variant_matches(None, None)
    assert vmf_variant_matches("", None)
    assert vmf_variant_matches("zeta-40", "zeta-40")
    assert not vmf_variant_matches(None, "zeta-40")
    assert not vmf_variant_matches("zeta-40", None)


def test_vmf_assignment_constants_and_normalization() -> None:
    from src.core.vmf_assignment import (
        VMF_ASSIGNMENTS,
        normalize_vmf_assignment,
        vmf_doc_topic_filename,
    )

    assert VMF_ASSIGNMENTS == ("hard", "soft", "foldin", "foldincounts")
    assert normalize_vmf_assignment("fold-in-counts") == "foldincounts"
    assert (
        vmf_doc_topic_filename("train", "foldincounts")
        == "doc_topic_train_foldin_counts.pkl"
    )
    assert normalize_vmf_assignment("fold-in") == "foldin"
    assert normalize_vmf_assignment(" Hard ") == "hard"
    assert vmf_doc_topic_filename("test", "foldin") == "doc_topic_test_foldin.pkl"
    assert vmf_doc_topic_filename("train", "hard") == "doc_topic_train.pkl"
    import pytest

    with pytest.raises(ValueError):
        normalize_vmf_assignment("argmax")
    with pytest.raises(ValueError):
        vmf_doc_topic_filename("dev", "hard")


def test_saem_settings_are_sweepable_after_the_mcem_keys() -> None:
    """The SAEM burn-in T_0 (t0-<n>) and decay a (a-<v>) label like the other axes."""
    saem = {**DEFAULTS, "saem_burn_in": 5, "saem_decay": 1.0}
    assert (
        format_vmf_parameter_variant({**saem, "saem_burn_in": 8}, ("saem_burn_in",))
        == "t0-8"
    )
    assert (
        format_vmf_parameter_variant({**saem, "saem_decay": 0.8}, ("saem_decay",))
        == "a-0p8"
    )
    label = format_vmf_parameter_variant(
        {**saem, "num_iterations": 30, "saem_burn_in": 10},
        ("saem_burn_in", "num_iterations"),
    )
    assert label == "t-30_t0-10"
    assert parse_vmf_parameter_variant(label) == {
        "num_iterations": 30.0,
        "saem_burn_in": 10.0,
    }
    assert parse_vmf_parameter_variant("t0-2") == {"saem_burn_in": 2.0}
    assert parse_vmf_parameter_variant("a-0p6") == {"saem_decay": 0.6}
    assert is_vmf_parameter_variant("t0-8_a-0p8")
    assert not is_vmf_parameter_variant("a-0p8_t0-8")  # canonical order
    assert not is_vmf_parameter_variant("t0-8_t0-8")
    assert normalize_vmf_parameter_variant("t0-8") == "t0-8"

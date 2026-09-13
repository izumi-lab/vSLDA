"""SAEM variant of the vMF Sentence LDA trainer (Kuhn & Lavielle 2004).

Covers the Newton concentration solver, the step-size schedule, the Robbins--Monro
averaging of the E-step statistics, the alpha freeze after burn-in, and the fact that
saem_burn_in=None with kappa_solver="banerjee" leaves the plain MCEM unchanged (the default
is the SAEM with T_0 = 5 and the Newton kappa solver).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.core.progress import NullProgressReporter
from src.models.vmf_sentence_lda import (
    VMFLDATrainer,
    mean_resultant_length,
    solve_kappa_newton,
)


class _FixedEncoder:
    def __init__(self, mapping: dict[str, np.ndarray], dim: int) -> None:
        self._mapping = mapping
        self._dim = dim

    def encode(self, sentences) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self._dim), dtype=np.float64)
        return np.vstack([self._mapping[item] for item in sentences]).astype(np.float64)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def _normalize(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return arr / np.linalg.norm(arr)


_MAPPING = {
    "a": _normalize(np.array([1.0, 0.0, 0.0])),
    "b": _normalize(np.array([0.0, 1.0, 0.0])),
    "c": _normalize(np.array([0.0, 0.0, 1.0])),
}


def _build(corpus=None, *, seed: int | None = None, **kwargs: object) -> VMFLDATrainer:
    if seed is not None:
        np.random.seed(seed)
    return VMFLDATrainer(
        corpus=corpus or [["a"], ["b"], ["c"]],
        encoder=_FixedEncoder(_MAPPING, dim=3),
        num_topics=2,
        alpha=1.0,
        kappa=2.0,
        num_components=1,
        pre_normalize_transform="none",
        log=logging.getLogger("test-vmf-saem"),
        progress=NullProgressReporter(),
        **kwargs,
    )


def test_newton_kappa_solves_mean_resultant_equation_within_tanabe_bound() -> None:
    for dim, rbar in ((3, 0.4), (384, 0.55), (768, 0.7)):
        banerjee = (rbar * dim - rbar**3) / (1.0 - rbar**2)
        kappa = solve_kappa_newton(rbar, dim, banerjee)
        assert abs(float(mean_resultant_length(kappa, dim)) - rbar) < 1e-10
        # Tanabe et al. (2007): the Banerjee value is within a factor 2/(M-2) of the root
        assert abs(kappa - banerjee) / kappa <= 2.0 / max(dim - 2, 1) + 1e-9


def test_trainer_defaults_to_saem_with_newton_kappa() -> None:
    trainer = _build()
    assert trainer.saem_burn_in == 5
    assert trainer.saem_decay == 1.0
    assert trainer.kappa_solver == "newton"


def test_m_step_newton_solver_reaches_exact_root() -> None:
    nk = np.array([4.0, 3.0])
    nk_comp = nk[:, None].copy()
    r = np.zeros((2, 1, 3))
    r[0, 0] = [2.0, 0.0, 0.0]
    r[1, 0] = [0.0, 1.5, 0.0]
    rbar = np.array([0.5, 0.5])
    expected_banerjee = (rbar * 3.0 - rbar**3) / (1.0 - rbar**2)

    banerjee = _build(kappa_solver="banerjee")
    assert banerjee.kappa_solver == "banerjee"
    banerjee._apply_m_step_updates(nk=nk, nk_comp=nk_comp, r=r)
    assert np.allclose(banerjee.kappa_per_topic, expected_banerjee)

    newton = _build(kappa_solver="newton")
    newton._apply_m_step_updates(nk=nk, nk_comp=nk_comp, r=r)
    assert np.allclose(
        mean_resultant_length(newton.kappa_per_topic, 3), rbar, atol=1e-10
    )
    assert not np.allclose(newton.kappa_per_topic, expected_banerjee)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(saem_burn_in=-1), "saem_burn_in"),
        (dict(saem_decay=0.5), "saem_decay"),
        (dict(saem_decay=1.5), "saem_decay"),
        (dict(kappa_solver="bisect"), "kappa_solver"),
    ],
)
def test_trainer_rejects_invalid_saem_settings(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build(**kwargs)


def test_saem_step_size_and_averaging_follow_robbins_monro_recursion() -> None:
    trainer = _build(saem_burn_in=None)
    assert [trainer.saem_step_size(t) for t in (1, 2, 3, 10)] == [1.0, 1.0, 1.0, 1.0]
    trainer.saem_burn_in = 2
    trainer.saem_decay = 1.0
    assert [trainer.saem_step_size(t) for t in (1, 2, 3, 4, 5)] == [
        1.0,
        1.0,
        1.0,
        0.5,
        1.0 / 3.0,
    ]
    trainer.saem_decay = 0.7
    assert trainer.saem_step_size(4) == pytest.approx(2.0**-0.7)

    stats = [
        (np.array([4.0, 2.0]), np.array([[4.0], [2.0]]), np.ones((2, 1, 3))),
        (np.array([2.0, 4.0]), np.array([[2.0], [4.0]]), 3.0 * np.ones((2, 1, 3))),
        (np.array([6.0, 0.0]), np.array([[6.0], [0.0]]), np.zeros((2, 1, 3))),
    ]
    # burn-in (gamma == 1): the state is the current statistic itself
    s = trainer._saem_update(1.0, *stats[0])
    assert s[0] is stats[0][0]
    # gamma = 1/2 -> mean of the first two
    s = trainer._saem_update(0.5, *stats[1])
    assert np.allclose(s[0], [3.0, 3.0]) and np.allclose(s[2], 2.0)
    # gamma = 1/3 -> equal-weight mean of the three statistics
    s = trainer._saem_update(1.0 / 3.0, *stats[2])
    assert np.allclose(s[0], np.mean([st[0] for st in stats], axis=0))
    assert np.allclose(s[2], np.mean([st[2] for st in stats], axis=0))


def test_saem_freezes_alpha_after_burn_in_and_plain_mcem_is_unchanged() -> None:
    corpus = [["a", "b"], ["b", "c"], ["c", "a"]]

    plain = _build(corpus, seed=7, saem_burn_in=None, kappa_solver="banerjee")
    plain.sample(4, num_sweeps=2, num_samples=1)
    saem = _build(
        corpus, seed=7, saem_burn_in=2, saem_decay=1.0, kappa_solver="banerjee"
    )
    saem.sample(4, num_sweeps=2, num_samples=1)

    assert [d.alpha_frozen for d in plain.iteration_diagnostics] == [False] * 4
    assert [d.alpha_updated for d in plain.iteration_diagnostics] == [True] * 4
    assert [d.saem_gamma for d in plain.iteration_diagnostics] == [1.0] * 4
    assert [d.kappa_solver for d in plain.iteration_diagnostics] == ["banerjee"] * 4

    assert [d.alpha_frozen for d in saem.iteration_diagnostics] == [
        False,
        False,
        True,
        True,
    ]
    assert [d.alpha_updated for d in saem.iteration_diagnostics] == [
        True,
        True,
        False,
        False,
    ]
    assert [d.saem_gamma for d in saem.iteration_diagnostics] == [1.0, 1.0, 1.0, 0.5]
    assert all(np.isfinite(d.saem_objective) for d in saem.iteration_diagnostics)
    assert saem.saem_gamma == [1.0, 1.0, 1.0, 0.5]

    # through the burn-in the SAEM run is the plain run
    ref = _build(corpus, seed=7, saem_burn_in=None)
    ref.sample(2, num_sweeps=2, num_samples=1)
    burn = _build(corpus, seed=7, saem_burn_in=2, saem_decay=1.0)
    burn.sample(2, num_sweeps=2, num_samples=1)
    assert np.array_equal(burn.topic_means, ref.topic_means)
    assert np.array_equal(burn.kappa_per_topic, ref.kappa_per_topic)
    assert np.array_equal(burn.alpha, ref.alpha)
    assert all(
        np.array_equal(x, y)
        for x, y in zip(burn.topic_assignments, ref.topic_assignments)
    )

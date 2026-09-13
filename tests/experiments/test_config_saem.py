from __future__ import annotations

import pytest

from src.experiments.config_parsers import parse_train_config


def test_parse_train_config_defaults_to_saem_with_newton_kappa() -> None:
    train = parse_train_config({"train": {"num_topics": 10, "num_iterations": 1}})
    assert train.saem_burn_in == 5
    assert train.saem_decay == 1.0
    assert train.kappa_solver == "newton"


def test_parse_train_config_null_burn_in_restores_plain_mcem() -> None:
    train = parse_train_config(
        {
            "train": {
                "num_topics": 10,
                "num_iterations": 1,
                "saem_burn_in": None,
                "kappa_solver": "banerjee",
            }
        }
    )
    assert train.saem_burn_in is None
    assert train.kappa_solver == "banerjee"


def test_parse_train_config_saem_settings() -> None:
    train = parse_train_config(
        {
            "train": {
                "num_topics": 10,
                "num_iterations": 10,
                "saem_burn_in": 5,
                "saem_decay": 0.7,
                "kappa_solver": "newton",
            }
        }
    )
    assert train.saem_burn_in == 5
    assert train.saem_decay == 0.7
    assert train.kappa_solver == "newton"


@pytest.mark.parametrize(
    "train, message",
    [
        ({"saem_burn_in": -1}, "saem_burn_in"),
        ({"saem_decay": 0.5}, "saem_decay"),
        ({"saem_decay": 1.2}, "saem_decay"),
        ({"kappa_solver": "bisect"}, "kappa_solver"),
    ],
)
def test_parse_train_config_rejects_invalid_saem_settings(
    train: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_train_config({"train": {"num_topics": 10, "num_iterations": 1, **train}})

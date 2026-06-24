from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.baselines.models.gaussian_internal import utils


def test_gaussian_internal_utils_imports_without_choldate() -> None:
    logger = utils.get_logger("test-gaussian-utils")
    assert logger.name == "test-gaussian-utils"


def test_gaussian_internal_utils_surfaces_clear_error_when_choldate_missing() -> None:
    original_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "choldate":
            raise ModuleNotFoundError("No module named 'choldate'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        try:
            utils.chol_rank1_update(
                np.eye(2, dtype=np.float64),
                np.asarray([1.0, 0.0], dtype=np.float64),
            )
        except ModuleNotFoundError as exc:
            assert "optional dependency `choldate`" in str(exc)
        else:
            raise AssertionError(
                "Expected missing choldate to raise ModuleNotFoundError"
            )

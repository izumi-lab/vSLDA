from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_RANDOM_SEED = 42


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """
    Set Python, NumPy (and Torch if available) seeds for reproducibility.

    Args:
        seed: Seed value.
        deterministic_torch: If True and torch is available, set deterministic flags.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch might not be installed; ignore silently to keep optional dependency.
        pass

    # For completeness, set PYTHONHASHSEED to make hashing deterministic in the process.
    os.environ["PYTHONHASHSEED"] = str(seed)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BaselineAdapterSpec:
    model: str
    runner_family: str
    infer_mode: Literal["standard", "no_params", "train_only"] = "standard"
    train_passes_test_csvs: bool = False
    train_passes_encoder_device: bool = False
    train_passes_effective_random_state: bool = False
    # The persist step receives ``foldin`` (request option ``vmf_foldin``, default
    # True) and ``condition_fingerprint`` so that it can write the collapsed
    # fold-in document-topic artifacts of the vMF family (MvTM).
    persist_passes_foldin: bool = False

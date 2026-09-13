"""Result-path variant of a vMF Sentence LDA run whose training hyperparameters were overridden.

The main experiments fix kappa_0, alpha_0, the Gibbs sweeps per E-step (zeta), the stored
sweeps (B), the MCEM iterations (T) and, since the SAEM version of the algorithm, the burn-in
T_0 (``saem_burn_in``) and the step-size exponent a (``saem_decay``) in the experiment YAML. A
hyperparameter sensitivity sweep changes one of them from the command line; without a path
label every such run would
share the display key of the default run and overwrite its ``latest/.../CURRENT.json`` pointer.

The label is built only from the keys that ``experiments run`` overrode *and* that differ from
the config value (``TrainConfig.hyperparameter_overrides``), so every run driven by the YAML
alone keeps its historical path and condition id. Labels mirror the Gaussian prior-scale
variant (``psi0-0p1``): ``kappa0-100``, ``alpha0-0p1``, ``zeta-40``, ``b-16``, ``t-5``, ``t0-8``,
``a-0p8``, joined by ``_`` in that order when several are overridden.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Mapping

import numpy as np

# Canonical key order of the label and the config-field name each key overrides.
VMF_HYPERPARAMETER_KEYS: tuple[str, ...] = (
    "kappa0",
    "alpha0",
    "gibbs_sweeps",
    "num_samples",
    "num_iterations",
    "saem_burn_in",
    "saem_decay",
)
VMF_HYPERPARAMETER_CONFIG_FIELDS: dict[str, str] = {
    "kappa0": "kappa_default",
    "alpha0": "alpha",
    "gibbs_sweeps": "gibbs_sweeps",
    "num_samples": "num_samples",
    "num_iterations": "num_iterations",
    "saem_burn_in": "saem_burn_in",
    "saem_decay": "saem_decay",
}
_VMF_VARIANT_LABELS: dict[str, str] = {
    "kappa0": "kappa0",
    "alpha0": "alpha0",
    "gibbs_sweeps": "zeta",
    "num_samples": "b",
    "num_iterations": "t",
    "saem_burn_in": "t0",
    "saem_decay": "a",
}
_LABEL_TO_KEY = {label: key for key, label in _VMF_VARIANT_LABELS.items()}
_VARIANT_COMPONENT = re.compile(r"^(kappa0|alpha0|zeta|b|t0|t|a)-([0-9]+(?:p[0-9]+)?)$")


def format_vmf_value(value: float | int) -> str:
    """``0.1 -> 0p1``, ``100.0 -> 100``, ``20 -> 20`` (same rule as ``format_prior_scale_variant``)."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("vMF hyperparameter values must be finite and > 0.")
    normalized = format(Decimal(str(number)).normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized.replace(".", "p")


def parse_vmf_value(text: str) -> float:
    return float(str(text).replace("p", "."))


def format_vmf_parameter_variant(
    hyperparameters: Mapping[str, Any],
    overrides: tuple[str, ...] | list[str] | None,
) -> str | None:
    """Label of the overridden keys, or ``None`` when nothing was overridden.

    ``hyperparameters`` maps the keys of :data:`VMF_HYPERPARAMETER_KEYS` to their values;
    ``overrides`` lists the keys that were overridden and differ from the config value.
    """
    if not overrides:
        return None
    unknown = sorted(set(overrides) - set(VMF_HYPERPARAMETER_KEYS))
    if unknown:
        raise ValueError(f"Unknown vMF hyperparameter override(s): {unknown}")
    parts: list[str] = []
    for key in VMF_HYPERPARAMETER_KEYS:
        if key not in overrides:
            continue
        value = hyperparameters.get(key)
        if value is None:
            raise ValueError(f"Overridden vMF hyperparameter {key!r} has no value.")
        if isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(
                f"A per-topic vector for {key!r} cannot be encoded in a result-path variant."
            )
        parts.append(f"{_VMF_VARIANT_LABELS[key]}-{format_vmf_value(value)}")
    return "_".join(parts) if parts else None


def normalize_vmf_parameter_variant(value: object) -> str | None:
    """Validate a variant label given on the command line; ``None``/empty means the default run."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not is_vmf_parameter_variant(text):
        raise ValueError(
            "vmf_variant must be one or more of kappa0-<v>, alpha0-<v>, zeta-<n>, b-<n>, "
            "t-<n>, t0-<n>, a-<v> joined by '_' in that order (e.g. kappa0-100, alpha0-0p1, "
            f"zeta-40, t0-8, a-0p8); got {text!r}."
        )
    return text


def is_vmf_parameter_variant(text: str) -> bool:
    """Whether a display-key suffix is a vMF hyperparameter label (in canonical key order)."""
    if not text:
        return False
    positions: list[int] = []
    for component in text.split("_"):
        match = _VARIANT_COMPONENT.match(component)
        if match is None:
            return False
        positions.append(VMF_HYPERPARAMETER_KEYS.index(_LABEL_TO_KEY[match.group(1)]))
    return positions == sorted(positions) and len(set(positions)) == len(positions)


def parse_vmf_parameter_variant(text: str | None) -> dict[str, float]:
    """``"kappa0-100_zeta-40" -> {"kappa0": 100.0, "gibbs_sweeps": 40.0}``."""
    if text is None or not str(text).strip():
        return {}
    if not is_vmf_parameter_variant(str(text)):
        raise ValueError(f"Not a vMF hyperparameter variant label: {text!r}")
    parsed: dict[str, float] = {}
    for component in str(text).split("_"):
        match = _VARIANT_COMPONENT.match(component)
        assert match is not None
        parsed[_LABEL_TO_KEY[match.group(1)]] = parse_vmf_value(match.group(2))
    return parsed


def vmf_variant_matches(recorded: object, requested: str | None) -> bool:
    """Whether an artifact's recorded variant (``None`` = default run) is the requested one."""
    recorded_text = "" if recorded is None else str(recorded).strip()
    requested_text = "" if requested is None else str(requested).strip()
    return recorded_text == requested_text

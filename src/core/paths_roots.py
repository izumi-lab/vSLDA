from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"
DATA_ROOT = REPO_ROOT / "data"
DATA_RESOURCES_ROOT = DATA_ROOT / "resources"
WIKIENTVEC_ROOT = DATA_RESOURCES_ROOT / "wikientvec"
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_RESULTS_ROOT = RESULTS_ROOT / "experiments"
BASELINE_RESULTS_ROOT = RESULTS_ROOT / "baselines"
CLASSIFICATION_RESULTS_ROOT = RESULTS_ROOT / "classification"
VISUALIZATION_RESULTS_ROOT = RESULTS_ROOT / "visualization"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def stringify_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

from __future__ import annotations

import re
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not locate repository root from test path")


REPO_ROOT = _find_repo_root()
MODELS_ROOT = REPO_ROOT / "src" / "baselines" / "models"
TRAINER_ATTR_RE = re.compile(r"\btrainer\.")
THIRD_PARTY_LOW_LEVEL_IMPORT_RE = re.compile(r"\.\.src\.(perplexity|prior|utils)\b")
ALLOWED_TRAINER_ACCESS = {
    "src/baselines/models/gaussian_state.py",
    "src/baselines/models/gaussianlda.py",
    "src/baselines/models/sentence_gaussianlda.py",
}


def test_gaussian_trainer_attribute_access_is_isolated_to_snapshot_boundary() -> None:
    offenders: list[str] = []
    for path in sorted(MODELS_ROOT.glob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if "gaussian" not in path.name:
            continue
        text = "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if "import" not in line
        )
        if TRAINER_ATTR_RE.search(text) and relative not in ALLOWED_TRAINER_ACCESS:
            offenders.append(relative)

    assert offenders == []


def test_gaussian_train_layers_import_repo_owned_trainers_directly() -> None:
    gaussian_text = (MODELS_ROOT / "gaussianlda.py").read_text(encoding="utf-8")
    sentence_text = (MODELS_ROOT / "sentence_gaussianlda.py").read_text(
        encoding="utf-8"
    )

    assert (
        "from src.baselines.models.gaussian_trainer import GaussianLDATrainer"
        in gaussian_text
    )
    assert (
        "from src.baselines.models.sentence_gaussian_trainer import GaussianLDATrainer"
        in sentence_text
    )
    assert "gaussian_boundaries" not in gaussian_text
    assert "gaussian_boundaries" not in sentence_text
    assert not (MODELS_ROOT / "gaussian_boundaries.py").exists()


def test_gaussian_repo_owned_trainers_use_repo_owned_low_level_modules() -> None:
    trainer_paths = [
        REPO_ROOT / "src" / "baselines" / "models" / "gaussian_trainer.py",
        REPO_ROOT / "src" / "baselines" / "models" / "sentence_gaussian_trainer.py",
    ]
    for path in trainer_paths:
        text = path.read_text(encoding="utf-8")
        assert THIRD_PARTY_LOW_LEVEL_IMPORT_RE.search(text) is None
        assert "src.baselines.models.gaussian_internal" in text
        assert "src.baselines.third_party" not in text

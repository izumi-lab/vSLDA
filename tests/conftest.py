from __future__ import annotations

import sys
from pathlib import Path
from shutil import copytree

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def materialize_smoke_fixture(tmp_path: Path):
    def _materialize(name: str, destination: str | Path) -> Path:
        source = TEST_FIXTURES_ROOT / "smoke" / name
        if not source.exists():
            pytest.skip(f"Smoke fixture not available in this checkout: {source}")
        destination_path = tmp_path / Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        copytree(source, destination_path, dirs_exist_ok=True)
        return destination_path

    return _materialize

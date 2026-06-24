from __future__ import annotations

from pathlib import Path


class MissingResourceError(FileNotFoundError):
    """Base error for missing project resources."""

    resource_label = "resource"

    def __init__(self, path: str | Path, *, detail: str | None = None) -> None:
        self.path = Path(path)
        message = f"Missing {self.resource_label}: {self.path}"
        if detail:
            message = f"{message}. {detail}"
        super().__init__(message)


class MissingDatasetError(MissingResourceError):
    """Raised when dataset inputs required by a workflow are not available."""

    resource_label = "dataset resource"


class MissingArtifactError(MissingResourceError):
    """Raised when an expected artifact is not available."""

    resource_label = "artifact"


def require_existing_path(
    path: str | Path,
    *,
    error_cls: type[MissingResourceError] = MissingResourceError,
    detail: str | None = None,
) -> Path:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise error_cls(resolved_path, detail=detail)
    return resolved_path


def require_dataset_path(path: str | Path, *, detail: str | None = None) -> Path:
    return require_existing_path(path, error_cls=MissingDatasetError, detail=detail)


def require_artifact_path(path: str | Path, *, detail: str | None = None) -> Path:
    return require_existing_path(path, error_cls=MissingArtifactError, detail=detail)

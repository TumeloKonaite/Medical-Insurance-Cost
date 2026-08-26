from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Protocol

from src.exceptions import (
    ArtifactLoadError,
    ArtifactNotFoundError,
    ArtifactSaveError,
    DeploymentError,
)


class ArtifactRepository(Protocol):
    def load_model(self) -> Any: ...

    def save_model(self, model: Any) -> None: ...


class LocalArtifactRepository:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

    def load_model(self) -> Any:
        model = self._load(self.model_path, "model")
        if not callable(getattr(model, "predict", None)):
            raise ArtifactLoadError("The model artifact is invalid.")
        return model

    def save_model(self, model: Any) -> None:
        self._save(self.model_path, model, "model")

    @staticmethod
    def _load(path: Path, artifact_name: str) -> Any:
        if not path.is_file():
            raise ArtifactNotFoundError(f"The {artifact_name} artifact is missing.")
        try:
            with path.open("rb") as artifact_file:
                return pickle.load(artifact_file)
        except Exception as exc:
            raise ArtifactLoadError(
                f"The {artifact_name} artifact could not be loaded."
            ) from exc

    @staticmethod
    def _save(path: Path, artifact: Any, artifact_name: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as artifact_file:
                pickle.dump(artifact, artifact_file)
        except Exception as exc:
            raise ArtifactSaveError(
                f"The {artifact_name} artifact could not be saved."
            ) from exc


class PackagedMlflowRepository:
    """Read an already validated local MLflow package; never access a registry."""

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)

    def load_model(self) -> Any:
        try:
            from src.mlops.runtime import get_validated_model

            return get_validated_model(self.package_dir)
        except DeploymentError as exc:
            raise ArtifactLoadError("The packaged production model is invalid.") from exc

    def save_model(self, model: Any) -> None:
        del model
        raise ArtifactSaveError("Production model packages are immutable.")

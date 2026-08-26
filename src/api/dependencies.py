import os
from functools import lru_cache
from pathlib import Path

from src.paths import ARTIFACTS_DIR
from src.repositories.artifact_repository import (
    LocalArtifactRepository,
    PackagedMlflowRepository,
)
from src.services.prediction_service import PredictionService


@lru_cache(maxsize=1)
def _build_prediction_service() -> PredictionService:
    package_dir = os.environ.get("MODEL_PACKAGE_DIR", "").strip()
    repository = (
        PackagedMlflowRepository(Path(package_dir))
        if package_dir
        else LocalArtifactRepository(model_path=ARTIFACTS_DIR / "model.pkl")
    )
    return PredictionService(repository)


async def get_prediction_service() -> PredictionService:
    return _build_prediction_service()

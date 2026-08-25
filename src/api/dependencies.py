from functools import lru_cache

from src.paths import ARTIFACTS_DIR
from src.repositories.artifact_repository import LocalArtifactRepository
from src.services.prediction_service import PredictionService


@lru_cache(maxsize=1)
def _build_prediction_service() -> PredictionService:
    repository = LocalArtifactRepository(
        model_path=ARTIFACTS_DIR / "model.pkl",
        preprocessor_path=ARTIFACTS_DIR / "preprocessor.pkl",
    )
    return PredictionService(repository)


async def get_prediction_service() -> PredictionService:
    return _build_prediction_service()

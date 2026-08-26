import logging
import os
from functools import lru_cache
from pathlib import Path

from src.database import create_database_engine
from src.paths import ARTIFACTS_DIR
from src.repositories.artifact_repository import (
    LocalArtifactRepository,
    PackagedMlflowRepository,
)
from src.repositories.prediction_event_repository import (
    DisabledPredictionEventRepository,
    SqlPredictionEventRepository,
)
from src.services.prediction_event_service import PredictionEventService
from src.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_prediction_service() -> PredictionService:
    # Choose the local model during development or the packaged production model.
    package_dir = os.environ.get("MODEL_PACKAGE_DIR", "").strip()
    repository = (
        PackagedMlflowRepository(Path(package_dir))
        if package_dir
        else LocalArtifactRepository(model_path=ARTIFACTS_DIR / "model.pkl")
    )
    return PredictionService(repository)


async def get_prediction_service() -> PredictionService:
    return _build_prediction_service()


@lru_cache(maxsize=1)
def _build_prediction_event_service() -> PredictionEventService:
    # Persistence stays disabled when DATABASE_URL has not been configured.
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        repository = DisabledPredictionEventRepository()
    else:
        try:
            # The engine connects lazily when the first event is written.
            repository = SqlPredictionEventRepository(
                create_database_engine(database_url)
            )
        except Exception as exc:
            logger.warning(
                "Prediction persistence configuration failed error_type=%s",
                type(exc).__name__,
            )
            repository = DisabledPredictionEventRepository()
    return PredictionEventService(repository)


async def get_prediction_event_service() -> PredictionEventService:
    return _build_prediction_event_service()

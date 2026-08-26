from __future__ import annotations

import logging
import math
from threading import Lock
from typing import Any

import pandas as pd

from src.exceptions import (
    ArtifactRepositoryError,
    ArtifactUnavailableError,
    PredictionError,
)
from src.model_contract import FEATURE_COLUMNS
from src.repositories.artifact_repository import ArtifactRepository
from src.schemas.prediction import PredictionRequest

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, artifact_repository: ArtifactRepository):
        self._artifact_repository = artifact_repository
        self._model: Any | None = None
        self._artifact_lock = Lock()

    def predict(self, prediction_input: PredictionRequest) -> float:
        model = self._get_model()
        features = pd.DataFrame([prediction_input.model_dump()], columns=FEATURE_COLUMNS)

        try:
            predictions = model.predict(features)
            prediction = float(predictions[0])
        except Exception as exc:
            logger.exception("Prediction inference failed")
            raise PredictionError("Prediction inference failed.") from exc

        if not math.isfinite(prediction):
            logger.error("Prediction inference returned a non-finite value")
            raise PredictionError("Prediction inference returned an invalid result.")
        return prediction

    def _get_model(self) -> Any:
        if self._model is None:
            with self._artifact_lock:
                if self._model is None:
                    try:
                        model = self._artifact_repository.load_model()
                    except ArtifactRepositoryError as exc:
                        logger.warning("Prediction model is unavailable: %s", exc)
                        raise ArtifactUnavailableError(
                            "Prediction model is unavailable."
                        ) from exc
                    self._model = model
        return self._model

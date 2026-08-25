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
from src.repositories.artifact_repository import ArtifactRepository
from src.schemas.prediction import PredictionRequest

logger = logging.getLogger(__name__)


class PredictionService:
    FEATURE_COLUMNS = ("age", "sex", "bmi", "children", "smoker", "region")

    def __init__(self, artifact_repository: ArtifactRepository):
        self._artifact_repository = artifact_repository
        self._model: Any | None = None
        self._preprocessor: Any | None = None
        self._artifact_lock = Lock()

    def predict(self, prediction_input: PredictionRequest) -> float:
        model, preprocessor = self._get_artifacts()
        features = pd.DataFrame(
            [prediction_input.model_dump()], columns=self.FEATURE_COLUMNS
        )

        try:
            transformed_features = preprocessor.transform(features)
            predictions = model.predict(transformed_features)
            prediction = float(predictions[0])
        except Exception as exc:
            logger.exception("Prediction inference failed")
            raise PredictionError("Prediction inference failed.") from exc

        if not math.isfinite(prediction):
            logger.error("Prediction inference returned a non-finite value")
            raise PredictionError("Prediction inference returned an invalid result.")
        return prediction

    def _get_artifacts(self) -> tuple[Any, Any]:
        if self._model is None or self._preprocessor is None:
            with self._artifact_lock:
                if self._model is None or self._preprocessor is None:
                    try:
                        model = self._artifact_repository.load_model()
                        preprocessor = self._artifact_repository.load_preprocessor()
                    except ArtifactRepositoryError as exc:
                        logger.warning("Prediction artifacts are unavailable: %s", exc)
                        raise ArtifactUnavailableError(
                            "Prediction artifacts are unavailable."
                        ) from exc
                    self._model = model
                    self._preprocessor = preprocessor
        return self._model, self._preprocessor

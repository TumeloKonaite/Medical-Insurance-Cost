from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Literal, Protocol

from src.model_contract import PREDICTION_CONTRACT_VERSION
from src.schemas.prediction import PredictionRequest
from src.schemas.prediction_event import PredictionEvent

logger = logging.getLogger(__name__)


class EventWriter(Protocol):
    def add(self, event: PredictionEvent) -> bool: ...


class PredictionEventService:
    def __init__(self, repository: EventWriter):
        self._repository = repository

    def record_success(
        self,
        *,
        request_id: uuid.UUID,
        source: Literal["web", "json"],
        payload: PredictionRequest,
        predicted_charges: float,
        model_version: str,
        inference_latency_ms: float,
    ) -> bool:
        # Convert the validated request and prediction result into one typed event.
        event = PredictionEvent(
            id=uuid.uuid4(),
            request_id=request_id,
            created_at=datetime.now(timezone.utc),
            source=source,
            predicted_charges=predicted_charges,
            model_version=model_version,
            prediction_contract_version=PREDICTION_CONTRACT_VERSION,
            inference_latency_ms=inference_latency_ms,
            **payload.model_dump(),
        )
        try:
            # This repository call is the point where persistence happens.
            return self._repository.add(event)
        except Exception as exc:
            # Deliberately omit exception text and feature values: driver errors can
            # contain connection details and request data must not enter logs.
            logger.warning(
                "Prediction event persistence failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            return False

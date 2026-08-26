import uuid
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_prediction_event_service, get_prediction_service
from src.exceptions import ArtifactUnavailableError, PredictionError
from src.schemas.prediction import PredictionRequest, PredictionResponse
from src.services.prediction_event_service import PredictionEventService
from src.services.prediction_service import PredictionService

router = APIRouter()


@router.post("/predict-json", response_model=PredictionResponse)
async def predict_json(
    payload: PredictionRequest,
    prediction_service: Annotated[
        PredictionService, Depends(get_prediction_service)
    ],
    prediction_event_service: Annotated[
        PredictionEventService, Depends(get_prediction_event_service)
    ],
) -> PredictionResponse:
    # Run the model and measure inference time before doing any database work.
    started_at = perf_counter()
    try:
        prediction = prediction_service.predict(payload)
    except ArtifactUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is unavailable.",
        ) from exc
    except PredictionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please try again later.",
        ) from exc

    inference_latency_ms = (perf_counter() - started_at) * 1000
    # Persist only successful JSON predictions. Persistence is fail-open.
    prediction_event_service.record_success(
        request_id=uuid.uuid4(),
        source="json",
        payload=payload,
        predicted_charges=prediction,
        model_version=getattr(prediction_service, "model_version", "unknown"),
        inference_latency_ms=inference_latency_ms,
    )
    return PredictionResponse(predicted_charges=prediction)

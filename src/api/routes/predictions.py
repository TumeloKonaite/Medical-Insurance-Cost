from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_prediction_service
from src.exceptions import ArtifactUnavailableError, PredictionError
from src.schemas.prediction import PredictionRequest, PredictionResponse
from src.services.prediction_service import PredictionService

router = APIRouter()


@router.post("/predict-json", response_model=PredictionResponse)
async def predict_json(
    payload: PredictionRequest,
    prediction_service: Annotated[
        PredictionService, Depends(get_prediction_service)
    ],
) -> PredictionResponse:
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

    return PredictionResponse(predicted_charges=prediction)

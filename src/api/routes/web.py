import uuid
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from src.api.dependencies import get_prediction_event_service, get_prediction_service
from src.exceptions import ArtifactUnavailableError, PredictionError
from src.paths import TEMPLATES_DIR
from src.schemas.prediction import PredictionRequest
from src.services.prediction_event_service import PredictionEventService
from src.services.prediction_service import PredictionService

router = APIRouter()
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _render_home(
    request: Request,
    *,
    results: str | None = None,
    error_message: str | None = None,
    status_code: int = status.HTTP_200_OK,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"results": results, "error_message": error_message},
        status_code=status_code,
    )


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return _render_home(request)


@router.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    prediction_service: Annotated[
        PredictionService, Depends(get_prediction_service)
    ],
    prediction_event_service: Annotated[
        PredictionEventService, Depends(get_prediction_event_service)
    ],
) -> HTMLResponse:
    form = await request.form()
    try:
        payload = PredictionRequest.model_validate(dict(form))
    except ValidationError:
        return _render_home(
            request,
            error_message="Invalid input supplied. Please check values.",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    # Run the model and measure inference time before doing any database work.
    started_at = perf_counter()
    try:
        prediction = prediction_service.predict(payload)
    except ArtifactUnavailableError:
        return _render_home(
            request,
            error_message="Prediction service is unavailable.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except PredictionError:
        return _render_home(
            request,
            error_message="Prediction failed. Please try again later.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    inference_latency_ms = (perf_counter() - started_at) * 1000
    # Persist only successful web-form predictions. Persistence is fail-open.
    prediction_event_service.record_success(
        request_id=uuid.uuid4(),
        source="web",
        payload=payload,
        predicted_charges=prediction,
        model_version=getattr(prediction_service, "model_version", "unknown"),
        inference_latency_ms=inference_latency_ms,
    )
    result = f"Estimated insurance charges: {prediction:.2f}"
    return _render_home(request, results=result)

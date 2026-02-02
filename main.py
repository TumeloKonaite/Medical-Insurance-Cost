from typing import Literal

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/health", response_class=JSONResponse)
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "home.html",
        {"results": None, "error_message": None},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_datapoint(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...),
):
    try:
        data = CustomData(
            age=age,
            sex=sex,
            bmi=bmi,
            children=children,
            smoker=smoker,
            region=region,
        )
    except (TypeError, ValueError):
        return templates.TemplateResponse(
            request,
            "home.html",
            {
                "results": None,
                "error_message": "Invalid input supplied. Please check values.",
            },
        )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()

    try:
        results = predict_pipeline.predict(pred_df)
    except Exception:
        return templates.TemplateResponse(
            request,
            "home.html",
            {
                "results": None,
                "error_message": "Prediction failed. Check logs for details.",
            },
        )

    result_text = f"Estimated insurance charges: {float(results[0]):.2f}"
    return templates.TemplateResponse(
        request,
        "home.html",
        {"results": result_text, "error_message": None},
    )


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0)
    sex: Literal["female", "male"]
    bmi: float = Field(..., gt=0)
    children: int = Field(..., ge=0)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]

    model_config = {"extra": "forbid"}


class PredictionResponse(BaseModel):
    predicted_charges: float
    currency: str = "USD"


@app.post("/predict-json", response_model=PredictionResponse)
async def predict_json(payload: PredictionRequest):
    data = CustomData(
        age=payload.age,
        sex=payload.sex,
        bmi=payload.bmi,
        children=payload.children,
        smoker=payload.smoker,
        region=payload.region,
    )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()

    try:
        results = predict_pipeline.predict(pred_df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check logs for details.",
        ) from exc

    return PredictionResponse(predicted_charges=float(results[0]))

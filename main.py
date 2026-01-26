from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")


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

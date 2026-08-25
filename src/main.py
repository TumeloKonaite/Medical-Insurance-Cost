from fastapi import FastAPI

from src.api.routes import health, predictions, web


def create_app() -> FastAPI:
    application = FastAPI(title="Medical Insurance Cost Prediction")
    application.include_router(health.router)
    application.include_router(web.router)
    application.include_router(predictions.router)
    return application


app = create_app()

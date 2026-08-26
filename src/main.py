import os
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, predictions


DEFAULT_CORS_ORIGINS = "http://localhost:5173"


def parse_cors_origins(value: str) -> list[str]:
    origins: list[str] = []
    for item in value.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        try:
            parsed = urlsplit(origin)
            is_valid = (
                origin != "*"
                and parsed.scheme in {"http", "https"}
                and bool(parsed.netloc)
                and not parsed.path
                and not parsed.query
                and not parsed.fragment
                and not parsed.username
                and not parsed.password
            )
        except ValueError:
            is_valid = False
        if not is_valid:
            raise RuntimeError(
                "CORS_ALLOWED_ORIGINS must contain only explicit HTTP(S) origins."
            )
        if origin not in origins:
            origins.append(origin)
    return origins


def create_app() -> FastAPI:
    application = FastAPI(title="Medical Insurance Cost Prediction")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=parse_cors_origins(
            os.getenv("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS)
        ),
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    application.include_router(health.router)
    application.include_router(predictions.router)
    return application


app = create_app()

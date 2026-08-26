from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MLFLOW_ENVIRONMENT_VARIABLES = (
    "ENABLE_MLFLOW_TRACKING",
    "ENABLE_MODEL_REGISTRATION",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTERED_MODEL_NAME",
    "MLFLOW_ALLOW_FILE_STORE",
)


@pytest.fixture(autouse=True)
def isolate_mlflow_environment(monkeypatch):
    """Prevent a developer's loaded DagsHub settings from changing test behavior."""
    for variable in MLFLOW_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(autouse=True)
def isolate_prediction_database(monkeypatch):
    """Unit tests must never connect to a developer or CI Neon database."""
    from src.api.dependencies import _build_prediction_event_service

    monkeypatch.delenv("DATABASE_URL", raising=False)
    _build_prediction_event_service.cache_clear()
    yield
    _build_prediction_event_service.cache_clear()

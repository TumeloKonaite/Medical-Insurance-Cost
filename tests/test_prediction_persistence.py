from __future__ import annotations

import asyncio
import logging
import uuid

import httpx
import sqlalchemy as sa
from sqlalchemy.pool import StaticPool

from src.api.dependencies import (
    get_prediction_event_service,
    get_prediction_service,
)
from src.database import secure_database_url
from src.main import app
from src.model_contract import PREDICTION_CONTRACT_VERSION
from src.models.prediction_event import metadata, prediction_events
from src.repositories.prediction_event_repository import SqlPredictionEventRepository
from src.schemas.prediction import PredictionRequest
from src.services.prediction_event_service import PredictionEventService

VALID_JSON = {
    "age": 41,
    "sex": "male",
    "bmi": 30.25,
    "children": 2,
    "smoker": "no",
    "region": "northwest",
}


class StubPredictionService:
    model_version = "42"

    def predict(self, payload):
        del payload
        return 9876.54


class RecordingRepository:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(event)
        return True


class FailingRepository:
    def add(self, event):
        del event
        raise RuntimeError(
            "postgresql://demo:secret-password@host/internal-db-error"
        )


def _request(method: str, url: str, **kwargs):
    async def send_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(send_request())


def _override_dependencies(event_service):
    prediction_service = StubPredictionService()

    async def override_prediction_service():
        return prediction_service

    async def override_event_service():
        return event_service

    app.dependency_overrides[get_prediction_service] = override_prediction_service
    app.dependency_overrides[get_prediction_event_service] = override_event_service


def test_successful_prediction_persists_expected_values_and_source():
    repository = RecordingRepository()
    _override_dependencies(PredictionEventService(repository))
    try:
        json_response = _request("POST", "/predict-json", json=VALID_JSON)
    finally:
        app.dependency_overrides.clear()

    assert json_response.status_code == 200
    assert [event.source for event in repository.events] == ["json"]

    for event in repository.events:
        assert event.age == 41
        assert event.sex == "male"
        assert event.bmi == 30.25
        assert event.children == 2
        assert event.smoker == "no"
        assert event.region == "northwest"
        assert event.predicted_charges == 9876.54
        assert event.model_version == "42"
        assert event.prediction_contract_version == PREDICTION_CONTRACT_VERSION
        assert event.inference_latency_ms >= 0
        assert event.created_at.tzinfo is not None
        assert event.actual_charges is None
        assert event.actual_recorded_at is None


def test_database_failure_is_fail_open_and_sanitized(caplog):
    _override_dependencies(PredictionEventService(FailingRepository()))
    with caplog.at_level(logging.WARNING):
        try:
            response = _request("POST", "/predict-json", json=VALID_JSON)
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["predicted_charges"] == 9876.54
    assert "secret-password" not in response.text
    assert "internal-db-error" not in response.text
    assert "secret-password" not in caplog.text
    assert "internal-db-error" not in caplog.text
    assert "error_type=RuntimeError" in caplog.text


def test_invalid_requests_are_not_persisted():
    repository = RecordingRepository()
    _override_dependencies(PredictionEventService(repository))
    try:
        json_response = _request(
            "POST", "/predict-json", json={**VALID_JSON, "bmi": 0}
        )
    finally:
        app.dependency_overrides.clear()

    assert json_response.status_code == 422
    assert repository.events == []


def test_duplicate_request_ids_do_not_create_duplicate_events():
    engine = sa.create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine)
    service = PredictionEventService(SqlPredictionEventRepository(engine))
    request_id = uuid.uuid4()
    payload = PredictionRequest(**VALID_JSON)

    first_inserted = service.record_success(
        request_id=request_id,
        source="json",
        payload=payload,
        predicted_charges=9876.54,
        model_version="42",
        inference_latency_ms=1.25,
    )
    second_inserted = service.record_success(
        request_id=request_id,
        source="json",
        payload=payload,
        predicted_charges=9876.54,
        model_version="42",
        inference_latency_ms=1.25,
    )

    with engine.connect() as connection:
        rows = connection.execute(sa.select(prediction_events)).mappings().all()

    assert first_inserted is True
    assert second_inserted is False
    assert len(rows) == 1
    assert rows[0]["request_id"] == request_id
    assert rows[0]["source"] == "json"
    assert rows[0]["model_version"] == "42"
    assert rows[0]["prediction_contract_version"] == PREDICTION_CONTRACT_VERSION
    assert float(rows[0]["predicted_charges"]) == 9876.54
    assert float(rows[0]["inference_latency_ms"]) == 1.25


def test_database_url_enforces_tls_and_neon_pooler():
    url = secure_database_url(
        "postgresql://demo:placeholder@ep-example-pooler.us-east-2.aws.neon.tech/"
        "demo?sslmode=disable"
    )

    assert url.drivername == "postgresql+psycopg"
    assert url.query["sslmode"] == "require"
    assert url.query["connect_timeout"] == "5"

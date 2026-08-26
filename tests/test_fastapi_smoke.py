import asyncio

import httpx
import pytest

from src.api.dependencies import get_prediction_service
from src.exceptions import ArtifactUnavailableError, PredictionError
from src.main import app, create_app, parse_cors_origins


VALID_JSON = {
    "age": 19,
    "sex": "female",
    "bmi": 27.9,
    "children": 0,
    "smoker": "yes",
    "region": "southwest",
}


class StubPredictionService:
    def __init__(self, result=12345.67, error=None):
        self.result = result
        self.error = error
        self.payloads = []

    def predict(self, payload):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return self.result


class ASGITestClient:
    def __init__(self, application=app):
        self.application = application

    def request(self, method, url, **kwargs):
        async def send_request():
            transport = httpx.ASGITransport(app=self.application)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send_request())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def options(self, url, **kwargs):
        return self.request("OPTIONS", url, **kwargs)


@pytest.fixture
def client():
    service = StubPredictionService()
    async def override_service():
        return service

    app.dependency_overrides[get_prediction_service] = override_service
    yield ASGITestClient(), service
    app.dependency_overrides.clear()


def test_health_docs_and_api_only_root(client):
    test_client, _ = client

    assert test_client.get("/health").json() == {"status": "ok"}
    assert test_client.get("/docs").status_code == 200
    assert test_client.get("/openapi.json").status_code == 200
    assert test_client.get("/").status_code == 404


def test_valid_json_prediction(client):
    test_client, _ = client

    response = test_client.post("/predict-json", json=VALID_JSON)

    assert response.status_code == 200
    assert response.json() == {
        "predicted_charges": 12345.67,
        "currency": "USD",
    }


def test_invalid_json_returns_422(client):
    test_client, _ = client

    response = test_client.post(
        "/predict-json", json={**VALID_JSON, "region": "invalid"}
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (
            PredictionError("internal inference details"),
            500,
            "Prediction failed. Please try again later.",
        ),
        (
            ArtifactUnavailableError("/private/artifacts/model.pkl"),
            503,
            "Prediction service is unavailable.",
        ),
    ],
)
def test_json_service_failures_are_sanitized(
    client, error, expected_status, expected_detail
):
    test_client, service = client
    service.error = error

    response = test_client.post("/predict-json", json=VALID_JSON)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert "private" not in response.text


def test_approved_cors_origin_receives_headers(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,https://medical-insurance-cost.vercel.app",
    )
    application = create_app()

    async def override_service():
        return StubPredictionService()

    application.dependency_overrides[get_prediction_service] = override_service
    test_client = ASGITestClient(application)

    response = test_client.options(
        "/predict-json",
        headers={
            "Origin": "https://medical-insurance-cost.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://medical-insurance-cost.vercel.app"
    )

    prediction = test_client.post(
        "/predict-json",
        json=VALID_JSON,
        headers={"Origin": "https://medical-insurance-cost.vercel.app"},
    )
    assert prediction.status_code == 200
    assert prediction.headers["access-control-allow-origin"] == (
        "https://medical-insurance-cost.vercel.app"
    )


def test_unapproved_cors_origin_receives_no_allow_origin_header(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173")
    test_client = ASGITestClient(create_app())

    response = test_client.post(
        "/predict-json",
        json=VALID_JSON,
        headers={"Origin": "https://unapproved.example"},
    )

    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "ftp://example.com",
        "https://user:pass@example.com",
        "https://[invalid",
        "example.com",
    ],
)
def test_invalid_cors_origins_fail_closed(value):
    with pytest.raises(RuntimeError, match="explicit HTTP"):
        parse_cors_origins(value)

import asyncio

import httpx
import pytest

from src.api.dependencies import get_prediction_service
from src.exceptions import ArtifactUnavailableError, PredictionError
from src.main import app


VALID_FORM = {
    "age": "19",
    "sex": "female",
    "bmi": "27.9",
    "children": "0",
    "smoker": "yes",
    "region": "southwest",
}
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
    @staticmethod
    def request(method, url, **kwargs):
        async def send_request():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send_request())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture
def client():
    service = StubPredictionService()
    async def override_service():
        return service

    app.dependency_overrides[get_prediction_service] = override_service
    yield ASGITestClient(), service
    app.dependency_overrides.clear()


def test_health_and_home_endpoints(client):
    test_client, _ = client

    assert test_client.get("/health").json() == {"status": "ok"}
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Medical Insurance Cost Prediction" in response.text


def test_valid_form_prediction_uses_injected_service(client):
    test_client, service = client

    response = test_client.post("/predict", data=VALID_FORM)

    assert response.status_code == 200
    assert "Estimated insurance charges: 12345.67" in response.text
    assert service.payloads[0].model_dump() == VALID_JSON


@pytest.mark.parametrize(
    "invalid_form",
    [
        {**VALID_FORM, "bmi": "0"},
        {**VALID_FORM, "region": "invalid"},
        {**VALID_FORM, "unknown": "field"},
    ],
)
def test_invalid_form_returns_safe_html_422(client, invalid_form):
    test_client, service = client

    response = test_client.post("/predict", data=invalid_form)

    assert response.status_code == 422
    assert "Invalid input supplied" in response.text
    assert service.payloads == []


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


def test_html_missing_artifacts_returns_safe_503(client):
    test_client, service = client
    service.error = ArtifactUnavailableError("/private/artifacts/model.pkl")

    response = test_client.post("/predict", data=VALID_FORM)

    assert response.status_code == 503
    assert "Prediction service is unavailable" in response.text
    assert "private" not in response.text

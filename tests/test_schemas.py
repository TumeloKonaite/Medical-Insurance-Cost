import math

import pytest
from pydantic import ValidationError

from src.schemas.prediction import PredictionRequest


VALID_INPUT = {
    "age": 29,
    "sex": "female",
    "bmi": 27.4,
    "children": 2,
    "smoker": "no",
    "region": "southeast",
}


def test_prediction_request_accepts_valid_input():
    request = PredictionRequest.model_validate(VALID_INPUT)

    assert request.model_dump() == VALID_INPUT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sex", "unknown"),
        ("region", "central"),
        ("smoker", "sometimes"),
        ("age", -1),
        ("bmi", 0),
        ("children", -1),
        ("bmi", math.inf),
        ("bmi", -math.inf),
        ("bmi", math.nan),
    ],
)
def test_prediction_request_rejects_invalid_values(field, value):
    payload = {**VALID_INPUT, field: value}

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_prediction_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({**VALID_INPUT, "unexpected": True})

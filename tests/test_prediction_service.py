import numpy as np
import pytest

from src.exceptions import (
    ArtifactNotFoundError,
    ArtifactUnavailableError,
    PredictionError,
)
from src.schemas.prediction import PredictionRequest
from src.services.prediction_service import PredictionService


class RecordingPreprocessor:
    def __init__(self):
        self.frames = []

    def transform(self, features):
        self.frames.append(features)
        return np.array([[1.0, 2.0]])


class RecordingModel:
    def __init__(self, result=4321.5):
        self.inputs = []
        self.result = result

    def predict(self, features):
        self.inputs.append(features)
        return np.array([self.result])


class FakeRepository:
    def __init__(self):
        self.model = RecordingModel()
        self.preprocessor = RecordingPreprocessor()
        self.model_loads = 0
        self.preprocessor_loads = 0

    def load_model(self):
        self.model_loads += 1
        return self.model

    def load_preprocessor(self):
        self.preprocessor_loads += 1
        return self.preprocessor

    def save_model(self, model):
        self.model = model

    def save_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor


class MissingRepository(FakeRepository):
    def load_model(self):
        raise ArtifactNotFoundError("sensitive/path/model.pkl")


def _payload():
    return PredictionRequest(
        age=29,
        sex="female",
        bmi=27.4,
        children=2,
        smoker="no",
        region="southeast",
    )


def test_service_builds_ordered_features_and_returns_scalar():
    repository = FakeRepository()
    service = PredictionService(repository)

    result = service.predict(_payload())

    assert result == 4321.5
    frame = repository.preprocessor.frames[0]
    assert list(frame.columns) == [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
    ]
    assert frame.iloc[0].to_dict() == _payload().model_dump()
    np.testing.assert_array_equal(repository.model.inputs[0], [[1.0, 2.0]])


def test_service_caches_artifacts_for_its_lifecycle():
    repository = FakeRepository()
    service = PredictionService(repository)

    service.predict(_payload())
    service.predict(_payload())

    assert repository.model_loads == 1
    assert repository.preprocessor_loads == 1


def test_service_maps_repository_errors():
    service = PredictionService(MissingRepository())

    with pytest.raises(ArtifactUnavailableError) as error:
        service.predict(_payload())

    assert "sensitive/path" not in str(error.value)


def test_service_rejects_non_finite_prediction():
    repository = FakeRepository()
    repository.model.result = np.inf

    with pytest.raises(PredictionError):
        PredictionService(repository).predict(_payload())

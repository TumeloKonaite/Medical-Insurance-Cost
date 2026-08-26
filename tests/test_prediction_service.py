import numpy as np
import pytest

from src.exceptions import (
    ArtifactNotFoundError,
    ArtifactUnavailableError,
    PredictionError,
)
from src.model_contract import FEATURE_COLUMNS
from src.schemas.prediction import PredictionRequest
from src.services.prediction_service import PredictionService


class RecordingPipeline:
    def __init__(self, result=4321.5):
        self.inputs = []
        self.result = result

    def predict(self, features):
        self.inputs.append(features)
        return np.array([self.result])


class FakeRepository:
    def __init__(self):
        self.model = RecordingPipeline()
        self.model_loads = 0

    def load_model(self):
        self.model_loads += 1
        return self.model

    def save_model(self, model):
        self.model = model


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


def test_service_passes_ordered_raw_features_to_one_pipeline():
    repository = FakeRepository()
    result = PredictionService(repository).predict(_payload())

    assert result == 4321.5
    frame = repository.model.inputs[0]
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert frame.iloc[0].to_dict() == _payload().model_dump()


def test_service_caches_pipeline_for_its_lifecycle():
    repository = FakeRepository()
    service = PredictionService(repository)
    service.predict(_payload())
    service.predict(_payload())
    assert repository.model_loads == 1


def test_service_maps_repository_errors_without_internal_details():
    service = PredictionService(MissingRepository())
    with pytest.raises(ArtifactUnavailableError) as error:
        service.predict(_payload())
    assert "sensitive/path" not in str(error.value)


def test_service_rejects_non_finite_prediction():
    repository = FakeRepository()
    repository.model.result = np.inf
    with pytest.raises(PredictionError):
        PredictionService(repository).predict(_payload())

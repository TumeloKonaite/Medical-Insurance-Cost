import pytest

from src.exceptions import ArtifactLoadError, ArtifactNotFoundError
from src.repositories.artifact_repository import LocalArtifactRepository


class SerializablePipeline:
    def predict(self, features):
        return [123.0]


def test_repository_saves_and_loads_one_pipeline(tmp_path):
    repository = LocalArtifactRepository(tmp_path / "model.pkl")
    repository.save_model(SerializablePipeline())
    assert isinstance(repository.load_model(), SerializablePipeline)


def test_repository_maps_missing_artifact_to_safe_error(tmp_path):
    repository = LocalArtifactRepository(tmp_path / "missing-model.pkl")
    with pytest.raises(ArtifactNotFoundError, match="model artifact is missing"):
        repository.load_model()


def test_repository_maps_corrupt_artifact_to_safe_error(tmp_path):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"not a pickle")
    repository = LocalArtifactRepository(model_path)
    with pytest.raises(ArtifactLoadError, match="model artifact could not be loaded"):
        repository.load_model()


def test_repository_rejects_artifact_with_wrong_interface(tmp_path):
    repository = LocalArtifactRepository(tmp_path / "model.pkl")
    repository.save_model({"not": "a pipeline"})
    with pytest.raises(ArtifactLoadError, match="model artifact is invalid"):
        repository.load_model()

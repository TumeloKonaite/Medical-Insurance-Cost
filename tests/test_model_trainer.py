import numpy as np

from src.repositories.artifact_repository import LocalArtifactRepository
from src.training.model_trainer import ModelTrainer


def _make_arrays(seed: int = 42):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(40, 10))
    X_test = rng.normal(size=(10, 10))
    y_train = X_train @ rng.normal(size=10) + rng.normal(scale=0.1, size=40)
    y_test = X_test @ rng.normal(size=10) + rng.normal(scale=0.1, size=10)

    train_arr = np.column_stack([X_train, y_train])
    test_arr = np.column_stack([X_test, y_test])
    return train_arr, test_arr


def test_model_trainer_saves_best_model(tmp_path):
    train_arr, test_arr = _make_arrays()

    model_path = tmp_path / "model.pkl"
    repository = LocalArtifactRepository(model_path, tmp_path / "preprocessor.pkl")
    trainer = ModelTrainer(repository)

    score = trainer.run(train_arr, test_arr)

    assert isinstance(score, float)
    assert model_path.exists()
    assert callable(repository.load_model().predict)

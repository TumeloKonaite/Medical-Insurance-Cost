import numpy as np

from src.components.model_trainer import ModelTrainer


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

    trainer = ModelTrainer()
    model_path = tmp_path / "model.pkl"
    trainer.model_trainer_config.trained_model_file_path = str(model_path)

    score = trainer.initiate_model_trainer(train_arr, test_arr)

    assert isinstance(score, float)
    assert model_path.exists()

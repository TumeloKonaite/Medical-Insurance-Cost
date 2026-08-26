from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.mlops.config import MlflowConfig
from src.model_contract import FEATURE_COLUMNS
from src.repositories.artifact_repository import LocalArtifactRepository
from src.training.model_trainer import ModelTrainer


def _data_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(__file__).resolve().parents[1] / "Data" / "medical_insurance.csv"
    data = pd.read_csv(data_path)
    return data.iloc[:1000].copy(), data.iloc[1000:].copy()


def test_model_trainer_evaluates_and_saves_complete_pipelines(tmp_path):
    train_data, test_data = _data_split()
    repository = LocalArtifactRepository(tmp_path / "model.pkl")
    trainer = ModelTrainer(repository, tracking_config=MlflowConfig(enabled=False))

    result = trainer.run(train_data, test_data)
    saved_pipeline = repository.load_model()

    assert isinstance(result.score, float)
    assert set(result.candidate_metrics) == {
        "Random Forest",
        "Linear Regression",
        "Support Vector Machine",
        "Bayesian Ridge",
        "AdaBoost",
    }
    assert isinstance(saved_pipeline, Pipeline)
    assert list(saved_pipeline.named_steps) == ["preprocessor", "regressor"]

    raw_features = test_data.loc[:, list(FEATURE_COLUMNS)].iloc[:1]
    prediction = float(saved_pipeline.predict(raw_features)[0])
    assert np.isfinite(prediction)

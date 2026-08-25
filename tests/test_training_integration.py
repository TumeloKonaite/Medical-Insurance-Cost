from pathlib import Path

from src.repositories.artifact_repository import LocalArtifactRepository
from src.schemas.prediction import PredictionRequest
from src.services.prediction_service import PredictionService
from src.training.data_ingestion import DataIngestion, DataIngestionConfig
from src.training.data_transformation import DataTransformation
from src.training.model_trainer import ModelTrainer


def test_training_artifacts_support_real_prediction(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    repository = LocalArtifactRepository(
        model_path=tmp_path / "model.pkl",
        preprocessor_path=tmp_path / "preprocessor.pkl",
    )
    ingestion = DataIngestion(
        DataIngestionConfig(
            source_data_path=project_root / "Data" / "medical_insurance.csv",
            train_data_path=tmp_path / "train.csv",
            test_data_path=tmp_path / "test.csv",
            raw_data_path=tmp_path / "data.csv",
        )
    )

    train_path, test_path, _ = ingestion.run()
    train_data, test_data = DataTransformation(repository).run(train_path, test_path)
    ModelTrainer(repository).run(train_data, test_data)
    prediction = PredictionService(repository).predict(
        PredictionRequest(
            age=19,
            sex="female",
            bmi=27.9,
            children=0,
            smoker="yes",
            region="southwest",
        )
    )

    assert isinstance(prediction, float)
    assert prediction > 0

import numpy as np
import pandas as pd

from pipeline.predict_pipeline import PredictPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def _train_pipeline(tmp_path):
    ingestion = DataIngestion()
    ingestion.ingestion_config.source_data_path = "Data/medical_insurance.csv"
    ingestion.ingestion_config.train_data_path = str(tmp_path / "train.csv")
    ingestion.ingestion_config.test_data_path = str(tmp_path / "test.csv")
    ingestion.ingestion_config.raw_data_path = str(tmp_path / "raw.csv")

    train_path, test_path, _ = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    transformer.data_transformation_config.preprocessor_obj_file_path = str(
        tmp_path / "preprocessor.pkl"
    )
    train_arr, test_arr, _ = transformer.initiate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()
    trainer.model_trainer_config.trained_model_file_path = str(tmp_path / "model.pkl")
    trainer.initiate_model_trainer(train_arr, test_arr)

    return tmp_path / "model.pkl", tmp_path / "preprocessor.pkl"


def test_predict_pipeline_returns_prediction(tmp_path):
    model_path, preprocessor_path = _train_pipeline(tmp_path)

    pipeline = PredictPipeline()
    pipeline.model_path = str(model_path)
    pipeline.preprocessor_path = str(preprocessor_path)

    sample = pd.DataFrame(
        {
            "age": [19],
            "sex": ["female"],
            "bmi": [27.9],
            "children": [0],
            "smoker": ["yes"],
            "region": ["southwest"],
        }
    )

    preds = pipeline.predict(sample)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1,)

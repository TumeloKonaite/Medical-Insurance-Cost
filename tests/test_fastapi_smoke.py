from fastapi.testclient import TestClient

from pipeline.predict_pipeline import PredictPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import main


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


def test_fastapi_smoke(tmp_path, monkeypatch):
    model_path, preprocessor_path = _train_pipeline(tmp_path)

    def _mock_init(self):
        self.model_path = str(model_path)
        self.preprocessor_path = str(preprocessor_path)
        self.s3_bucket = None
        self.model_s3_key = "model.pkl"
        self.preprocessor_s3_key = "preprocessor.pkl"
        self.feature_columns = [
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region",
        ]

    monkeypatch.setattr(PredictPipeline, "__init__", _mock_init)

    client = TestClient(main.app)

    response = client.get("/")
    assert response.status_code == 200

    payload = {
        "age": "19",
        "sex": "female",
        "bmi": "27.9",
        "children": "0",
        "smoker": "yes",
        "region": "southwest",
    }
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert "Estimated insurance charges" in response.text

    json_payload = {
        "age": 19,
        "sex": "female",
        "bmi": 27.9,
        "children": 0,
        "smoker": "yes",
        "region": "southwest",
    }
    response = client.post("/predict-json", json=json_payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_charges" in body
    assert body["currency"] == "USD"

    invalid_payload = {
        "age": 19,
        "sex": "female",
        "bmi": 27.9,
        "children": 0,
        "smoker": "yes",
        "region": "invalid",
    }
    response = client.post("/predict-json", json=invalid_payload)
    assert response.status_code == 422

import os
import sys

import boto3
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.feature_columns = ["age", "sex", "bmi", "children", "smoker", "region"]

        self.s3_bucket = os.getenv("MODEL_S3_BUCKET")
        self.model_s3_key = os.getenv("MODEL_S3_MODEL_KEY", "model.pkl")
        self.preprocessor_s3_key = os.getenv(
            "MODEL_S3_PREPROCESSOR_KEY", "preprocessor.pkl"
        )

    def _download_from_s3(self, bucket: str, key: str, dest_path: str) -> None:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        client = boto3.client("s3")
        client.download_file(bucket, key, dest_path)

    def _ensure_artifacts(self) -> None:
        if not self.s3_bucket:
            return

        if not os.path.exists(self.model_path):
            self._download_from_s3(self.s3_bucket, self.model_s3_key, self.model_path)

        if not os.path.exists(self.preprocessor_path):
            self._download_from_s3(
                self.s3_bucket, self.preprocessor_s3_key, self.preprocessor_path
            )

    def predict(self, features):
        try:
            self._ensure_artifacts()
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            df = features.copy()
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                raise CustomException(
                    f"Missing required columns for prediction: {missing_cols}",
                    sys,
                )

            X = df[self.feature_columns].copy()
            X_transformed = preprocessor.transform(X)
            preds = model.predict(X_transformed)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        age: int,
        sex: str,
        bmi: float,
        children: int,
        smoker: str,
        region: str,
    ):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column_name = "charges"
        self.numeric_scaled_columns = ["children", "bmi"]
        self.categorical_columns = ["sex", "region", "smoker"]

    def _make_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "encode_multicategorical",
                    OneHotEncoder(sparse_output=False),
                    self.categorical_columns,
                ),
                ("feature_scaling", MinMaxScaler(), self.numeric_scaled_columns),
            ],
            remainder="passthrough",
        )

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_column_name = self.target_column_name
            required_columns = self.numeric_scaled_columns + self.categorical_columns

            missing_cols = [
                col for col in required_columns if col not in train_df.columns
            ]
            if missing_cols:
                raise CustomException(
                    f"Missing required columns for transformation: {missing_cols}", sys
                )

            input_feature_train_df = train_df.drop(columns=[target_column_name]).copy()
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name]).copy()
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying MinMax scaling and one-hot encoding.")

            preprocessor = self._make_preprocessor()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessor artifact.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

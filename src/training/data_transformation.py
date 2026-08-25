from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.exceptions import ArtifactRepositoryError, TrainingError
from src.repositories.artifact_repository import ArtifactRepository

logger = logging.getLogger(__name__)


class DataTransformation:
    TARGET_COLUMN = "charges"
    PASSTHROUGH_COLUMNS = ["age"]
    NUMERIC_COLUMNS = ["children", "bmi"]
    CATEGORICAL_COLUMNS = ["sex", "region", "smoker"]

    def __init__(self, artifact_repository: ArtifactRepository):
        self._artifact_repository = artifact_repository

    def run(
        self, train_path: str | Path, test_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            self._validate_columns(train_data)
            self._validate_columns(test_data)

            train_features = train_data.drop(columns=[self.TARGET_COLUMN])
            train_target = train_data[self.TARGET_COLUMN]
            test_features = test_data.drop(columns=[self.TARGET_COLUMN])
            test_target = test_data[self.TARGET_COLUMN]

            preprocessor = self._make_preprocessor()
            transformed_train = preprocessor.fit_transform(train_features)
            transformed_test = preprocessor.transform(test_features)
            self._artifact_repository.save_preprocessor(preprocessor)
        except ArtifactRepositoryError:
            raise
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            logger.exception("Data transformation failed")
            raise TrainingError("Data transformation failed.") from exc

        return (
            np.c_[transformed_train, np.asarray(train_target)],
            np.c_[transformed_test, np.asarray(test_target)],
        )

    @classmethod
    def _make_preprocessor(cls) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "encode_multicategorical",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    cls.CATEGORICAL_COLUMNS,
                ),
                ("feature_scaling", MinMaxScaler(), cls.NUMERIC_COLUMNS),
            ],
            remainder="passthrough",
        )

    @classmethod
    def _validate_columns(cls, data: pd.DataFrame) -> None:
        required = {
            cls.TARGET_COLUMN,
            *cls.PASSTHROUGH_COLUMNS,
            *cls.NUMERIC_COLUMNS,
            *cls.CATEGORICAL_COLUMNS,
        }
        missing = sorted(required.difference(data.columns))
        if missing:
            raise TrainingError(f"Training data is missing columns: {missing}")

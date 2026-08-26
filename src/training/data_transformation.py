from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.exceptions import TrainingError
from src.model_contract import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class DataTransformation:
    TARGET_COLUMN = TARGET_COLUMN
    PASSTHROUGH_COLUMNS = ("age",)
    NUMERIC_COLUMNS = ("children", "bmi")
    CATEGORICAL_COLUMNS = ("sex", "smoker", "region")

    def run(
        self, train_path: str | Path, test_path: str | Path
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            self._validate_columns(train_data)
            self._validate_columns(test_data)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            logger.exception("Data transformation failed")
            raise TrainingError("Data transformation failed.") from exc

        columns = [*FEATURE_COLUMNS, self.TARGET_COLUMN]
        return train_data.loc[:, columns].copy(), test_data.loc[:, columns].copy()

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
                ("age_passthrough", "passthrough", cls.PASSTHROUGH_COLUMNS),
            ],
            remainder="drop",
        )

    @classmethod
    def _validate_columns(cls, data: pd.DataFrame) -> None:
        required = {cls.TARGET_COLUMN, *FEATURE_COLUMNS}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise TrainingError(f"Training data is missing columns: {missing}")

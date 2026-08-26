from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.exceptions import ArtifactRepositoryError, TrainingError
from src.mlops.config import MlflowConfig
from src.mlops.tracking import TrackingContext, TrackingRecord, track_training
from src.model_contract import FEATURE_COLUMNS, TARGET_COLUMN
from src.repositories.artifact_repository import ArtifactRepository
from src.training.data_transformation import DataTransformation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    mae: float
    mse: float
    rmse: float


@dataclass(frozen=True)
class TrainingResult:
    selected_model_name: str
    selected_pipeline: Pipeline
    candidate_metrics: dict[str, RegressionMetrics]
    candidate_hyperparameters: dict[str, dict[str, Any]]
    tracking: TrackingRecord | None = None

    @property
    def score(self) -> float:
        return self.candidate_metrics[self.selected_model_name].r2


class ModelTrainer:
    RANDOM_STATE = 42

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        tracking_config: MlflowConfig | None = None,
        tracking_context: TrackingContext | None = None,
    ):
        self._artifact_repository = artifact_repository
        self._tracking_config = tracking_config or MlflowConfig.from_env()
        self._tracking_context = tracking_context

    def run(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> TrainingResult:
        try:
            self._validate_data(train_data)
            self._validate_data(test_data)

            train_features = train_data.loc[:, list(FEATURE_COLUMNS)]
            train_target = train_data.loc[:, TARGET_COLUMN]
            test_features = test_data.loc[:, list(FEATURE_COLUMNS)]
            test_target = test_data.loc[:, TARGET_COLUMN]

            estimators = self._candidate_estimators()
            pipelines = {
                name: Pipeline(
                    steps=[
                        ("preprocessor", DataTransformation._make_preprocessor()),
                        ("regressor", estimator),
                    ]
                )
                for name, estimator in estimators.items()
            }
            metrics: dict[str, RegressionMetrics] = {}

            for name, pipeline in pipelines.items():
                pipeline.fit(train_features, train_target)
                predictions = pipeline.predict(test_features)
                metrics[name] = self._calculate_metrics(test_target, predictions)
                candidate_result = metrics[name]
                logger.info(
                    "%s MAE %.4f | MSE %.4f | RMSE %.4f | R2 %.4f",
                    name,
                    candidate_result.mae,
                    candidate_result.mse,
                    candidate_result.rmse,
                    candidate_result.r2,
                )

            selected_name = max(metrics, key=lambda name: metrics[name].r2)
            selected_pipeline = pipelines[selected_name]
            result = TrainingResult(
                selected_model_name=selected_name,
                selected_pipeline=selected_pipeline,
                candidate_metrics=metrics,
                candidate_hyperparameters={
                    name: estimator.get_params(deep=False)
                    for name, estimator in estimators.items()
                },
            )
            self._artifact_repository.save_model(selected_pipeline)

            total_rows = len(train_data) + len(test_data)
            context = self._tracking_context or TrackingContext(
                dataset_row_count=total_rows,
                test_split_ratio=len(test_data) / total_rows,
                random_seed=self.RANDOM_STATE,
            )
            tracking = track_training(result, context, self._tracking_config)
            result = replace(result, tracking=tracking)
        except (ArtifactRepositoryError, TrainingError):
            raise
        except Exception as exc:
            logger.exception("Model training failed")
            raise TrainingError("Model training failed.") from exc

        logger.info("Saved best pipeline: %s", selected_name)
        return result

    @classmethod
    def _candidate_estimators(cls) -> dict[str, Any]:
        return {
            "Random Forest": RandomForestRegressor(random_state=cls.RANDOM_STATE),
            "Linear Regression": LinearRegression(),
            "Support Vector Machine": SVR(),
            "Bayesian Ridge": BayesianRidge(),
            "AdaBoost": AdaBoostRegressor(random_state=cls.RANDOM_STATE),
        }

    @staticmethod
    def _calculate_metrics(target: pd.Series, predictions: Any) -> RegressionMetrics:
        mse = float(mean_squared_error(target, predictions))
        return RegressionMetrics(
            r2=float(r2_score(target, predictions)),
            mae=float(mean_absolute_error(target, predictions)),
            mse=mse,
            rmse=mse**0.5,
        )

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        required = {TARGET_COLUMN, *FEATURE_COLUMNS}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise TrainingError(f"Training data is missing columns: {missing}")

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

from src.exceptions import ArtifactRepositoryError, TrainingError
from src.repositories.artifact_repository import ArtifactRepository

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, artifact_repository: ArtifactRepository):
        self._artifact_repository = artifact_repository

    def run(self, train_data: np.ndarray, test_data: np.ndarray) -> float:
        try:
            train_features, train_target = train_data[:, :-1], train_data[:, -1]
            test_features, test_target = test_data[:, :-1], test_data[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Support Vector Machine": SVR(),
                "Bayesian Ridge": BayesianRidge(),
                "AdaBoost": AdaBoostRegressor(random_state=42),
            }
            scores: dict[str, float] = {}
            trained_models = {}

            for name, model in models.items():
                model.fit(train_features, train_target)
                predictions = model.predict(test_features)
                scores[name] = float(r2_score(test_target, predictions))
                trained_models[name] = model
                logger.info(
                    "%s MAE %.4f | MSE %.4f | R2 %.4f",
                    name,
                    mean_absolute_error(test_target, predictions),
                    mean_squared_error(test_target, predictions),
                    scores[name],
                )

            best_model_name = max(scores, key=lambda name: scores[name])
            self._artifact_repository.save_model(trained_models[best_model_name])
        except ArtifactRepositoryError:
            raise
        except Exception as exc:
            logger.exception("Model training failed")
            raise TrainingError("Model training failed.") from exc

        logger.info("Saved best model: %s", best_model_name)
        return scores[best_model_name]

import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _evaluate(self, y_true, y_pred):
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Support Vector Machine": SVR(),
                "Bayesian Ridge": BayesianRidge(),
                "AdaBoost": AdaBoostRegressor(random_state=42),
            }

            metrics = {}
            trained_models = {}

            for name, model in models.items():
                logging.info("Training %s", name)
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                metrics[name] = self._evaluate(y_test, y_test_pred)
                trained_models[name] = model
                logging.info(
                    "%s Test MAE: %.4f | MSE: %.4f | R2: %.4f",
                    name,
                    metrics[name]["mae"],
                    metrics[name]["mse"],
                    metrics[name]["r2"],
                )

            if not metrics:
                raise CustomException("Model evaluation did not return results.")

            best_model_name = max(metrics, key=lambda k: metrics[k]["r2"])
            best_model_score = metrics[best_model_name]["r2"]
            best_model = trained_models[best_model_name]

            logging.info(
                "Best model: %s with R2 %.4f", best_model_name, best_model_score
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)

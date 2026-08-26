from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sklearn

from src.exceptions import TrainingError
from src.mlops.config import MlflowConfig
from src.model_contract import FEATURE_COLUMNS, TARGET_COLUMN

if TYPE_CHECKING:
    from src.training.model_trainer import TrainingResult

logger = logging.getLogger(__name__)

DATASET_NAME = "medical_insurance"
INPUT_COLUMN_TYPES = {
    "age": "long",
    "sex": "string",
    "bmi": "double",
    "children": "long",
    "smoker": "string",
    "region": "string",
}
MODEL_INPUT_EXAMPLE = pd.DataFrame(
    [
        {
            "age": 29,
            "sex": "female",
            "bmi": 27.4,
            "children": 2,
            "smoker": "no",
            "region": "southeast",
        }
    ],
    columns=FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class TrackingContext:
    dataset_row_count: int
    test_split_ratio: float = 0.2
    random_seed: int = 42
    dataset_name: str = DATASET_NAME
    selection_metric: str = "r2"


def track_training(
    result: TrainingResult, context: TrackingContext, config: MlflowConfig
) -> str | None:
    """Log one completed training experiment, importing MLflow only when enabled."""
    if not config.enabled:
        return None

    try:
        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.types import ColSpec, Schema

        if config.tracking_uri:
            if config.tracking_uri.startswith("file:"):
                # MLflow 3.15+ requires an explicit opt-in for its maintained
                # local file-store compatibility mode.
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

        with mlflow.start_run() as run:
            selected_estimator = result.selected_pipeline.named_steps["regressor"]
            parameters: dict[str, Any] = {
                "dataset_name": context.dataset_name,
                "dataset_row_count": context.dataset_row_count,
                "feature_names": json.dumps(FEATURE_COLUMNS),
                "target_name": TARGET_COLUMN,
                "test_split_ratio": context.test_split_ratio,
                "random_seed": context.random_seed,
                "selection_metric": context.selection_metric,
                "selected_model_name": result.selected_model_name,
                "selected_model_hyperparameters": _json_params(
                    selected_estimator.get_params(deep=False)
                ),
                "candidate_model_names": json.dumps(
                    list(result.candidate_metrics), separators=(",", ":")
                ),
            }
            for name, hyperparameters in result.candidate_hyperparameters.items():
                parameters[f"candidate/{_slug(name)}/hyperparameters"] = _json_params(
                    hyperparameters
                )
            mlflow.log_params(parameters)

            for name, metrics in result.candidate_metrics.items():
                prefix = f"candidate/{_slug(name)}"
                mlflow.log_metrics(
                    {
                        f"{prefix}/r2": metrics.r2,
                        f"{prefix}/mae": metrics.mae,
                        f"{prefix}/mse": metrics.mse,
                        f"{prefix}/rmse": metrics.rmse,
                    }
                )

            selected_metrics = result.candidate_metrics[result.selected_model_name]
            mlflow.log_metrics(
                {
                    "selected/r2": selected_metrics.r2,
                    "selected/mae": selected_metrics.mae,
                    "selected/mse": selected_metrics.mse,
                    "selected/rmse": selected_metrics.rmse,
                }
            )
            tags = {
                "problem_type": "regression",
                "dataset": DATASET_NAME,
                "selected_model": result.selected_model_name,
                "tracking_backend": config.tracking_backend,
            }
            git_sha = _git_commit_sha()
            if git_sha:
                tags["git_commit_sha"] = git_sha
            mlflow.set_tags(tags)

            signature = ModelSignature(
                inputs=Schema(
                    [ColSpec(INPUT_COLUMN_TYPES[name], name) for name in FEATURE_COLUMNS]
                ),
                outputs=Schema([ColSpec("double")]),
            )
            mlflow.sklearn.log_model(
                sk_model=result.selected_pipeline,
                name="model",
                signature=signature,
                input_example=MODEL_INPUT_EXAMPLE,
                pip_requirements=[
                    f"scikit-learn=={sklearn.__version__}",
                    f"pandas=={pd.__version__}",
                    f"numpy=={np.__version__}",
                ],
            )
            return run.info.run_id
    except Exception as exc:
        logger.exception("MLflow experiment tracking failed")
        raise TrainingError("MLflow experiment tracking failed.") from exc


def _json_params(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, default=str, separators=(",", ":"))


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _git_commit_sha() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return process.stdout.strip() or None

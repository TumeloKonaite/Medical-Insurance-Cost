from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sklearn

from src.exceptions import TrainingError
from src.mlops.config import MlflowConfig
from src.model_contract import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    PREDICTION_CONTRACT_VERSION,
    TARGET_COLUMN,
)

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

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
    dataset_filename: str | None = None
    dataset_sha256: str | None = None

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str | Path,
        *,
        dataset_row_count: int,
        test_split_ratio: float = 0.2,
        random_seed: int = 42,
        selection_metric: str = "r2",
    ) -> TrackingContext:
        path = Path(dataset_path)
        return cls(
            dataset_row_count=dataset_row_count,
            test_split_ratio=test_split_ratio,
            random_seed=random_seed,
            selection_metric=selection_metric,
            dataset_filename=path.name,
            dataset_sha256=_sha256_file(path),
        )


@dataclass(frozen=True)
class TrackingRecord:
    run_id: str
    model_uri: str
    git_commit_sha: str | None
    dataset_sha256: str | None
    registered_model_name: str | None = None
    model_version: str | None = None
    pipeline_sha256: str | None = None


def track_training(
    result: TrainingResult, context: TrackingContext, config: MlflowConfig
) -> TrackingRecord | None:
    """Log one completed training experiment, importing MLflow only when enabled."""
    if not config.enabled:
        return None

    try:
        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.types import ColSpec, Schema

        config.validate()
        git_sha = _git_commit_sha()
        if config.is_remote:
            _validate_remote_lineage(context, git_sha)

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
                "dataset_filename": context.dataset_filename or context.dataset_name,
                "dataset_sha256": context.dataset_sha256 or "unavailable",
                "canonical_feature_list": json.dumps(FEATURE_COLUMNS),
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
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
            }
            if git_sha:
                tags["git_commit_sha"] = git_sha
                tags["source_commit_sha"] = git_sha
            if context.dataset_sha256:
                tags["dataset_sha256"] = context.dataset_sha256
            mlflow.set_tags(tags)

            signature = ModelSignature(
                inputs=Schema(
                    [ColSpec(INPUT_COLUMN_TYPES[name], name) for name in FEATURE_COLUMNS]
                ),
                outputs=Schema([ColSpec("double")]),
            )
            logged_model_info = mlflow.sklearn.log_model(
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
            run_id = run.info.run_id

        model_uri = logged_model_info.model_uri
        record = TrackingRecord(
            run_id=run_id,
            model_uri=model_uri,
            git_commit_sha=git_sha,
            dataset_sha256=context.dataset_sha256,
        )
        if config.enable_model_registration:
            _validate_registration_readiness(result, logged_model_info)
            from src.mlops.registry import register_logged_model

            registration = register_logged_model(
                config=config,
                run_id=run_id,
                source_model_uri=model_uri,
                source_commit_sha=git_sha,
                dataset_sha256=context.dataset_sha256,
                selected_model=result.selected_model_name,
                selection_metric=context.selection_metric,
            )
            record = TrackingRecord(
                run_id=run_id,
                model_uri=model_uri,
                git_commit_sha=git_sha,
                dataset_sha256=context.dataset_sha256,
                registered_model_name=registration.model_name,
                model_version=registration.model_version,
                pipeline_sha256=registration.pipeline_sha256,
            )
        return record
    except TrainingError:
        raise
    except Exception as exc:
        logger.error("MLflow tracking failed; sensitive exception details were suppressed.")
        message = (
            "Remote MLflow tracking failed. Verify MLFLOW_TRACKING_URI, DagsHub "
            "credentials, repository access, and network connectivity."
            if config.is_remote
            else "MLflow experiment tracking failed. Verify the local tracking store."
        )
        raise TrainingError(message) from exc


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as dataset:
            for chunk in iter(lambda: dataset.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrainingError("The tracked source dataset could not be checksummed.") from exc
    return digest.hexdigest()


def _validate_remote_lineage(
    context: TrackingContext, git_commit_sha: str | None
) -> None:
    missing = []
    if not git_commit_sha:
        missing.append("Git commit SHA")
    if not context.dataset_filename:
        missing.append("dataset filename")
    if not context.dataset_sha256:
        missing.append("dataset SHA-256")
    if missing:
        raise TrainingError(
            "Remote MLflow tracking requires lineage metadata: " + ", ".join(missing) + "."
        )


def _validate_registration_readiness(
    result: TrainingResult, logged_model_info: ModelInfo
) -> None:
    metric_values = [
        value
        for metrics in result.candidate_metrics.values()
        for value in (metrics.r2, metrics.mae, metrics.mse, metrics.rmse)
    ]
    if not all(np.isfinite(metric_values)):
        raise TrainingError("Model registration requires finite candidate metrics.")

    prediction = np.asarray(result.selected_pipeline.predict(MODEL_INPUT_EXAMPLE))
    if prediction.size == 0 or not np.isfinite(prediction).all():
        raise TrainingError(
            "Model registration requires a finite raw-feature smoke prediction."
        )

    if logged_model_info.signature is None:
        raise TrainingError("Model registration requires an MLflow model signature.")
    if logged_model_info.saved_input_example_info is None:
        raise TrainingError("Model registration requires an MLflow input example.")

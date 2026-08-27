from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.exceptions import DeploymentError
from src.mlops.runtime import validate_local_package
from src.model_contract import FEATURE_COLUMNS, TARGET_COLUMN
from src.monitoring.arize_client import (
    ACTUAL_COLUMN,
    PREDICTION_COLUMN,
    PREDICTION_ID_COLUMN,
    ArizeBatchClient,
    BatchClient,
)
from src.monitoring.config import ArizeExportConfig
from src.schemas.prediction import PredictionRequest


class BaselineUploadError(ValueError):
    """A sanitized validation or upload error for the baseline workflow."""


@dataclass(frozen=True)
class BaselineBatch:
    dataframe: pd.DataFrame
    model_version: str
    batch_id: str
    dataset_sha256: str
    mlflow_run_id: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BaselineUploadError("The baseline dataset could not be read.") from exc
    return digest.hexdigest()


def build_baseline_batch(
    *, test_data: str | Path, model_package: str | Path
) -> BaselineBatch:
    data_path = Path(test_data)
    dataset_sha256 = _sha256_file(data_path)
    try:
        source = pd.read_csv(data_path)
    except Exception as exc:
        raise BaselineUploadError("The baseline dataset is not a valid CSV file.") from exc
    expected_columns = (*FEATURE_COLUMNS, TARGET_COLUMN)
    if tuple(source.columns) != expected_columns or source.empty:
        raise BaselineUploadError(
            "The baseline dataset must contain exactly the six canonical features "
            "followed by charges."
        )

    validated_rows: list[dict[str, object]] = []
    actuals: list[float] = []
    try:
        for row in source.to_dict(orient="records"):
            request = PredictionRequest.model_validate(
                {name: row[name] for name in FEATURE_COLUMNS}
            )
            actual = float(row[TARGET_COLUMN])
            if not math.isfinite(actual) or actual < 0:
                raise ValueError
            validated_rows.append(request.model_dump())
            actuals.append(actual)
    except Exception as exc:
        raise BaselineUploadError(
            "The baseline dataset contains invalid feature or target values."
        ) from exc

    try:
        deployment = validate_local_package(model_package)
    except DeploymentError as exc:
        raise BaselineUploadError(
            "The production model package is invalid."
        ) from exc
    features = pd.DataFrame(validated_rows, columns=FEATURE_COLUMNS)
    try:
        predictions = np.asarray(
            deployment.model.predict(features), dtype=float
        ).reshape(-1)
    except Exception as exc:
        raise BaselineUploadError("Baseline prediction generation failed.") from exc
    if len(predictions) != len(features) or not np.isfinite(predictions).all():
        raise BaselineUploadError(
            "Baseline predictions must be finite and match the dataset rows."
        )

    frame = features.copy()
    frame.insert(
        0,
        PREDICTION_ID_COLUMN,
        [
            hashlib.sha256(f"{dataset_sha256}:{index}".encode()).hexdigest()
            for index in range(len(frame))
        ],
    )
    frame[PREDICTION_COLUMN] = predictions
    frame[ACTUAL_COLUMN] = actuals
    metadata = deployment.metadata
    return BaselineBatch(
        dataframe=frame,
        model_version=metadata["model_version"],
        batch_id=metadata["mlflow_run_id"] or dataset_sha256,
        dataset_sha256=dataset_sha256,
        mlflow_run_id=metadata["mlflow_run_id"],
    )


def upload_baseline(
    *,
    config: ArizeExportConfig,
    test_data: str | Path,
    model_package: str | Path,
    client: BatchClient | None = None,
) -> dict[str, str | int]:
    batch = build_baseline_batch(
        test_data=test_data, model_package=model_package
    )
    uploader = client or ArizeBatchClient(config)
    try:
        status_code = uploader.upload(
            batch.dataframe,
            event_type="baseline",
            model_version=batch.model_version,
            environment="validation",
            batch_id=batch.batch_id,
        )
    except Exception:
        raise BaselineUploadError("The baseline upload failed.") from None
    return {
        "records_sent": len(batch.dataframe),
        "model_version": batch.model_version,
        "batch_id": batch.batch_id,
        "dataset_sha256": batch.dataset_sha256,
        "status_code": status_code,
    }

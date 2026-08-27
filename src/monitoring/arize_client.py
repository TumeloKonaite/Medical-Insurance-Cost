from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import logging
from typing import Any, Literal, Protocol

import pandas as pd

from src.model_contract import FEATURE_COLUMNS
from src.monitoring.config import ArizeExportConfig

PREDICTION_ID_COLUMN = "prediction_id"
TIMESTAMP_COLUMN = "prediction_timestamp"
PREDICTION_COLUMN = "prediction"
ACTUAL_COLUMN = "actual"
TAG_COLUMNS = (
    "source",
    "prediction_contract_version",
    "inference_latency_ms",
)


class ArizeUploadError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class BatchClient(Protocol):
    def upload(
        self,
        dataframe: pd.DataFrame,
        *,
        event_type: Literal["prediction", "actual", "baseline"],
        model_version: str,
        environment: Literal["production", "validation"],
        batch_id: str = "",
    ) -> int: ...


def unix_seconds(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def prediction_dataframe(records: Sequence[Any]) -> pd.DataFrame:
    rows = [
        {
            PREDICTION_ID_COLUMN: str(record.request_id),
            TIMESTAMP_COLUMN: unix_seconds(record.created_at),
            "age": int(record.age),
            "sex": str(record.sex),
            "bmi": float(record.bmi),
            "children": int(record.children),
            "smoker": str(record.smoker),
            "region": str(record.region),
            PREDICTION_COLUMN: float(record.predicted_charges),
            "source": str(record.source),
            "prediction_contract_version": str(
                record.prediction_contract_version
            ),
            "inference_latency_ms": float(record.inference_latency_ms),
        }
        for record in records
    ]
    columns = [
        PREDICTION_ID_COLUMN,
        TIMESTAMP_COLUMN,
        *FEATURE_COLUMNS,
        PREDICTION_COLUMN,
        *TAG_COLUMNS,
    ]
    return pd.DataFrame(rows, columns=columns)


def actual_dataframe(records: Sequence[Any]) -> pd.DataFrame:
    rows = [
        {
            PREDICTION_ID_COLUMN: str(record.request_id),
            TIMESTAMP_COLUMN: unix_seconds(record.created_at),
            ACTUAL_COLUMN: float(record.actual_charges),
        }
        for record in records
    ]
    return pd.DataFrame(
        rows, columns=[PREDICTION_ID_COLUMN, TIMESTAMP_COLUMN, ACTUAL_COLUMN]
    )


def _schema(event_type: Literal["prediction", "actual", "baseline"]):
    from arize.ml.types import Schema

    common = {
        "prediction_id_column_name": PREDICTION_ID_COLUMN,
    }
    if event_type == "prediction":
        return Schema(
            **common,
            timestamp_column_name=TIMESTAMP_COLUMN,
            prediction_label_column_name=PREDICTION_COLUMN,
            feature_column_names=list(FEATURE_COLUMNS),
            tag_column_names=list(TAG_COLUMNS),
        )
    if event_type == "actual":
        return Schema(
            **common,
            timestamp_column_name=TIMESTAMP_COLUMN,
            actual_label_column_name=ACTUAL_COLUMN,
        )
    return Schema(
        **common,
        prediction_label_column_name=PREDICTION_COLUMN,
        actual_label_column_name=ACTUAL_COLUMN,
        feature_column_names=list(FEATURE_COLUMNS),
    )


class ArizeBatchClient:
    def __init__(self, config: ArizeExportConfig):
        try:
            from arize import ArizeClient

            # SDK diagnostics can contain server-provided detail. Our own adapter
            # emits only the sanitized batch metadata required for operations.
            logging.getLogger("arize").disabled = True
            self._client = ArizeClient(api_key=config.api_key)
        except Exception:
            raise ArizeUploadError(
                "The Arize client could not be initialized."
            ) from None
        self._config = config

    def upload(
        self,
        dataframe: pd.DataFrame,
        *,
        event_type: Literal["prediction", "actual", "baseline"],
        model_version: str,
        environment: Literal["production", "validation"],
        batch_id: str = "",
    ) -> int:
        from arize.ml.types import Environments, ModelTypes

        arize_environment = (
            Environments.PRODUCTION
            if environment == "production"
            else Environments.VALIDATION
        )
        try:
            response = self._client.ml.log(
                space_id=self._config.space_id,
                model_name=self._config.model_name,
                model_type=ModelTypes.NUMERIC,
                dataframe=dataframe,
                schema=_schema(event_type),
                environment=arize_environment,
                model_version=model_version,
                batch_id=batch_id,
            )
            status_code = int(getattr(response, "status_code", 0))
        except Exception:
            raise ArizeUploadError("The Arize upload failed.") from None
        if not 200 <= status_code < 300:
            raise ArizeUploadError(
                "Arize did not acknowledge the upload.", status_code=status_code
            )
        return status_code

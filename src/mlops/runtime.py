from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.exceptions import DeploymentError
from src.model_contract import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    PREDICTION_CONTRACT_VERSION,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = "medical-insurance-cost"
MODAL_APPLICATION = "medical-insurance-cost"
PIPELINE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
NUMERIC_VERSION_PATTERN = re.compile(r"^[1-9][0-9]*$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
EXACT_MODEL_URI_PATTERN = re.compile(
    rf"^models:/{re.escape(MODEL_NAME)}/(?P<version>[1-9][0-9]*)$"
)
REQUIRED_METADATA_FIELDS = frozenset(
    {
        "deployment_id",
        "deployment_timestamp_utc",
        "environment",
        "modal_application",
        "model_name",
        "model_version",
        "model_uri",
        "mlflow_run_id",
        "source_commit_sha",
        "dataset_sha256",
        "pipeline_sha256",
        "feature_schema_version",
        "prediction_contract_version",
        "validation_status",
    }
)


@dataclass(frozen=True)
class ValidatedDeployment:
    metadata: Mapping[str, str]
    model: Any


def parse_exact_model_uri(model_uri: str) -> tuple[str, str]:
    """Return the only registered-model identity accepted for production."""
    match = EXACT_MODEL_URI_PATTERN.fullmatch(model_uri)
    if match is None:
        raise DeploymentError(
            "Production packaging requires an exact numeric model URI in the form "
            f"models:/{MODEL_NAME}/<positive-integer>."
        )
    return MODEL_NAME, match.group("version")


def validate_local_package(package_dir: str | Path) -> ValidatedDeployment:
    """Validate and load a package using local files only."""
    root = Path(package_dir).resolve()
    metadata = _read_metadata(root / "deployment_metadata.json")
    _validate_metadata(metadata)

    model_dir = root / "model"
    if not model_dir.is_dir() or not (model_dir / "MLmodel").is_file():
        raise DeploymentError("The packaged MLflow model is missing or incomplete.")

    model_metadata, serialized_pipeline = _inspect_mlflow_model(model_dir)
    actual_checksum = _sha256_file(serialized_pipeline)
    if actual_checksum != metadata["pipeline_sha256"]:
        raise DeploymentError(
            "The packaged pipeline checksum does not match deployment metadata."
        )

    input_example = _load_input_example(model_metadata, model_dir)
    model = _load_sklearn_model(model_dir)
    _smoke_predict(model, input_example)
    return ValidatedDeployment(metadata=metadata, model=model)


@lru_cache(maxsize=4)
def _cached_local_package(package_dir: str) -> ValidatedDeployment:
    return validate_local_package(package_dir)


def validate_production_startup(
    package_dir: str | Path = "/app/build/model",
) -> Mapping[str, str]:
    """Fail container startup unless the baked package is complete and untampered."""
    deployment = _cached_local_package(str(Path(package_dir).resolve()))
    identity_fields = (
        "deployment_id",
        "model_name",
        "model_version",
        "mlflow_run_id",
        "source_commit_sha",
        "pipeline_sha256",
    )
    identity = " ".join(
        f"{field}={deployment.metadata[field]}" for field in identity_fields
    )
    logger.info(
        "Validated production model package %s",
        identity,
        extra={field: deployment.metadata[field] for field in identity_fields},
    )
    print(f"Validated production model package {identity}", flush=True)
    return deployment.metadata


def get_validated_model(package_dir: str | Path) -> Any:
    """Return the process-cached model established by startup validation."""
    return _cached_local_package(str(Path(package_dir).resolve())).model


def get_validated_metadata(package_dir: str | Path) -> Mapping[str, str]:
    """Return metadata from the process-cached, validated model package."""
    return _cached_local_package(str(Path(package_dir).resolve())).metadata


def clear_runtime_cache() -> None:
    """Clear process state for isolated tests."""
    _cached_local_package.cache_clear()


def _read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise DeploymentError("Deployment metadata is missing.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Deployment metadata is unreadable or invalid.") from exc
    if not isinstance(payload, dict):
        raise DeploymentError("Deployment metadata must be a JSON object.")
    if set(payload) != REQUIRED_METADATA_FIELDS:
        missing = sorted(REQUIRED_METADATA_FIELDS - set(payload))
        extra = sorted(set(payload) - REQUIRED_METADATA_FIELDS)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise DeploymentError(
            "Deployment metadata fields are invalid (" + "; ".join(details) + ")."
        )
    if not all(
        isinstance(value, str) and value.strip() == value and value
        for value in payload.values()
    ):
        raise DeploymentError("Every deployment metadata value must be a non-empty string.")
    return payload


def _validate_metadata(metadata: Mapping[str, str]) -> None:
    model_name, version = parse_exact_model_uri(metadata["model_uri"])
    if metadata["model_name"] != model_name or metadata["model_version"] != version:
        raise DeploymentError("Deployment metadata contains inconsistent model identity.")
    if not NUMERIC_VERSION_PATTERN.fullmatch(metadata["model_version"]):
        raise DeploymentError("Deployment metadata requires a positive numeric model version.")
    expected_values = {
        "environment": "production",
        "modal_application": MODAL_APPLICATION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "validation_status": "validated",
    }
    for field, expected in expected_values.items():
        if metadata[field] != expected:
            raise DeploymentError(f"Deployment metadata has an invalid {field} value.")
    for field in ("dataset_sha256", "pipeline_sha256"):
        if not PIPELINE_SHA256_PATTERN.fullmatch(metadata[field]):
            raise DeploymentError(f"Deployment metadata has an invalid {field} value.")
    if not GIT_SHA_PATTERN.fullmatch(metadata["source_commit_sha"]):
        raise DeploymentError("Deployment metadata has an invalid source_commit_sha value.")
    try:
        uuid.UUID(metadata["deployment_id"])
    except ValueError as exc:
        raise DeploymentError("Deployment metadata has an invalid deployment_id value.") from exc
    try:
        timestamp = datetime.fromisoformat(
            metadata["deployment_timestamp_utc"].replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise DeploymentError(
            "Deployment metadata has an invalid deployment_timestamp_utc value."
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise DeploymentError("The deployment timestamp must be expressed in UTC.")


def _inspect_mlflow_model(model_dir: Path) -> tuple[Any, Path]:
    try:
        import mlflow

        metadata = mlflow.models.Model.load(str(model_dir))
    except Exception as exc:
        raise DeploymentError("The packaged MLflow model metadata could not be loaded.") from exc

    flavor = metadata.flavors.get("sklearn") if metadata.flavors else None
    if not isinstance(flavor, dict):
        raise DeploymentError("The packaged model does not contain the supported sklearn flavor.")
    serialized_name = flavor.get("pickled_model")
    if not isinstance(serialized_name, str) or not serialized_name:
        raise DeploymentError("The sklearn flavor does not identify its serialized pipeline.")
    serialized_pipeline = (model_dir / serialized_name).resolve()
    if not serialized_pipeline.is_relative_to(model_dir.resolve()):
        raise DeploymentError("The sklearn flavor declares an unsafe pipeline path.")
    if not serialized_pipeline.is_file():
        raise DeploymentError("The serialized sklearn pipeline is missing.")

    signature = metadata.signature
    if signature is None or signature.inputs is None:
        raise DeploymentError("The packaged MLflow model does not contain a signature.")
    signature_columns = tuple(
        column.name for column in signature.inputs.inputs if column.name is not None
    )
    if signature_columns != tuple(FEATURE_COLUMNS):
        raise DeploymentError(
            "The model signature does not match the six canonical raw feature fields."
        )
    if metadata.saved_input_example_info is None:
        raise DeploymentError("The packaged MLflow model does not contain an input example.")
    return metadata, serialized_pipeline


def _load_input_example(model_metadata: Any, model_dir: Path) -> pd.DataFrame:
    try:
        example = model_metadata.load_input_example(str(model_dir))
    except Exception as exc:
        raise DeploymentError("The packaged model input example could not be loaded.") from exc
    if example is None:
        raise DeploymentError("The packaged MLflow model does not contain an input example.")
    if isinstance(example, pd.DataFrame):
        frame = example.copy()
    elif isinstance(example, Mapping):
        try:
            frame = pd.DataFrame(example)
        except ValueError:
            frame = pd.DataFrame([example])
    else:
        raise DeploymentError("The model input example is not a raw-feature table.")
    if tuple(frame.columns) != tuple(FEATURE_COLUMNS) or frame.empty:
        raise DeploymentError("The model input example does not match the feature contract.")
    return frame


def _load_sklearn_model(model_dir: Path) -> Any:
    try:
        import mlflow

        model = mlflow.sklearn.load_model(str(model_dir))
    except Exception as exc:
        raise DeploymentError("The packaged sklearn model could not be loaded.") from exc
    if not callable(getattr(model, "predict", None)):
        raise DeploymentError("The packaged sklearn model has no prediction interface.")
    return model


def _smoke_predict(model: Any, raw_features: pd.DataFrame) -> float:
    try:
        result = np.asarray(model.predict(raw_features), dtype=float).reshape(-1)
    except Exception as exc:
        raise DeploymentError("The packaged model smoke prediction failed.") from exc
    if result.size != 1 or not math.isfinite(float(result[0])):
        raise DeploymentError("The packaged model smoke prediction is not finite and numeric.")
    return float(result[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DeploymentError("The serialized pipeline could not be checksummed.") from exc
    return digest.hexdigest()

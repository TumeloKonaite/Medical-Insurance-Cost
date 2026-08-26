from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.exceptions import DeploymentError, ModelRegistryError
from src.mlops.config import MlflowConfig
from src.mlops.registry import _configure_mlflow, inspect_version
from src.mlops.runtime import (
    GIT_SHA_PATTERN,
    MODEL_NAME,
    PIPELINE_SHA256_PATTERN,
    _inspect_mlflow_model,
    _load_input_example,
    _load_sklearn_model,
    _sha256_file,
    _smoke_predict,
    parse_exact_model_uri,
    validate_local_package,
)
from src.model_contract import FEATURE_SCHEMA_VERSION, PREDICTION_CONTRACT_VERSION


@dataclass(frozen=True)
class DeploymentMetadata:
    deployment_id: str
    deployment_timestamp_utc: str
    environment: str
    modal_application: str
    model_name: str
    model_version: str
    model_uri: str
    mlflow_run_id: str
    source_commit_sha: str
    dataset_sha256: str
    pipeline_sha256: str
    feature_schema_version: str
    prediction_contract_version: str
    validation_status: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def prepare_deployment(
    *,
    config: MlflowConfig,
    model_uri: str,
    output_dir: str | Path,
    expected_run_id: str,
    expected_pipeline_sha256: str,
) -> DeploymentMetadata:
    """Build and atomically publish one validated, inference-only model package."""
    model_name, model_version = parse_exact_model_uri(model_uri)
    if model_name != MODEL_NAME or config.registered_model_name != MODEL_NAME:
        raise DeploymentError(f"The production model name must be {MODEL_NAME}.")
    if not expected_run_id.strip():
        raise DeploymentError("An expected MLflow run ID is required.")
    if not PIPELINE_SHA256_PATTERN.fullmatch(expected_pipeline_sha256):
        raise DeploymentError(
            "The expected pipeline checksum must be 64 lowercase hexadecimal characters."
        )

    destination = Path(output_dir).resolve()
    _require_clean_destination(destination)
    version = _validate_registry_lineage(
        config=config,
        model_name=model_name,
        model_version=model_version,
        expected_run_id=expected_run_id,
        expected_pipeline_sha256=expected_pipeline_sha256,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}-staging-", dir=destination.parent
        ) as staging_name:
            staging = Path(staging_name)
            downloaded_model = _download_exact_model(model_uri, staging / "download")
            _validate_downloaded_model(
                downloaded_model, expected_pipeline_sha256=version.pipeline_sha256
            )

            package = staging / "package"
            shutil.copytree(downloaded_model, package / "model")
            metadata = DeploymentMetadata(
                deployment_id=str(uuid.uuid4()),
                deployment_timestamp_utc=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                environment="production",
                modal_application="medical-insurance-cost",
                model_name=model_name,
                model_version=model_version,
                model_uri=model_uri,
                mlflow_run_id=version.mlflow_run_id,
                source_commit_sha=version.source_commit_sha,
                dataset_sha256=version.dataset_sha256,
                pipeline_sha256=version.pipeline_sha256,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                prediction_contract_version=PREDICTION_CONTRACT_VERSION,
                validation_status="validated",
            )
            (package / "deployment_metadata.json").write_text(
                json.dumps(metadata.as_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            # The same local-only validation used at container startup is the final gate.
            validate_local_package(package)
            if destination.exists():
                destination.rmdir()
            package.replace(destination)
    except DeploymentError:
        raise
    except OSError as exc:
        raise DeploymentError("The deployment package could not be written safely.") from exc
    return metadata


def _require_clean_destination(destination: Path) -> None:
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise DeploymentError("The deployment output path must be a new or empty directory.")
    if destination.is_dir():
        try:
            if next(destination.iterdir(), None) is not None:
                raise DeploymentError(
                    "The deployment output directory is not empty; refusing to mix packages."
                )
        except OSError as exc:
            raise DeploymentError("The deployment output directory cannot be inspected.") from exc


def _validate_registry_lineage(
    *,
    config: MlflowConfig,
    model_name: str,
    model_version: str,
    expected_run_id: str,
    expected_pipeline_sha256: str,
) -> Any:
    try:
        version = inspect_version(config, model_name, model_version)
    except ModelRegistryError as exc:
        raise DeploymentError("The exact registered model version is invalid.") from exc
    if version.model_uri != f"models:/{MODEL_NAME}/{model_version}":
        raise DeploymentError("The registry returned an inconsistent exact model URI.")
    if version.validation_status != "validated":
        raise DeploymentError("Only a validated registered model version can be packaged.")
    if version.mlflow_run_id != expected_run_id:
        raise DeploymentError("The registered model run ID does not match the expected run ID.")
    if version.pipeline_sha256 != expected_pipeline_sha256:
        raise DeploymentError(
            "The registered pipeline checksum does not match the expected checksum."
        )
    if version.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise DeploymentError("The registered model feature contract is unsupported.")
    if version.prediction_contract_version != PREDICTION_CONTRACT_VERSION:
        raise DeploymentError("The registered model prediction contract is unsupported.")
    if not GIT_SHA_PATTERN.fullmatch(version.source_commit_sha):
        raise DeploymentError("The registered source Git commit is missing or invalid.")

    try:
        client = _configure_mlflow(config)
        source_run = client.get_run(version.mlflow_run_id)
    except Exception as exc:
        raise DeploymentError("The registered model source run could not be verified.") from exc
    if source_run.info.run_id != version.mlflow_run_id:
        raise DeploymentError("The source run identity does not match the model version.")
    run_tags = dict(source_run.data.tags or {})
    source_commit = run_tags.get("source_commit_sha") or run_tags.get(
        "mlflow.source.git.commit"
    )
    if not source_commit or source_commit != version.source_commit_sha:
        raise DeploymentError("The source Git commit tag is missing or inconsistent.")
    dataset_checksum = run_tags.get("dataset_sha256")
    if not dataset_checksum or dataset_checksum != version.dataset_sha256:
        raise DeploymentError("The source dataset checksum is missing or inconsistent.")
    return version


def _download_exact_model(model_uri: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        import mlflow

        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(destination),
            )
        ).resolve()
    except Exception as exc:
        raise DeploymentError("The exact registered model artifact could not be downloaded.") from exc
    if not downloaded.is_relative_to(destination.resolve()) or not downloaded.is_dir():
        raise DeploymentError("MLflow returned an unsafe or incomplete artifact path.")
    return downloaded


def _validate_downloaded_model(
    model_dir: Path, *, expected_pipeline_sha256: str
) -> None:
    metadata, serialized_pipeline = _inspect_mlflow_model(model_dir)
    actual_checksum = _sha256_file(serialized_pipeline)
    if actual_checksum != expected_pipeline_sha256:
        raise DeploymentError(
            "The downloaded pipeline checksum does not match the registered version."
        )
    input_example = _load_input_example(metadata, model_dir)
    model = _load_sklearn_model(model_dir)
    _smoke_predict(model, input_example)

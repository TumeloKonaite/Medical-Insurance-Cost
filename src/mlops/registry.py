from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.exceptions import ModelRegistryError
from src.mlops.config import MlflowConfig
from src.model_contract import FEATURE_SCHEMA_VERSION, PREDICTION_CONTRACT_VERSION

PIPELINE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
NUMERIC_VERSION_PATTERN = re.compile(r"^[1-9][0-9]*$")
REQUIRED_VERSION_TAGS = (
    "training_run_id",
    "source_commit_sha",
    "dataset_sha256",
    "feature_schema_version",
    "prediction_contract_version",
    "selected_model",
    "selection_metric",
    "validation_status",
    "pipeline_sha256",
)


@dataclass(frozen=True)
class RegistrationResult:
    model_name: str
    model_version: str
    run_id: str
    source_model_uri: str
    source_commit_sha: str
    dataset_sha256: str
    pipeline_sha256: str


@dataclass(frozen=True)
class RegistryVersion:
    model_name: str
    model_version: str
    model_uri: str
    mlflow_run_id: str
    source_commit_sha: str
    dataset_sha256: str
    pipeline_sha256: str
    validation_status: str
    feature_schema_version: str
    prediction_contract_version: str
    selected_model: str
    selection_metric: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AliasResolution:
    model_name: str
    alias: str
    model_version: str
    model_uri: str
    mlflow_run_id: str
    source_commit_sha: str
    dataset_sha256: str
    pipeline_sha256: str
    validation_status: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def register_logged_model(
    *,
    config: MlflowConfig,
    run_id: str,
    source_model_uri: str,
    source_commit_sha: str | None,
    dataset_sha256: str | None,
    selected_model: str,
    selection_metric: str,
) -> RegistrationResult:
    """Register one selected run artifact, then checksum that exact version."""
    if not source_commit_sha or not dataset_sha256:
        raise ModelRegistryError(
            "Model registration requires a Git commit SHA and dataset SHA-256."
        )
    _require_sha256(dataset_sha256, "dataset_sha256")

    try:
        import mlflow

        client = _configure_mlflow(config)
        tags = {
            "training_run_id": run_id,
            "source_commit_sha": source_commit_sha,
            "dataset_sha256": dataset_sha256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
            "selected_model": selected_model,
            "selection_metric": selection_metric,
            "validation_status": "candidate",
        }
        version = mlflow.register_model(
            model_uri=source_model_uri,
            name=config.registered_model_name,
            tags=tags,
        )
        version_number = _require_numeric_version(str(version.version))
        pipeline_sha256 = _checksum_registered_pipeline(
            config=config,
            model_name=config.registered_model_name,
            model_version=version_number,
        )
        client.set_model_version_tag(
            config.registered_model_name,
            version_number,
            "pipeline_sha256",
            pipeline_sha256,
        )
        return RegistrationResult(
            model_name=config.registered_model_name,
            model_version=version_number,
            run_id=run_id,
            source_model_uri=source_model_uri,
            source_commit_sha=source_commit_sha,
            dataset_sha256=dataset_sha256,
            pipeline_sha256=pipeline_sha256,
        )
    except ModelRegistryError:
        raise
    except Exception as exc:
        raise ModelRegistryError(
            "Model registration failed. Verify registry access and the logged model artifact."
        ) from exc


def inspect_version(
    config: MlflowConfig, model_name: str, model_version: str
) -> RegistryVersion:
    version_number = _require_numeric_version(model_version)
    try:
        client = _configure_mlflow(config)
        version = client.get_model_version(model_name, version_number)
    except ModelRegistryError:
        raise
    except Exception as exc:
        raise ModelRegistryError(
            "The requested registered model version could not be read. Verify its name, "
            "numeric version, credentials, and registry access."
        ) from exc
    return _validated_version(model_name, version)


def resolve_alias(
    config: MlflowConfig, model_name: str, alias: str
) -> AliasResolution:
    if not alias.strip():
        raise ModelRegistryError("The model alias cannot be empty.")
    try:
        client = _configure_mlflow(config)
        version = client.get_model_version_by_alias(model_name, alias)
    except ModelRegistryError:
        raise
    except Exception as exc:
        raise ModelRegistryError(
            "The model alias could not be resolved. Verify that it exists and that "
            "the configured credentials can read the registry."
        ) from exc

    inspected = _validated_version(model_name, version)
    if inspected.validation_status != "validated":
        raise ModelRegistryError(
            "The resolved model version is not validated and cannot be deployed."
        )
    return AliasResolution(
        model_name=model_name,
        alias=alias,
        model_version=inspected.model_version,
        model_uri=inspected.model_uri,
        mlflow_run_id=inspected.mlflow_run_id,
        source_commit_sha=inspected.source_commit_sha,
        dataset_sha256=inspected.dataset_sha256,
        pipeline_sha256=inspected.pipeline_sha256,
        validation_status=inspected.validation_status,
    )


def promote_version(
    config: MlflowConfig,
    model_name: str,
    model_version: str,
    alias: str,
    *,
    confirmed: bool,
) -> AliasResolution:
    if not confirmed:
        raise ModelRegistryError("Promotion requires the explicit --confirm flag.")
    version_number = _require_numeric_version(model_version)
    if not alias.strip():
        raise ModelRegistryError("The model alias cannot be empty.")

    inspected = inspect_version(config, model_name, version_number)
    try:
        client = _configure_mlflow(config)
        client.set_model_version_tag(
            model_name, version_number, "validation_status", "validated"
        )
        client.set_registered_model_alias(model_name, alias, version_number)
    except ModelRegistryError:
        raise
    except Exception as exc:
        raise ModelRegistryError(
            "Promotion failed. Verify write access to the registered model."
        ) from exc

    return AliasResolution(
        model_name=model_name,
        alias=alias,
        model_version=version_number,
        model_uri=inspected.model_uri,
        mlflow_run_id=inspected.mlflow_run_id,
        source_commit_sha=inspected.source_commit_sha,
        dataset_sha256=inspected.dataset_sha256,
        pipeline_sha256=inspected.pipeline_sha256,
        validation_status="validated",
    )


def verify_alias_and_numeric(
    config: MlflowConfig, model_name: str, alias: str
) -> dict[str, Any]:
    """Load the mutable and immutable references with the deployment sklearn flavor."""
    resolution = resolve_alias(config, model_name, alias)
    alias_uri = f"models:/{model_name}@{alias}"
    try:
        import mlflow

        _configure_mlflow(config)
        alias_model = mlflow.sklearn.load_model(alias_uri)
        numeric_model = mlflow.sklearn.load_model(resolution.model_uri)
        from src.mlops.tracking import MODEL_INPUT_EXAMPLE

        alias_prediction = np.asarray(alias_model.predict(MODEL_INPUT_EXAMPLE))
        numeric_prediction = np.asarray(numeric_model.predict(MODEL_INPUT_EXAMPLE))
        if not (
            alias_prediction.size
            and numeric_prediction.size
            and np.isfinite(alias_prediction).all()
            and np.isfinite(numeric_prediction).all()
            and np.allclose(alias_prediction, numeric_prediction)
        ):
            raise ModelRegistryError(
                "Alias and numeric model predictions are not compatible."
            )
        client = _configure_mlflow(config)
        alias_version = client.get_model_version_by_alias(model_name, alias)
        numeric_version = client.get_model_version(
            model_name, resolution.model_version
        )
        if alias_version.run_id != numeric_version.run_id:
            raise ModelRegistryError(
                "Alias and numeric model references do not identify the same MLflow run."
            )
    except ModelRegistryError:
        raise
    except Exception as exc:
        raise ModelRegistryError(
            "Model loading verification failed. Verify registry and artifact access."
        ) from exc

    return {
        **resolution.as_dict(),
        "alias_model_uri": alias_uri,
        "predictions_compatible": True,
        "alias_prediction": float(alias_prediction[0]),
        "numeric_prediction": float(numeric_prediction[0]),
    }


def _configure_mlflow(config: MlflowConfig):
    if not config.enabled:
        raise ModelRegistryError(
            "Registry commands require ENABLE_MLFLOW_TRACKING=true."
        )
    config.validate()
    import mlflow

    if config.tracking_uri and config.tracking_uri.startswith("file:"):
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(config.tracking_uri)
    return mlflow.MlflowClient(
        tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri
    )


def _checksum_registered_pipeline(
    *, config: MlflowConfig, model_name: str, model_version: str
) -> str:
    import mlflow

    exact_uri = f"models:/{model_name}/{model_version}"
    with tempfile.TemporaryDirectory(prefix="registered-model-") as destination:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=exact_uri,
                dst_path=destination,
            )
        )
        model_metadata = mlflow.models.Model.load(str(downloaded))
        sklearn_flavor = model_metadata.flavors.get("sklearn", {})
        serialized_name = sklearn_flavor.get("pickled_model")
        if not serialized_name:
            raise ModelRegistryError(
                "The registered artifact does not declare a serialized sklearn pipeline."
            )
        serialized_pipeline = (downloaded / serialized_name).resolve()
        if downloaded.resolve() not in serialized_pipeline.parents:
            raise ModelRegistryError(
                "The registered artifact declares an unsafe sklearn pipeline path."
            )
        if not serialized_pipeline.is_file():
            raise ModelRegistryError(
                "The registered sklearn pipeline artifact could not be found."
            )
        checksum = _sha256_file(serialized_pipeline)
    _require_sha256(checksum, "pipeline_sha256")
    return checksum


def _validated_version(model_name: str, version: Any) -> RegistryVersion:
    version_number = _require_numeric_version(str(version.version))
    tags = dict(version.tags or {})
    missing = [name for name in REQUIRED_VERSION_TAGS if not tags.get(name)]
    if missing:
        raise ModelRegistryError(
            "The registered model version is missing required lineage tags: "
            + ", ".join(missing)
            + "."
        )
    _require_sha256(tags["dataset_sha256"], "dataset_sha256")
    _require_sha256(tags["pipeline_sha256"], "pipeline_sha256")
    if not SOURCE_COMMIT_PATTERN.fullmatch(tags["source_commit_sha"]):
        raise ModelRegistryError(
            "The source_commit_sha value must be a full Git commit SHA."
        )
    if version.run_id != tags["training_run_id"]:
        raise ModelRegistryError(
            "The registered model run ID does not match its training_run_id tag."
        )
    return RegistryVersion(
        model_name=model_name,
        model_version=version_number,
        model_uri=f"models:/{model_name}/{version_number}",
        mlflow_run_id=tags["training_run_id"],
        source_commit_sha=tags["source_commit_sha"],
        dataset_sha256=tags["dataset_sha256"],
        pipeline_sha256=tags["pipeline_sha256"],
        validation_status=tags["validation_status"],
        feature_schema_version=tags["feature_schema_version"],
        prediction_contract_version=tags["prediction_contract_version"],
        selected_model=tags["selected_model"],
        selection_metric=tags["selection_metric"],
    )


def _require_numeric_version(version: str) -> str:
    if not NUMERIC_VERSION_PATTERN.fullmatch(version):
        raise ModelRegistryError("The model version must be an exact positive integer.")
    return version


def _require_sha256(value: str, name: str) -> None:
    if not PIPELINE_SHA256_PATTERN.fullmatch(value):
        raise ModelRegistryError(
            f"The {name} value must be 64 lowercase hexadecimal characters."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

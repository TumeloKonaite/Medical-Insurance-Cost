from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.exceptions import DeploymentError
from src.mlops.config import MlflowConfig
from src.mlops.deployment import prepare_deployment
from src.mlops.runtime import (
    REQUIRED_METADATA_FIELDS,
    clear_runtime_cache,
    parse_exact_model_uri,
    validate_production_startup,
    validate_local_package,
)
from src.mlops.tracking import INPUT_COLUMN_TYPES, MODEL_INPUT_EXAMPLE
from src.model_contract import FEATURE_COLUMNS
from src.repositories.artifact_repository import PackagedMlflowRepository


@pytest.fixture
def deployment_package(tmp_path):
    package = tmp_path / "package"
    model_dir = package / "model"
    frame = pd.concat([MODEL_INPUT_EXAMPLE] * 2, ignore_index=True)
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", ["age", "bmi", "children"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["sex", "smoker", "region"],
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", DummyRegressor(strategy="constant", constant=12345.67)),
        ]
    ).fit(frame, np.array([10000.0, 20000.0]))
    signature = ModelSignature(
        inputs=Schema(
            [ColSpec(INPUT_COLUMN_TYPES[name], name) for name in FEATURE_COLUMNS]
        ),
        outputs=Schema([ColSpec("double")]),
    )
    mlflow.sklearn.save_model(
        pipeline,
        model_dir,
        signature=signature,
        input_example=MODEL_INPUT_EXAMPLE,
        pip_requirements=[],
    )
    mlflow_metadata = mlflow.models.Model.load(str(model_dir))
    serialized = model_dir / mlflow_metadata.flavors["sklearn"]["pickled_model"]
    pipeline_sha256 = hashlib.sha256(serialized.read_bytes()).hexdigest()
    metadata = {
        "deployment_id": str(uuid.uuid4()),
        "deployment_timestamp_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "environment": "production",
        "modal_application": "medical-insurance-cost",
        "model_name": "medical-insurance-cost",
        "model_version": "7",
        "model_uri": "models:/medical-insurance-cost/7",
        "mlflow_run_id": "run-123",
        "source_commit_sha": "a" * 40,
        "dataset_sha256": "b" * 64,
        "pipeline_sha256": pipeline_sha256,
        "feature_schema_version": "1",
        "prediction_contract_version": "1",
        "validation_status": "validated",
    }
    (package / "deployment_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    clear_runtime_cache()
    yield package, metadata
    clear_runtime_cache()


@pytest.mark.parametrize(
    "model_uri",
    [
        "models:/medical-insurance-cost@champion",
        "models:/medical-insurance-cost/latest",
        "models:/medical-insurance-cost/Production",
        "models:/medical-insurance-cost/0",
        "models:/medical-insurance-cost/-1",
        "models:/medical-insurance-cost/1.0",
        "models:/another-model/7",
        "runs:/run-id/model",
        "/tmp/model",
        "./model",
    ],
)
def test_only_exact_positive_numeric_model_uri_is_accepted(model_uri):
    with pytest.raises(DeploymentError, match="exact numeric model URI"):
        parse_exact_model_uri(model_uri)


def test_complete_package_validates_without_registry_or_network(
    deployment_package, monkeypatch
):
    package, metadata = deployment_package

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("runtime attempted registry or artifact network access")

    monkeypatch.setattr(mlflow, "MlflowClient", network_forbidden)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", network_forbidden)

    validated = validate_local_package(package)
    assert validated.metadata == metadata
    assert validated.model.predict(MODEL_INPUT_EXAMPLE)[0] == pytest.approx(12345.67)


def test_production_provider_reuses_model_loaded_at_startup(
    deployment_package, monkeypatch
):
    package, _ = deployment_package
    original_loader = mlflow.sklearn.load_model
    loads = 0

    def recording_loader(model_uri, *args, **kwargs):
        nonlocal loads
        loads += 1
        return original_loader(model_uri, *args, **kwargs)

    monkeypatch.setattr(mlflow.sklearn, "load_model", recording_loader)
    validate_production_startup(package)
    repository = PackagedMlflowRepository(package)

    assert repository.load_model() is repository.load_model()
    assert loads == 1


def test_metadata_has_exact_immutable_identity_and_no_credentials(deployment_package):
    package, metadata = deployment_package
    stored = json.loads((package / "deployment_metadata.json").read_text())

    assert set(stored) == REQUIRED_METADATA_FIELDS
    assert stored == metadata
    assert stored["model_uri"] == "models:/medical-insurance-cost/7"
    serialized = json.dumps(stored).lower()
    for prohibited in ("password", "token", "credential", "dagshub"):
        assert prohibited not in serialized


def test_startup_validation_detects_pipeline_tampering(deployment_package):
    package, _ = deployment_package
    model_metadata = mlflow.models.Model.load(str(package / "model"))
    pipeline = package / "model" / model_metadata.flavors["sklearn"]["pickled_model"]
    pipeline.write_bytes(pipeline.read_bytes() + b"tampered")

    with pytest.raises(DeploymentError, match="checksum"):
        validate_local_package(package)


def test_checksum_mismatch_blocks_package(deployment_package):
    package, metadata = deployment_package
    metadata["pipeline_sha256"] = "0" * 64
    (package / "deployment_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(DeploymentError, match="checksum"):
        validate_local_package(package)


@pytest.mark.parametrize(
    ("field", "message"),
    [("signature", "signature"), ("saved_input_example_info", "input example")],
)
def test_signature_and_input_example_are_required(deployment_package, field, message):
    package, _ = deployment_package
    mlmodel_path = package / "model" / "MLmodel"
    import yaml

    mlmodel = yaml.safe_load(mlmodel_path.read_text(encoding="utf-8"))
    mlmodel.pop(field, None)
    mlmodel_path.write_text(yaml.safe_dump(mlmodel), encoding="utf-8")

    with pytest.raises(DeploymentError, match=message):
        validate_local_package(package)


def test_non_finite_smoke_prediction_blocks_package(deployment_package, monkeypatch):
    package, _ = deployment_package

    class NonFiniteModel:
        @staticmethod
        def predict(_features):
            return np.array([np.inf])

    monkeypatch.setattr(
        "src.mlops.runtime._load_sklearn_model", lambda _model_dir: NonFiniteModel()
    )
    with pytest.raises(DeploymentError, match="not finite"):
        validate_local_package(package)


def test_stale_output_directory_blocks_packaging_before_registry_access(tmp_path):
    output = tmp_path / "model"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(DeploymentError, match="not empty"):
        prepare_deployment(
            config=MlflowConfig(
                enabled=True,
                tracking_uri=tmp_path.as_uri(),
                registered_model_name="medical-insurance-cost",
            ),
            model_uri="models:/medical-insurance-cost/7",
            output_dir=output,
            expected_run_id="run-123",
            expected_pipeline_sha256="a" * 64,
        )


def test_missing_required_metadata_blocks_startup(deployment_package):
    package, metadata = deployment_package
    del metadata["source_commit_sha"]
    (package / "deployment_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(DeploymentError, match="missing: source_commit_sha"):
        validate_local_package(package)


def test_package_fixture_is_isolated(deployment_package, tmp_path):
    """Guard against accidental mutation shared between parameterized tests."""
    package, _ = deployment_package
    copied = tmp_path / "copy"
    shutil.copytree(package, copied)
    assert validate_local_package(copied).metadata["model_version"] == "7"

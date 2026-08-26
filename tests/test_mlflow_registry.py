from pathlib import Path

import pandas as pd
import pytest
import mlflow

from src.exceptions import ModelRegistryError
from src.mlops.config import MlflowConfig
from src.mlops.deployment import prepare_deployment
from src.mlops.registry import (
    PIPELINE_SHA256_PATTERN,
    inspect_version,
    promote_version,
    resolve_alias,
    verify_alias_and_numeric,
)
from src.mlops.tracking import TrackingContext
from src.repositories.artifact_repository import LocalArtifactRepository
from src.training.model_trainer import ModelTrainer


def test_selected_model_registration_promotion_and_resolution(tmp_path):
    data_path = Path(__file__).resolve().parents[1] / "Data" / "medical_insurance.csv"
    data = pd.read_csv(data_path)
    train_data = data.iloc[:500].copy()
    test_data = data.iloc[500:650].copy()
    config = MlflowConfig(
        enabled=True,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        experiment_name="registry-test",
        enable_model_registration=True,
        registered_model_name="medical-insurance-cost",
    )
    context = TrackingContext.from_dataset(
        data_path,
        dataset_row_count=len(data),
        test_split_ratio=0.2,
        random_seed=42,
    )

    result = ModelTrainer(
        LocalArtifactRepository(tmp_path / "model.pkl"),
        tracking_config=config,
        tracking_context=context,
    ).run(train_data, test_data)

    assert result.tracking is not None
    assert result.tracking.model_version == "1"
    assert result.tracking.model_uri.startswith("models:/m-")
    assert PIPELINE_SHA256_PATTERN.fullmatch(result.tracking.pipeline_sha256 or "")

    candidate = inspect_version(config, "medical-insurance-cost", "1")
    assert candidate.validation_status == "candidate"
    assert candidate.mlflow_run_id == result.tracking.run_id
    assert candidate.dataset_sha256 == context.dataset_sha256
    assert candidate.pipeline_sha256 == result.tracking.pipeline_sha256

    client = mlflow.MlflowClient(
        tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri
    )
    client.set_registered_model_alias("medical-insurance-cost", "champion", "1")
    with pytest.raises(ModelRegistryError, match="not validated"):
        resolve_alias(config, "medical-insurance-cost", "champion")

    promoted = promote_version(
        config,
        "medical-insurance-cost",
        "1",
        "champion",
        confirmed=True,
    )
    resolution = resolve_alias(config, "medical-insurance-cost", "champion")

    assert promoted.validation_status == "validated"
    assert resolution.model_version == "1"
    assert resolution.model_uri == "models:/medical-insurance-cost/1"
    assert "@champion" not in resolution.model_uri
    assert resolution.pipeline_sha256 == candidate.pipeline_sha256

    verification = verify_alias_and_numeric(
        config, "medical-insurance-cost", "champion"
    )
    assert verification["predictions_compatible"] is True
    assert verification["mlflow_run_id"] == result.tracking.run_id

    package_dir = tmp_path / "build" / "model"
    deployment = prepare_deployment(
        config=config,
        model_uri=resolution.model_uri,
        output_dir=package_dir,
        expected_run_id=resolution.mlflow_run_id,
        expected_pipeline_sha256=resolution.pipeline_sha256,
    )
    assert deployment.model_version == "1"
    assert deployment.model_uri == "models:/medical-insurance-cost/1"
    assert deployment.mlflow_run_id == result.tracking.run_id
    assert deployment.pipeline_sha256 == resolution.pipeline_sha256
    assert (package_dir / "model" / "MLmodel").is_file()
    assert (package_dir / "deployment_metadata.json").is_file()


def test_promotion_requires_explicit_confirmation(tmp_path):
    config = MlflowConfig(enabled=True, tracking_uri=tmp_path.as_uri())

    with pytest.raises(ModelRegistryError, match="--confirm"):
        promote_version(
            config,
            "medical-insurance-cost",
            "1",
            "champion",
            confirmed=False,
        )


@pytest.mark.parametrize("version", ["latest", "champion", "1.0", "0", "-1"])
def test_registry_rejects_non_numeric_exact_versions(tmp_path, version):
    config = MlflowConfig(enabled=True, tracking_uri=tmp_path.as_uri())

    with pytest.raises(ModelRegistryError, match="positive integer"):
        inspect_version(config, "medical-insurance-cost", version)

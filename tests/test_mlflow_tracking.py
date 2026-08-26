from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.exceptions import MlflowConfigurationError
from src.mlops.config import MlflowConfig
from src.mlops.tracking import MODEL_INPUT_EXAMPLE, TrackingContext
from src.model_contract import FEATURE_COLUMNS
from src.repositories.artifact_repository import LocalArtifactRepository
from src.training.model_trainer import ModelTrainer


def _training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(__file__).resolve().parents[1] / "Data" / "medical_insurance.csv"
    data = pd.read_csv(data_path)
    return data.iloc[:500].copy(), data.iloc[500:650].copy()


def test_enabled_config_defaults_to_local_file_store(tmp_path):
    config = MlflowConfig.from_env(
        {"ENABLE_MLFLOW_TRACKING": "true"}, working_directory=tmp_path
    )

    assert config.enabled is True
    assert config.tracking_uri == (tmp_path / "mlruns").as_uri()
    assert config.tracking_backend == "local"
    assert config.enable_model_registration is False


def test_dagshub_config_uses_standard_mlflow_credentials():
    config = MlflowConfig.from_env(
        {
            "ENABLE_MLFLOW_TRACKING": "true",
            "ENABLE_MODEL_REGISTRATION": "true",
            "MLFLOW_TRACKING_URI": "https://dagshub.com/example/project.mlflow",
            "MLFLOW_TRACKING_USERNAME": "example",
            "MLFLOW_TRACKING_PASSWORD": "secret-placeholder",
        }
    )

    assert config.tracking_backend == "dagshub"
    assert config.enable_model_registration is True
    assert "secret-placeholder" not in repr(config)


@pytest.mark.parametrize(
    "environment",
    [
        {
            "ENABLE_MLFLOW_TRACKING": "true",
            "MLFLOW_TRACKING_USERNAME": "example",
            "MLFLOW_TRACKING_PASSWORD": "secret-placeholder",
        },
        {
            "ENABLE_MLFLOW_TRACKING": "true",
            "MLFLOW_TRACKING_URI": "https://example:secret@dagshub.com/a/b.mlflow",
            "MLFLOW_TRACKING_USERNAME": "example",
            "MLFLOW_TRACKING_PASSWORD": "secret-placeholder",
        },
        {
            "ENABLE_MLFLOW_TRACKING": "true",
            "MLFLOW_TRACKING_URI": "https://dagshub.com/a/b.mlflow",
            "MLFLOW_TRACKING_USERNAME": "example",
        },
    ],
)
def test_remote_config_rejects_unsafe_or_incomplete_settings(environment):
    with pytest.raises(MlflowConfigurationError) as error:
        MlflowConfig.from_env(environment)

    assert "secret-placeholder" not in str(error.value)


def test_disabled_tracking_creates_no_mlflow_store(tmp_path):
    train_data, test_data = _training_data()
    store = tmp_path / "disabled-mlruns"
    config = MlflowConfig(enabled=False, tracking_uri=store.as_uri())

    ModelTrainer(
        LocalArtifactRepository(tmp_path / "model.pkl"), tracking_config=config
    ).run(train_data, test_data)

    assert not store.exists()


def test_enabled_tracking_logs_reloadable_raw_feature_model(tmp_path):
    import mlflow

    train_data, test_data = _training_data()
    store = tmp_path / "mlruns"
    experiment_name = "pipeline-test"
    config = MlflowConfig(
        enabled=True,
        tracking_uri=store.as_uri(),
        experiment_name=experiment_name,
        enable_model_registration=False,
    )

    data_path = Path(__file__).resolve().parents[1] / "Data" / "medical_insurance.csv"
    context = TrackingContext.from_dataset(
        data_path,
        dataset_row_count=len(train_data) + len(test_data),
        test_split_ratio=len(test_data) / (len(train_data) + len(test_data)),
    )
    ModelTrainer(
        LocalArtifactRepository(tmp_path / "model.pkl"),
        tracking_config=config,
        tracking_context=context,
    ).run(train_data, test_data)

    mlflow.set_tracking_uri(store.as_uri())
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run_id = runs.iloc[0]["run_id"]
    run = mlflow.get_run(run_id)

    assert run.data.params["dataset_name"] == "medical_insurance"
    assert run.data.params["dataset_filename"] == "medical_insurance.csv"
    assert len(run.data.params["dataset_sha256"]) == 64
    assert "canonical_feature_list" in run.data.params
    assert run.data.params["target_name"] == "charges"
    assert run.data.params["selection_metric"] == "r2"
    assert "candidate/random_forest/hyperparameters" in run.data.params
    assert "candidate/random_forest/r2" in run.data.metrics
    assert "selected/rmse" in run.data.metrics
    assert run.data.tags["problem_type"] == "regression"
    assert run.data.tags["tracking_backend"] == "local"
    assert run.data.tags["feature_schema_version"] == "1"
    assert run.data.tags["prediction_contract_version"] == "1"

    model_uri = f"runs:/{run_id}/model"
    logged_model = mlflow.models.Model.load(model_uri)
    assert logged_model.signature is not None
    assert logged_model.saved_input_example_info is not None
    assert [item.name for item in logged_model.signature.inputs.inputs] == list(
        FEATURE_COLUMNS
    )

    pipeline = mlflow.sklearn.load_model(model_uri)
    prediction = float(pipeline.predict(MODEL_INPUT_EXAMPLE)[0])
    assert np.isfinite(prediction)

import sys

from src.exceptions import ApplicationError
from src.mlops.config import MlflowConfig
from src.mlops.tracking import TrackingContext
from src.paths import ARTIFACTS_DIR
from src.repositories.artifact_repository import LocalArtifactRepository
from src.training.data_ingestion import DataIngestion
from src.training.data_transformation import DataTransformation
from src.training.model_trainer import ModelTrainer


def main():
    # Run ingestion, validation, training, artifact storage, and optional tracking.
    repository = LocalArtifactRepository(model_path=ARTIFACTS_DIR / "model.pkl")
    ingestion = DataIngestion()
    train_path, test_path, _ = ingestion.run()
    train_data, test_data = DataTransformation().run(train_path, test_path)
    # Capture dataset details used for MLflow lineage and reproducibility.
    total_rows = len(train_data) + len(test_data)
    context = TrackingContext.from_dataset(
        ingestion.config.source_data_path,
        dataset_row_count=total_rows,
        test_split_ratio=ingestion.config.test_size,
        random_seed=ingestion.config.random_state,
    )
    result = ModelTrainer(
        repository,
        tracking_config=MlflowConfig.from_env(),
        tracking_context=context,
    ).run(train_data, test_data)

    # Print the important artifact and tracking details for the person training it.
    print("Train:", train_path)
    print("Test:", test_path)
    print("Pipeline:", repository.model_path)
    print("Selected model:", result.selected_model_name)
    print("Model score (R2):", result.score)
    if result.tracking:
        print("MLflow run ID:", result.tracking.run_id)
        print("Source model URI:", result.tracking.model_uri)
        print("Git commit SHA:", result.tracking.git_commit_sha)
        print("Dataset SHA-256:", result.tracking.dataset_sha256)
        if result.tracking.model_version:
            print("Registered model:", result.tracking.registered_model_name)
            print("Numeric model version:", result.tracking.model_version)
            print("Pipeline SHA-256:", result.tracking.pipeline_sha256)


if __name__ == "__main__":
    try:
        main()
    except ApplicationError as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

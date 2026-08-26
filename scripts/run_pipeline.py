from src.paths import ARTIFACTS_DIR
from src.repositories.artifact_repository import LocalArtifactRepository
from src.training.data_ingestion import DataIngestion
from src.training.data_transformation import DataTransformation
from src.training.model_trainer import ModelTrainer


def main():
    repository = LocalArtifactRepository(model_path=ARTIFACTS_DIR / "model.pkl")
    train_path, test_path, _ = DataIngestion().run()
    train_data, test_data = DataTransformation().run(train_path, test_path)
    result = ModelTrainer(repository).run(train_data, test_data)

    print("Train:", train_path)
    print("Test:", test_path)
    print("Pipeline:", repository.model_path)
    print("Selected model:", result.selected_model_name)
    print("Model score (R2):", result.score)


if __name__ == "__main__":
    main()

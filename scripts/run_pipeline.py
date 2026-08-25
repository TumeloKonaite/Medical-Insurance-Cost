from src.paths import ARTIFACTS_DIR
from src.repositories.artifact_repository import LocalArtifactRepository
from src.training.data_ingestion import DataIngestion
from src.training.data_transformation import DataTransformation
from src.training.model_trainer import ModelTrainer


def main():
    repository = LocalArtifactRepository(
        model_path=ARTIFACTS_DIR / "model.pkl",
        preprocessor_path=ARTIFACTS_DIR / "preprocessor.pkl",
    )
    train_path, test_path, _ = DataIngestion().run()
    train_data, test_data = DataTransformation(repository).run(train_path, test_path)
    score = ModelTrainer(repository).run(train_data, test_data)

    print("Train:", train_path)
    print("Test:", test_path)
    print("Preprocessor:", repository.preprocessor_path)
    print("Model:", repository.model_path)
    print("Model score (R2):", score)


if __name__ == "__main__":
    main()

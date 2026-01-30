from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    train_path, test_path, _ = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, preprocessor_path = DataTransformation().initiate_data_transformation(
        train_path, test_path
    )
    score = ModelTrainer().initiate_model_trainer(train_arr, test_arr)

    print("Train:", train_path)
    print("Test:", test_path)
    print("Preprocessor:", preprocessor_path)
    print("Model score (R2):", score)


if __name__ == "__main__":
    main()

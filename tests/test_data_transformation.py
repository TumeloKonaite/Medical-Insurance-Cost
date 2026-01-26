import pandas as pd

from src.components.data_transformation import DataTransformation


def _make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [19, 18, 28, 33, 32, 40, 21, 45],
            "sex": ["female", "male", "male", "male", "female", "female", "male", "female"],
            "bmi": [27.9, 33.77, 33.0, 22.705, 28.88, 30.1, 25.3, 31.2],
            "children": [0, 1, 3, 0, 1, 2, 0, 3],
            "smoker": ["yes", "no", "no", "no", "yes", "no", "yes", "no"],
            "region": [
                "southwest",
                "southeast",
                "southeast",
                "northwest",
                "northwest",
                "northeast",
                "southwest",
                "northeast",
            ],
            "charges": [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552, 11090.7178, 2020.5523, 25476.829],
        }
    )


def test_data_transformation_matches_notebook_preprocessing(tmp_path):
    df = _make_dataset()
    train_df = df.iloc[:6].copy()
    test_df = df.iloc[6:].copy()

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    preprocessor_path = tmp_path / "preprocessor.pkl"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    transformer = DataTransformation()
    transformer.data_transformation_config.preprocessor_obj_file_path = str(
        preprocessor_path
    )

    train_arr, test_arr, saved_path = transformer.initiate_data_transformation(
        str(train_path), str(test_path)
    )

    assert saved_path == str(preprocessor_path)
    assert preprocessor_path.exists()

    expected_feature_count = 11
    assert train_arr.shape == (len(train_df), expected_feature_count + 1)
    assert test_arr.shape == (len(test_df), expected_feature_count + 1)

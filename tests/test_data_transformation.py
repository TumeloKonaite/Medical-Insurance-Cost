import pandas as pd

from src.model_contract import FEATURE_COLUMNS, TARGET_COLUMN
from src.training.data_transformation import DataTransformation


def _make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [19, 18, 28, 33],
            "sex": ["female", "male", "male", "female"],
            "bmi": [27.9, 33.77, 33.0, 22.705],
            "children": [0, 1, 3, 0],
            "smoker": ["yes", "no", "no", "no"],
            "region": ["southwest", "southeast", "southeast", "northwest"],
            "charges": [16884.924, 1725.5523, 4449.462, 21984.47061],
        }
    )


def test_data_transformation_returns_canonical_raw_columns(tmp_path):
    dataset = _make_dataset()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    dataset.iloc[:3].to_csv(train_path, index=False)
    dataset.iloc[3:].to_csv(test_path, index=False)

    train_data, test_data = DataTransformation().run(train_path, test_path)

    assert list(train_data.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert list(test_data.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]


def test_preprocessor_excludes_target_and_handles_unknown_categories():
    dataset = _make_dataset()
    preprocessor = DataTransformation._make_preprocessor()
    preprocessor.fit(dataset.loc[:, list(FEATURE_COLUMNS)])

    unknown = dataset.loc[[0], list(FEATURE_COLUMNS)].copy()
    unknown.loc[:, "region"] = "unknown-region"
    transformed = preprocessor.transform(unknown)

    assert TARGET_COLUMN not in preprocessor.feature_names_in_
    assert transformed.shape[0] == 1

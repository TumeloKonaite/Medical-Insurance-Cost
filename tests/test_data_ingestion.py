import pandas as pd

from src.training.data_ingestion import DataIngestion, DataIngestionConfig


def _make_sample_df(rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": list(range(20, 20 + rows)),
            "sex": ["female", "male"] * (rows // 2) + (["female"] if rows % 2 else []),
            "bmi": [25.0 + i * 0.1 for i in range(rows)],
            "children": [i % 3 for i in range(rows)],
            "smoker": ["yes", "no"] * (rows // 2) + (["no"] if rows % 2 else []),
            "region": ["southwest", "southeast", "northwest", "northeast"] * (rows // 4)
            + (["southwest"] * (rows % 4)),
            "charges": [1000.0 + i * 10 for i in range(rows)],
        }
    )


def test_data_ingestion_creates_artifacts(tmp_path):
    source_path = tmp_path / "medical_insurance.csv"
    raw_path = tmp_path / "data.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    df = _make_sample_df(10)
    df.to_csv(source_path, index=False)

    ingestion = DataIngestion(
        DataIngestionConfig(
            source_data_path=source_path,
            raw_data_path=raw_path,
            train_data_path=train_path,
            test_data_path=test_path,
            test_size=0.2,
            random_state=42,
        )
    )

    train_out, test_out, raw_out = ingestion.run()

    assert train_path.exists()
    assert test_path.exists()
    assert raw_path.exists()
    assert train_out == train_path
    assert test_out == test_path
    assert raw_out == raw_path

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    raw_df = pd.read_csv(raw_path)

    assert len(raw_df) == 10
    assert len(train_df) == 8
    assert len(test_df) == 2

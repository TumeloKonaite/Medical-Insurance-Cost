from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import TrainingError
from src.paths import ARTIFACTS_DIR, DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    source_data_path: Path = DATA_DIR / "medical_insurance.csv"
    train_data_path: Path = ARTIFACTS_DIR / "train.csv"
    test_data_path: Path = ARTIFACTS_DIR / "test.csv"
    raw_data_path: Path = ARTIFACTS_DIR / "data.csv"
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def run(self) -> tuple[Path, Path, Path]:
        # Load the source CSV and stop early if it is missing.
        source_path = Path(self.config.source_data_path)
        if not source_path.is_file():
            raise TrainingError("The source dataset could not be found.")

        try:
            data = pd.read_csv(source_path)
            # Use a fixed seed so the train/test split can be reproduced.
            train_data, test_data = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            train_path = Path(self.config.train_data_path)
            test_path = Path(self.config.test_data_path)
            raw_path = Path(self.config.raw_data_path)
            for output_path in (train_path, test_path, raw_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the raw copy and both datasets for the remaining training stages.
            data.to_csv(raw_path, index=False)
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logger.exception("Data ingestion failed")
            raise TrainingError("Data ingestion failed.") from exc

        logger.info("Created train, test, and raw datasets")
        return train_path, test_path, raw_path

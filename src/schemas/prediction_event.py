from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class PredictionEvent:
    id: uuid.UUID
    request_id: uuid.UUID
    created_at: datetime
    source: Literal["web", "json"]
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    predicted_charges: float
    model_version: str
    prediction_contract_version: str
    inference_latency_ms: float
    actual_charges: float | None = None
    actual_recorded_at: datetime | None = None

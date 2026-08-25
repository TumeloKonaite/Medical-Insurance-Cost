from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0)
    sex: Literal["female", "male"]
    bmi: float = Field(gt=0, allow_inf_nan=False)
    children: int = Field(ge=0)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class PredictionResponse(BaseModel):
    predicted_charges: float = Field(allow_inf_nan=False)
    currency: Literal["USD"] = "USD"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"

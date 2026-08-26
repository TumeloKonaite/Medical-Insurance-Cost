from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _read_bool(environment: Mapping[str, str], name: str, default: bool) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "medical-insurance-cost"
    enable_model_registration: bool = False
    registered_model_name: str = "medical-insurance-cost"

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
        working_directory: str | Path | None = None,
    ) -> MlflowConfig:
        env = os.environ if environment is None else environment
        enabled = _read_bool(env, "ENABLE_MLFLOW_TRACKING", False)
        supplied_uri = env.get("MLFLOW_TRACKING_URI", "").strip()
        tracking_uri: str | None = supplied_uri or None
        if enabled and tracking_uri is None:
            root = Path(working_directory or Path.cwd()).resolve()
            tracking_uri = (root / "mlruns").as_uri()

        return cls(
            enabled=enabled,
            tracking_uri=tracking_uri,
            experiment_name=env.get(
                "MLFLOW_EXPERIMENT_NAME", "medical-insurance-cost"
            ),
            enable_model_registration=_read_bool(
                env, "ENABLE_MODEL_REGISTRATION", False
            ),
            registered_model_name=env.get(
                "MLFLOW_REGISTERED_MODEL_NAME", "medical-insurance-cost"
            ),
        )

    @property
    def tracking_backend(self) -> str:
        uri = self.tracking_uri or ""
        if uri.startswith(("file:", "sqlite:")) or "://" not in uri:
            return "local"
        return "remote"


# Preserve MLflow's conventional capitalization for callers that prefer it.
MLflowConfig = MlflowConfig

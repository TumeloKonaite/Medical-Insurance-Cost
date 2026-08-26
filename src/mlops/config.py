from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qsl, urlsplit

from src.exceptions import MlflowConfigurationError


def _read_bool(environment: Mapping[str, str], name: str, default: bool) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise MlflowConfigurationError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0."
    )


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
        credentials_supplied = bool(
            env.get("MLFLOW_TRACKING_USERNAME", "").strip()
            or env.get("MLFLOW_TRACKING_PASSWORD", "").strip()
        )
        if enabled and tracking_uri is None and credentials_supplied:
            raise MlflowConfigurationError(
                "MLFLOW_TRACKING_URI is required when remote MLflow credentials "
                "are configured; refusing to fall back to local tracking."
            )
        if enabled and tracking_uri is None:
            root = Path(working_directory or Path.cwd()).resolve()
            tracking_uri = (root / "mlruns").as_uri()

        config = cls(
            enabled=enabled,
            tracking_uri=tracking_uri,
            experiment_name=env.get("MLFLOW_EXPERIMENT_NAME", "medical-insurance-cost").strip(),
            enable_model_registration=_read_bool(
                env, "ENABLE_MODEL_REGISTRATION", False
            ),
            registered_model_name=env.get(
                "MLFLOW_REGISTERED_MODEL_NAME", "medical-insurance-cost"
            ).strip(),
        )
        config.validate(
            username_configured=bool(
                env.get("MLFLOW_TRACKING_USERNAME", "").strip()
            ),
            password_configured=bool(
                env.get("MLFLOW_TRACKING_PASSWORD", "").strip()
            ),
        )
        return config

    def validate(
        self,
        *,
        username_configured: bool | None = None,
        password_configured: bool | None = None,
    ) -> None:
        if self.tracking_uri:
            _reject_secret_bearing_uri(self.tracking_uri)
        if self.enable_model_registration and not self.enabled:
            raise MlflowConfigurationError(
                "ENABLE_MODEL_REGISTRATION requires ENABLE_MLFLOW_TRACKING=true."
            )
        if not self.enabled:
            return
        if not self.tracking_uri:
            raise MlflowConfigurationError(
                "MLFLOW_TRACKING_URI is required when MLflow tracking is enabled."
            )
        if not self.experiment_name:
            raise MlflowConfigurationError("MLFLOW_EXPERIMENT_NAME cannot be empty.")
        if self.enable_model_registration and not self.registered_model_name:
            raise MlflowConfigurationError(
                "MLFLOW_REGISTERED_MODEL_NAME cannot be empty when registration is enabled."
            )
        if self.tracking_backend == "dagshub" and username_configured is not None:
            if not username_configured or not password_configured:
                raise MlflowConfigurationError(
                    "DagsHub tracking requires both MLFLOW_TRACKING_USERNAME and "
                    "MLFLOW_TRACKING_PASSWORD."
                )

    @property
    def tracking_backend(self) -> str:
        uri = self.tracking_uri or ""
        if uri.startswith(("file:", "sqlite:")) or "://" not in uri:
            return "local"
        if (urlsplit(uri).hostname or "").lower() == "dagshub.com":
            return "dagshub"
        return "remote"

    @property
    def is_remote(self) -> bool:
        return self.tracking_backend != "local"


def _reject_secret_bearing_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    secret_markers = (
        "token",
        "password",
        "secret",
        "credential",
        "access_key",
        "api_key",
    )
    if parsed.username or parsed.password or any(
        any(marker in key.lower() for marker in secret_markers)
        for key, _ in parse_qsl(parsed.query)
    ):
        raise MlflowConfigurationError(
            "MLFLOW_TRACKING_URI must not contain credentials or secret query parameters; "
            "use the standard MLflow username and password environment variables."
        )


# Preserve MLflow's conventional capitalization for callers that prefer it.
MLflowConfig = MlflowConfig

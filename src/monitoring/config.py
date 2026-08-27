from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_MODEL_NAME = "medical-insurance-cost"


class MonitoringConfigurationError(ValueError):
    """A sanitized monitoring configuration error."""


def _positive_integer(
    environment: Mapping[str, str], name: str, default: int, maximum: int
) -> int:
    raw = environment.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise MonitoringConfigurationError(
            f"{name} must be a positive integer."
        ) from exc
    if value < 1 or value > maximum:
        raise MonitoringConfigurationError(
            f"{name} must be between 1 and {maximum}."
        )
    return value


@dataclass(frozen=True, repr=False)
class ArizeExportConfig:
    api_key: str
    space_id: str
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 500
    stale_claim_minutes: int = 30
    retry_base_seconds: int = 60
    retry_max_seconds: int = 3600

    def __repr__(self) -> str:
        return (
            "ArizeExportConfig(api_key=<redacted>, space_id=<redacted>, "
            f"model_name={self.model_name!r}, batch_size={self.batch_size}, "
            f"stale_claim_minutes={self.stale_claim_minutes}, "
            f"retry_base_seconds={self.retry_base_seconds}, "
            f"retry_max_seconds={self.retry_max_seconds})"
        )

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> ArizeExportConfig:
        values = os.environ if environment is None else environment
        api_key = values.get("ARIZE_API_KEY", "").strip()
        space_id = values.get("ARIZE_SPACE_ID", "").strip()
        missing = [
            name
            for name, value in (
                ("ARIZE_API_KEY", api_key),
                ("ARIZE_SPACE_ID", space_id),
            )
            if not value
        ]
        if missing:
            raise MonitoringConfigurationError(
                "Missing required monitoring setting(s): " + ", ".join(missing)
            )

        model_name = values.get("ARIZE_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
        if not model_name:
            raise MonitoringConfigurationError("ARIZE_MODEL_NAME must not be empty.")

        batch_size = _positive_integer(
            values, "ARIZE_EXPORT_BATCH_SIZE", 500, 10_000
        )
        stale_claim_minutes = _positive_integer(
            values, "ARIZE_CLAIM_STALE_MINUTES", 30, 24 * 60
        )
        retry_base_seconds = _positive_integer(
            values, "ARIZE_RETRY_BASE_SECONDS", 60, 24 * 60 * 60
        )
        retry_max_seconds = _positive_integer(
            values, "ARIZE_RETRY_MAX_SECONDS", 3600, 7 * 24 * 60 * 60
        )
        if retry_max_seconds < retry_base_seconds:
            raise MonitoringConfigurationError(
                "ARIZE_RETRY_MAX_SECONDS must be at least ARIZE_RETRY_BASE_SECONDS."
            )
        return cls(
            api_key=api_key,
            space_id=space_id,
            model_name=model_name,
            batch_size=batch_size,
            stale_claim_minutes=stale_claim_minutes,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )

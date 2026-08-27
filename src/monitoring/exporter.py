from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from itertools import groupby
from time import perf_counter

from src.database import create_database_engine
from src.monitoring.arize_client import (
    ArizeBatchClient,
    ArizeUploadError,
    BatchClient,
    actual_dataframe,
    prediction_dataframe,
)
from src.monitoring.config import (
    ArizeExportConfig,
    MonitoringConfigurationError,
)
from src.monitoring.outbox import ExportRecord, OutboxRepository

logger = logging.getLogger(__name__)
NUMERIC_MODEL_VERSION = re.compile(r"^[1-9][0-9]*$")


@dataclass(frozen=True)
class ExportSummary:
    records_claimed: int
    records_sent: int
    records_retried: int
    records_failed: int
    remaining_backlog: int
    oldest_pending_age_seconds: int | None

    def as_dict(self) -> dict[str, int | None]:
        return asdict(self)


class ExportRunFailed(RuntimeError):
    def __init__(self, summary: ExportSummary):
        super().__init__("One or more Arize batches failed.")
        self.summary = summary


class ExportInfrastructureError(RuntimeError):
    """A sanitized exporter infrastructure failure."""


class ArizeExporter:
    def __init__(
        self,
        *,
        config: ArizeExportConfig,
        repository: OutboxRepository,
        client: BatchClient,
    ):
        self._config = config
        self._repository = repository
        self._client = client

    def run(self, *, now: datetime | None = None) -> ExportSummary:
        started = now or datetime.now(timezone.utc)
        records = self._repository.claim(
            limit=self._config.batch_size,
            now=started,
            stale_after=timedelta(minutes=self._config.stale_claim_minutes),
        )
        sent = 0
        failed = 0
        retried = sum(record.attempt_count > 1 for record in records)
        def group_key(record: ExportRecord) -> tuple[str, str]:
            return record.model_version, record.event_type

        for (model_version, event_type), grouped in groupby(
            sorted(records, key=group_key), key=group_key
        ):
            batch = list(grouped)
            timer = perf_counter()
            status_code = 0
            try:
                if NUMERIC_MODEL_VERSION.fullmatch(model_version) is None:
                    raise ArizeUploadError(
                        "The persisted model version is not numeric."
                    )
                dataframe = (
                    prediction_dataframe(batch)
                    if event_type == "prediction"
                    else actual_dataframe(batch)
                )
                status_code = self._client.upload(
                    dataframe,
                    event_type=event_type,
                    model_version=model_version,
                    environment="production",
                )
                self._repository.mark_sent(
                    [record.outbox_id for record in batch],
                    sent_at=datetime.now(timezone.utc),
                )
                sent += len(batch)
                outcome = "sent"
            except Exception as exc:
                if isinstance(exc, ArizeUploadError):
                    status_code = exc.status_code
                self._repository.reschedule_failed(
                    batch,
                    failed_at=datetime.now(timezone.utc),
                    base_seconds=self._config.retry_base_seconds,
                    maximum_seconds=self._config.retry_max_seconds,
                )
                failed += len(batch)
                outcome = "failed"
            logger.info(
                "arize_batch batch_size=%d model_version=%s event_type=%s "
                "outcome=%s status_code=%d attempt_count=%d elapsed_ms=%.2f",
                len(batch),
                model_version,
                event_type,
                outcome,
                status_code,
                max(record.attempt_count for record in batch),
                (perf_counter() - timer) * 1000,
            )

        backlog = self._repository.backlog(now=datetime.now(timezone.utc))
        summary = ExportSummary(
            records_claimed=len(records),
            records_sent=sent,
            records_retried=retried,
            records_failed=failed,
            remaining_backlog=backlog.remaining,
            oldest_pending_age_seconds=backlog.oldest_pending_age_seconds,
        )
        logger.info("arize_export_summary %s", summary.as_dict())
        if failed:
            raise ExportRunFailed(summary)
        return summary


def run_exporter_from_env() -> dict[str, int | None]:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise MonitoringConfigurationError(
            "DATABASE_URL is required for the Arize exporter."
        )
    config = ArizeExportConfig.from_environment()
    try:
        engine = create_database_engine(database_url)
        exporter = ArizeExporter(
            config=config,
            repository=OutboxRepository(engine),
            client=ArizeBatchClient(config),
        )
        return exporter.run().as_dict()
    except ExportRunFailed:
        raise
    except Exception:
        raise ExportInfrastructureError(
            "The Arize exporter encountered an infrastructure failure."
        ) from None

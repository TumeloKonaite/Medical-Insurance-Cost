from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import sqlalchemy as sa
from sqlalchemy import Engine

from src.database import create_database_engine
from src.models.prediction_event import arize_export_events, prediction_events


class GroundTruthError(ValueError):
    """A sanitized delayed-ground-truth validation error."""


def parse_actual_charges(value: str | float | Decimal) -> Decimal:
    try:
        actual = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise GroundTruthError(
            "Actual charges must be a finite non-negative number."
        ) from exc
    if not actual.is_finite() or actual < 0:
        raise GroundTruthError(
            "Actual charges must be a finite non-negative number."
        )
    return actual


def record_actual(
    engine: Engine,
    *,
    request_id: uuid.UUID,
    actual_charges: str | float | Decimal,
    recorded_at: datetime | None = None,
) -> str:
    actual = parse_actual_charges(actual_charges)
    timestamp = recorded_at or datetime.now(timezone.utc)
    with engine.begin() as connection:
        event = connection.execute(
            sa.select(
                prediction_events.c.id,
                prediction_events.c.actual_charges,
            )
            .where(prediction_events.c.request_id == request_id)
            .with_for_update()
        ).mappings().one_or_none()
        if event is None:
            raise GroundTruthError("No prediction exists for that request ID.")

        existing = event["actual_charges"]
        if existing is not None and Decimal(existing) != actual:
            raise GroundTruthError(
                "A different actual value has already been recorded."
            )
        changed = existing is None
        if changed:
            connection.execute(
                sa.update(prediction_events)
                .where(prediction_events.c.id == event["id"])
                .values(actual_charges=actual, actual_recorded_at=timestamp)
            )

        outbox_exists = connection.scalar(
            sa.select(sa.literal(True)).where(
                arize_export_events.c.prediction_event_id == event["id"],
                arize_export_events.c.event_type == "actual",
            )
        )
        if not outbox_exists:
            connection.execute(
                sa.insert(arize_export_events).values(
                    id=uuid.uuid4(),
                    prediction_event_id=event["id"],
                    event_type="actual",
                    status="pending",
                )
            )
    return "recorded" if changed else "unchanged"


def record_actual_from_env(
    *, request_id_text: str, actual_charges_text: str
) -> dict[str, str]:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise GroundTruthError("DATABASE_URL is required to record actual charges.")
    try:
        request_id = uuid.UUID(request_id_text)
    except ValueError as exc:
        raise GroundTruthError("Request ID must be a valid UUID.") from exc
    try:
        result = record_actual(
            create_database_engine(database_url),
            request_id=request_id,
            actual_charges=actual_charges_text,
        )
    except GroundTruthError:
        raise
    except Exception:
        raise GroundTruthError("Ground-truth persistence failed.") from None
    return {"request_id": str(request_id), "status": result}

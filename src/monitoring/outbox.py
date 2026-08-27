from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy import Engine

from src.models.prediction_event import arize_export_events, prediction_events


@dataclass(frozen=True)
class ExportRecord:
    outbox_id: uuid.UUID
    prediction_event_id: uuid.UUID
    event_type: str
    attempt_count: int
    request_id: uuid.UUID
    created_at: datetime
    source: str
    age: int
    sex: str
    bmi: Decimal
    children: int
    smoker: str
    region: str
    predicted_charges: Decimal
    actual_charges: Decimal | None
    model_version: str
    prediction_contract_version: str
    inference_latency_ms: Decimal


@dataclass(frozen=True)
class Backlog:
    remaining: int
    oldest_pending_age_seconds: int | None


class OutboxRepository:
    def __init__(self, engine: Engine):
        self._engine = engine

    def claim(
        self,
        *,
        limit: int,
        now: datetime,
        stale_after: timedelta,
    ) -> list[ExportRecord]:
        stale_before = now - stale_after
        with self._engine.begin() as connection:
            connection.execute(
                sa.update(arize_export_events)
                .where(
                    arize_export_events.c.status == "processing",
                    sa.or_(
                        arize_export_events.c.claimed_at.is_(None),
                        arize_export_events.c.claimed_at < stale_before,
                    ),
                )
                .values(
                    status="pending", claimed_at=None, next_attempt_at=now
                )
            )
            eligible = sa.and_(
                arize_export_events.c.status == "pending",
                sa.or_(
                    arize_export_events.c.next_attempt_at.is_(None),
                    arize_export_events.c.next_attempt_at <= now,
                ),
            )
            ids = list(
                connection.execute(
                    sa.select(arize_export_events.c.id)
                    .where(eligible)
                    .order_by(arize_export_events.c.created_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                ).scalars()
            )
            if not ids:
                return []
            connection.execute(
                sa.update(arize_export_events)
                .where(
                    arize_export_events.c.id.in_(ids),
                    arize_export_events.c.status == "pending",
                )
                .values(
                    status="processing",
                    claimed_at=now,
                    attempt_count=arize_export_events.c.attempt_count + 1,
                )
            )
            selected = (
                sa.select(
                    arize_export_events.c.id.label("outbox_id"),
                    arize_export_events.c.prediction_event_id,
                    arize_export_events.c.event_type,
                    arize_export_events.c.attempt_count,
                    prediction_events.c.request_id,
                    prediction_events.c.created_at,
                    prediction_events.c.source,
                    prediction_events.c.age,
                    prediction_events.c.sex,
                    prediction_events.c.bmi,
                    prediction_events.c.children,
                    prediction_events.c.smoker,
                    prediction_events.c.region,
                    prediction_events.c.predicted_charges,
                    prediction_events.c.actual_charges,
                    prediction_events.c.model_version,
                    prediction_events.c.prediction_contract_version,
                    prediction_events.c.inference_latency_ms,
                )
                .join(
                    prediction_events,
                    prediction_events.c.id
                    == arize_export_events.c.prediction_event_id,
                )
                .where(arize_export_events.c.id.in_(ids))
                .order_by(arize_export_events.c.created_at)
            )
            rows = connection.execute(selected).mappings().all()
        return [ExportRecord(**dict(row)) for row in rows]

    def mark_sent(
        self, outbox_ids: list[uuid.UUID], *, sent_at: datetime
    ) -> None:
        if not outbox_ids:
            return
        with self._engine.begin() as connection:
            connection.execute(
                sa.update(arize_export_events)
                .where(
                    arize_export_events.c.id.in_(outbox_ids),
                    arize_export_events.c.status == "processing",
                )
                .values(
                    status="sent",
                    sent_at=sent_at,
                    claimed_at=None,
                    next_attempt_at=None,
                )
            )

    def reschedule_failed(
        self,
        records: list[ExportRecord],
        *,
        failed_at: datetime,
        base_seconds: int,
        maximum_seconds: int,
    ) -> None:
        with self._engine.begin() as connection:
            for record in records:
                exponent = min(max(record.attempt_count - 1, 0), 30)
                delay = min(base_seconds * (2**exponent), maximum_seconds)
                connection.execute(
                    sa.update(arize_export_events)
                    .where(
                        arize_export_events.c.id == record.outbox_id,
                        arize_export_events.c.status == "processing",
                    )
                    .values(
                        status="pending",
                        claimed_at=None,
                        next_attempt_at=failed_at + timedelta(seconds=delay),
                    )
                )

    def backlog(self, *, now: datetime | None = None) -> Backlog:
        current = now or datetime.now(timezone.utc)
        with self._engine.connect() as connection:
            remaining = connection.scalar(
                sa.select(sa.func.count())
                .select_from(arize_export_events)
                .where(arize_export_events.c.status != "sent")
            )
            oldest = connection.scalar(
                sa.select(sa.func.min(arize_export_events.c.created_at)).where(
                    arize_export_events.c.status == "pending"
                )
            )
        if oldest is None:
            age = None
        else:
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            age = max(0, int((current - oldest).total_seconds()))
        return Backlog(remaining=int(remaining or 0), oldest_pending_age_seconds=age)

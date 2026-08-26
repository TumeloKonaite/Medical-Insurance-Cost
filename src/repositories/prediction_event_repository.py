from __future__ import annotations

from typing import Protocol

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import IntegrityError

from src.models.prediction_event import prediction_events
from src.schemas.prediction_event import PredictionEvent


class PredictionEventRepository(Protocol):
    def add(self, event: PredictionEvent) -> bool: ...


class DisabledPredictionEventRepository:
    def add(self, event: PredictionEvent) -> bool:
        # Local inference can run without a configured prediction database.
        del event
        return False


class SqlPredictionEventRepository:
    def __init__(self, engine: Engine):
        self._engine = engine

    def add(self, event: PredictionEvent) -> bool:
        # Keep every model feature in its own typed database column.
        values = {
            "id": event.id,
            "request_id": event.request_id,
            "created_at": event.created_at,
            "source": event.source,
            "age": event.age,
            "sex": event.sex,
            "bmi": event.bmi,
            "children": event.children,
            "smoker": event.smoker,
            "region": event.region,
            "predicted_charges": event.predicted_charges,
            "model_version": event.model_version,
            "prediction_contract_version": event.prediction_contract_version,
            "inference_latency_ms": event.inference_latency_ms,
            "actual_charges": event.actual_charges,
            "actual_recorded_at": event.actual_recorded_at,
        }

        if self._engine.dialect.name == "postgresql":
            # PostgreSQL ignores a retry that reuses the same request ID.
            statement = (
                postgresql_insert(prediction_events)
                .values(**values)
                .on_conflict_do_nothing(index_elements=[prediction_events.c.request_id])
            )
            with self._engine.begin() as connection:
                result = connection.execute(statement)
            return result.rowcount == 1

        # This branch supports isolated repository tests without PostgreSQL.
        try:
            with self._engine.begin() as connection:
                connection.execute(sa.insert(prediction_events).values(**values))
            return True
        except IntegrityError:
            return False

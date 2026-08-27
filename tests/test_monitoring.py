from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.pool import StaticPool

from src.model_contract import FEATURE_COLUMNS, PREDICTION_CONTRACT_VERSION
from src.models.prediction_event import (
    arize_export_events,
    metadata,
    prediction_events,
)
from src.monitoring.arize_client import (
    ACTUAL_COLUMN,
    PREDICTION_COLUMN,
    PREDICTION_ID_COLUMN,
    ArizeBatchClient,
    ArizeUploadError,
    _schema,
    prediction_dataframe,
)
from src.monitoring.baseline import (
    BaselineUploadError,
    build_baseline_batch,
    upload_baseline,
)
from src.monitoring.config import (
    ArizeExportConfig,
    MonitoringConfigurationError,
)
from src.monitoring.exporter import ArizeExporter, ExportRunFailed
from src.monitoring.ground_truth import GroundTruthError, record_actual
from src.monitoring.outbox import OutboxRepository
from src.repositories.prediction_event_repository import SqlPredictionEventRepository
from src.schemas.prediction import PredictionRequest
from src.services.prediction_event_service import PredictionEventService

VALID_INPUT = {
    "age": 41,
    "sex": "male",
    "bmi": 30.25,
    "children": 2,
    "smoker": "no",
    "region": "northwest",
}


@pytest.fixture
def engine():
    database = sa.create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(database)
    return database


def add_prediction(engine, *, model_version="7", request_id=None):
    request_id = request_id or uuid.uuid4()
    service = PredictionEventService(SqlPredictionEventRepository(engine))
    assert service.record_success(
        request_id=request_id,
        source="json",
        payload=PredictionRequest(**VALID_INPUT),
        predicted_charges=9876.54,
        model_version=model_version,
        inference_latency_ms=1.25,
    )
    return request_id


class FakeClient:
    def __init__(self, *, fail_event_type=None, failure_message="upload failed"):
        self.calls = []
        self.fail_event_type = fail_event_type
        self.failure_message = failure_message

    def upload(self, dataframe, **kwargs):
        self.calls.append((dataframe.copy(), kwargs))
        if kwargs["event_type"] == self.fail_event_type:
            raise ArizeUploadError(self.failure_message, status_code=503)
        return 200


def config(**overrides):
    values = {
        "api_key": "test-secret",
        "space_id": "test-space",
        "batch_size": 500,
        "stale_claim_minutes": 30,
        "retry_base_seconds": 60,
        "retry_max_seconds": 3600,
    }
    values.update(overrides)
    return ArizeExportConfig(**values)


def test_configuration_is_fail_closed_and_repr_redacts_secrets():
    with pytest.raises(MonitoringConfigurationError, match="ARIZE_API_KEY"):
        ArizeExportConfig.from_environment({})

    parsed = ArizeExportConfig.from_environment(
        {
            "ARIZE_API_KEY": "super-secret",
            "ARIZE_SPACE_ID": "private-space",
            "ARIZE_EXPORT_BATCH_SIZE": "25",
        }
    )
    rendered = repr(parsed)
    assert parsed.batch_size == 25
    assert parsed.model_name == "medical-insurance-cost"
    assert "super-secret" not in rendered
    assert "private-space" not in rendered


def test_missing_arize_credentials_do_not_affect_fastapi_startup(monkeypatch):
    from src.main import create_app

    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    monkeypatch.delenv("ARIZE_SPACE_ID", raising=False)
    application = create_app()
    assert any(route.path == "/predict-json" for route in application.routes)


def test_prediction_insert_and_outbox_insert_are_atomic(engine):
    request_id = add_prediction(engine)
    with engine.connect() as connection:
        event = connection.execute(
            sa.select(prediction_events).where(
                prediction_events.c.request_id == request_id
            )
        ).mappings().one()
        outbox = connection.execute(sa.select(arize_export_events)).mappings().one()
    assert outbox["prediction_event_id"] == event["id"]
    assert outbox["event_type"] == "prediction"
    assert outbox["status"] == "pending"


def test_outbox_failure_rolls_back_prediction_insert(engine):
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TRIGGER reject_outbox BEFORE INSERT ON arize_export_events "
                "BEGIN SELECT RAISE(ABORT, 'reject'); END"
            )
        )
    service = PredictionEventService(SqlPredictionEventRepository(engine))
    inserted = service.record_success(
        request_id=uuid.uuid4(),
        source="json",
        payload=PredictionRequest(**VALID_INPUT),
        predicted_charges=1.0,
        model_version="7",
        inference_latency_ms=1.0,
    )
    with engine.connect() as connection:
        count = connection.scalar(sa.select(sa.func.count()).select_from(prediction_events))
    assert inserted is False
    assert count == 0


def test_duplicate_prediction_does_not_duplicate_outbox(engine):
    request_id = add_prediction(engine)
    service = PredictionEventService(SqlPredictionEventRepository(engine))
    assert not service.record_success(
        request_id=request_id,
        source="json",
        payload=PredictionRequest(**VALID_INPUT),
        predicted_charges=9876.54,
        model_version="7",
        inference_latency_ms=1.25,
    )
    with engine.connect() as connection:
        count = connection.scalar(
            sa.select(sa.func.count()).select_from(arize_export_events)
        )
    assert count == 1


def test_dataframe_mapping_converts_ids_timestamps_features_and_tags(engine):
    request_id = add_prediction(engine)
    now = datetime.now(timezone.utc)
    records = OutboxRepository(engine).claim(
        limit=10, now=now, stale_after=timedelta(minutes=30)
    )
    frame = prediction_dataframe(records)
    assert frame.loc[0, PREDICTION_ID_COLUMN] == str(request_id)
    assert isinstance(frame.loc[0, "prediction_timestamp"], np.integer | int)
    assert list(frame.columns[2:8]) == list(FEATURE_COLUMNS)
    assert frame.loc[0, PREDICTION_COLUMN] == pytest.approx(9876.54)
    assert frame.loc[0, "source"] == "json"
    assert frame.loc[0, "prediction_contract_version"] == PREDICTION_CONTRACT_VERSION
    assert frame.loc[0, "inference_latency_ms"] == pytest.approx(1.25)


def test_arize_v8_schemas_and_numeric_production_upload():
    pytest.importorskip("arize")
    prediction_schema = _schema("prediction")
    actual_schema = _schema("actual")
    baseline_schema = _schema("baseline")
    assert prediction_schema.feature_column_names == list(FEATURE_COLUMNS)
    assert prediction_schema.tag_column_names == [
        "source",
        "prediction_contract_version",
        "inference_latency_ms",
    ]
    assert prediction_schema.prediction_label_column_name == PREDICTION_COLUMN
    assert actual_schema.actual_label_column_name == ACTUAL_COLUMN
    assert actual_schema.prediction_label_column_name is None
    assert baseline_schema.prediction_label_column_name == PREDICTION_COLUMN
    assert baseline_schema.actual_label_column_name == ACTUAL_COLUMN

    captured = {}

    class MLClient:
        @staticmethod
        def log(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(status_code=200)

    client = object.__new__(ArizeBatchClient)
    client._config = config()
    client._client = SimpleNamespace(ml=MLClient())
    status = client.upload(
        pd.DataFrame(),
        event_type="prediction",
        model_version="7",
        environment="production",
    )
    from arize.ml.types import Environments, ModelTypes

    assert status == 200
    assert captured["model_type"] is ModelTypes.NUMERIC
    assert captured["environment"] is Environments.PRODUCTION
    assert captured["model_version"] == "7"


def test_exporter_groups_versions_and_separates_actual_batches(engine):
    first = add_prediction(engine, model_version="7")
    add_prediction(engine, model_version="8")
    assert record_actual(
        engine, request_id=first, actual_charges="10000.25"
    ) == "recorded"
    fake = FakeClient()
    summary = ArizeExporter(
        config=config(), repository=OutboxRepository(engine), client=fake
    ).run()

    call_keys = {
        (call[1]["model_version"], call[1]["event_type"]) for call in fake.calls
    }
    assert call_keys == {("7", "prediction"), ("7", "actual"), ("8", "prediction")}
    actual_frame = next(
        frame for frame, args in fake.calls if args["event_type"] == "actual"
    )
    assert list(actual_frame.columns) == [
        PREDICTION_ID_COLUMN,
        "prediction_timestamp",
        ACTUAL_COLUMN,
    ]
    assert actual_frame.loc[0, PREDICTION_ID_COLUMN] == str(first)
    assert summary.records_claimed == 3
    assert summary.records_sent == 3
    assert summary.remaining_backlog == 0


def test_failed_upload_is_rescheduled_with_backoff_and_sanitized_logs(
    engine, caplog
):
    add_prediction(engine)
    fake = FakeClient(
        fail_event_type="prediction",
        failure_message=(
            "api-key=test-secret feature_age=41 "
            "postgresql://user:password@host/database"
        ),
    )
    exporter = ArizeExporter(
        config=config(), repository=OutboxRepository(engine), client=fake
    )
    before = datetime.now(timezone.utc)
    with caplog.at_level(logging.INFO), pytest.raises(ExportRunFailed) as failure:
        exporter.run()
    after = datetime.now(timezone.utc)
    with engine.connect() as connection:
        row = connection.execute(sa.select(arize_export_events)).mappings().one()
    assert failure.value.summary.records_failed == 1
    assert row["status"] == "pending"
    assert row["attempt_count"] == 1
    assert row["next_attempt_at"] is not None
    next_attempt = row["next_attempt_at"].replace(tzinfo=timezone.utc)
    assert before + timedelta(seconds=60) <= next_attempt
    assert next_attempt <= after + timedelta(seconds=60)
    for prohibited in ("test-secret", "feature_age", "password", "database"):
        assert prohibited not in caplog.text
    assert "status_code=503" in caplog.text


def test_retry_backoff_is_bounded(engine):
    add_prediction(engine)
    repository = OutboxRepository(engine)
    now = datetime.now(timezone.utc)
    record = repository.claim(
        limit=1, now=now, stale_after=timedelta(minutes=30)
    )[0]
    repository.reschedule_failed(
        [replace(record, attempt_count=99)],
        failed_at=now,
        base_seconds=60,
        maximum_seconds=3600,
    )
    with engine.connect() as connection:
        next_attempt = connection.scalar(
            sa.select(arize_export_events.c.next_attempt_at)
        )
    next_attempt = next_attempt.replace(tzinfo=timezone.utc)
    assert next_attempt == now.replace(tzinfo=None).replace(
        tzinfo=timezone.utc
    ) + timedelta(seconds=3600)


def test_crashed_claim_is_recovered_only_after_stale_period(engine):
    add_prediction(engine)
    repository = OutboxRepository(engine)
    now = datetime.now(timezone.utc)
    claimed = repository.claim(
        limit=10, now=now, stale_after=timedelta(minutes=30)
    )
    assert len(claimed) == 1
    assert repository.claim(
        limit=10,
        now=now + timedelta(minutes=29),
        stale_after=timedelta(minutes=30),
    ) == []
    recovered = repository.claim(
        limit=10,
        now=now + timedelta(minutes=31),
        stale_after=timedelta(minutes=30),
    )
    assert len(recovered) == 1
    assert recovered[0].attempt_count == 2


def test_ground_truth_is_idempotent_and_refuses_conflict(engine):
    request_id = add_prediction(engine)
    assert record_actual(
        engine, request_id=request_id, actual_charges="12345.67"
    ) == "recorded"
    assert record_actual(
        engine, request_id=request_id, actual_charges="12345.670"
    ) == "unchanged"
    with pytest.raises(GroundTruthError, match="different actual"):
        record_actual(engine, request_id=request_id, actual_charges="12345.68")
    with engine.connect() as connection:
        actual_outboxes = connection.scalar(
            sa.select(sa.func.count())
            .select_from(arize_export_events)
            .where(arize_export_events.c.event_type == "actual")
        )
    assert actual_outboxes == 1


@pytest.mark.parametrize("value", ["nan", "inf", "-0.01"])
def test_ground_truth_rejects_invalid_actuals(engine, value):
    request_id = add_prediction(engine)
    with pytest.raises(GroundTruthError, match="finite non-negative"):
        record_actual(engine, request_id=request_id, actual_charges=value)


def test_baseline_validates_contract_builds_stable_ids_and_uploads(
    tmp_path, monkeypatch
):
    source = pd.DataFrame(
        [{**VALID_INPUT, "charges": 10000.0}, {**VALID_INPUT, "charges": 12000.0}]
    )
    path = tmp_path / "test.csv"
    source.to_csv(path, index=False)

    class Model:
        @staticmethod
        def predict(features):
            return np.array([9000.0] * len(features))

    deployment = SimpleNamespace(
        model=Model(),
        metadata={"model_version": "7", "mlflow_run_id": "run-123"},
    )
    monkeypatch.setattr(
        "src.monitoring.baseline.validate_local_package", lambda _path: deployment
    )
    first = build_baseline_batch(test_data=path, model_package=tmp_path)
    second = build_baseline_batch(test_data=path, model_package=tmp_path)
    assert first.dataframe[PREDICTION_ID_COLUMN].tolist() == second.dataframe[
        PREDICTION_ID_COLUMN
    ].tolist()
    assert list(first.dataframe.columns) == [
        PREDICTION_ID_COLUMN,
        *FEATURE_COLUMNS,
        PREDICTION_COLUMN,
        ACTUAL_COLUMN,
    ]

    fake = FakeClient()
    summary = upload_baseline(
        config=config(),
        test_data=path,
        model_package=tmp_path,
        client=fake,
    )
    assert summary["records_sent"] == 2
    assert fake.calls[0][1] == {
        "event_type": "baseline",
        "model_version": "7",
        "environment": "validation",
        "batch_id": "run-123",
    }


def test_baseline_refuses_noncanonical_dataset(tmp_path, monkeypatch):
    path = tmp_path / "bad.csv"
    pd.DataFrame([VALID_INPUT]).to_csv(path, index=False)
    with pytest.raises(BaselineUploadError, match="exactly"):
        build_baseline_batch(test_data=path, model_package=tmp_path)

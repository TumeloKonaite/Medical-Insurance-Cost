import uuid

import sqlalchemy as sa

metadata = sa.MetaData()

# This definition mirrors the table created by the Alembic migration.
prediction_events = sa.Table(
    "prediction_events",
    metadata,
    sa.Column("id", sa.Uuid(), primary_key=True, default=uuid.uuid4),
    sa.Column("request_id", sa.Uuid(), nullable=False, unique=True),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column("source", sa.String(), nullable=False),
    sa.Column("age", sa.Integer(), nullable=False),
    sa.Column("sex", sa.String(), nullable=False),
    sa.Column("bmi", sa.Numeric(), nullable=False),
    sa.Column("children", sa.Integer(), nullable=False),
    sa.Column("smoker", sa.String(), nullable=False),
    sa.Column("region", sa.String(), nullable=False),
    sa.Column("predicted_charges", sa.Numeric(), nullable=False),
    sa.Column("model_version", sa.String(), nullable=False),
    sa.Column("prediction_contract_version", sa.String(), nullable=False),
    sa.Column("inference_latency_ms", sa.Numeric(), nullable=False),
    sa.Column("actual_charges", sa.Numeric(), nullable=True),
    sa.Column("actual_recorded_at", sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("source IN ('web', 'json')", name="ck_prediction_events_source"),
    sa.Index("ix_prediction_events_request_id", "request_id"),
    sa.Index("ix_prediction_events_created_at", "created_at"),
    sa.Index("ix_prediction_events_model_version", "model_version"),
)

arize_export_events = sa.Table(
    "arize_export_events",
    metadata,
    sa.Column("id", sa.Uuid(), primary_key=True, default=uuid.uuid4),
    sa.Column(
        "prediction_event_id",
        sa.Uuid(),
        sa.ForeignKey("prediction_events.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("event_type", sa.String(), nullable=False),
    sa.Column("status", sa.String(), nullable=False, server_default="pending"),
    sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.CheckConstraint(
        "event_type IN ('prediction', 'actual')",
        name="ck_arize_export_events_event_type",
    ),
    sa.CheckConstraint(
        "status IN ('pending', 'processing', 'sent')",
        name="ck_arize_export_events_status",
    ),
    sa.CheckConstraint(
        "attempt_count >= 0", name="ck_arize_export_events_attempt_count"
    ),
    sa.UniqueConstraint(
        "prediction_event_id",
        "event_type",
        name="uq_arize_export_events_prediction_event_type",
    ),
    sa.Index(
        "ix_arize_export_events_pending",
        "status",
        "next_attempt_at",
        "created_at",
    ),
    sa.Index(
        "ix_arize_export_events_stale_claims",
        "status",
        "claimed_at",
    ),
)

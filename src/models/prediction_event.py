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

"""Create prediction events.

Revision ID: 20260826_01
Revises:
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260826_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create the typed event table and its monitoring lookup indexes.
    op.create_table(
        "prediction_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
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
        sa.CheckConstraint(
            "source IN ('web', 'json')", name="ck_prediction_events_source"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    op.create_index(
        "ix_prediction_events_request_id",
        "prediction_events",
        ["request_id"],
    )
    op.create_index(
        "ix_prediction_events_created_at",
        "prediction_events",
        ["created_at"],
    )
    op.create_index(
        "ix_prediction_events_model_version",
        "prediction_events",
        ["model_version"],
    )


def downgrade() -> None:
    # Remove indexes before removing the table during an intentional rollback.
    op.drop_index("ix_prediction_events_model_version", table_name="prediction_events")
    op.drop_index("ix_prediction_events_created_at", table_name="prediction_events")
    op.drop_index("ix_prediction_events_request_id", table_name="prediction_events")
    op.drop_table("prediction_events")

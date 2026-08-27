"""Create the Arize transactional outbox.

Revision ID: 20260827_01
Revises: 20260826_01
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260827_01"
down_revision: str | None = "20260826_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "arize_export_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("prediction_event_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column(
            "status", sa.String(), server_default="pending", nullable=False
        ),
        sa.Column(
            "attempt_count", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
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
        sa.ForeignKeyConstraint(
            ["prediction_event_id"], ["prediction_events.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "prediction_event_id",
            "event_type",
            name="uq_arize_export_events_prediction_event_type",
        ),
    )
    op.create_index(
        "ix_arize_export_events_pending",
        "arize_export_events",
        ["status", "next_attempt_at", "created_at"],
    )
    op.create_index(
        "ix_arize_export_events_stale_claims",
        "arize_export_events",
        ["status", "claimed_at"],
    )

    # Existing durable predictions should enter the outbox exactly once. Reusing
    # the prediction UUID here avoids requiring a database UUID extension.
    op.execute(
        sa.text(
            """
            INSERT INTO arize_export_events
                (id, prediction_event_id, event_type, status, attempt_count)
            SELECT id, id, 'prediction', 'pending', 0
            FROM prediction_events
            ON CONFLICT (prediction_event_id, event_type) DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.drop_index(
        "ix_arize_export_events_stale_claims",
        table_name="arize_export_events",
    )
    op.drop_index(
        "ix_arize_export_events_pending", table_name="arize_export_events"
    )
    op.drop_table("arize_export_events")

"""Add append-only memory_versions table.

Revision ID: 0002_memory_versions
Revises: 0001_initial_schema
Create Date: 2026-04-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from siqueira_memo.models.types import GUID, JSONB

revision: str = "0002_memory_versions"
down_revision: str | Sequence[str] | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    return sa.inspect(bind).has_table(table_name)


def upgrade() -> None:
    if _has_table("memory_versions"):
        return
    op.create_table(
        "memory_versions",
        sa.Column("profile_id", sa.String(length=128), nullable=False),
        sa.Column("target_type", sa.String(length=24), nullable=False),
        sa.Column("target_id", GUID(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.String(length=1024), nullable=True),
        sa.Column("event_id", GUID(), nullable=True),
        sa.Column("rollback_to_version", sa.Integer(), nullable=True),
        sa.Column("before_snapshot", JSONB(), nullable=True),
        sa.Column("after_snapshot", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["memory_events.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "profile_id",
            "target_type",
            "target_id",
            "version",
            name="uq_memory_versions_target_version",
        ),
    )
    op.create_index("ix_memory_versions_profile_id", "memory_versions", ["profile_id"])
    op.create_index("ix_memory_versions_target_type", "memory_versions", ["target_type"])
    op.create_index("ix_memory_versions_target_id", "memory_versions", ["target_id"])
    op.create_index("ix_memory_versions_operation", "memory_versions", ["operation"])
    op.create_index("ix_memory_versions_event_id", "memory_versions", ["event_id"])
    op.create_index(
        "ix_memory_versions_target_created",
        "memory_versions",
        ["target_type", "target_id", "created_at"],
    )


def downgrade() -> None:
    if not _has_table("memory_versions"):
        return
    op.drop_index("ix_memory_versions_target_created", table_name="memory_versions")
    op.drop_index("ix_memory_versions_event_id", table_name="memory_versions")
    op.drop_index("ix_memory_versions_operation", table_name="memory_versions")
    op.drop_index("ix_memory_versions_target_id", table_name="memory_versions")
    op.drop_index("ix_memory_versions_target_type", table_name="memory_versions")
    op.drop_index("ix_memory_versions_profile_id", table_name="memory_versions")
    op.drop_table("memory_versions")

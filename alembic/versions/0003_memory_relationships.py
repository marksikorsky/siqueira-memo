"""Add first-class memory_relationships graph table.

Revision ID: 0003_memory_relationships
Revises: 0002_memory_versions
Create Date: 2026-04-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from siqueira_memo.models.types import GUID, JSONB, UUIDArray

revision: str = "0003_memory_relationships"
down_revision: str | Sequence[str] | None = "0002_memory_versions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_table(table_name: str) -> bool:
    bind = op.get_bind()
    return sa.inspect(bind).has_table(table_name)


def upgrade() -> None:
    if _has_table("memory_relationships"):
        return
    op.create_table(
        "memory_relationships",
        sa.Column("profile_id", sa.String(length=128), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_id", GUID(), nullable=False),
        sa.Column("relationship_type", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", GUID(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("source_event_ids", UUIDArray(), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("extra_metadata", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "profile_id",
            "source_type",
            "source_id",
            "relationship_type",
            "target_type",
            "target_id",
            "status",
            name="uq_memory_relationships_directed_status",
        ),
    )
    for name, cols in (
        ("ix_memory_relationships_profile_id", ["profile_id"]),
        ("ix_memory_relationships_source_type", ["source_type"]),
        ("ix_memory_relationships_source_id", ["source_id"]),
        ("ix_memory_relationships_relationship_type", ["relationship_type"]),
        ("ix_memory_relationships_target_type", ["target_type"]),
        ("ix_memory_relationships_target_id", ["target_id"]),
        ("ix_memory_relationships_status", ["status"]),
        ("ix_memory_relationships_source", ["profile_id", "source_type", "source_id", "status"]),
        ("ix_memory_relationships_target", ["profile_id", "target_type", "target_id", "status"]),
    ):
        op.create_index(name, "memory_relationships", cols)


def downgrade() -> None:
    if not _has_table("memory_relationships"):
        return
    for name in (
        "ix_memory_relationships_target",
        "ix_memory_relationships_source",
        "ix_memory_relationships_status",
        "ix_memory_relationships_target_id",
        "ix_memory_relationships_target_type",
        "ix_memory_relationships_relationship_type",
        "ix_memory_relationships_source_id",
        "ix_memory_relationships_source_type",
        "ix_memory_relationships_profile_id",
    ):
        op.drop_index(name, table_name="memory_relationships")
    op.drop_table("memory_relationships")

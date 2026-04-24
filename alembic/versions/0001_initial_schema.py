"""Initial Siqueira Memo schema.

Creates every ORM table from ``siqueira_memo.models`` plus Postgres-only
extensions (pgvector, pg_trgm) and per-dialect indexes (HNSW, GIN FTS). On
SQLite the extension/index steps are silently skipped.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-24
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from alembic import op

# Ensure ``src`` is on sys.path so ``siqueira_memo`` resolves when alembic runs
# from a clean shell (``alembic upgrade head``).
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from siqueira_memo.models import Base  # noqa: E402

revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade() -> None:
    bind = op.get_bind()
    if _is_postgres():
        # pgvector for embeddings, pg_trgm for trigram lookup on aliases/FTS.
        bind.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        bind.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # Creating tables from metadata is safe on every dialect we support.
    Base.metadata.create_all(bind=bind)

    if _is_postgres():
        # HNSW indexes for embedding tables (plan §28.4). pgvector's vector
        # HNSW opclass supports up to 2,000 dimensions; text-embedding-3-large
        # is 3,072 dimensions, so that table is deliberately left unindexed
        # until we switch it to halfvec/subvector indexing in a later migration.
        bind.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS chunk_embeddings_bge_m3_hnsw_idx "
            "ON chunk_embeddings_bge_m3 "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        bind.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS chunk_embeddings_mock_hnsw_idx "
            "ON chunk_embeddings_mock "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        # Russian + English + simple FTS on chunks (plan §28.3).
        bind.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS chunks_fts_ru_idx "
            "ON chunks USING gin (to_tsvector('russian', chunk_text))"
        )
        bind.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS chunks_fts_en_idx "
            "ON chunks USING gin (to_tsvector('english', chunk_text))"
        )
        bind.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS chunks_fts_simple_idx "
            "ON chunks USING gin (to_tsvector('simple', chunk_text))"
        )
        # Entity alias trigram.
        bind.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS entity_aliases_trgm_idx "
            "ON entity_aliases USING gin (alias_normalized gin_trgm_ops)"
        )
        # Partial-unique active canonical indexes (plan §18.2.6).
        bind.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS facts_active_canonical_key_idx "
            "ON facts (profile_id, canonical_key) WHERE status = 'active'"
        )
        bind.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS decisions_active_canonical_key_idx "
            "ON decisions (profile_id, canonical_key) WHERE status = 'active'"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if _is_postgres():
        for sql in (
            "DROP INDEX IF EXISTS decisions_active_canonical_key_idx",
            "DROP INDEX IF EXISTS facts_active_canonical_key_idx",
            "DROP INDEX IF EXISTS entity_aliases_trgm_idx",
            "DROP INDEX IF EXISTS chunks_fts_simple_idx",
            "DROP INDEX IF EXISTS chunks_fts_en_idx",
            "DROP INDEX IF EXISTS chunks_fts_ru_idx",
            "DROP INDEX IF EXISTS chunk_embeddings_mock_hnsw_idx",
            "DROP INDEX IF EXISTS chunk_embeddings_bge_m3_hnsw_idx",
        ):
            bind.exec_driver_sql(sql)
    Base.metadata.drop_all(bind=bind)

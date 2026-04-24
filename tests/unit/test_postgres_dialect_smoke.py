"""Postgres-dialect smoke tests.

These verify that the cross-dialect type adapters emit the expected Postgres
types (UUID, JSONB, ARRAY, vector) and that the initial Alembic migration
renders pgvector + partial-unique DDL when compiled against the Postgres
dialect — all without a running Postgres instance.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_mock_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.schema import CreateTable

from siqueira_memo.models import Base, Chunk, Fact
from siqueira_memo.models.types import GUID, JSONB, StringArray, UUIDArray, Vector

PG_DIALECT = postgresql.dialect()


def test_guid_resolves_to_native_uuid_on_postgres():
    impl = GUID().dialect_impl(PG_DIALECT)
    assert isinstance(impl.impl_instance, PG_UUID)
    value = uuid.uuid4()
    bound = GUID().process_bind_param(value, PG_DIALECT)
    assert isinstance(bound, uuid.UUID)


def test_jsonb_resolves_to_pg_jsonb():
    impl = JSONB().dialect_impl(PG_DIALECT)
    assert isinstance(impl.impl_instance, PG_JSONB)


def test_string_array_resolves_to_text_array():
    impl = StringArray().dialect_impl(PG_DIALECT)
    assert isinstance(impl.impl_instance, ARRAY)
    assert impl.impl_instance.item_type.__class__.__name__ == "String"


def test_uuid_array_resolves_to_uuid_array():
    impl = UUIDArray().dialect_impl(PG_DIALECT)
    assert isinstance(impl.impl_instance, ARRAY)
    assert isinstance(impl.impl_instance.item_type, PG_UUID)


def test_vector_resolves_to_pgvector_when_extension_available():
    try:
        from pgvector.sqlalchemy import Vector as PGVector
    except ImportError:
        pytest.skip("pgvector not installed")
    impl = Vector(16).dialect_impl(PG_DIALECT)
    assert isinstance(impl.impl_instance, PGVector)


def test_facts_active_canonical_index_renders_partial_on_postgres():
    """The unique index must carry a WHERE clause on Postgres (plan §18.2.6)."""
    from sqlalchemy import inspect
    from sqlalchemy.schema import CreateIndex

    mapper = inspect(Fact)
    idx = next(i for i in mapper.local_table.indexes if i.name == "uq_facts_active_canonical")
    compiled = str(CreateIndex(idx).compile(dialect=PG_DIALECT))
    assert "WHERE" in compiled.upper()
    assert "active" in compiled.lower()


def test_create_table_renders_pg_types():
    """Compile ``CREATE TABLE messages`` against Postgres — must emit UUID/JSONB."""
    messages_table = Base.metadata.tables["messages"]
    ddl = str(CreateTable(messages_table).compile(dialect=PG_DIALECT))
    assert "UUID" in ddl
    assert "JSONB" in ddl
    # Plan §3.2 expects TEXT[] for entities.
    assert "TEXT[]" in ddl or "VARCHAR[]" in ddl


def test_chunks_table_has_vector_column_on_postgres():
    try:
        from pgvector.sqlalchemy import Vector as PGVector  # noqa: F401
    except ImportError:
        pytest.skip("pgvector not installed")
    from siqueira_memo.models import ChunkEmbeddingMock

    ddl = str(CreateTable(ChunkEmbeddingMock.__table__).compile(dialect=PG_DIALECT))
    # Dim 16 for the mock table.
    assert "VECTOR(16)" in ddl.upper()


def test_base_metadata_create_all_captured_by_pg_mock_engine():
    """Run metadata.create_all against a mock Postgres engine and capture DDL.

    This catches accidental regressions where adding a new column or table
    would fail Postgres-specific compilation (e.g. a type that lacks a PG
    implementation).
    """
    captured: list[str] = []

    def dump(sql, *multiparams, **params):  # noqa: ARG001
        captured.append(str(sql.compile(dialect=PG_DIALECT)))

    engine = create_mock_engine("postgresql://", dump)
    Base.metadata.create_all(engine, checkfirst=False)
    body = "\n".join(captured)
    # Spot-check a few expected DDL fragments.
    assert "CREATE TABLE memory_events" in body
    assert "CREATE TABLE messages" in body
    assert "CREATE TABLE chunks" in body
    assert "CREATE TABLE chunk_embeddings_openai_text_embedding_3_large" in body
    # Array + JSONB columns rendered correctly.
    assert "JSONB" in body
    assert "UUID" in body


def test_vector_round_trip_on_sqlite_json_stores_floats():
    """Sanity: on non-PG dialects Vector serialises to JSON floats."""
    v = Vector(4)
    sqlite_dialect = type("FakeSQLite", (), {"name": "sqlite"})()
    encoded = v.process_bind_param([0.1, -0.2, 0.3, 0.0], sqlite_dialect)
    decoded = v.process_result_value(encoded, sqlite_dialect)
    assert decoded == [0.1, -0.2, 0.3, 0.0]


def test_guid_round_trip_accepts_string_on_non_pg():
    sqlite_dialect = type("FakeSQLite", (), {"name": "sqlite"})()
    g = GUID()
    u = uuid.uuid4()
    bound = g.process_bind_param(u, sqlite_dialect)
    assert isinstance(bound, str) and len(bound) == 36
    back = g.process_result_value(bound, sqlite_dialect)
    assert back == u


def test_uuid_array_round_trip_on_sqlite_is_json():
    sqlite_dialect = type("FakeSQLite", (), {"name": "sqlite"})()
    arr = UUIDArray()
    uids = [uuid.uuid4(), uuid.uuid4()]
    bound = arr.process_bind_param(uids, sqlite_dialect)
    decoded = arr.process_result_value(bound, sqlite_dialect)
    assert decoded == uids


def test_chunk_embedding_tables_cover_plan_models():
    """Plan §31.6: the registry must enumerate the three physical tables we ship."""
    from siqueira_memo.models import EMBEDDING_TABLE_BY_MODEL

    keys = set(EMBEDDING_TABLE_BY_MODEL.keys())
    assert ("mock", 16) in keys
    assert ("text-embedding-3-large", 3072) in keys
    assert ("bge-m3", 1024) in keys


def test_chunk_source_type_column_uses_string():
    # Guard against someone regressing chunk.source_type to Enum and breaking
    # the per-dialect text storage assumption.
    col = Chunk.__table__.columns["source_type"]
    assert col.type.__class__.__name__ == "String"


def test_alembic_initial_migration_compiles_pg_ddl(monkeypatch):
    """Execute the initial migration's DDL against a mock PG engine.

    We load the migration script by path (alembic's ``versions/`` is not a
    Python package) and patch ``op.get_bind`` to return a dialect stub so the
    migration's CREATE EXTENSION / HNSW / FTS / partial-unique statements are
    captured.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_sqmemo_initial_migration",
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0001_initial_schema.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    statements: list[str] = []

    class _Bind:
        class dialect:
            name = "postgresql"

        def exec_driver_sql(self, sql: str) -> None:
            statements.append(sql)

    class _OpShim:
        @staticmethod
        def get_bind():
            return _Bind()

    monkeypatch.setattr(module, "op", _OpShim)
    monkeypatch.setattr(
        module.Base.metadata,
        "create_all",
        lambda *a, **kw: statements.append("-- create_all skipped in mock --"),
    )
    module.upgrade()

    body = "\n".join(statements)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in body
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in body
    assert "chunk_embeddings_bge_m3_hnsw_idx" in body
    assert "chunk_embeddings_mock_hnsw_idx" in body
    assert "chunk_embeddings_openai_te3l_hnsw_idx" not in body
    assert "hnsw (embedding vector_cosine_ops)" in body
    assert "to_tsvector('russian', chunk_text)" in body
    assert "facts_active_canonical_key_idx" in body
    assert "WHERE status = 'active'" in body

"""Cross-dialect column type adapters.

Production runs on PostgreSQL 16 with pgvector. Tests run on SQLite. The
adapters below expose a single column type that picks the right backend
implementation at bind time.

These adapters never hide semantics: the Postgres variant is always the
authoritative one; the SQLite variant is a JSON-serialised approximation used
only for unit testing.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import JSON, String, TypeDecorator
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.types import CHAR, Text


class GUID(TypeDecorator[uuid.UUID]):
    """Platform-independent UUID stored as native UUID on Postgres and CHAR(36) on SQLite."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:  # noqa: D401
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pg.UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: Any, dialect: Any) -> Any:  # noqa: D401
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(str(value)))

    def process_result_value(self, value: Any, dialect: Any) -> Any:  # noqa: D401
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class JSONB(TypeDecorator[Any]):
    """JSONB on Postgres, JSON on other dialects."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:  # noqa: D401
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pg.JSONB())
        return dialect.type_descriptor(JSON())


class StringArray(TypeDecorator[list[str]]):
    """TEXT[] on Postgres, JSON-encoded list on SQLite."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:  # noqa: D401
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pg.ARRAY(String()))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:  # noqa: D401
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return json.dumps(list(value), ensure_ascii=False)

    def process_result_value(self, value: Any, dialect: Any) -> Any:  # noqa: D401
        if value is None:
            return []
        if dialect.name == "postgresql":
            return list(value)
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            try:
                loaded = json.loads(value)
                if isinstance(loaded, list):
                    return [str(x) for x in loaded]
            except json.JSONDecodeError:
                return []
        return []


class UUIDArray(TypeDecorator[list[uuid.UUID]]):
    """UUID[] on Postgres, JSON-encoded list on SQLite."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:  # noqa: D401
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pg.ARRAY(pg.UUID(as_uuid=True)))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:  # noqa: D401
        if value is None:
            return None
        coerced = [v if isinstance(v, uuid.UUID) else uuid.UUID(str(v)) for v in value]
        if dialect.name == "postgresql":
            return coerced
        return json.dumps([str(v) for v in coerced])

    def process_result_value(self, value: Any, dialect: Any) -> Any:  # noqa: D401
        if value is None:
            return []
        if dialect.name == "postgresql":
            return [v if isinstance(v, uuid.UUID) else uuid.UUID(str(v)) for v in value]
        if isinstance(value, list):
            return [uuid.UUID(str(v)) for v in value]
        if isinstance(value, str) and value:
            try:
                loaded = json.loads(value)
                return [uuid.UUID(str(v)) for v in loaded]
            except json.JSONDecodeError:
                return []
        return []


class Vector(TypeDecorator[list[float]]):
    """pgvector ``vector(N)`` on Postgres, JSON-encoded list of floats elsewhere.

    The Postgres implementation is fully realised by the pgvector extension and
    is used by HNSW indexes defined in migrations. On SQLite the column is only
    a serialization of the embedding; similarity scoring is still possible via
    Python cosine helpers for in-memory tests.
    """

    impl = Text
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect: Any) -> Any:  # noqa: D401
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector as PGVector  # type: ignore[import-not-found]

                return dialect.type_descriptor(PGVector(self.dimensions))
            except ImportError:
                return dialect.type_descriptor(Text())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:  # noqa: D401
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return json.dumps([float(v) for v in value])

    def process_result_value(self, value: Any, dialect: Any) -> Any:  # noqa: D401
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value) if value is not None else None
        if isinstance(value, list):
            return [float(v) for v in value]
        if isinstance(value, str) and value:
            try:
                return [float(x) for x in json.loads(value)]
            except (json.JSONDecodeError, ValueError):
                return None
        return None

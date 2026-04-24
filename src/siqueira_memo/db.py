"""Database engine and session plumbing.

Siqueira targets PostgreSQL 16 + pgvector in production. Tests run against
in-memory SQLite so the engine helpers intentionally cover both dialects.
Postgres-specific indexing (HNSW, tstzrange, GIN) is created by Alembic
migrations; the ORM layer stays portable across dialects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from siqueira_memo.config import Settings, get_settings

_engines: dict[str, AsyncEngine] = {}
_factories: dict[str, async_sessionmaker[AsyncSession]] = {}


def build_engine(settings: Settings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    kwargs: dict[str, Any] = {
        "echo": settings.database_echo,
        "future": True,
    }
    if settings.is_sqlite:
        # SQLite async needs check_same_thread=False and we disable pooling
        # for :memory: so every session shares the same connection.
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in settings.database_url:
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    return create_async_engine(settings.database_url, **kwargs)


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    key = settings.database_url
    if key not in _engines:
        _engines[key] = build_engine(settings)
        _factories[key] = async_sessionmaker(
            _engines[key], expire_on_commit=False, class_=AsyncSession
        )
    return _engines[key]


def get_session_factory(
    settings: Settings | None = None,
) -> async_sessionmaker[AsyncSession]:
    get_engine(settings)
    settings = settings or get_settings()
    return _factories[settings.database_url]


@asynccontextmanager
async def session_scope(
    settings: Settings | None = None,
) -> AsyncIterator[AsyncSession]:
    factory = get_session_factory(settings)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


async def dispose_engines() -> None:
    """Dispose all cached engines. Called at app shutdown and from tests."""
    for engine in list(_engines.values()):
        await engine.dispose()
    _engines.clear()
    _factories.clear()


async def create_all_for_tests(settings: Settings | None = None) -> None:
    """Create all ORM tables. Production uses Alembic; this is for SQLite tests."""
    from siqueira_memo.models.base import Base  # noqa: WPS433

    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all_for_tests(settings: Settings | None = None) -> None:
    from siqueira_memo.models.base import Base  # noqa: WPS433

    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

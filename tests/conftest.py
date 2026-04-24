"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from siqueira_memo.config import Settings, settings_for_tests
from siqueira_memo.db import (
    create_all_for_tests,
    dispose_engines,
    drop_all_for_tests,
    get_session_factory,
)
from siqueira_memo.workers.queue import MemoryJobQueue, set_default_queue


@pytest.fixture(scope="session")
def event_loop():
    """Share one event loop across the test session — avoids fresh-loop churn."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def settings() -> Settings:
    return settings_for_tests()


@pytest_asyncio.fixture
async def db(settings: Settings) -> AsyncIterator[Settings]:
    await create_all_for_tests(settings)
    try:
        yield settings
    finally:
        await drop_all_for_tests(settings)
        await dispose_engines()


@pytest_asyncio.fixture
async def session(db):
    factory = get_session_factory(db)
    async with factory() as s:
        yield s


@pytest.fixture
def queue() -> MemoryJobQueue:
    q = MemoryJobQueue()
    set_default_queue(q)
    try:
        yield q
    finally:
        set_default_queue(None)

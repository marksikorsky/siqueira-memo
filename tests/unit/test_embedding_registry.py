"""Embedding registry tests. Plan §31.6 / §33.11."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from siqueira_memo.models import Chunk, ChunkEmbeddingMock, EmbeddingIndex
from siqueira_memo.models.constants import CHUNK_SOURCE_MESSAGE
from siqueira_memo.services.embedding_registry import (
    EmbeddingRegistry,
    sync_from_provider,
)
from siqueira_memo.services.embedding_service import (
    MockEmbeddingProvider,
    cosine,
)


@pytest.mark.asyncio
async def test_register_activates_provider_spec(db, session):
    provider = MockEmbeddingProvider()
    info = await sync_from_provider(session, provider)
    assert info.active is True
    assert info.table_name == "chunk_embeddings_mock"
    rows = (await session.execute(select(EmbeddingIndex))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_registration_is_idempotent(db, session):
    registry = EmbeddingRegistry()
    await registry.register(session, MockEmbeddingProvider().spec)
    await registry.register(session, MockEmbeddingProvider().spec)
    rows = (await session.execute(select(EmbeddingIndex))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_registering_new_model_deactivates_old(db, session):
    """Two physical tables for the same model_name → newer one becomes active."""
    registry = EmbeddingRegistry()
    from siqueira_memo.services.embedding_service import EmbeddingSpec

    old_spec = EmbeddingSpec(
        provider="mock",
        model_name="mock",
        model_version="1",
        dimensions=16,
        table_name="chunk_embeddings_mock",
    )
    newer_spec = EmbeddingSpec(
        provider="mock",
        model_name="mock",
        model_version="2",
        dimensions=1024,
        table_name="chunk_embeddings_bge_m3",
    )
    await registry.register(session, old_spec)
    await registry.register(session, newer_spec)

    active = await registry.active(session)
    assert len(active) == 1
    assert active[0].model_version == "2"
    assert active[0].table_name == "chunk_embeddings_bge_m3"


@pytest.mark.asyncio
async def test_store_embedding_validates_dimensions(db, session):
    registry = EmbeddingRegistry()
    mock = MockEmbeddingProvider()
    chunk = Chunk(
        profile_id="p1",
        source_type=CHUNK_SOURCE_MESSAGE,
        source_id=uuid.uuid4(),
        chunk_text="hi",
        token_count=1,
        tokenizer_name="t",
    )
    session.add(chunk)
    await session.flush()

    with pytest.raises(ValueError):
        await registry.store_embedding(
            session, chunk_id=chunk.id, vector=[0.0], spec=mock.spec
        )

    vec = mock.embed("hello")
    emb_id = await registry.store_embedding(
        session, chunk_id=chunk.id, vector=vec, spec=mock.spec
    )
    stored = (
        await session.execute(
            select(ChunkEmbeddingMock).where(ChunkEmbeddingMock.id == emb_id)
        )
    ).scalar_one()
    assert stored.chunk_id == chunk.id
    assert len(stored.embedding) == mock.spec.dimensions


@pytest.mark.asyncio
async def test_store_embedding_is_unique_per_chunk_version(db, session):
    mock = MockEmbeddingProvider()
    registry = EmbeddingRegistry()
    chunk = Chunk(
        profile_id="p1",
        source_type=CHUNK_SOURCE_MESSAGE,
        source_id=uuid.uuid4(),
        chunk_text="hi",
        token_count=1,
        tokenizer_name="t",
    )
    session.add(chunk)
    await session.flush()
    vec = mock.embed("hello")
    await registry.store_embedding(session, chunk_id=chunk.id, vector=vec, spec=mock.spec)
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        await registry.store_embedding(session, chunk_id=chunk.id, vector=vec, spec=mock.spec)


def test_mock_provider_embeddings_are_deterministic():
    mock = MockEmbeddingProvider()
    a = mock.embed("hello")
    b = mock.embed("hello")
    c = mock.embed("goodbye")
    assert a == b
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(a, c) < 1.0

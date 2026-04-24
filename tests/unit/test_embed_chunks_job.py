"""embed_chunks worker + rebuild helper. Plan §4.3 / §31.6."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from siqueira_memo.models import Chunk, ChunkEmbeddingMock, EmbeddingIndex
from siqueira_memo.models.constants import CHUNK_SOURCE_MESSAGE
from siqueira_memo.services.embedding_registry import EmbeddingRegistry
from siqueira_memo.services.embedding_service import MockEmbeddingProvider
from siqueira_memo.workers.jobs import embed_chunks_for_source


@pytest.mark.asyncio
async def test_embed_chunks_for_source_fills_table(db, session):
    profile = "p1"
    source_id = uuid.uuid4()
    session.add_all(
        [
            Chunk(
                profile_id=profile,
                source_type=CHUNK_SOURCE_MESSAGE,
                source_id=source_id,
                chunk_text="Siqueira Memo uses pgvector.",
                chunk_index=0,
                token_count=5,
                tokenizer_name="test",
            ),
            Chunk(
                profile_id=profile,
                source_type=CHUNK_SOURCE_MESSAGE,
                source_id=source_id,
                chunk_text="It is integrated via MemoryProvider plugin.",
                chunk_index=1,
                token_count=6,
                tokenizer_name="test",
            ),
        ]
    )
    await session.flush()
    provider = MockEmbeddingProvider()
    registry = EmbeddingRegistry()
    await registry.register(session, provider.spec)

    embedded = await embed_chunks_for_source(
        session,
        profile_id=profile,
        source_type=CHUNK_SOURCE_MESSAGE,
        source_id=source_id,
        provider=provider,
        registry=registry,
    )
    assert embedded == 2

    # Re-running must not duplicate embeddings (unique (chunk_id, model_version)).
    embedded_again = await embed_chunks_for_source(
        session,
        profile_id=profile,
        source_type=CHUNK_SOURCE_MESSAGE,
        source_id=source_id,
        provider=provider,
        registry=registry,
    )
    assert embedded_again == 0

    rows = (await session.execute(select(ChunkEmbeddingMock))).scalars().all()
    assert len(rows) == 2
    assert all(len(r.embedding) == provider.spec.dimensions for r in rows)

    indexes = (await session.execute(select(EmbeddingIndex))).scalars().all()
    assert any(i.table_name == provider.spec.table_name and i.active for i in indexes)


@pytest.mark.asyncio
async def test_embed_chunks_skips_sensitive(db, session):
    profile = "p1"
    source_id = uuid.uuid4()
    session.add(
        Chunk(
            profile_id=profile,
            source_type=CHUNK_SOURCE_MESSAGE,
            source_id=source_id,
            chunk_text="contains redacted [SECRET_REF:openai_api_key/x/abc]",
            chunk_index=0,
            token_count=5,
            tokenizer_name="test",
            sensitivity="sensitive",
        )
    )
    await session.flush()
    provider = MockEmbeddingProvider()
    registry = EmbeddingRegistry()
    await registry.register(session, provider.spec)
    embedded = await embed_chunks_for_source(
        session,
        profile_id=profile,
        source_type=CHUNK_SOURCE_MESSAGE,
        source_id=source_id,
        provider=provider,
        registry=registry,
    )
    assert embedded == 0
    rows = (await session.execute(select(ChunkEmbeddingMock))).scalars().all()
    assert rows == []

"""Embedding index registry. Plan §31.6 / §33.11.

Maintains rows in ``embedding_indexes`` describing which physical embedding
table each model/dimension pair writes to, and whether it is the active one
for new retrievals.

Because production can run multiple embedding backends simultaneously (e.g.
migrating from ``text-embedding-3-large`` → ``bge-m3``), the registry stores:

* ``model_name``, ``model_version``, ``dimensions``, ``distance_metric``
* ``table_name`` (the physical ``chunk_embeddings_*`` table)
* ``active`` flag — only one active row per ``(model_name, model_version,
  dimensions)`` tuple.

Write-side helpers deliberately insert into the specific physical table (via
``EMBEDDING_TABLE_BY_MODEL``) so tests can verify vectors land in the correct
place without depending on pgvector.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.logging import get_logger
from siqueira_memo.models import (
    ChunkEmbeddingBGEM3,
    ChunkEmbeddingMock,
    ChunkEmbeddingOpenAITEL3,
    EmbeddingIndex,
)
from siqueira_memo.services.embedding_service import EmbeddingProvider, EmbeddingSpec

log = get_logger(__name__)


_SUPPORTED_TABLES: dict[str, type] = {
    ChunkEmbeddingMock.__tablename__: ChunkEmbeddingMock,
    ChunkEmbeddingOpenAITEL3.__tablename__: ChunkEmbeddingOpenAITEL3,
    ChunkEmbeddingBGEM3.__tablename__: ChunkEmbeddingBGEM3,
}


@dataclass
class EmbeddingIndexInfo:
    id: uuid.UUID
    table_name: str
    model_name: str
    model_version: str
    dimensions: int
    distance_metric: str
    active: bool


class EmbeddingRegistry:
    async def register(
        self,
        session: AsyncSession,
        spec: EmbeddingSpec,
        *,
        distance_metric: str = "cosine",
        active: bool = True,
    ) -> EmbeddingIndexInfo:
        row = (
            await session.execute(
                select(EmbeddingIndex).where(
                    EmbeddingIndex.table_name == spec.table_name,
                    EmbeddingIndex.model_name == spec.model_name,
                    EmbeddingIndex.model_version == spec.model_version,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = EmbeddingIndex(
                id=uuid.uuid4(),
                table_name=spec.table_name,
                model_name=spec.model_name,
                model_version=spec.model_version,
                dimensions=spec.dimensions,
                distance_metric=distance_metric,
                active=active,
            )
            session.add(row)
        else:
            row.distance_metric = distance_metric
            row.active = active
        if active:
            await self._deactivate_others(
                session, model_name=spec.model_name, keep_id=row.id
            )
        await session.flush()
        log.info(
            "embedding.registry.register",
            extra={
                "table_name": spec.table_name,
                "model_name": spec.model_name,
                "active": active,
            },
        )
        return _to_info(row)

    async def _deactivate_others(
        self, session: AsyncSession, *, model_name: str, keep_id: uuid.UUID
    ) -> None:
        await session.execute(
            update(EmbeddingIndex)
            .where(EmbeddingIndex.model_name == model_name)
            .where(EmbeddingIndex.id != keep_id)
            .values(active=False)
        )

    async def list_all(self, session: AsyncSession) -> list[EmbeddingIndexInfo]:
        rows = (await session.execute(select(EmbeddingIndex))).scalars().all()
        return [_to_info(row) for row in rows]

    async def active(self, session: AsyncSession) -> list[EmbeddingIndexInfo]:
        rows = (
            await session.execute(
                select(EmbeddingIndex).where(EmbeddingIndex.active.is_(True))
            )
        ).scalars().all()
        return [_to_info(row) for row in rows]

    async def store_embedding(
        self,
        session: AsyncSession,
        *,
        chunk_id: uuid.UUID,
        vector: list[float],
        spec: EmbeddingSpec,
    ) -> uuid.UUID:
        table = _SUPPORTED_TABLES.get(spec.table_name)
        if table is None:
            raise ValueError(f"no ORM table registered for {spec.table_name}")
        if len(vector) != spec.dimensions:
            raise ValueError(
                f"dimension mismatch: expected {spec.dimensions}, got {len(vector)}"
            )
        row = table(
            id=uuid.uuid4(),
            chunk_id=chunk_id,
            model_name=spec.model_name,
            model_version=spec.model_version,
            dimensions=spec.dimensions,
            embedding=vector,
        )
        session.add(row)
        await session.flush()
        row_id: uuid.UUID = row.id
        return row_id


def _to_info(row: EmbeddingIndex) -> EmbeddingIndexInfo:
    return EmbeddingIndexInfo(
        id=row.id,
        table_name=row.table_name,
        model_name=row.model_name,
        model_version=row.model_version,
        dimensions=row.dimensions,
        distance_metric=row.distance_metric,
        active=row.active,
    )


async def sync_from_provider(
    session: AsyncSession, provider: EmbeddingProvider
) -> EmbeddingIndexInfo:
    """Convenience: register the provider's spec as the active table."""
    registry = EmbeddingRegistry()
    return await registry.register(session, provider.spec)

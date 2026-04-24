#!/usr/bin/env python3
"""Rebuild embeddings for existing chunks. Plan §4.3 / §22.3.

Iterates chunks for a given profile and embedding model, writes embeddings
into the matching per-model physical table, and reports how many were
created. Idempotent: re-running only fills gaps.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select  # noqa: E402

from siqueira_memo.config import get_settings  # noqa: E402
from siqueira_memo.db import get_session_factory  # noqa: E402
from siqueira_memo.services.embedding_registry import EmbeddingRegistry  # noqa: E402
from siqueira_memo.services.embedding_service import build_embedding_provider  # noqa: E402
from siqueira_memo.workers.jobs import embed_chunks_for_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="rebuild_embeddings")
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "message", "tool_output", "artifact", "session_summary"],
    )
    parser.add_argument("--profile-id", default=None)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    from siqueira_memo.models import Chunk

    settings = get_settings()
    factory = get_session_factory(settings)
    provider = build_embedding_provider(settings)
    registry = EmbeddingRegistry()
    profile_id = args.profile_id or settings.derive_profile_id()

    async with factory() as session:
        await registry.register(session, provider.spec)

        stmt = select(Chunk.source_type, Chunk.source_id).where(
            Chunk.profile_id == profile_id
        )
        if args.source != "all":
            stmt = stmt.where(Chunk.source_type == args.source)
        distinct_sources = (await session.execute(stmt.distinct())).all()

        total = 0
        for source_type, source_id in distinct_sources:
            embedded = await embed_chunks_for_source(
                session,
                profile_id=profile_id,
                source_type=source_type,
                source_id=source_id,
                provider=provider,
                registry=registry,
            )
            total += embedded
        await session.commit()
        print(
            f"rebuilt {total} embedding(s) across {len(distinct_sources)} source(s) "
            f"using {provider.spec.model_name}@{provider.spec.model_version}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(_main(parse_args())))

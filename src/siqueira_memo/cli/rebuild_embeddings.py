"""``siqueira-memo-rebuild-embeddings`` entrypoint. Plan §4.3 / §22.3."""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from siqueira_memo.config import get_settings
from siqueira_memo.db import get_session_factory
from siqueira_memo.models import Chunk
from siqueira_memo.services.embedding_registry import EmbeddingRegistry
from siqueira_memo.services.embedding_service import build_embedding_provider
from siqueira_memo.workers.jobs import embed_chunks_for_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siqueira-memo-rebuild-embeddings")
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "message", "tool_output", "artifact", "session_summary"],
    )
    parser.add_argument("--profile-id", default=None)
    return parser


async def _run(args: argparse.Namespace) -> int:
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
            total += await embed_chunks_for_source(
                session,
                profile_id=profile_id,
                source_type=source_type,
                source_id=source_id,
                provider=provider,
                registry=registry,
            )
        await session.commit()
        print(
            f"rebuilt {total} embedding(s) across {len(distinct_sources)} source(s) "
            f"using {provider.spec.model_name}@{provider.spec.model_version}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

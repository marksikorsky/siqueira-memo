"""``siqueira-memo-import-hermes`` entrypoint. Plan §33.7."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from siqueira_memo.config import get_settings
from siqueira_memo.db import get_session_factory
from siqueira_memo.services.ingest_service import IngestService
from siqueira_memo.services.session_importer import (
    SessionImporter,
    iter_jsonl,
    iter_sqlite,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siqueira-memo-import-hermes")
    parser.add_argument("path", type=Path)
    parser.add_argument("--format", choices=["jsonl", "sqlite"], required=True)
    parser.add_argument("--profile-id", default=None)
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    factory = get_session_factory(settings)
    ingest = IngestService(profile_id=args.profile_id or settings.derive_profile_id())
    importer = SessionImporter(ingest=ingest)
    rows = iter_jsonl(args.path) if args.format == "jsonl" else iter_sqlite(args.path)
    async with factory() as session:
        result = await importer.import_iter(session, rows)
        await session.commit()
    print(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

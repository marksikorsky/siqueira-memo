#!/usr/bin/env python3
"""Import Hermes session transcripts into Siqueira Memo. Plan §33.7."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from siqueira_memo.config import get_settings  # noqa: E402
from siqueira_memo.db import get_session_factory  # noqa: E402
from siqueira_memo.services.ingest_service import IngestService  # noqa: E402
from siqueira_memo.services.session_importer import (  # noqa: E402
    SessionImporter,
    iter_jsonl,
    iter_sqlite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--format", choices=["jsonl", "sqlite"], required=True)
    parser.add_argument("--profile-id", default=None)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    settings = get_settings()
    factory = get_session_factory(settings)
    ingest = IngestService(profile_id=args.profile_id or settings.derive_profile_id())
    importer = SessionImporter(ingest=ingest)

    rows_iter = iter_jsonl(args.path) if args.format == "jsonl" else iter_sqlite(args.path)

    async with factory() as session:
        result = await importer.import_iter(session, rows_iter)
        await session.commit()
    print(result)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(_main(args))

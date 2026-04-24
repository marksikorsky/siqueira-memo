#!/usr/bin/env python3
"""Export project/topic memory as Markdown. Plan §9.2 Task 9.3."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from siqueira_memo.config import get_settings  # noqa: E402
from siqueira_memo.db import get_session_factory  # noqa: E402
from siqueira_memo.services.markdown_export import ExportFilter, export_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="export_markdown")
    parser.add_argument("--project", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--output", default="-")
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    factory = get_session_factory(settings)
    filt = ExportFilter(
        profile_id=args.profile_id or settings.derive_profile_id(),
        project=args.project,
        topic=args.topic,
    )
    async with factory() as session:
        body = await export_markdown(session, filt)
    if args.output == "-":
        print(body)
    else:
        Path(args.output).write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(_main(parse_args())))

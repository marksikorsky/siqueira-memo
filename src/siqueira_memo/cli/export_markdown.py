"""``siqueira-memo-export-markdown`` entrypoint. Plan §9.2 Task 9.3."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from siqueira_memo.config import get_settings
from siqueira_memo.db import get_session_factory
from siqueira_memo.services.markdown_export import ExportFilter, export_markdown


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siqueira-memo-export-markdown")
    parser.add_argument("--project", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--output", default="-")
    return parser


async def _run(args: argparse.Namespace) -> int:
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

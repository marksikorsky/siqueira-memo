"""Provider-specific CLI shim.

Exposes a handful of operational commands invokable via
``hermes memory siqueira-memo <cmd>`` where Hermes chains this into its own
plugin-cli loader.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the service package is importable when this CLI is invoked directly.
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[3] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _cmd_status() -> int:
    from siqueira_memo.config import get_settings

    settings = get_settings()
    print(f"profile_id: {settings.derive_profile_id()}")
    print(f"database_url: {settings.database_url.split('://', 1)[0]}://...")
    print(f"embedding_provider: {settings.embedding_provider}")
    print(f"queue_backend: {settings.queue_backend}")
    return 0


def _cmd_migrate() -> int:
    import subprocess

    return subprocess.call(["alembic", "upgrade", "head"])


def _cmd_drain() -> int:
    from siqueira_memo.workers.queue import get_default_queue

    queue = get_default_queue()
    drained = asyncio.run(queue.drain())
    print(f"drained {drained} job(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="siqueira-memo-cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("migrate")
    sub.add_parser("drain")
    args = parser.parse_args(argv)
    if args.cmd == "status":
        return _cmd_status()
    if args.cmd == "migrate":
        return _cmd_migrate()
    if args.cmd == "drain":
        return _cmd_drain()
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

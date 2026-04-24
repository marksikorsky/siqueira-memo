#!/usr/bin/env python3
"""Dev-only helper: drop and recreate Siqueira Memo schema."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from siqueira_memo.db import (  # noqa: E402
    create_all_for_tests,
    dispose_engines,
    drop_all_for_tests,
)


async def main() -> None:
    await drop_all_for_tests()
    await create_all_for_tests()
    await dispose_engines()
    print("schema reset")


if __name__ == "__main__":
    asyncio.run(main())

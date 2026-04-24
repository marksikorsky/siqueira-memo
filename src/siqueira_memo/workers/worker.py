"""Standalone worker entrypoint.

For now it supports only the in-memory queue backend: it loops, drains the
queue, and sleeps briefly between iterations. When Redis/Dramatiq is
available the deployment runs ``dramatiq`` instead and this loop acts as the
local development fallback.
"""

from __future__ import annotations

import asyncio

from siqueira_memo.config import get_settings
from siqueira_memo.logging import configure_logging, get_logger
from siqueira_memo.workers.jobs import register_default_handlers
from siqueira_memo.workers.queue import get_default_queue

log = get_logger(__name__)


async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    queue = get_default_queue()
    register_default_handlers(queue)
    log.info("worker.started", extra={"queue_backend": settings.queue_backend})
    try:
        while True:
            drained = await queue.drain()
            if drained == 0:
                await asyncio.sleep(0.5)
    except asyncio.CancelledError:  # pragma: no cover
        log.info("worker.cancelled")


def main() -> None:  # pragma: no cover
    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()

"""Async job queue abstraction.

Two backends are supported:

* ``memory`` — jobs are appended to an in-process list and drained synchronously
  by ``drain()``. Tests use this because Hermes ``sync_turn`` must stay
  non-blocking but the tests still need the downstream work to run.
* ``redis`` — publishes tasks to a Dramatiq broker. Only instantiated when the
  runtime has redis/dramatiq available; otherwise the queue falls back to
  memory mode with a warning.

Every enqueue is idempotent at the caller level via ``dedup_key``: the
in-memory queue ignores duplicate dedup keys while they remain pending.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from siqueira_memo.logging import get_logger

log = get_logger(__name__)

HandlerFn = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class Job:
    name: str
    payload: dict[str, Any]
    dedup_key: str | None = None
    priority: int = 5
    meta: dict[str, Any] = field(default_factory=dict)


class JobQueue(Protocol):
    def enqueue(self, job: Job) -> None:
        ...

    def register(self, name: str, handler: HandlerFn) -> None:
        ...

    async def drain(self) -> int:
        ...


class MemoryJobQueue:
    """In-process queue. Delivery is FIFO and execution happens in ``drain``.

    The queue is thread-safe enough for the typical test harness (single
    process, single event loop). Use the Redis backend for concurrent workers.
    """

    def __init__(self) -> None:
        self._queue: list[Job] = []
        self._dedup: set[str] = set()
        self._handlers: dict[str, HandlerFn] = {}

    def enqueue(self, job: Job) -> None:
        if job.dedup_key and job.dedup_key in self._dedup:
            log.debug(
                "queue.dedup",
                extra={"job_name": job.name, "dedup_key": job.dedup_key},
            )
            return
        if job.dedup_key:
            self._dedup.add(job.dedup_key)
        self._queue.append(job)

    def register(self, name: str, handler: HandlerFn) -> None:
        self._handlers[name] = handler

    async def drain(self) -> int:
        count = 0
        # Snapshot so handlers may enqueue follow-up jobs without infinite loops.
        while self._queue:
            job = self._queue.pop(0)
            if job.dedup_key:
                self._dedup.discard(job.dedup_key)
            handler = self._handlers.get(job.name)
            if handler is None:
                log.warning("queue.no_handler", extra={"job_name": job.name})
                continue
            try:
                await handler(job.payload)
                count += 1
            except Exception:
                log.exception("queue.job_failed", extra={"job_name": job.name})
        return count

    def pending(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()
        self._dedup.clear()


def build_queue(backend: str) -> JobQueue:
    if backend == "memory":
        return MemoryJobQueue()
    if backend == "redis":
        # The Redis/Dramatiq adapter is optional. It lives under a
        # conditionally-imported module so mypy would otherwise flag the
        # ``import-not-found`` permanently. When present, the adapter returns
        # a ``JobQueue`` duck-type compatible with the protocol.
        try:
            module = importlib.import_module("siqueira_memo.workers.redis_queue")
        except ModuleNotFoundError:  # pragma: no cover
            log.warning("queue.redis_unavailable_fallback_memory")
            return MemoryJobQueue()
        queue_cls = module.RedisJobQueue
        queue: JobQueue = queue_cls()
        return queue
    raise ValueError(f"unsupported queue backend: {backend}")


_default_queue: JobQueue | None = None


def get_default_queue() -> JobQueue:
    global _default_queue
    if _default_queue is None:
        from siqueira_memo.config import get_settings

        _default_queue = build_queue(get_settings().queue_backend)
    return _default_queue


def set_default_queue(queue: JobQueue | None) -> None:
    global _default_queue
    _default_queue = queue

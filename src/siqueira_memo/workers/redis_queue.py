"""Redis-backed job queue for cross-process Siqueira workers.

This is intentionally small: Hermes/API processes enqueue JSON-safe ``Job``
objects into a Redis list, and ``siqueira-memo-worker`` drains that list with
registered async handlers. It preserves the same ``JobQueue`` protocol as the
in-memory backend, so unit tests can stay hermetic while Docker/prod uses real
cross-process delivery.
"""

from __future__ import annotations

import json
from typing import Any

from siqueira_memo.config import get_settings
from siqueira_memo.logging import get_logger
from siqueira_memo.workers.queue import HandlerFn, Job

log = get_logger(__name__)

_DEFAULT_QUEUE_KEY = "siqueira:jobs"
_DEFAULT_DEDUP_PREFIX = "siqueira:jobs:dedup:"
_DEFAULT_DEDUP_TTL_SECONDS = 3600


class RedisJobQueue:
    """Redis list-backed queue implementing ``JobQueue``.

    The class uses redis-py's synchronous client because enqueue/drain operations
    are tiny and the worker already wraps handler execution in async code. This
    keeps the API surface identical to ``MemoryJobQueue``.
    """

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        queue_key: str = _DEFAULT_QUEUE_KEY,
        dedup_prefix: str = _DEFAULT_DEDUP_PREFIX,
        dedup_ttl_seconds: int = _DEFAULT_DEDUP_TTL_SECONDS,
    ) -> None:
        try:
            import redis
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("redis package is required for RedisJobQueue") from exc

        settings = get_settings()
        self.queue_key = queue_key
        self.dedup_prefix = dedup_prefix
        self.dedup_ttl_seconds = dedup_ttl_seconds
        self._handlers: dict[str, HandlerFn] = {}
        self._redis = redis.Redis.from_url(redis_url or settings.redis_url)
        # Fail fast at construction so build_queue() can fall back cleanly.
        self._redis.ping()

    def enqueue(self, job: Job) -> None:
        if job.dedup_key:
            dedup_key = self._dedup_key(job.dedup_key)
            was_set = self._redis.set(dedup_key, "1", nx=True, ex=self.dedup_ttl_seconds)
            if not was_set:
                log.debug(
                    "queue.redis_dedup",
                    extra={"job_name": job.name, "dedup_key": job.dedup_key},
                )
                return
        self._redis.rpush(self.queue_key, json.dumps(_job_to_dict(job), separators=(",", ":")))

    def register(self, name: str, handler: HandlerFn) -> None:
        self._handlers[name] = handler

    async def drain(self) -> int:
        count = 0
        while True:
            raw = self._redis.lpop(self.queue_key)
            if raw is None:
                return count
            try:
                job = _job_from_raw(raw)
            except Exception:
                log.exception("queue.redis_bad_job")
                continue
            handler = self._handlers.get(job.name)
            if handler is None:
                log.warning("queue.no_handler", extra={"job_name": job.name})
                self._release_dedup(job)
                continue
            try:
                await handler(job.payload)
                count += 1
            except Exception:
                log.exception("queue.job_failed", extra={"job_name": job.name})
            finally:
                self._release_dedup(job)

    def pending(self) -> int:
        return int(self._redis.llen(self.queue_key))

    def clear(self) -> None:
        self._redis.delete(self.queue_key)

    def _dedup_key(self, dedup_key: str) -> str:
        return f"{self.dedup_prefix}{dedup_key}"

    def _release_dedup(self, job: Job) -> None:
        if job.dedup_key:
            self._redis.delete(self._dedup_key(job.dedup_key))


def _job_to_dict(job: Job) -> dict[str, Any]:
    return {
        "name": job.name,
        "payload": job.payload,
        "dedup_key": job.dedup_key,
        "priority": job.priority,
        "meta": job.meta,
    }


def _job_from_raw(raw: bytes | str) -> Job:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    doc = json.loads(raw)
    return Job(
        name=str(doc["name"]),
        payload=dict(doc.get("payload") or {}),
        dedup_key=doc.get("dedup_key"),
        priority=int(doc.get("priority", 5)),
        meta=dict(doc.get("meta") or {}),
    )

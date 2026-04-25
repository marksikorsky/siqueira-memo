"""Small process-local cache for Hermes prefetch warm jobs.

Hermes calls ``queue_prefetch`` after a completed turn and ``prefetch`` before a
later model call in the same provider process. Redis workers cannot share this
in-memory cache across processes, but the handler still performs the recall so
memory-backed dev/test runs and single-process deployments get real warmups.
"""

from __future__ import annotations

from typing import Any

_CACHE: dict[str, dict[str, Any]] = {}


def cache_key(profile_id: str, session_id: str, query: str) -> str:
    return f"{profile_id}:{session_id}:{query}"


def get_prefetch_cache(profile_id: str, session_id: str, query: str) -> dict[str, Any] | None:
    cached = _CACHE.get(cache_key(profile_id, session_id, query))
    return dict(cached) if cached is not None else None


def set_prefetch_cache(
    profile_id: str,
    session_id: str,
    query: str,
    value: dict[str, Any],
) -> None:
    _CACHE[cache_key(profile_id, session_id, query)] = dict(value)


def clear_prefetch_cache() -> None:
    _CACHE.clear()

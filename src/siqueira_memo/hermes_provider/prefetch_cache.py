"""Prefetch warm-cache shared by Hermes provider and worker jobs.

The local dict covers in-process memory-queue runs. When Redis is configured,
workers also write the shaped context pack to Redis so a separate Hermes/API
process can read the warmed pack before the next model call.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any

from siqueira_memo.config import Settings

_CACHE: dict[str, dict[str, Any]] = {}
_TTL_SECONDS = 900


def cache_key(profile_id: str, session_id: str, query: str) -> str:
    return f"{profile_id}:{session_id}:{query}"


def get_prefetch_cache(
    profile_id: str,
    session_id: str,
    query: str,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    key = cache_key(profile_id, session_id, query)
    cached = _CACHE.get(key)
    if cached is not None:
        return dict(cached)
    cached = _redis_get(key, settings)
    if cached is not None:
        _CACHE[key] = dict(cached)
        return cached
    return None


def set_prefetch_cache(
    profile_id: str,
    session_id: str,
    query: str,
    value: dict[str, Any],
    settings: Settings | None = None,
) -> None:
    key = cache_key(profile_id, session_id, query)
    _CACHE[key] = dict(value)
    _redis_set(key, value, settings)


def clear_prefetch_cache() -> None:
    _CACHE.clear()


def _redis_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"siqueira:prefetch:{digest}"


def _redis_get(key: str, settings: Settings | None) -> dict[str, Any] | None:
    if settings is None or settings.queue_backend != "redis":
        return None
    try:
        redis_mod: Any = importlib.import_module("redis")
        raw: Any = redis_mod.Redis.from_url(settings.redis_url).get(_redis_key(key))
        if raw is None:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _redis_set(key: str, value: dict[str, Any], settings: Settings | None) -> None:
    if settings is None or settings.queue_backend != "redis":
        return
    try:
        redis_mod: Any = importlib.import_module("redis")
        redis_mod.Redis.from_url(settings.redis_url).setex(
            _redis_key(key),
            _TTL_SECONDS,
            json.dumps(value, ensure_ascii=False, default=str),
        )
    except Exception:
        return

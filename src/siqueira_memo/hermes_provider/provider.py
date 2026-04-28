"""``SiqueiraMemoProvider`` — Hermes MemoryProvider plugin. Plan §5 / §32 / §33.

Behavioural contract:

* ``is_available()`` performs no network calls (plan §32.1).
* ``initialize()`` accepts the documented keyword fields (plan §33.6).
* ``sync_turn()`` must be non-blocking (plan §5.5). The provider enqueues a
  persistence task through the in-process or Redis queue and returns
  immediately.
* ``prefetch()`` returns a ``ContextPack`` shaped to Hermes prefetch budgets
  (plan §33.5) and must never emit deep/forensic payloads.
* ``handle_tool_call()`` dispatches the ``siqueira_*`` tools and always returns
  a JSON string — never raises into Hermes.
* Write-side hooks (``on_pre_compress``, ``on_session_end``, ``on_memory_write``,
  ``on_delegation``) persist observations but never mutate live memory without
  going through the extraction lifecycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.db import dispose_engines, get_session_factory
from siqueira_memo.hermes_provider.prefetch_cache import (
    clear_prefetch_cache,
    get_prefetch_cache,
    set_prefetch_cache,
)
from siqueira_memo.hermes_provider.tools import TOOL_NAMES, tool_schemas
from siqueira_memo.logging import get_logger
from siqueira_memo.models.constants import (
    AGENT_CONTEXT_PRIMARY,
)
from siqueira_memo.schemas.memory import (
    CorrectRequest,
    ForgetRequest,
    RememberRequest,
    SourcesRequest,
    TimelineRequest,
)
from siqueira_memo.schemas.recall import RecallRequest
from siqueira_memo.services.context_pack_service import ContextPackShaper
from siqueira_memo.services.deletion_service import DeletionService
from siqueira_memo.services.embedding_service import build_embedding_provider
from siqueira_memo.services.extraction_service import ExtractionService
from siqueira_memo.services.retrieval_service import RetrievalService
from siqueira_memo.workers.queue import Job, JobQueue, get_default_queue

log = get_logger(__name__)


_HERMES_AUX_COMPACTION_PREFIXES = (
    "[CONTEXT COMPACTION",
    "[context compaction",
)


@dataclass
class SiqueiraMemoProvider:
    """Main Hermes memory-provider entrypoint.

    The class is written to be independently importable/testable — it does not
    require a running Hermes instance. Integration with Hermes happens via the
    ``register(ctx)`` function inside ``plugins/memory/siqueira-memo``.
    """

    name: str = "siqueira-memo"
    _settings: Settings | None = None
    _initialised: bool = False
    _session_id: str = ""
    _hermes_home: str | None = None
    _agent_identity: str | None = None
    _agent_context: str = AGENT_CONTEXT_PRIMARY
    _profile_id: str = "default"
    _prefetch_cache: dict[str, Any] = field(default_factory=dict)
    _queue: JobQueue | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Pure, local check. No DB, no network. Plan §32.1."""
        try:
            settings = self._settings or get_settings()
            return bool(settings.database_url)
        except Exception:
            return False

    def initialize(self, session_id: str = "", **kwargs: Any) -> None:
        settings = self._settings or get_settings()
        self._settings = settings
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home") or settings.hermes_home
        self._agent_identity = kwargs.get("agent_identity") or settings.agent_identity
        self._agent_context = kwargs.get("agent_context") or AGENT_CONTEXT_PRIMARY

        # Plan §33.6: derive profile_id.
        if self._agent_identity:
            self._profile_id = self._agent_identity
        elif self._hermes_home:
            normalized = str(Path(self._hermes_home).expanduser().resolve())
            self._profile_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
        else:
            self._profile_id = settings.default_profile_id

        self._queue = get_default_queue()
        self._initialised = True

        log.info(
            "siqueira.provider.initialize",
            extra={
                "session_id": session_id,
                "profile_id": self._profile_id,
                "agent_context": self._agent_context,
            },
        )

    def shutdown(self) -> None:
        # Drain any pending in-memory jobs so sync_turn work does not get lost.
        queue = self._queue
        if queue is not None:
            with suppress(RuntimeError):
                _run(queue.drain())
        with suppress(RuntimeError):
            _run(dispose_engines())
        _RUNNER.close()

    # ------------------------------------------------------------------
    # Tool schemas & dispatch
    # ------------------------------------------------------------------
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return tool_schemas()

    def get_config_schema(self) -> list[dict[str, Any]]:
        """Return Hermes setup/status-compatible field metadata.

        Hermes' ``memory setup/status`` code expects a list of field dicts, not
        a JSON Schema object. The plugin YAML can still expose JSON Schema-like
        metadata for humans, but this runtime hook must match Hermes' provider
        contract.
        """
        return [
            {
                "key": "database_url",
                "description": "Siqueira Postgres/SQLite database URL",
                "env_var": "SIQUEIRA_DATABASE_URL",
                "default": "sqlite+aiosqlite:///./siqueira_memo.db",
            },
            {
                "key": "embedding_provider",
                "description": "Embedding provider",
                "choices": ["mock", "openai", "local"],
                "env_var": "SIQUEIRA_EMBEDDING_PROVIDER",
                "default": "mock",
            },
            {
                "key": "embedding_model",
                "description": "Embedding model name",
                "env_var": "SIQUEIRA_EMBEDDING_MODEL",
                "default": "text-embedding-3-large",
            },
            {
                "key": "queue_backend",
                "description": "Background queue backend",
                "choices": ["memory", "redis"],
                "env_var": "SIQUEIRA_QUEUE_BACKEND",
                "default": "redis",
            },
            {
                "key": "api_token",
                "description": "Siqueira API token",
                "env_var": "SIQUEIRA_API_TOKEN",
                "secret": True,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str | None = None) -> None:
        """Persist operator-supplied configuration.

        Hermes passes the result of its config-management UI through this hook.
        We write a best-effort ``.env`` style file under ``hermes_home`` so
        subsequent starts pick it up; the provider never crashes if the path
        is not writable.
        """
        if not hermes_home:
            return
        target = Path(hermes_home).expanduser() / "siqueira-memo.env"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"SIQUEIRA_{key.upper()}={value}" for key, value in values.items() if value
            ]
            target.write_text("\n".join(lines) + "\n")
        except OSError:  # pragma: no cover
            log.warning("siqueira.provider.save_config_failed", extra={"path": str(target)})

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        if tool_name not in TOOL_NAMES:
            return json.dumps({"ok": False, "error": f"unknown tool {tool_name}"})
        try:
            coro = self._dispatch_tool(tool_name, args, **kwargs)
            return _run(coro)
        except Exception as exc:
            log.exception("siqueira.tool.failed", extra={"tool_name": tool_name})
            return json.dumps({"ok": False, "error": str(exc), "tool": tool_name})

    async def _dispatch_tool(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        assert self._settings is not None
        factory = get_session_factory(self._settings)
        body: dict[str, Any]
        async with factory() as session:
            try:
                if tool_name == "siqueira_memory_recall":
                    recall_payload = RecallRequest(
                        profile_id=self._profile_id,
                        query=args.get("query", ""),
                        project=args.get("project"),
                        topic=args.get("topic"),
                        mode=args.get("mode", "balanced"),
                        limit=int(args.get("limit", 15)),
                        include_sources=bool(args.get("include_sources", True)),
                        allow_secret_recall=bool(args.get("allow_secret_recall", False)),
                        session_id=self._session_id or None,
                    )
                    embedding = build_embedding_provider(self._settings)
                    retrieval_svc = RetrievalService(
                        profile_id=self._profile_id, embedding_provider=embedding
                    )
                    retrieval_result = await retrieval_svc.recall(session, recall_payload)
                    pack = (
                        ContextPackShaper(self._settings).shape_for_prefetch(
                            retrieval_result.context_pack, recall_payload.mode
                        )
                        if recall_payload.mode in {"fast", "balanced"}
                        else retrieval_result.context_pack
                    )
                    body = pack.model_dump(mode="json")

                elif tool_name == "siqueira_memory_remember":
                    remember_req = RememberRequest(
                        profile_id=self._profile_id,
                        session_id=self._session_id or None,
                        kind=args.get("kind", "fact"),
                        subject=args.get("subject"),
                        predicate=args.get("predicate"),
                        object=args.get("object"),
                        statement=args.get("statement", ""),
                        project=args.get("project"),
                        topic=args.get("topic"),
                        rationale=args.get("rationale"),
                        confidence=float(args.get("confidence", 0.9)),
                        source_event_ids=[uuid.UUID(x) for x in args.get("source_event_ids", [])],
                    )
                    remember_result = await ExtractionService(
                        profile_id=self._profile_id
                    ).remember(session, remember_req)
                    body = remember_result.model_dump(mode="json")

                elif tool_name == "siqueira_memory_correct":
                    replacement = args.get("replacement")
                    replacement_req: RememberRequest | None = None
                    if replacement:
                        replacement_req = RememberRequest.model_validate(
                            {**replacement, "profile_id": self._profile_id}
                        )
                    correct_req = CorrectRequest(
                        profile_id=self._profile_id,
                        session_id=self._session_id or None,
                        target_type=args.get("target_type", "fact"),
                        target_id=(
                            uuid.UUID(args["target_id"]) if args.get("target_id") else None
                        ),
                        correction_text=args.get("correction_text", ""),
                        replacement=replacement_req,
                    )
                    correct_result = await ExtractionService(
                        profile_id=self._profile_id
                    ).apply_correction(session, correct_req)
                    body = correct_result.model_dump(mode="json")

                elif tool_name == "siqueira_memory_forget":
                    forget_req = ForgetRequest(
                        profile_id=self._profile_id,
                        target_type=args.get("target_type"),
                        target_id=uuid.UUID(args["target_id"]),
                        mode=args.get("mode", "soft"),
                        reason=args.get("reason"),
                        scrub_raw=bool(args.get("scrub_raw", False)),
                    )
                    forget_result = await DeletionService(
                        profile_id=self._profile_id
                    ).forget(session, forget_req)
                    body = forget_result.model_dump(mode="json")

                elif tool_name == "siqueira_memory_timeline":
                    # The timeline route logic is duplicated in a slim form
                    # here so the tool does not need an HTTP round-trip.
                    from siqueira_memo.api.routes_memory import (
                        timeline as route_timeline,  # noqa: WPS433
                    )

                    timeline_req = TimelineRequest(
                        profile_id=self._profile_id,
                        project=args.get("project"),
                        topic=args.get("topic"),
                        entity=args.get("entity"),
                        since=_parse_dt(args.get("since")),
                        until=_parse_dt(args.get("until")),
                        limit=int(args.get("limit", 50)),
                    )
                    timeline_response = await route_timeline(
                        timeline_req,
                        session=session,
                        profile_id=self._profile_id,
                        _token="internal",
                    )
                    body = timeline_response.model_dump(mode="json")

                elif tool_name == "siqueira_memory_sources":
                    sources_req = SourcesRequest(
                        profile_id=self._profile_id,
                        target_type=args.get("target_type"),
                        target_id=uuid.UUID(args["target_id"]),
                    )
                    from siqueira_memo.api.routes_memory import (
                        sources as route_sources,  # noqa: WPS433
                    )

                    sources_response = await route_sources(
                        sources_req,
                        session=session,
                        profile_id=self._profile_id,
                        _token="internal",
                    )
                    body = sources_response.model_dump(mode="json")
                else:
                    body = {"ok": False, "error": f"unimplemented tool {tool_name}"}

                await session.commit()
            except BaseException:
                await session.rollback()
                raise
        return json.dumps({"ok": True, "tool": tool_name, "result": body}, default=str)

    # ------------------------------------------------------------------
    # Prompt block (plan §5.2)
    # ------------------------------------------------------------------
    def system_prompt_block(self) -> str:
        path = Path(__file__).resolve().parents[0] / "system_prompt.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return (
                "Siqueira Memo is your long-term memory provider. Use "
                "siqueira_memory_recall before relying on old context. Live user "
                "instructions and tool output override memory."
            )

    # ------------------------------------------------------------------
    # Prefetch / sync / lifecycle hooks
    # ------------------------------------------------------------------
    def prefetch(self, query: str, *, session_id: str = "") -> dict[str, Any]:
        """Return cached/prewarmed prefetch context. Plan §5.4 / §33.5."""
        cached: dict[str, Any] | None = self._prefetch_cache.get(
            self._cache_key(query, session_id)
        )
        if cached is None:
            cached = get_prefetch_cache(
                self._profile_id,
                session_id or self._session_id,
                query,
                self._settings,
            )
        if cached is not None:
            return cached
        return {
            "answer_context": "",
            "decisions": [],
            "facts": [],
            "chunks": [],
            "source_snippets": [],
            "confidence": "low",
            "warnings": ["prefetch cache cold — call siqueira_memory_recall explicitly"],
        }

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Schedule asynchronous prefetch of the provided query for the next turn."""
        if not self._queue:
            return
        self._queue.enqueue(
            Job(
                name="siqueira.prefetch_warm",
                payload={
                    "profile_id": self._profile_id,
                    "session_id": session_id or self._session_id,
                    "query": query,
                },
                dedup_key=f"prefetch:{self._profile_id}:{session_id}:{hash(query)}",
            )
        )

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Non-blocking turn ingestion. Plan §5.5."""
        if not self._queue:
            return
        payload = {
            "profile_id": self._profile_id,
            "session_id": session_id or self._session_id,
            "agent_context": self._agent_context,
            "user_content": user_content,
            "assistant_content": assistant_content,
        }
        self._queue.enqueue(
            Job(name="siqueira.sync_turn", payload=payload, dedup_key=None)
        )

    def on_pre_compress(self, messages: list[dict[str, Any]] | Any) -> str:
        """Enqueue high-priority extraction before compression. Plan §5.6 / §32.6."""
        if not self._queue:
            return ""
        try:
            message_count = len(messages) if hasattr(messages, "__len__") else 0
        except TypeError:
            message_count = 0
        self._queue.enqueue(
            Job(
                name="siqueira.pre_compress_extract",
                payload={
                    "profile_id": self._profile_id,
                    "session_id": self._session_id,
                    "message_count": message_count,
                    "transcript_tail": _compact_messages_for_memory(messages),
                },
                priority=1,
            )
        )
        return ""

    def on_session_end(self, messages: list[dict[str, Any]] | Any) -> None:
        if not self._queue:
            return
        self._queue.enqueue(
            Job(
                name="siqueira.session_end_summarise",
                payload={
                    "profile_id": self._profile_id,
                    "session_id": self._session_id,
                    "transcript_tail": _compact_messages_for_memory(messages),
                },
                priority=2,
            )
        )

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror Hermes built-in memory writes. Plan §33.10."""
        if not self._queue:
            return
        self._queue.enqueue(
            Job(
                name="siqueira.builtin_memory_mirror",
                payload={
                    "profile_id": self._profile_id,
                    "session_id": self._session_id,
                    "action": action,
                    "target": target,
                    "content": content,
                },
                dedup_key=None,
            )
        )

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Record parent-observed subagent completions. Plan §33.8."""
        if not self._queue:
            return
        self._queue.enqueue(
            Job(
                name="siqueira.delegation_observed",
                payload={
                    "profile_id": self._profile_id,
                    "parent_session_id": self._session_id,
                    "child_session_id": child_session_id,
                    "task": task,
                    "result": result,
                    "toolsets": list(kwargs.get("toolsets", [])),
                    "model": kwargs.get("model"),
                },
                priority=2,
            )
        )

    def on_turn_start(self, turn_number: int, message: dict[str, Any], **kwargs: Any) -> None:
        """Best-effort hook; Hermes build may or may not invoke it (plan §33.13)."""
        return None

    # ------------------------------------------------------------------
    # Observation / compaction detection
    # ------------------------------------------------------------------
    @staticmethod
    def is_hermes_auxiliary_compaction(text: str) -> bool:
        """Detect Hermes ContextCompressor summaries. Plan §33.2."""
        if not text:
            return False
        stripped = text.lstrip()
        return any(stripped.startswith(prefix) for prefix in _HERMES_AUX_COMPACTION_PREFIXES)

    # ------------------------------------------------------------------
    # Helpers used by workers and tests
    # ------------------------------------------------------------------
    def _cache_key(self, query: str, session_id: str) -> str:
        return f"{self._profile_id}:{session_id or self._session_id}:{query}"

    def set_prefetch_cache(self, query: str, session_id: str, value: dict[str, Any]) -> None:
        self._prefetch_cache[self._cache_key(query, session_id)] = value
        set_prefetch_cache(
            self._profile_id,
            session_id or self._session_id,
            query,
            value,
            self._settings,
        )

    def clear_prefetch_cache(self) -> None:
        self._prefetch_cache.clear()
        clear_prefetch_cache()


def _parse_dt(value: Any) -> Any:
    from datetime import datetime

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _compact_messages_for_memory(
    messages: list[dict[str, Any]] | Any,
    *,
    limit: int = 80,
    content_limit: int = 4000,
) -> list[dict[str, Any]]:
    """Return a JSON-safe tail of messages for compression/session-end capture.

    Hermes compaction can interrupt a long, tool-heavy session before the usual
    completed-turn path has persisted enough context. Keep this small and let
    the worker perform redaction before durable storage.
    """
    if not isinstance(messages, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in messages[-limit:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown")
        content = item.get("content")
        if content is None:
            content_text = ""
        elif isinstance(content, str):
            content_text = content
        else:
            try:
                import json

                content_text = json.dumps(content, ensure_ascii=False)
            except TypeError:
                content_text = str(content)
        if not content_text and item.get("tool_calls"):
            try:
                import json

                content_text = json.dumps(item.get("tool_calls"), ensure_ascii=False)
            except TypeError:
                content_text = str(item.get("tool_calls"))
        compact.append({"role": role, "content": content_text[:content_limit]})
    return compact


class _ProviderLoopRunner:
    """Run provider coroutines on one long-lived event loop.

    Hermes calls MemoryProvider tool hooks through a synchronous interface while
    the agent itself is usually already inside an event loop. Creating a fresh
    thread+loop per tool call leaves SQLAlchemy's asyncpg pool holding
    connections bound to dead/different loops, which can surface as asyncpg's
    ``another operation is in progress`` error on later recalls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = False

    def run(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            if self._closed:
                self._closed = False
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="siqueira-memo-provider-loop",
                daemon=True,
            )
            self._thread.start()
        self._ready.wait(timeout=5)
        if self._loop is None:  # pragma: no cover - catastrophic startup failure
            raise RuntimeError("Siqueira provider event loop did not start")
        return self._loop

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def close(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
            self._closed = True
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)


_RUNNER = _ProviderLoopRunner()


def _run(coro: Any) -> str:
    """Run a provider coroutine through one stable background event loop.

    The stable loop keeps asyncpg/SQLAlchemy pooled connections on the same
    event loop across consecutive Hermes tool calls and avoids overlapping work
    on loop-bound driver state.
    """
    result = _RUNNER.run(coro)
    return str(result)

"""Job handlers registered against the queue.

The in-memory queue is sufficient for unit tests and single-process dev runs.
Production should swap to the Redis queue; job payloads must stay JSON-safe so
both backends behave identically.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from siqueira_memo.config import Settings, get_settings
from siqueira_memo.db import get_session_factory
from siqueira_memo.hermes_provider.prefetch_cache import set_prefetch_cache
from siqueira_memo.logging import get_logger
from siqueira_memo.models import Chunk, MemoryEvent, Message, SessionSummary, ToolEvent
from siqueira_memo.models.constants import (
    CHUNK_SOURCE_MESSAGE,
    CHUNK_SOURCE_TOOL_OUTPUT,
    MESSAGE_SOURCE_SYNC_TURN,
    ROLE_ASSISTANT,
    ROLE_USER,
)
from siqueira_memo.schemas.ingest import (
    BuiltinMemoryMirrorIn,
    DelegationObservationIn,
    GenericEventIn,
    MessageIngestIn,
)
from siqueira_memo.schemas.memory import RememberRequest
from siqueira_memo.schemas.recall import RecallRequest
from siqueira_memo.services.chunking_service import ChunkingService
from siqueira_memo.services.context_pack_service import ContextPackShaper
from siqueira_memo.services.embedding_registry import EmbeddingRegistry
from siqueira_memo.services.embedding_service import (
    EmbeddingProvider,
    build_embedding_provider,
)
from siqueira_memo.services.extraction_gate import default_gate
from siqueira_memo.services.extraction_service import ExtractionService
from siqueira_memo.services.ingest_service import IngestService
from siqueira_memo.services.memory_capture_classifier import (
    MemoryCaptureDecision,
    classify_turn_memory,
)
from siqueira_memo.services.redaction_service import RedactionService
from siqueira_memo.services.retrieval_service import RetrievalService
from siqueira_memo.services.scope_classifier import classify_memory_scope
from siqueira_memo.workers.queue import Job, JobQueue, get_default_queue

_settings_override: Settings | None = None


def set_worker_settings(settings: Settings | None) -> None:
    """Override the settings used by background jobs.

    Tests inject their ephemeral settings here so handlers write to the same
    DB the assertions read from. In production this stays ``None`` and the
    handlers fall back to the cached ``get_settings()``.
    """
    global _settings_override
    _settings_override = settings


def _current_settings() -> Settings:
    return _settings_override or get_settings()

log = get_logger(__name__)


@dataclass
class JobContext:
    settings: Settings
    chunker: ChunkingService

    @classmethod
    def default(cls) -> JobContext:
        return cls(settings=_current_settings(), chunker=ChunkingService())


async def chunk_message_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    message_id = uuid.UUID(payload["message_id"])
    profile_id = payload["profile_id"]
    async with factory() as session:
        message = (
            await session.execute(
                select(Message).where(
                    Message.id == message_id, Message.profile_id == profile_id
                )
            )
        ).scalar_one_or_none()
        if message is None:
            log.debug("chunk_message.missing", extra={"message_id": str(message_id)})
            return
        if message.sensitivity == "sensitive":
            log.debug("chunk_message.skip_sensitive")
            return
        chunks = ctx.chunker.chunk_message(
            message.content_redacted,
            source_id=str(message.id),
            sensitivity=message.sensitivity,
        )
        for chunk in chunks:
            session.add(
                Chunk(
                    profile_id=profile_id,
                    source_type=CHUNK_SOURCE_MESSAGE,
                    source_id=message.id,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    tokenizer_name=chunk.tokenizer_name,
                    tokenizer_version=chunk.tokenizer_version,
                    project=message.project,
                    topic=message.topic,
                    entities=list(message.entities or []),
                    sensitivity=message.sensitivity,
                    extra_metadata={**chunk.extra_metadata, **(message.extra_metadata or {})},
                )
            )
        await session.commit()
    get_default_queue().enqueue(
        Job(
            name="embed_chunks",
            payload={
                "profile_id": profile_id,
                "source_type": CHUNK_SOURCE_MESSAGE,
                "source_id": str(message_id),
            },
            dedup_key=f"embed:message:{message_id}",
        )
    )
    log.info(
        "chunk_message.done",
        extra={"message_id": str(message_id), "chunks": len(chunks)},
    )


async def chunk_tool_output_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    tool_event_id = uuid.UUID(payload["tool_event_id"])
    profile_id = payload["profile_id"]
    async with factory() as session:
        tool_event = (
            await session.execute(
                select(ToolEvent).where(
                    ToolEvent.id == tool_event_id, ToolEvent.profile_id == profile_id
                )
            )
        ).scalar_one_or_none()
        if tool_event is None:
            return
        text = tool_event.output_redacted or tool_event.output_summary
        if not text:
            return
        chunks = ctx.chunker.chunk_log(text, source_id=str(tool_event.id))
        for chunk in chunks:
            session.add(
                Chunk(
                    profile_id=profile_id,
                    source_type=CHUNK_SOURCE_TOOL_OUTPUT,
                    source_id=tool_event.id,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    tokenizer_name=chunk.tokenizer_name,
                    tokenizer_version=chunk.tokenizer_version,
                    sensitivity=tool_event.sensitivity,
                    extra_metadata={**chunk.extra_metadata},
                )
            )
        await session.commit()


async def extraction_gate_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    message_id = uuid.UUID(payload["message_id"])
    profile_id = payload["profile_id"]
    async with factory() as session:
        message = (
            await session.execute(
                select(Message).where(
                    Message.id == message_id, Message.profile_id == profile_id
                )
            )
        ).scalar_one_or_none()
        if message is None:
            return
        result = default_gate.classify(message.content_redacted, role=message.role)
        message.extra_metadata = {
            **(message.extra_metadata or {}),
            "gate_labels": list(result.labels),
            "gate_confidence": result.confidence,
        }
        await session.commit()


async def embed_chunks_handler(payload: dict[str, Any]) -> None:
    """Embed chunks for a single source (message/tool_event/artifact).

    Payload keys:
    - ``profile_id`` (required)
    - ``source_type`` (required, e.g. "message")
    - ``source_id`` (required, UUID string)
    """
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    profile_id = payload["profile_id"]
    source_type = payload["source_type"]
    source_id = uuid.UUID(payload["source_id"])
    async with factory() as session:
        provider = build_embedding_provider(ctx.settings)
        registry = EmbeddingRegistry()
        spec = provider.spec
        await registry.register(session, spec)
        embedded = await embed_chunks_for_source(
            session,
            profile_id=profile_id,
            source_type=source_type,
            source_id=source_id,
            provider=provider,
            registry=registry,
        )
        await session.commit()
    log.info(
        "embed_chunks.done",
        extra={
            "profile_id": profile_id,
            "source_id": str(source_id),
            "embedded": embedded,
        },
    )


async def embed_chunks_for_source(
    session: AsyncSession,
    *,
    profile_id: str,
    source_type: str,
    source_id: uuid.UUID,
    provider: EmbeddingProvider,
    registry: EmbeddingRegistry,
) -> int:
    """Embed every unembedded chunk matching ``(profile_id, source_type, source_id)``.

    Returns the number of new embeddings written. Idempotent: relies on the
    per-embedding table's unique ``(chunk_id, model_version)`` constraint to
    skip chunks that already have a current-version embedding.
    """
    from sqlalchemy.exc import IntegrityError

    rows = (
        await session.execute(
            select(Chunk).where(
                Chunk.profile_id == profile_id,
                Chunk.source_type == source_type,
                Chunk.source_id == source_id,
            )
        )
    ).scalars().all()
    embedded = 0
    for chunk in rows:
        if chunk.sensitivity == "sensitive":
            continue
        try:
            async with session.begin_nested():
                vec = provider.embed(chunk.chunk_text)
                await registry.store_embedding(
                    session, chunk_id=chunk.id, vector=vec, spec=provider.spec
                )
                embedded += 1
        except IntegrityError:
            # Already embedded for this model version.
            continue
    return embedded


async def prefetch_warm_handler(payload: dict[str, Any]) -> None:
    """Run a balanced recall and cache the prompt-safe context pack."""
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    profile_id = str(payload["profile_id"])
    session_id = str(payload.get("session_id") or "")
    query = str(payload.get("query") or "")
    if not query.strip():
        return
    async with factory() as session:
        retrieval = RetrievalService(
            profile_id=profile_id,
            embedding_provider=build_embedding_provider(ctx.settings),
        )
        result = await retrieval.recall(
            session,
            RecallRequest(
                profile_id=profile_id,
                session_id=session_id or None,
                query=query,
                mode="balanced",
                limit=15,
                include_sources=True,
            ),
        )
        shaped = ContextPackShaper(ctx.settings).shape_for_prefetch(
            result.context_pack, "balanced"
        )
        await session.commit()
    set_prefetch_cache(
        profile_id,
        session_id,
        query,
        shaped.model_dump(mode="json"),
        ctx.settings,
    )


async def builtin_memory_mirror_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    profile_id = str(payload["profile_id"])
    async with factory() as session:
        await IngestService(profile_id=profile_id).ingest_builtin_memory_mirror(
            session,
            BuiltinMemoryMirrorIn(
                profile_id=profile_id,
                session_id=str(payload.get("session_id") or ""),
                action=payload.get("action", "add"),
                target=payload.get("target", "memory"),
                content=payload.get("content"),
                selector=payload.get("selector"),
            ),
        )
        await session.commit()


async def delegation_observed_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    profile_id = str(payload["profile_id"])
    async with factory() as session:
        await IngestService(profile_id=profile_id).ingest_delegation(
            session,
            DelegationObservationIn(
                profile_id=profile_id,
                parent_session_id=str(payload.get("parent_session_id") or ""),
                child_session_id=payload.get("child_session_id"),
                task=str(payload.get("task") or ""),
                result=str(payload.get("result") or ""),
                toolsets=list(payload.get("toolsets") or []),
                model=payload.get("model"),
            ),
        )
        await session.commit()


def _redact_transcript_tail(raw_tail: Any) -> list[dict[str, str]]:
    if not isinstance(raw_tail, list):
        return []
    redactor = RedactionService()
    out: list[dict[str, str]] = []
    for item in raw_tail:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown")
        content = str(item.get("content") or "")
        if not content.strip():
            continue
        out.append({"role": role, "content": redactor.redact(content).redacted})
    return out


def _transcript_preview(transcript_tail: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in transcript_tail
    )


async def pre_compress_extract_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    profile_id = str(payload["profile_id"])
    session_id = str(payload.get("session_id") or "")
    transcript_tail = _redact_transcript_tail(payload.get("transcript_tail") or [])
    summary_text = _transcript_preview(transcript_tail)
    async with factory() as session:
        ingest = IngestService(profile_id=profile_id)
        result = await ingest.ingest_event(
            session,
            GenericEventIn(
                profile_id=profile_id,
                session_id=session_id,
                event_type="pre_compress_extract",
                source="hermes_pre_compress",
                actor="hermes",
                agent_context="primary",
                payload={
                    "event_type": "pre_compress_extract",
                    "message_count": payload.get("message_count", 0),
                    "transcript_tail_count": len(transcript_tail),
                    "transcript_tail": transcript_tail,
                },
            ),
        )
        if transcript_tail:
            session.add(
                SessionSummary(
                    profile_id=profile_id,
                    session_id=session_id,
                    summary_short=(
                        f"Compaction transcript tail captured {len(transcript_tail)} messages: "
                        f"{summary_text[:420]}"
                    ),
                    summary_long=summary_text,
                    source_event_ids=[result.event_id],
                    model="heuristic",
                    model_version="compaction-tail-v1",
                    prompt_version="v1",
                )
            )
        await session.commit()


async def session_end_summarise_handler(payload: dict[str, Any]) -> None:
    ctx = JobContext.default()
    factory = get_session_factory(ctx.settings)
    profile_id = str(payload["profile_id"])
    session_id = str(payload.get("session_id") or "")
    async with factory() as session:
        rows = (
            await session.execute(
                select(Message)
                .where(Message.profile_id == profile_id, Message.session_id == session_id)
                .order_by(Message.created_at.asc())
            )
        ).scalars().all()
        transcript_tail = _redact_transcript_tail(payload.get("transcript_tail") or [])
        if rows:
            preview = "\n".join(f"{m.role}: {m.content_redacted}" for m in rows)[-2000:]
        else:
            preview = _transcript_preview(transcript_tail)[-2000:]
        event_id = uuid.uuid4()
        session.add(
            MemoryEvent(
                id=event_id,
                event_type="summary_created",
                source="session_end_summarise",
                actor="siqueira-worker",
                session_id=session_id,
                profile_id=profile_id,
                payload={
                    "event_type": "summary_created",
                    "message_count": len(rows),
                },
            )
        )
        session.add(
            SessionSummary(
                profile_id=profile_id,
                session_id=session_id,
                summary_short=(preview[:500] or "No messages captured for session."),
                summary_long=(preview or "No messages captured for session."),
                source_event_ids=[event_id],
                model="heuristic",
                model_version="aggressive-v1",
                prompt_version="v1",
            )
        )
        await session.commit()


async def sync_turn_handler(payload: dict[str, Any]) -> None:
    """Persist a completed Hermes turn and enqueue aggressive extraction."""
    ctx = JobContext.default()
    if ctx.settings.memory_capture_mode == "off" or not ctx.settings.memory_capture_save_raw_turns:
        return

    factory = get_session_factory(ctx.settings)
    profile_id = str(payload["profile_id"])
    session_id = str(payload.get("session_id") or "")
    agent_context = payload.get("agent_context")
    user_content = str(payload.get("user_content") or "")
    assistant_content = str(payload.get("assistant_content") or "")
    combined = "\n".join(part for part in (user_content, assistant_content) if part.strip())
    scope = classify_memory_scope(combined)
    ingest = IngestService(profile_id=profile_id)
    source_event_ids: list[str] = []
    source_message_ids: list[str] = []

    async with factory() as session:
        if user_content.strip():
            user_result = await ingest.ingest_message(
                session,
                MessageIngestIn(
                    profile_id=profile_id,
                    session_id=session_id,
                    platform="hermes",
                    role=ROLE_USER,
                    content=user_content,
                    source=MESSAGE_SOURCE_SYNC_TURN,
                    project=scope.project,
                    topic=scope.topic,
                    agent_context=agent_context,
                    metadata={
                        "scope_confidence": scope.confidence,
                        "scope_reason": scope.reason,
                    },
                ),
            )
            source_event_ids.append(str(user_result.event_id))
            source_message_ids.append(str(user_result.message_id))
        if assistant_content.strip():
            assistant_result = await ingest.ingest_message(
                session,
                MessageIngestIn(
                    profile_id=profile_id,
                    session_id=session_id,
                    platform="hermes",
                    role=ROLE_ASSISTANT,
                    content=assistant_content,
                    source=MESSAGE_SOURCE_SYNC_TURN,
                    project=scope.project,
                    topic=scope.topic,
                    agent_context=agent_context,
                    metadata={
                        "scope_confidence": scope.confidence,
                        "scope_reason": scope.reason,
                    },
                ),
            )
            source_event_ids.append(str(assistant_result.event_id))
            source_message_ids.append(str(assistant_result.message_id))
        await session.commit()

    if ctx.settings.memory_capture_extract_structured and source_message_ids:
        get_default_queue().enqueue(
            Job(
                name="siqueira.extract_turn_memory",
                payload={
                    "profile_id": profile_id,
                    "session_id": session_id,
                    "user_content": user_content,
                    "assistant_content": assistant_content,
                    "source_event_ids": source_event_ids,
                    "source_message_ids": source_message_ids,
                    "project": scope.project,
                    "topic": scope.topic,
                    "scope_reason": scope.reason,
                    "mode": ctx.settings.memory_capture_mode,
                },
                dedup_key=f"extract-turn:{profile_id}:{session_id}:{':'.join(source_message_ids)}",
            )
        )


async def extract_turn_memory_handler(payload: dict[str, Any]) -> None:
    """Deterministic aggressive v1 structured extraction for a turn."""
    ctx = JobContext.default()
    if ctx.settings.memory_capture_mode == "off":
        return
    profile_id = str(payload["profile_id"])
    session_id = str(payload.get("session_id") or "")
    user_content = str(payload.get("user_content") or "")
    assistant_content = str(payload.get("assistant_content") or "")
    combined = "\n".join(part for part in (user_content, assistant_content) if part.strip())
    source_event_ids = [uuid.UUID(str(x)) for x in payload.get("source_event_ids", [])]
    source_message_ids = [uuid.UUID(str(x)) for x in payload.get("source_message_ids", [])]
    project = payload.get("project")
    topic = payload.get("topic") or "conversation"

    llm_capture = classify_turn_memory(
        ctx.settings,
        user_content=user_content,
        assistant_content=assistant_content,
        default_project=project,
        default_topic=topic,
    )
    if llm_capture is not None:
        if not llm_capture.save:
            factory = get_session_factory(ctx.settings)
            async with factory() as session:
                await IngestService(profile_id=profile_id).ingest_event(
                    session,
                    GenericEventIn(
                        profile_id=profile_id,
                        session_id=session_id,
                        event_type="capture_classifier_skip",
                        source="memory_capture_classifier",
                        actor="siqueira-capture",
                        agent_context="primary",
                        payload={
                            "event_type": "capture_classifier_skip",
                            "reason": llm_capture.rationale,
                            "confidence": llm_capture.confidence,
                            "source_event_ids": [str(x) for x in source_event_ids],
                            "source_message_ids": [str(x) for x in source_message_ids],
                        },
                    ),
                )
                await session.commit()
            return
        capture = llm_capture
        model_provider = "openai-compatible"
        model_name = ctx.settings.memory_capture_llm_model
    else:
        if not _looks_useful_for_structured_memory(combined):
            return
        capture = _heuristic_capture(combined, user_content, assistant_content, project, topic)
        model_provider = "heuristic"
        model_name = "aggressive-v1"

    svc = ExtractionService(
        profile_id=profile_id,
        actor="siqueira-capture",
        model_provider=model_provider,
        model_name=model_name,
    )
    factory = get_session_factory(ctx.settings)
    async with factory() as session:
        if capture.kind == "decision":
            await svc.remember(
                session,
                RememberRequest(
                    profile_id=profile_id,
                    session_id=session_id,
                    kind="decision",
                    statement=capture.statement,
                    project=capture.project,
                    topic=capture.topic or topic,
                    rationale=capture.rationale,
                    confidence=capture.confidence or 0.9,
                    source_event_ids=source_event_ids,
                    source_message_ids=source_message_ids,
                    metadata={
                        "capture_mode": payload.get("mode", "aggressive"),
                        "capture_classifier": model_provider,
                    },
                ),
            )
        else:
            await svc.remember(
                session,
                RememberRequest(
                    profile_id=profile_id,
                    session_id=session_id,
                    kind="fact",
                    subject=capture.subject or "conversation",
                    predicate=capture.predicate or "captured",
                    object=capture.object or _fact_object(capture.statement),
                    statement=capture.statement,
                    project=capture.project,
                    topic=capture.topic or topic,
                    confidence=capture.confidence or 0.82,
                    source_event_ids=source_event_ids,
                    source_message_ids=source_message_ids,
                    metadata={
                        "capture_mode": payload.get("mode", "aggressive"),
                        "capture_classifier": model_provider,
                    },
                ),
            )
        await session.commit()


def _heuristic_capture(
    combined: str,
    user_content: str,
    assistant_content: str,
    project: str | None,
    topic: str | None,
) -> MemoryCaptureDecision:
    if _looks_like_decision(combined):
        return MemoryCaptureDecision(
            save=True,
            kind="decision",
            statement=_decision_statement(user_content, assistant_content),
            project=project,
            topic=topic,
            confidence=0.9,
            rationale="Aggressive turn capture detected decision/preference language.",
        )
    return MemoryCaptureDecision(
        save=True,
        kind="fact",
        statement=_fact_statement(combined),
        subject="conversation",
        predicate="captured",
        object=_fact_object(combined),
        project=project,
        topic=topic,
        confidence=0.82,
        rationale="Aggressive turn capture detected useful factual content.",
    )


def _looks_useful_for_structured_memory(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "запомни",
        "не забывай",
        "решение",
        "мы решили",
        "надо",
        "хочет",
        "предпочитает",
        "memory",
        "siqueira",
        "recall",
        "deploy",
        "auth",
        "tax",
        "crypto",
        # Useful-link/project-analysis turns: Mark often sends a repo/paper and
        # expects the resulting architecture/risk/TODO analysis to become memory.
        "github.com/",
        "arxiv.org/",
        "архитектура",
        "architecture",
        "implementation",
        "roadmap",
        "todo",
        "вердикт",
        "использовать стоит",
        "research layer",
        "source-backed",
        "conflict scan",
        "tailscale",
        "tailnet",
        "таилскейл",
        "tail4e3571",
        ".ts.net",
        "server inventory",
        "linux-сервер",
        "серверы",
        "tailscale ip",
    )
    return len(text.strip()) >= 40 and any(marker in lowered for marker in markers)


def _looks_like_decision(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "решение",
            "мы решили",
            "надо",
            "хочет",
            "предпочитает",
            "should",
            "must",
            "decision",
            "policy",
            "use ",
            "включаем",
            "вердикт",
            "использовать стоит",
            "архитектура",
            "architecture",
            "research layer",
            "should use",
            "must run",
        )
    )


def _decision_statement(user_content: str, assistant_content: str) -> str:
    preferred = assistant_content.strip() or user_content.strip()
    return preferred[:900]


def _fact_statement(text: str) -> str:
    return text.strip().replace("\n", " ")[:900]


def _fact_object(text: str) -> str:
    compact = text.strip().replace("\n", " ")
    return compact[:180]


def register_default_handlers(queue: JobQueue) -> None:
    queue.register("chunk_message", chunk_message_handler)
    queue.register("chunk_tool_output", chunk_tool_output_handler)
    queue.register("extraction_gate", extraction_gate_handler)
    queue.register("embed_chunks", embed_chunks_handler)
    queue.register("siqueira.sync_turn", sync_turn_handler)
    queue.register("siqueira.extract_turn_memory", extract_turn_memory_handler)
    queue.register("siqueira.prefetch_warm", prefetch_warm_handler)
    queue.register("siqueira.builtin_memory_mirror", builtin_memory_mirror_handler)
    queue.register("siqueira.delegation_observed", delegation_observed_handler)
    queue.register("siqueira.pre_compress_extract", pre_compress_extract_handler)
    queue.register("siqueira.session_end_summarise", session_end_summarise_handler)

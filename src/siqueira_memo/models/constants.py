"""Enumerations and constants used across the memory schema.

Plain string constants rather than Python ``Enum`` so the values are stable in
JSON payloads and reasonable as Postgres ``TEXT`` columns.
"""

from __future__ import annotations

from typing import Final

# plan §3.1 event types
EVENT_TYPE_MESSAGE_RECEIVED: Final = "message_received"
EVENT_TYPE_ASSISTANT_MESSAGE_SENT: Final = "assistant_message_sent"
EVENT_TYPE_TOOL_CALLED: Final = "tool_called"
EVENT_TYPE_TOOL_RESULT_RECEIVED: Final = "tool_result_received"
EVENT_TYPE_ARTIFACT_CREATED: Final = "artifact_created"
EVENT_TYPE_ARTIFACT_MODIFIED: Final = "artifact_modified"
EVENT_TYPE_SUMMARY_CREATED: Final = "summary_created"
EVENT_TYPE_FACT_EXTRACTED: Final = "fact_extracted"
EVENT_TYPE_DECISION_RECORDED: Final = "decision_recorded"
EVENT_TYPE_FACT_INVALIDATED: Final = "fact_invalidated"
EVENT_TYPE_DECISION_SUPERSEDED: Final = "decision_superseded"
EVENT_TYPE_MEMORY_DELETED: Final = "memory_deleted"
EVENT_TYPE_USER_CORRECTION: Final = "user_correction_received"
EVENT_TYPE_HINDSIGHT_IMPORTED: Final = "hindsight_imported"
EVENT_TYPE_DELEGATION_OBSERVED: Final = "delegation_observed"
EVENT_TYPE_BUILTIN_MEMORY_MIRROR: Final = "builtin_memory_mirror"
EVENT_TYPE_HERMES_AUX_COMPACTION_OBSERVED: Final = "hermes_auxiliary_compaction_observed"
EVENT_TYPE_ANSWER_CARD_CREATED: Final = "answer_card_created"

ALL_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        EVENT_TYPE_MESSAGE_RECEIVED,
        EVENT_TYPE_ASSISTANT_MESSAGE_SENT,
        EVENT_TYPE_TOOL_CALLED,
        EVENT_TYPE_TOOL_RESULT_RECEIVED,
        EVENT_TYPE_ARTIFACT_CREATED,
        EVENT_TYPE_ARTIFACT_MODIFIED,
        EVENT_TYPE_SUMMARY_CREATED,
        EVENT_TYPE_FACT_EXTRACTED,
        EVENT_TYPE_DECISION_RECORDED,
        EVENT_TYPE_FACT_INVALIDATED,
        EVENT_TYPE_DECISION_SUPERSEDED,
        EVENT_TYPE_MEMORY_DELETED,
        EVENT_TYPE_USER_CORRECTION,
        EVENT_TYPE_HINDSIGHT_IMPORTED,
        EVENT_TYPE_DELEGATION_OBSERVED,
        EVENT_TYPE_BUILTIN_MEMORY_MIRROR,
        EVENT_TYPE_HERMES_AUX_COMPACTION_OBSERVED,
        EVENT_TYPE_ANSWER_CARD_CREATED,
    }
)


# Message roles
ROLE_USER: Final = "user"
ROLE_ASSISTANT: Final = "assistant"
ROLE_SYSTEM: Final = "system"
ROLE_TOOL: Final = "tool"
ALL_ROLES: Final[frozenset[str]] = frozenset({ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_TOOL})


# Message sources
MESSAGE_SOURCE_LIVE_TURN: Final = "live_turn"
MESSAGE_SOURCE_SYNC_TURN: Final = "sync_turn"
MESSAGE_SOURCE_HERMES_SESSION_IMPORT: Final = "hermes_session_import"
MESSAGE_SOURCE_HINDSIGHT_IMPORT: Final = "hindsight_import"
MESSAGE_SOURCE_HERMES_AUX_COMPACTION: Final = "hermes_auxiliary_compaction"


# Fact / decision / entity statuses (plan §3.7, §3.8, §19.1, §31.7)
STATUS_CANDIDATE: Final = "candidate"
STATUS_ACTIVE: Final = "active"
STATUS_PROPOSED: Final = "proposed"
STATUS_REJECTED: Final = "rejected"
STATUS_SUPERSEDED: Final = "superseded"
STATUS_INVALIDATED: Final = "invalidated"
STATUS_DELETED: Final = "deleted"
STATUS_UNVERIFIED: Final = "unverified"
STATUS_NEEDS_REVIEW: Final = "needs_review"
STATUS_DEDUPED: Final = "deduped"
STATUS_PROMOTED_ACTIVE: Final = "promoted_active"
STATUS_STALE: Final = "stale"
STATUS_MERGED: Final = "merged"


# Sensitivity classes (plan §3.5)
SENSITIVITY_NORMAL: Final = "normal"
SENSITIVITY_ELEVATED: Final = "elevated"
SENSITIVITY_SENSITIVE: Final = "sensitive"


# Source/trust markers for imports (plan §6.2)
TRUST_PRIMARY: Final = "primary"
TRUST_SECONDARY: Final = "secondary"


# Chunk source types
CHUNK_SOURCE_MESSAGE: Final = "message"
CHUNK_SOURCE_TOOL_OUTPUT: Final = "tool_output"
CHUNK_SOURCE_ARTIFACT: Final = "artifact"
CHUNK_SOURCE_SESSION_SUMMARY: Final = "session_summary"
CHUNK_SOURCE_TOPIC_SUMMARY: Final = "topic_summary"
CHUNK_SOURCE_PROJECT_STATE: Final = "project_state"
CHUNK_SOURCE_FACT: Final = "fact"
CHUNK_SOURCE_DECISION: Final = "decision"


# Entity types (plan §3.6)
ENTITY_TYPE_PERSON: Final = "person"
ENTITY_TYPE_PROJECT: Final = "project"
ENTITY_TYPE_SERVER: Final = "server"
ENTITY_TYPE_REPO: Final = "repo"
ENTITY_TYPE_WALLET: Final = "wallet"
ENTITY_TYPE_COMPANY: Final = "company"
ENTITY_TYPE_PRODUCT: Final = "product"
ENTITY_TYPE_API: Final = "api"
ENTITY_TYPE_MODEL: Final = "model"
ENTITY_TYPE_DOCUMENT: Final = "document"
ENTITY_TYPE_TOPIC: Final = "topic"
ALL_ENTITY_TYPES: Final[frozenset[str]] = frozenset(
    {
        ENTITY_TYPE_PERSON,
        ENTITY_TYPE_PROJECT,
        ENTITY_TYPE_SERVER,
        ENTITY_TYPE_REPO,
        ENTITY_TYPE_WALLET,
        ENTITY_TYPE_COMPANY,
        ENTITY_TYPE_PRODUCT,
        ENTITY_TYPE_API,
        ENTITY_TYPE_MODEL,
        ENTITY_TYPE_DOCUMENT,
        ENTITY_TYPE_TOPIC,
    }
)


# Agent contexts (plan §32.4)
AGENT_CONTEXT_PRIMARY: Final = "primary"
AGENT_CONTEXT_SUBAGENT: Final = "subagent"
AGENT_CONTEXT_CRON: Final = "cron"
AGENT_CONTEXT_FLUSH: Final = "flush"


# Recall modes
RECALL_MODE_FAST: Final = "fast"
RECALL_MODE_BALANCED: Final = "balanced"
RECALL_MODE_DEEP: Final = "deep"
RECALL_MODE_FORENSIC: Final = "forensic"
ALL_RECALL_MODES: Final[frozenset[str]] = frozenset(
    {RECALL_MODE_FAST, RECALL_MODE_BALANCED, RECALL_MODE_DEEP, RECALL_MODE_FORENSIC}
)


# Conflict statuses (plan §21.6)
CONFLICT_STATUS_OPEN: Final = "open"
CONFLICT_STATUS_AUTO_RESOLVED: Final = "auto_resolved"
CONFLICT_STATUS_NEEDS_REVIEW: Final = "needs_review"
CONFLICT_STATUS_IGNORED: Final = "ignored"


# Tool event exit statuses
EXIT_STATUS_OK: Final = "ok"
EXIT_STATUS_ERROR: Final = "error"
EXIT_STATUS_TIMEOUT: Final = "timeout"
EXIT_STATUS_UNKNOWN: Final = "unknown"

"""Tool schemas exposed by the Hermes MemoryProvider plugin. Plan §5.3.

Schemas follow Anthropic's tool specification: each entry has ``name``,
``description``, and ``input_schema`` (JSON Schema draft 2020-12). The schemas
are kept intentionally tight (plan §7.2) so the model cannot misuse them.
"""

from __future__ import annotations

from typing import Any

RECALL_MODES = ["fast", "balanced", "deep", "forensic"]


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    # Hermes MemoryProvider expects OpenAI-style `parameters`. Keep
    # `input_schema` as a compatibility alias for adapters that use Anthropic's
    # native tool schema naming.
    return {
        "name": name,
        "description": description,
        "parameters": schema,
        "input_schema": schema,
    }


def tool_schemas() -> list[dict[str, Any]]:
    return [
        _tool(
            "siqueira_memory_recall",
            "Retrieve source-backed memory. Returns a compact context pack with "
            "decisions, facts, chunks, summaries, and conflicts. Use for questions "
            "that need long-term history. Pass mode='deep' or 'forensic' only when "
            "you explicitly need extended/source-heavy context.",
            {
                "query": {"type": "string", "description": "Question or topic to recall."},
                "project": {"type": ["string", "null"]},
                "topic": {"type": ["string", "null"]},
                "mode": {"type": "string", "enum": RECALL_MODES, "default": "balanced"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 15},
                "include_sources": {"type": "boolean", "default": True},
                "allow_secret_recall": {
                    "type": "boolean",
                    "default": True,
                    "description": "Trusted Hermes/internal recall includes raw secret memory when relevant. Set false for external/public-safe presentation.",
                },
            },
            required=["query"],
        ),
        _tool(
            "siqueira_memory_remember",
            "Persist a source-backed fact or decision. Use when the user asks you "
            "to remember something or confirms a decision. Provide source_event_ids "
            "if available so provenance is preserved.",
            {
                "kind": {"type": "string", "enum": ["fact", "decision"]},
                "subject": {"type": ["string", "null"]},
                "predicate": {"type": ["string", "null"]},
                "object": {"type": ["string", "null"]},
                "statement": {"type": "string"},
                "project": {"type": ["string", "null"]},
                "topic": {"type": ["string", "null"]},
                "rationale": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.9},
                "source_event_ids": {"type": "array", "items": {"type": "string", "format": "uuid"}, "default": []},
            },
            required=["kind", "statement"],
        ),
        _tool(
            "siqueira_memory_correct",
            "Correct or supersede a previously stored fact or decision. Provide "
            "the target id and, when the user has supplied a replacement, a new "
            "fact/decision body.",
            {
                "target_type": {"type": "string", "enum": ["fact", "decision", "message", "summary"]},
                "target_id": {"type": ["string", "null"], "format": "uuid"},
                "correction_text": {"type": "string"},
                "replacement": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                    "description": "Optional remember-shaped replacement body.",
                },
            },
            required=["target_type", "correction_text"],
        ),
        _tool(
            "siqueira_memory_forget",
            "Delete or invalidate memory. Use mode='soft' for reversible invalidation "
            "and mode='hard' when the user explicitly asks to erase content.",
            {
                "target_type": {
                    "type": "string",
                    "enum": ["fact", "decision", "message", "chunk", "summary", "entity", "session"],
                },
                "target_id": {"type": "string", "format": "uuid"},
                "mode": {"type": "string", "enum": ["soft", "hard"], "default": "soft"},
                "reason": {"type": ["string", "null"]},
                "scrub_raw": {"type": "boolean", "default": False},
            },
            required=["target_type", "target_id"],
        ),
        _tool(
            "siqueira_memory_timeline",
            "Return the chronological timeline for a project/topic/entity.",
            {
                "project": {"type": ["string", "null"]},
                "topic": {"type": ["string", "null"]},
                "entity": {"type": ["string", "null"]},
                "since": {"type": ["string", "null"], "format": "date-time"},
                "until": {"type": ["string", "null"], "format": "date-time"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
            },
        ),
        _tool(
            "siqueira_memory_sources",
            "Return the source events/messages that back a fact, decision, or summary.",
            {
                "target_type": {"type": "string", "enum": ["fact", "decision", "summary"]},
                "target_id": {"type": "string", "format": "uuid"},
            },
            required=["target_type", "target_id"],
        ),
    ]


TOOL_NAMES: tuple[str, ...] = tuple(t["name"] for t in tool_schemas())

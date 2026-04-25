"""LLM-assisted memory capture classifier.

The heuristic marker list is a fallback. This module asks an OpenAI-compatible
chat completion endpoint whether a completed turn contains durable information
worth promoting into structured memory.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from siqueira_memo.config import Settings
from siqueira_memo.logging import get_logger
from siqueira_memo.services.redaction_service import redact

log = get_logger(__name__)

MemoryKind = Literal["fact", "decision"]


@dataclass(frozen=True)
class MemoryCaptureDecision:
    save: bool
    kind: MemoryKind = "fact"
    statement: str = ""
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    project: str | None = None
    topic: str | None = None
    confidence: float = 0.0
    rationale: str | None = None
    sensitivity: str = "normal"


def classify_turn_memory(
    settings: Settings,
    *,
    user_content: str,
    assistant_content: str,
    default_project: str | None = None,
    default_topic: str | None = None,
) -> MemoryCaptureDecision | None:
    """Classify one completed turn with an OpenAI-compatible LLM.

    Returns ``None`` when the LLM classifier is disabled or not configured, so
    callers can fall back to deterministic heuristics.
    """

    if not settings.memory_capture_llm_enabled:
        return None

    base_url = (settings.memory_capture_llm_base_url or "").strip()
    api_key = settings.memory_capture_llm_api_key.get_secret_value().strip()
    api_key = api_key or os.environ.get("SIQUEIRA_MEMORY_CAPTURE_LLM_API_KEY", "").strip()
    if not base_url:
        log.warning("memory_capture.llm.disabled_no_base_url")
        return None

    redacted_user = redact(user_content).redacted
    redacted_assistant = redact(assistant_content).redacted
    payload = _build_payload(
        settings,
        user_content=redacted_user,
        assistant_content=redacted_assistant,
        default_project=default_project,
        default_topic=default_topic,
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=settings.memory_capture_llm_timeout_seconds) as client:
            resp = client.post(_chat_completions_url(base_url), headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_json_object(str(content))
        return _decision_from_payload(parsed, default_project=default_project, default_topic=default_topic)
    except Exception as exc:  # pragma: no cover - network/provider failures are fallback path
        log.warning("memory_capture.llm.failed", extra={"error": str(exc)})
        return None


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _build_payload(
    settings: Settings,
    *,
    user_content: str,
    assistant_content: str,
    default_project: str | None,
    default_topic: str | None,
) -> dict[str, Any]:
    system = """You are a memory-capture classifier for an AI assistant.
Decide if a completed user/assistant turn contains durable information worth saving for future sessions.
Return ONLY valid JSON. No markdown.

Save durable information: user preferences/corrections, project decisions, stable infrastructure inventory, API/workflow conventions, tax/accounting facts, deployment state, architecture choices, recurring procedures, source-backed conclusions.
Ignore: casual acknowledgements, thanks, transient chatter, duplicate summaries with no new durable fact, tool noise, empty messages.
Never save raw secrets. If the turn contains secrets, set sensitivity="secret" and either save only redacted metadata or save=false.

JSON schema:
{
  "save": true|false,
  "kind": "fact"|"decision",
  "importance": 0.0-1.0,
  "statement": "single concise memory statement",
  "subject": "for facts, optional subject",
  "predicate": "for facts, optional predicate",
  "object": "for facts, optional object",
  "project": "project name or null",
  "topic": "topic name or null",
  "reason": "why this should or should not be saved",
  "sensitivity": "normal"|"private"|"secret"
}
"""
    user = {
        "default_project": default_project,
        "default_topic": default_topic,
        "turn": {
            "user": user_content,
            "assistant": assistant_content,
        },
    }
    return {
        "model": settings.memory_capture_llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def _decision_from_payload(
    data: dict[str, Any], *, default_project: str | None, default_topic: str | None
) -> MemoryCaptureDecision:
    importance = _float_in_range(data.get("importance", data.get("confidence", 0.0)))
    save = bool(data.get("save")) and importance >= 0.5
    sensitivity = str(data.get("sensitivity") or "normal").lower()
    if sensitivity == "secret":
        save = False
    kind = "decision" if str(data.get("kind") or "fact").lower() == "decision" else "fact"
    statement = str(data.get("statement") or "").strip()
    if save and not statement:
        save = False
    return MemoryCaptureDecision(
        save=save,
        kind=kind,
        statement=statement[:900],
        subject=_optional_str(data.get("subject")),
        predicate=_optional_str(data.get("predicate")),
        object=_optional_str(data.get("object")),
        project=_optional_str(data.get("project")) or default_project,
        topic=_optional_str(data.get("topic")) or default_topic,
        confidence=importance,
        rationale=_optional_str(data.get("reason") or data.get("rationale")),
        sensitivity=sensitivity,
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text[:300]


def _float_in_range(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))

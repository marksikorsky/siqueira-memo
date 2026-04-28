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
from siqueira_memo.schemas.memory_capture import MemoryCandidate, MemoryCaptureResult
from siqueira_memo.services.redaction_service import redact

log = get_logger(__name__)

MemoryKind = Literal["fact", "decision"]
CAPTURE_PROMPT_VERSION = "capture-v2"


@dataclass(frozen=True)
class MemoryCaptureDecision:
    """Legacy v1 single-candidate classifier result.

    Kept until all worker/tests/importers have migrated to ``MemoryCaptureResult``.
    The worker normalizes both shapes into the v2 result contract.
    """

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


CaptureClassifierOutput = MemoryCaptureResult | MemoryCaptureDecision


def classify_turn_memory(
    settings: Settings,
    *,
    user_content: str,
    assistant_content: str,
    default_project: str | None = None,
    default_topic: str | None = None,
) -> CaptureClassifierOutput | None:
    """Classify one completed turn with an OpenAI-compatible LLM.

    Returns ``None`` when the LLM classifier is disabled or not configured, so
    callers can fall back to deterministic heuristics. Valid v2 responses return
    ``MemoryCaptureResult``. Legacy v1 single-object responses are normalized to
    a one-candidate ``MemoryCaptureResult`` for backward compatibility.
    """

    if not settings.memory_capture_llm_enabled:
        return None

    base_url = (settings.memory_capture_llm_base_url or "").strip()
    api_key = settings.memory_capture_llm_api_key.get_secret_value().strip()
    api_key = api_key or os.environ.get("SIQUEIRA_MEMORY_CAPTURE_LLM_API_KEY", "").strip()
    if not base_url:
        log.warning("memory_capture.llm.disabled_no_base_url")
        return None

    # The LLM sees redacted content, not raw secrets. Secret candidates should
    # describe the operational reference and sensitivity, not expose the value.
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
        return _capture_result_from_payload(
            parsed,
            default_project=default_project,
            default_topic=default_topic,
            classifier_model=settings.memory_capture_llm_model,
        )
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
    system = f"""You are a memory-capture classifier for an AI assistant.
Decide if a completed user/assistant turn contains durable information worth saving for future sessions.
Return ONLY valid JSON. No markdown.

Save durable information: user preferences/corrections, project decisions, stable infrastructure inventory, API/workflow conventions, tax/accounting facts, deployment state, architecture choices, recurring procedures, source-backed conclusions.
Ignore: casual acknowledgements, thanks, transient chatter, duplicate summaries with no new durable fact, tool noise, empty messages.
Secrets are allowed as operational memory candidates when useful, but the input is redacted: never invent or expose raw secret values. Tag them with sensitivity="secret", kind="secret", and a high/critical risk when appropriate.

Return this v2 JSON schema:
{{
  "prompt_version": "{CAPTURE_PROMPT_VERSION}",
  "classifier_model": "model name or null",
  "skipped_reason": "optional whole-turn skip reason",
  "candidates": [
    {{
      "action": "auto_save"|"skip_noise"|"merge"|"supersede"|"flag_conflict"|"needs_review",
      "kind": "fact"|"decision"|"preference"|"secret"|"entity"|"relationship"|"summary",
      "statement": "single concise source-backed candidate statement",
      "subject": "for facts/secrets, optional subject",
      "predicate": "for facts/secrets, optional predicate",
      "object": "for facts/secrets, optional object",
      "project": "project name or null",
      "topic": "topic name or null",
      "entity_names": [],
      "confidence": 0.0-1.0,
      "importance": 0.0-1.0,
      "sensitivity": "public"|"internal"|"private"|"secret",
      "risk": "low"|"medium"|"high"|"critical",
      "rationale": "why this action is correct",
      "source_message_ids": [],
      "source_event_ids": [],
      "relation_to_existing": [],
      "review_reason": "optional reason for exception review"
    }}
  ]
}}

Backward-compatible v1 JSON with save/kind/statement is still accepted by the caller, but you should emit v2.
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
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        loaded = json.loads(match.group(0))
    if not isinstance(loaded, dict):
        raise ValueError("memory capture classifier must return a JSON object")
    return loaded


def _capture_result_from_payload(
    data: dict[str, Any], *, default_project: str | None, default_topic: str | None, classifier_model: str | None = None
) -> MemoryCaptureResult:
    """Parse either v2 multi-candidate payloads or legacy v1 single payloads."""

    if "candidates" in data:
        candidates_raw = data.get("candidates") or []
        if not isinstance(candidates_raw, list):
            raise ValueError("memory capture candidates must be a list")
        candidates = [
            _candidate_from_payload(item, default_project=default_project, default_topic=default_topic)
            for item in candidates_raw
            if isinstance(item, dict)
        ]
        return MemoryCaptureResult(
            candidates=candidates,
            skipped_reason=_optional_str(data.get("skipped_reason")),
            classifier_model=_optional_str(data.get("classifier_model")) or classifier_model,
            prompt_version=_optional_str(data.get("prompt_version")) or CAPTURE_PROMPT_VERSION,
        )

    # Legacy v1 response support. This is intentionally lossless enough for the
    # old worker/tests, while exposing the new candidate array downstream.
    legacy = _decision_from_payload(data, default_project=default_project, default_topic=default_topic)
    action = "auto_save" if legacy.save else "skip_noise"
    statement = legacy.statement or (legacy.rationale or "Classifier skipped this turn.")
    sensitivity = _normalize_sensitivity(legacy.sensitivity)
    candidate = MemoryCandidate(
        action=action,
        kind=legacy.kind,
        statement=statement,
        subject=legacy.subject,
        predicate=legacy.predicate,
        object=legacy.object,
        project=legacy.project or default_project,
        topic=legacy.topic or default_topic,
        confidence=legacy.confidence,
        importance=legacy.confidence,
        sensitivity=sensitivity,
        risk="high" if sensitivity == "secret" else "low",
        rationale=legacy.rationale or ("Legacy classifier save." if legacy.save else "Legacy classifier skip."),
    )
    return MemoryCaptureResult(
        candidates=[candidate],
        skipped_reason=None if legacy.save else candidate.rationale,
        classifier_model=classifier_model,
        prompt_version="capture-v1-compatible",
    )


def _candidate_from_payload(
    data: dict[str, Any], *, default_project: str | None, default_topic: str | None
) -> MemoryCandidate:
    payload = dict(data)
    payload["confidence"] = _float_in_range(payload.get("confidence", payload.get("importance", 0.0)))
    payload["importance"] = _float_in_range(payload.get("importance", payload.get("confidence", 0.0)))
    payload["project"] = _optional_str(payload.get("project")) or default_project
    payload["topic"] = _optional_str(payload.get("topic")) or default_topic
    payload["statement"] = str(payload.get("statement") or "").strip()[:900]
    payload["rationale"] = str(payload.get("rationale") or payload.get("reason") or "No rationale supplied.").strip()
    payload["sensitivity"] = _normalize_sensitivity(payload.get("sensitivity"))
    payload["risk"] = _normalize_risk(payload.get("risk"))
    payload["kind"] = _normalize_kind(payload.get("kind"))
    payload["action"] = _normalize_action(payload.get("action"))
    for field in ("subject", "predicate", "object"):
        payload[field] = _optional_str(payload.get(field))
    return MemoryCandidate.model_validate(payload)


def _decision_from_payload(
    data: dict[str, Any], *, default_project: str | None, default_topic: str | None
) -> MemoryCaptureDecision:
    importance = _float_in_range(data.get("importance", data.get("confidence", 0.0)))
    save = bool(data.get("save")) and importance >= 0.5
    sensitivity = str(data.get("sensitivity") or "normal").lower()
    if _normalize_sensitivity(sensitivity) == "secret":
        save = False
    kind: MemoryKind = "decision" if str(data.get("kind") or "fact").lower() == "decision" else "fact"
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


def _normalize_action(value: Any) -> str:
    action = str(value or "auto_save").lower().strip()
    allowed = {"auto_save", "skip_noise", "merge", "supersede", "flag_conflict", "needs_review"}
    if action not in allowed:
        # Let Pydantic produce the strict contract error expected by tests.
        return action
    return action


def _normalize_kind(value: Any) -> str:
    kind = str(value or "fact").lower().strip()
    allowed = {"fact", "decision", "preference", "secret", "entity", "relationship", "summary"}
    return kind if kind in allowed else "fact"


def _normalize_sensitivity(value: Any) -> str:
    sensitivity = str(value or "internal").lower().strip()
    if sensitivity == "normal":
        return "internal"
    allowed = {"public", "internal", "private", "secret"}
    return sensitivity if sensitivity in allowed else "internal"


def _normalize_risk(value: Any) -> str:
    risk = str(value or "low").lower().strip()
    allowed = {"low", "medium", "high", "critical"}
    return risk if risk in allowed else "low"


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

"""Secret-aware memory helpers.

This is not a hardened vault. It is a policy layer that prevents accidental raw
secret leakage into recall, admin default views, exports, and prefetch while
still allowing explicit audited reveal for trusted operators.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from siqueira_memo.services.redaction_service import redact

SECRET_MASK = "[SECRET_MASKED]"
_SECRET_VALUE_KEYS = {
    "secret_value",
    "raw_secret",
    "secret",
    "password",
    "token",
    "api_key",
    "connection_string",
}
_SECRET_HINT_KEYS = {"secret_ref", "secret_kind", "masked_preview"}
_SECRET_POLICIES = {"explicit_or_high_relevance", "never_prefetch"}


class MemoryLike(Protocol):
    statement: str
    extra_metadata: dict[str, Any]


def is_secret_metadata(metadata: dict[str, Any] | None) -> bool:
    meta = metadata or {}
    sensitivity = str(meta.get("sensitivity") or "").lower()
    if sensitivity == "secret":
        return True
    recall_policy = str(meta.get("recall_policy") or "").lower()
    if recall_policy in _SECRET_POLICIES:
        return True
    return any(key in meta and meta.get(key) for key in _SECRET_HINT_KEYS | _SECRET_VALUE_KEYS)


def recall_policy(metadata: dict[str, Any] | None) -> str:
    policy = str((metadata or {}).get("recall_policy") or "task_relevant").strip().lower()
    if policy in {"always", "task_relevant", "explicit_or_high_relevance", "never_prefetch"}:
        return policy
    return "task_relevant"


def mask_secret_value(value: str | None) -> str:
    text = str(value or "")
    if not text:
        return SECRET_MASK
    if len(text) <= 8:
        return "••••"
    if text.startswith("sk-proj-") and len(text) > 16:
        return f"sk-proj-...{text[-4:]}"
    if text.startswith("sk-") and len(text) > 12:
        return f"sk-...{text[-4:]}"
    return f"{text[:3]}...{text[-4:]}" if len(text) > 12 else f"...{text[-4:]}"


def masked_preview(text: str, metadata: dict[str, Any] | None) -> str:
    meta = metadata or {}
    configured = meta.get("masked_preview")
    if configured:
        return str(configured)
    redacted = redact(text or "").redacted
    secret_value = _secret_value_from_metadata(meta)
    if secret_value:
        redacted = redacted.replace(secret_value, mask_secret_value(secret_value))
    if redacted != (text or ""):
        return redacted
    if is_secret_metadata(meta):
        return SECRET_MASK
    return redacted


def secret_ref(metadata: dict[str, Any] | None) -> str | None:
    ref = (metadata or {}).get("secret_ref")
    return str(ref) if ref else None


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    sanitized = _sanitize_metadata_value(metadata or {}, key_hint=None)
    return cast(dict[str, Any], sanitized) if isinstance(sanitized, dict) else {}


def _sanitize_metadata_value(value: Any, *, key_hint: str | None) -> Any:
    key = (key_hint or "").lower()
    if key in _SECRET_VALUE_KEYS:
        return SECRET_MASK
    if isinstance(value, dict):
        return {str(k): _sanitize_metadata_value(v, key_hint=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_metadata_value(item, key_hint=key_hint) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_metadata_value(item, key_hint=key_hint) for item in value]
    if isinstance(value, str):
        return redact(value).redacted
    return value


def secret_value_for_reveal(statement: str, metadata: dict[str, Any] | None) -> str | None:
    meta = metadata or {}
    value = _secret_value_from_metadata(meta)
    if value:
        return value
    if is_secret_metadata(meta):
        return statement
    return None


def _secret_value_from_metadata(metadata: dict[str, Any]) -> str | None:
    for key in _SECRET_VALUE_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    return None

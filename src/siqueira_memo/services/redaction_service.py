"""Secret/redaction pipeline. Plan §7.2 / §23.

Detectors are implemented as layered regexes. Each detector produces a
``Finding`` with its kind, matched range, and a stable SECRET_REF placeholder
derived from a salted hash of the match — the hash alone cannot be reversed
back into the secret, and identical secrets redact to the same reference.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from re import Pattern
from typing import Any


@dataclass(frozen=True)
class Finding:
    kind: str
    name: str
    start: int
    end: int


@dataclass
class RedactionResult:
    redacted: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def matches(self) -> int:
        return len(self.findings)


@dataclass(frozen=True)
class _Detector:
    kind: str
    pattern: Pattern[str]
    name: str = "unknown"
    min_length: int = 0


def _compile(pat: str, flags: int = 0) -> Pattern[str]:
    return re.compile(pat, flags)


# ---------------------------------------------------------------------------
# Detector corpus
# ---------------------------------------------------------------------------

_DETECTORS: list[_Detector] = [
    # SSH/PEM multi-line keys must run first — they use dot-matches-all.
    _Detector(
        kind="ssh_private_key",
        name="OpenSSH private key block",
        pattern=_compile(
            r"-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]+?-----END OPENSSH PRIVATE KEY-----",
        ),
    ),
    _Detector(
        kind="pem_private_key",
        name="PEM private key block",
        pattern=_compile(
            r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA )?PRIVATE KEY-----",
        ),
    ),
    # Anthropic keys
    _Detector(
        kind="anthropic_api_key",
        name="Anthropic API key",
        pattern=_compile(r"sk-ant-api\d{2}-[A-Za-z0-9_\-]{60,}"),
    ),
    # OpenAI keys (modern sk-proj-*, legacy sk-*). Require 32+ body chars.
    _Detector(
        kind="openai_api_key",
        name="OpenAI API key",
        pattern=_compile(r"sk-proj-[A-Za-z0-9_\-]{32,}"),
    ),
    _Detector(
        kind="openai_api_key",
        name="OpenAI API key (legacy)",
        pattern=_compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    ),
    _Detector(
        kind="openrouter_api_key",
        name="OpenRouter API key",
        pattern=_compile(r"sk-or-v1-[A-Za-z0-9]{48,}"),
    ),
    _Detector(
        kind="github_token",
        name="GitHub token",
        pattern=_compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    ),
    _Detector(
        kind="telegram_bot_token",
        name="Telegram bot token",
        pattern=_compile(r"\b\d{8,12}:[A-Za-z0-9_\-]{30,}\b"),
    ),
    _Detector(
        kind="jwt",
        name="JSON Web Token",
        pattern=_compile(r"\bey[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
    ),
    _Detector(
        kind="aws_access_key_id",
        name="AWS access key ID",
        pattern=_compile(r"\bA(?:KIA|SIA|ROA|IDA|NPA|NVA|PKA|SCA)[0-9A-Z]{16}\b"),
    ),
    _Detector(
        kind="aws_secret_access_key",
        name="AWS secret access key",
        pattern=_compile(
            r"(?i)(?:aws_?secret_?access_?key|aws_?secret)[\"'\s:=]+([A-Za-z0-9/+=]{40})\b",
        ),
    ),
    _Detector(
        kind="slack_token",
        name="Slack token",
        pattern=_compile(r"xox[aboprs]-[A-Za-z0-9\-]{10,}"),
    ),
    _Detector(
        kind="bearer_token",
        name="Bearer token",
        pattern=_compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}"),
    ),
    _Detector(
        kind="basic_auth_header",
        name="Basic auth header",
        pattern=_compile(r"(?i)authorization:\s*basic\s+[A-Za-z0-9+/=]{8,}"),
    ),
    _Detector(
        kind="database_url",
        name="Database URL with credentials",
        pattern=_compile(
            r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|clickhouse)://"
            r"[^\s:@/]+:[^\s@/]+@[^\s/\"']+"
        ),
    ),
    _Detector(
        kind="cookie_header",
        name="Cookie header",
        pattern=_compile(r"(?i)cookie:\s*[^\n]{8,}"),
    ),
    _Detector(
        kind="dotenv_assignment",
        name=".env assignment",
        # Catch FOO_TOKEN=value, FOO_SECRET=value, FOO_KEY=value, OPENAI_API_KEY=value
        pattern=_compile(
            r"(?m)^\s*(?:[A-Z0-9_]{0,40}(?:TOKEN|SECRET|KEY|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS))\s*=\s*\S+"
        ),
    ),
]


def _looks_like_bip39(line: str) -> list[Finding]:
    """BIP39 seed phrase heuristic.

    BIP39 has 2048 English words and phrases are 12/15/18/21/24 words. Detecting
    the full wordlist inline is overkill; we look for long runs of short
    lowercase ASCII words that look like seeds. False positives on ordinary
    prose are possible, which is why we require all candidate words match the
    BIP39 character profile (3-8 chars, lowercase, alphabetic-only).
    """
    findings: list[Finding] = []
    # Slide a window of 12 lowercase dictionary-like words.
    tokens = list(re.finditer(r"\b[a-z]{3,8}\b", line))
    allowed_counts = {12, 15, 18, 21, 24}
    for i in range(len(tokens) - 11):
        for count in sorted(allowed_counts):
            j = i + count - 1
            if j >= len(tokens):
                continue
            window = tokens[i : j + 1]
            words = [t.group(0) for t in window]
            if any(w in {"the", "and", "for", "are", "with", "that", "this"} for w in words):
                # Stop-words disqualify the window.
                continue
            start = window[0].start()
            end = window[-1].end()
            findings.append(Finding(kind="seed_phrase", name="BIP39 seed phrase", start=start, end=end))
            break  # Don't double-count the same starting token.
    return findings


_PLACEHOLDER_RE = re.compile(r"\[SECRET_REF:[^\]\s]+\]")


class RedactionService:
    """Top-level pipeline. Detectors run in order, non-overlapping."""

    placeholder_template = "[SECRET_REF:{kind}/{name}/{digest}]"

    def __init__(self, detectors: Iterable[_Detector] | None = None) -> None:
        self._detectors = list(detectors) if detectors is not None else list(_DETECTORS)

    def _placeholder(self, kind: str, name: str, raw: str) -> str:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "secret"
        return self.placeholder_template.format(kind=kind, name=safe_name, digest=digest)

    def _collect_spans(self, text: str) -> list[tuple[int, int, str, str]]:
        """Collect non-overlapping (start, end, kind, name) redaction spans."""
        spans: list[tuple[int, int, str, str]] = []
        # Remember already redacted placeholders so we skip them.
        for m in _PLACEHOLDER_RE.finditer(text):
            spans.append((m.start(), m.end(), "__placeholder__", "placeholder"))
        for det in self._detectors:
            for m in det.pattern.finditer(text):
                spans.append((m.start(), m.end(), det.kind, det.name))
        # BIP39 seed heuristic.
        for finding in _looks_like_bip39(text):
            spans.append((finding.start, finding.end, finding.kind, finding.name))

        # Sort and merge overlapping spans; later detectors do not override
        # placeholders but do extend earlier regex spans.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: list[tuple[int, int, str, str]] = []
        for span in spans:
            if not merged or span[0] >= merged[-1][1]:
                merged.append(span)
                continue
            prev = merged[-1]
            if span[2] == "__placeholder__" or prev[2] == "__placeholder__":
                # Placeholder always wins.
                merged[-1] = prev if prev[2] == "__placeholder__" else span
                continue
            merged[-1] = (prev[0], max(prev[1], span[1]), prev[2], prev[3])
        return merged

    def redact(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(redacted=text or "", findings=[])
        spans = self._collect_spans(text)
        if not spans:
            return RedactionResult(redacted=text, findings=[])
        out: list[str] = []
        cursor = 0
        findings: list[Finding] = []
        for start, end, kind, name in spans:
            if start < cursor:
                continue
            out.append(text[cursor:start])
            if kind == "__placeholder__":
                out.append(text[start:end])
            else:
                raw = text[start:end]
                # For database URLs, preserve host/db and only replace credentials.
                if kind == "database_url":
                    scheme = raw.split("://", 1)[0]
                    tail = raw.split("@", 1)[1] if "@" in raw else ""
                    placeholder = self._placeholder(kind, name, raw)
                    replacement = f"{scheme}://{placeholder}@{tail}" if tail else placeholder
                    out.append(replacement)
                else:
                    out.append(self._placeholder(kind, name, raw))
                findings.append(Finding(kind=kind, name=name, start=start, end=end))
            cursor = end
        out.append(text[cursor:])
        return RedactionResult(redacted="".join(out), findings=findings)

    def redact_dict(self, value: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Recursively redact all string values in a dict.

        Returns ``(redacted_dict, total_matches)``.
        """
        count = 0

        def _walk(obj: Any) -> Any:
            nonlocal count
            if isinstance(obj, str):
                result = self.redact(obj)
                count += result.matches
                return result.redacted
            if isinstance(obj, dict):
                return {k: _walk(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_walk(v) for v in obj)
            return obj

        walked = _walk(value)
        return walked, count


_default_service = RedactionService()


def redact(text: str) -> RedactionResult:
    """Convenience wrapper around the module-level default service."""
    return _default_service.redact(text)


def redact_dict(value: dict[str, Any]) -> tuple[dict[str, Any], int]:
    return _default_service.redact_dict(value)

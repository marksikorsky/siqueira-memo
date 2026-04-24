"""Deterministic text + key normalisation. Plan §31.3 / §18.2.6."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_MARKDOWN_TOKENS = re.compile(r"[`*_~>|#\[\]()]")
_DASHES = re.compile(r"[‐-―−]")
_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = " .,:;!? "


def normalize_text(text: str) -> str:
    """Normalise free-form text for equality comparisons.

    Steps mirror plan §31.3 exactly:

    1. Unicode NFKC
    2. Case-fold (Unicode-safe lower)
    3. Strip markdown punctuation that does not change identity
    4. Normalize whitespace and Unicode dashes
    5. Remove trailing punctuation
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _MARKDOWN_TOKENS.sub(" ", text)
    text = _DASHES.sub("-", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = text.strip(_TRAILING_PUNCT)
    return text


def fact_canonical_key(
    subject: str,
    predicate: str,
    obj: str,
    *,
    project: str | None = None,
    profile_id: str = "default",
) -> str:
    """Canonical key for a fact row. Plan §18.2.6 extended with profile scoping."""
    parts = [
        normalize_text(subject),
        normalize_text(predicate),
        normalize_text(obj),
        normalize_text(project or "__global__"),
        profile_id,
    ]
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def decision_canonical_key(
    project: str | None,
    topic: str,
    decision_summary: str,
    *,
    profile_id: str = "default",
) -> str:
    """Canonical key for a decision row. Plan §18.2.6."""
    parts = [
        normalize_text(project or "__global__"),
        normalize_text(topic),
        normalize_text(decision_summary),
        profile_id,
    ]
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Stable SHA-256 of normalised content for dedupe/audit purposes."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def advisory_lock_key(canonical_key: str) -> int:
    """64-bit advisory lock key per plan §33.12.

    We emulate Postgres ``hashtextextended`` by taking the first 8 bytes of a
    SHA-256 hash and interpreting them as a signed 64-bit integer (matching
    ``pg_advisory_lock(bigint)``'s expected input).
    """
    digest = hashlib.sha256(canonical_key.encode("utf-8")).digest()
    unsigned = int.from_bytes(digest[:8], "big", signed=False)
    # Convert to signed 64-bit range Postgres expects.
    if unsigned >= 2**63:
        return unsigned - 2**64
    return unsigned


def normalize_path(path: str) -> str:
    from pathlib import Path

    return str(Path(path).expanduser().resolve())

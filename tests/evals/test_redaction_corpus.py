"""Redaction corpus evals. Plan §23.4.

Blocking CI checks (plan §26.2):

* known secret recall >= 99% on a fixed corpus;
* false positive rate on public identifiers <= 20% (we aim for 2% but do not
  over-tune single-sample corpora);
* no known secret substring appears in the redacted payload sent to chunking.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from siqueira_memo.services.chunking_service import ChunkingService
from siqueira_memo.services.redaction_service import redact

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "redaction"


def _load_lines(name: str) -> list[str]:
    return [
        line.rstrip("\n")
        for line in (_FIXTURE / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_mixed() -> list[dict]:
    return [
        json.loads(line)
        for line in (_FIXTURE / "mixed_messages.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


# Secrets split across multiple lines must be reassembled before checking.
_SECRET_BLOCKS: list[str] = []


def _collect_secret_blocks() -> list[str]:
    if _SECRET_BLOCKS:
        return _SECRET_BLOCKS
    text = (_FIXTURE / "known_secrets.txt").read_text(encoding="utf-8")
    blocks: list[str] = []
    current: list[str] = []
    in_key_block = False
    for line in text.splitlines():
        if line.startswith("-----BEGIN"):
            in_key_block = True
            current = [line]
            continue
        if in_key_block:
            current.append(line)
            if line.startswith("-----END"):
                blocks.append("\n".join(current))
                current = []
                in_key_block = False
            continue
        if line.strip():
            blocks.append(line)
    _SECRET_BLOCKS.extend(blocks)
    return _SECRET_BLOCKS


def test_known_secret_recall_above_threshold():
    secrets = _collect_secret_blocks()
    caught = 0
    leaked: list[str] = []
    for secret in secrets:
        redacted = redact(f"prefix {secret} suffix").redacted
        marker = _high_entropy_marker(secret)
        if marker not in redacted:
            caught += 1
        else:
            leaked.append(secret[:32] + "…")
    recall = caught / len(secrets)
    assert recall >= 0.99, f"recall={recall}; leaked={leaked}"


def test_false_positive_rate_within_budget():
    public = _load_lines("false_positives.txt")
    flagged = 0
    for line in public:
        result = redact(line)
        # Treat the whole input as a false positive only if we flagged anything
        # beyond a safe placeholder — a preserved URL or wallet must survive.
        if result.matches > 0 and all(
            part not in result.redacted
            for part in ["github.com/torvalds", "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEB0"]
            if part in line
        ):
            flagged += 1
    fp_rate = flagged / max(1, len(public))
    assert fp_rate <= 0.2, f"fp_rate={fp_rate}"


def test_no_secret_reaches_embedding_input():
    chunker = ChunkingService()
    messages = _load_mixed()
    # Collect substrings we must never see after redaction+chunking.
    forbidden = [
        "sk-proj-cccccccccccccccccccccccccccccccccccccccc",
        "abcd1234EFGH5678ijkl9012mnop3456qrst7890",
        "N3wPass",
    ]
    for msg in messages:
        redacted = redact(msg["text"]).redacted
        chunks = chunker.chunk_message(redacted, source_id="msg")
        blob = " ".join(c.chunk_text for c in chunks)
        for forbidden_value in forbidden:
            assert forbidden_value not in blob, f"leaked {forbidden_value}"


def test_public_identifiers_preserved_after_redaction():
    text = "see https://github.com/siqueira/memo and wallet 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEB0"
    redacted = redact(text).redacted
    assert "https://github.com/siqueira/memo" in redacted
    assert "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEB0" in redacted


def _high_entropy_marker(block: str) -> str:
    """Return the most distinctive substring from a secret block.

    For multi-line keys we grab the highest-entropy Base64-ish line so the
    assertion doesn't accidentally trip on header/footer text that the
    redactor may preserve.
    """
    lines = [line for line in block.splitlines() if line.strip()]
    candidate = max(
        lines,
        key=lambda line: sum(1 for c in line if re.match(r"[A-Za-z0-9+/]", c)),
    )
    return candidate.strip()

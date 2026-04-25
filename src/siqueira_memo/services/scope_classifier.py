"""Project/global scope classifier for automatic memory capture.

The classifier intentionally starts deterministic. It should be predictable,
cheap, and safe on the hot path; an LLM classifier can be layered on later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryScope:
    project: str | None
    topic: str | None
    confidence: float
    reason: str


_PROJECT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "siqueira-memo",
        (
            "siqueira",
            "memory",
            "recall",
            "dashboard",
            "capture",
            "provider",
            "sync_turn",
        ),
    ),
    (
        "brazil-tax-crypto",
        (
            "tax",
            "crypto",
            "brazil",
            "receita",
            "binance",
            "bybit",
            "avantis",
            "brl",
            "irpf",
            "gcap",
        ),
    ),
    ("shannon", ("shannon",)),
    ("clawik-ai", ("clawik", "clawik.ai")),
    ("draftmotion-ai", ("draftmotion", "draftmotion.ai")),
)

_GLOBAL_MARKERS = (
    "марк хочет",
    "марк предпочитает",
    "user prefers",
    "assistant behavior",
    "memory policy",
    "system prompt",
    "prompt",
    "запомни",
    "не забывай",
    "ты забыл",
    "сохранять почти",
)

_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("memory-write-policy", ("memory policy", "сохраня", "памят", "remember")),
    ("admin-ui", ("dashboard", "admin", "ui", "login")),
    ("deployment", ("deploy", "docker", "caddy", "tailscale", "container")),
    ("tax", ("tax", "receita", "irpf", "gcap")),
)


def classify_memory_scope(text: str, *, default_project: str | None = None) -> MemoryScope:
    """Classify text into project/topic or global scope.

    Global wins for explicit user/assistant-behavior preferences. Otherwise we
    pick the strongest keyword project. If unsure, return ``project=None`` —
    fake projects are worse than global memory.
    """

    lowered = text.lower()
    topic = _classify_topic(lowered)
    if any(marker in lowered for marker in _GLOBAL_MARKERS):
        return MemoryScope(
            project=None,
            topic=topic or "memory-write-policy",
            confidence=0.85,
            reason="global preference/policy marker",
        )

    best_project: str | None = None
    best_score = 0
    for project, keywords in _PROJECT_KEYWORDS:
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_project = project
            best_score = score

    if best_project and best_score > 0:
        return MemoryScope(
            project=best_project,
            topic=topic,
            confidence=min(0.95, 0.65 + best_score * 0.1),
            reason=f"matched {best_score} project keyword(s)",
        )

    if default_project:
        return MemoryScope(
            project=default_project,
            topic=topic,
            confidence=0.55,
            reason="inherited default project",
        )

    return MemoryScope(project=None, topic=topic, confidence=0.5, reason="no project match")


def _classify_topic(lowered: str) -> str | None:
    for topic, keywords in _TOPIC_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return topic
    return None

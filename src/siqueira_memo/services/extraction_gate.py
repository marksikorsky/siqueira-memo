"""Deterministic extraction gate. Plan §18.2.2 / §31.2.

Stage A: local regex/keyword classifier. Returns one or more labels without
calling any LLM. Stage B (cheap LLM classifier) is a hook that callers may
override but is intentionally optional — production wires a Haiku-class
classifier in via dependency injection.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Labels defined by plan §18.2.2.
LABEL_IGNORE = "ignore"
LABEL_CASUAL_ACK = "casual_ack"
LABEL_TOOL_NOISE = "tool_noise"
LABEL_POSSIBLE_FACT = "possible_fact"
LABEL_POSSIBLE_DECISION = "possible_decision"
LABEL_EXPLICIT_MEMORY_REQUEST = "explicit_memory_request"
LABEL_USER_CORRECTION = "user_correction"
LABEL_PROJECT_STATE_UPDATE = "project_state_update"
LABEL_SENSITIVE_SECRET_CANDIDATE = "sensitive_secret_candidate"


FULL_EXTRACTION_LABELS = frozenset(
    {
        LABEL_POSSIBLE_FACT,
        LABEL_POSSIBLE_DECISION,
        LABEL_EXPLICIT_MEMORY_REQUEST,
        LABEL_USER_CORRECTION,
        LABEL_PROJECT_STATE_UPDATE,
    }
)


@dataclass(frozen=True)
class GateResult:
    labels: tuple[str, ...]
    confidence: float = 1.0
    reason: str = ""
    requires_window_context: bool = False

    def needs_full_extraction(self) -> bool:
        return any(label in FULL_EXTRACTION_LABELS for label in self.labels)


# --------------------------------------------------------------------------
# Keyword corpora. Keeping lists short and explicit is intentional: the gate
# is deterministic and should not over-classify.
# --------------------------------------------------------------------------
_ACK_SET = frozenset(
    {
        "ok",
        "ок",
        "окей",
        "да",
        "нет",
        "спасибо",
        "понял",
        "продолжай",
        "thanks",
        "thx",
        "yes",
        "no",
    }
)

_EXPLICIT_MEMORY_VERBS = [
    r"\bзапомни\b",
    r"\bсохрани\b",
    r"\bне забудь\b",
    r"\bзабудь\b",
    r"\bудали из памяти\b",
    r"\bremember this\b",
    r"\bsave this\b",
    r"\bdon['’]t forget\b",
    r"\bforget this\b",
    r"\bdelete from memory\b",
]

_USER_CORRECTION_VERBS = [
    r"\bисправь\b",
    r"\bэто неверно\b",
    r"\bнет,\s",
    r"\bне так\b",
    r"\bcorrect me\b",
    r"\bthat['’]s wrong\b",
    r"\bactually,\s",
    r"\bнеправильно\b",
]

_DECISION_MARKERS = [
    r"\bрешаем\b",
    r"\bрешили\b",
    r"\bвыбираем\b",
    r"\bоставляем\b",
    r"\bделаем так\b",
    r"\bне используем\b",
    r"\bprimary\b",
    r"\bsource of truth\b",
    r"\blet's go with\b",
    r"\blet us use\b",
    r"\bwe['’]ll use\b",
    r"\bwe won['’]t use\b",
    r"\bне надо\b",
    r"\bне нужно\b",
]

_PROJECT_MARKERS = [
    r"\bproject\s+state\b",
    r"\bстатус проекта\b",
    r"\brelease\b",
    r"\bmilestone\b",
    r"\bdeploy(ed)?\b",
    r"\bschedule\b",
    r"\broadmap\b",
]

_FACT_MARKERS = [
    r"\bis\b",
    r"\bare\b",
    r"\bесть\b",
    r"\bявляется\b",
    r"\buses\b",
    r"\bruns on\b",
    r"\bstores\b",
    r"\bport\s+\d{2,5}\b",
    r"@\w+\.\w{2,}",
]

_SECRET_HINTS = [
    r"sk-proj-",
    r"sk-ant-",
    r"bearer\s+",
    r"api[_-]?key",
    r"password\s*=",
    r"passwd\s*=",
    r"-----BEGIN",
]

_TOOL_NOISE_HINTS = [
    r"^exit (?:code|status): \d+$",
    r"^\s*$",
]


def _has_any(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


@dataclass
class ExtractionGate:
    """Deterministic prefilter."""

    def classify(
        self,
        text: str,
        *,
        role: str = "user",
        previous_assistant_requested_confirmation: bool = False,
        is_tool_output: bool = False,
    ) -> GateResult:
        stripped = (text or "").strip()
        lowered = stripped.lower()

        if not stripped:
            return GateResult(labels=(LABEL_IGNORE,), reason="empty message")

        if is_tool_output and _has_any(_TOOL_NOISE_HINTS, stripped):
            return GateResult(labels=(LABEL_TOOL_NOISE,), reason="tool noise")

        if _has_any(_SECRET_HINTS, stripped):
            return GateResult(
                labels=(LABEL_SENSITIVE_SECRET_CANDIDATE,),
                reason="secret/credential-like content",
            )

        if _has_any(_EXPLICIT_MEMORY_VERBS, stripped):
            explicit_labels: list[str] = [LABEL_EXPLICIT_MEMORY_REQUEST]
            if _has_any(_USER_CORRECTION_VERBS, stripped):
                explicit_labels.insert(0, LABEL_USER_CORRECTION)
            return GateResult(
                labels=tuple(explicit_labels),
                reason="explicit memory verb",
                requires_window_context=True,
            )

        if _has_any(_USER_CORRECTION_VERBS, stripped):
            return GateResult(
                labels=(LABEL_USER_CORRECTION,),
                reason="user correction verb",
                requires_window_context=True,
            )

        if lowered in _ACK_SET or (
            len(stripped) < 5 and not previous_assistant_requested_confirmation
        ):
            return GateResult(labels=(LABEL_CASUAL_ACK,), reason="casual acknowledgement")

        labels: list[str] = []
        if _has_any(_DECISION_MARKERS, stripped):
            labels.append(LABEL_POSSIBLE_DECISION)
        if _has_any(_PROJECT_MARKERS, stripped):
            labels.append(LABEL_PROJECT_STATE_UPDATE)
        if _has_any(_FACT_MARKERS, stripped):
            labels.append(LABEL_POSSIBLE_FACT)

        if labels:
            return GateResult(
                labels=tuple(labels),
                reason="content markers matched",
                requires_window_context=LABEL_POSSIBLE_DECISION in labels,
            )

        # Default: benign free-text. Plan §18.2.2 treats this as ignorable for
        # extraction purposes. Downstream summarisation still runs at session
        # scope.
        return GateResult(labels=(LABEL_IGNORE,), reason="no extraction markers")


default_gate = ExtractionGate()

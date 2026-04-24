"""Deterministic extraction gate. Plan §18.2.2 / §31.2."""

from __future__ import annotations

import pytest

from siqueira_memo.services.extraction_gate import (
    LABEL_CASUAL_ACK,
    LABEL_EXPLICIT_MEMORY_REQUEST,
    LABEL_IGNORE,
    LABEL_POSSIBLE_DECISION,
    LABEL_POSSIBLE_FACT,
    LABEL_PROJECT_STATE_UPDATE,
    LABEL_SENSITIVE_SECRET_CANDIDATE,
    LABEL_TOOL_NOISE,
    LABEL_USER_CORRECTION,
    default_gate,
)


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("ок", LABEL_CASUAL_ACK),
        ("да", LABEL_CASUAL_ACK),
        ("ok", LABEL_CASUAL_ACK),
        ("thanks", LABEL_CASUAL_ACK),
        ("", LABEL_IGNORE),
        ("решили делать Hermes MemoryProvider primary", LABEL_POSSIBLE_DECISION),
        ("primary будет MemoryProvider", LABEL_POSSIBLE_DECISION),
        ("запомни: пароль хранить отдельно", LABEL_EXPLICIT_MEMORY_REQUEST),
        ("remember this: do not use MCP", LABEL_EXPLICIT_MEMORY_REQUEST),
        ("нет, это неверно", LABEL_USER_CORRECTION),
        ("actually, we use Postgres not Mongo", LABEL_USER_CORRECTION),
        ("milestone 3 deployed to staging", LABEL_PROJECT_STATE_UPDATE),
        ("token: sk-proj-aaaaaaaaaaaaaaaaaaaa", LABEL_SENSITIVE_SECRET_CANDIDATE),
    ],
)
def test_labels_basic(text: str, expected_label: str) -> None:
    result = default_gate.classify(text)
    assert expected_label in result.labels


def test_tool_noise():
    result = default_gate.classify("exit code: 0", is_tool_output=True)
    assert LABEL_TOOL_NOISE in result.labels


def test_plain_factual_sentence_flagged_possible_fact():
    result = default_gate.classify("the server runs on port 8080")
    assert LABEL_POSSIBLE_FACT in result.labels
    assert result.needs_full_extraction()


def test_benign_chat_is_ignored():
    result = default_gate.classify("hmm, interesting")
    assert LABEL_IGNORE in result.labels
    assert not result.needs_full_extraction()


def test_decision_requires_window_context():
    result = default_gate.classify("выбираем второй вариант")
    assert LABEL_POSSIBLE_DECISION in result.labels
    assert result.requires_window_context is True

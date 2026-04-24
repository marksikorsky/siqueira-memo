"""Discriminated union payload validation. Plan §31.9."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from siqueira_memo.schemas.event_payloads import (
    DelegationObservedPayload,
    HermesAuxCompactionPayload,
    MessageReceivedPayload,
    UserCorrectionReceivedPayload,
    validate_event_payload,
)


def test_discriminator_picks_message_received():
    payload = validate_event_payload(
        {
            "event_type": "message_received",
            "message_id": "abc",
            "role": "user",
            "platform": "cli",
            "content_hash": "deadbeef",
        }
    )
    assert isinstance(payload, MessageReceivedPayload)


def test_delegation_payload_required_fields():
    payload = validate_event_payload(
        {
            "event_type": "delegation_observed",
            "parent_session_id": "p",
            "task": "do X",
            "result": "done",
        }
    )
    assert isinstance(payload, DelegationObservedPayload)
    assert payload.toolsets == []


def test_rejects_unknown_event_type():
    with pytest.raises(ValidationError):
        validate_event_payload({"event_type": "not_a_real_type"})


def test_hermes_aux_compaction_defaults():
    payload = validate_event_payload(
        {
            "event_type": "hermes_auxiliary_compaction_observed",
            "summary_text": "[CONTEXT COMPACTION] …",
        }
    )
    assert isinstance(payload, HermesAuxCompactionPayload)
    assert payload.prefix == "[CONTEXT COMPACTION]"
    assert payload.source_message_count == 0


def test_builtin_memory_mirror_enforces_action_enum():
    with pytest.raises(ValidationError):
        validate_event_payload(
            {
                "event_type": "builtin_memory_mirror",
                "action": "frobnicate",
                "target": "memory",
                "content_hash": "deadbeef",
            }
        )


def test_user_correction_minimal():
    payload = validate_event_payload(
        {"event_type": "user_correction_received", "correction_text": "no, it's X"}
    )
    assert isinstance(payload, UserCorrectionReceivedPayload)
    assert payload.correction_text == "no, it's X"

"""Memory capture v2 classifier contract tests."""

from __future__ import annotations

from pydantic import ValidationError

from siqueira_memo.services.memory_capture_classifier import _capture_result_from_payload


def test_capture_parser_accepts_multi_candidate_payload():
    result = _capture_result_from_payload(
        {
            "prompt_version": "capture-v2",
            "classifier_model": "test-model",
            "candidates": [
                {
                    "action": "auto_save",
                    "kind": "fact",
                    "statement": "Hermes admin is exposed only over Tailscale.",
                    "subject": "Hermes admin",
                    "predicate": "exposure",
                    "object": "Tailscale only",
                    "project": "siqueira-memo",
                    "topic": "admin",
                    "confidence": 0.93,
                    "importance": 0.88,
                    "sensitivity": "internal",
                    "risk": "low",
                    "rationale": "Stable operational detail.",
                },
                {
                    "action": "auto_save",
                    "kind": "decision",
                    "statement": "Siqueira should prefer source-backed facts over summaries.",
                    "project": "siqueira-memo",
                    "topic": "memory-policy",
                    "confidence": 0.91,
                    "importance": 0.9,
                    "rationale": "Explicit design decision.",
                },
            ],
        },
        default_project=None,
        default_topic="conversation",
    )

    assert result.prompt_version == "capture-v2"
    assert len(result.candidates) == 2
    assert result.candidates[0].action == "auto_save"
    assert result.candidates[0].kind == "fact"
    assert result.candidates[1].kind == "decision"
    assert result.candidates[1].topic == "memory-policy"


def test_capture_parser_keeps_legacy_v1_payload_compatible():
    result = _capture_result_from_payload(
        {
            "save": True,
            "kind": "decision",
            "importance": 0.87,
            "statement": "Use aggressive memory capture for durable infrastructure details.",
            "project": "siqueira-memo",
            "topic": "capture",
            "reason": "Legacy classifier response.",
            "sensitivity": "normal",
        },
        default_project=None,
        default_topic="conversation",
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.action == "auto_save"
    assert candidate.kind == "decision"
    assert candidate.confidence == 0.87
    assert candidate.rationale == "Legacy classifier response."


def test_capture_parser_turns_legacy_skip_into_skip_noise_candidate():
    result = _capture_result_from_payload(
        {
            "save": False,
            "kind": "fact",
            "importance": 0.95,
            "statement": "",
            "reason": "Casual acknowledgement/no durable content.",
        },
        default_project="siqueira-memo",
        default_topic="conversation",
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.action == "skip_noise"
    assert candidate.kind == "fact"
    assert candidate.project == "siqueira-memo"
    assert candidate.rationale == "Casual acknowledgement/no durable content."


def test_capture_parser_rejects_unknown_actions_in_v2_payload():
    try:
        _capture_result_from_payload(
            {
                "candidates": [
                    {
                        "action": "silently_drop",
                        "kind": "fact",
                        "statement": "Bad action must not parse.",
                        "confidence": 0.9,
                        "importance": 0.9,
                        "rationale": "Invalid contract.",
                    }
                ]
            },
            default_project=None,
            default_topic=None,
        )
    except ValidationError:
        return

    raise AssertionError("invalid v2 capture action should fail validation")

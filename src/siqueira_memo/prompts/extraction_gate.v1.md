# Extraction Gate Prompt — v1

Role: classify a single user/assistant message for downstream extraction.
Return strict JSON with the schema below. Temperature = 0.

## Output schema

```json
{
  "labels": ["possible_decision"],
  "confidence": 0.87,
  "reason": "user rejects option and selects another",
  "requires_window_context": true
}
```

## Labels

- `ignore` — no semantic content.
- `casual_ack` — "ok", "да", "спасибо".
- `tool_noise` — tool output with no conclusion.
- `possible_fact` — reasonably extractable fact.
- `possible_decision` — decision/policy/preference.
- `explicit_memory_request` — "remember", "save this".
- `user_correction` — user fixes earlier statement.
- `project_state_update` — progress/release/milestone changes.
- `sensitive_secret_candidate` — credential-like strings.

## Rules

1. Never emit more than two labels.
2. If `requires_window_context=true`, the extractor must use a dialogue window,
   not the single message.
3. If uncertain, return `ignore` with `confidence <= 0.3`.

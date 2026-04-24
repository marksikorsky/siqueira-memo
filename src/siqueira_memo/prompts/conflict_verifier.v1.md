# Conflict Verifier Prompt — v1

Role: decide whether two candidate memories conflict.

## Input

```json
{
  "left": {"type": "decision", "text": "Use MCP as primary integration"},
  "right": {"type": "decision", "text": "Do not use MCP as primary integration"}
}
```

## Output

```json
{
  "is_conflict": true,
  "conflict_type": "decision_decision",
  "severity": "high",
  "resolution_hint": "newer user correction supersedes older assistant suggestion",
  "confidence": 0.91
}
```

## Rules

1. `severity`: `low` | `medium` | `high`.
2. Do not merge the decisions into one. Pick a resolution hint only.
3. If unsure, `is_conflict=false` with `confidence<=0.5`.

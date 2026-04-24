# Decision Extractor Prompt — v1

Role: extract concrete decisions from a dialogue window (10-30 messages).
Return strict JSON. Temperature = 0. Schema version: v1.

## Output schema

```json
{
  "decisions": [
    {
      "topic": "memory integration",
      "project": "siqueira-memo",
      "decision": "Use Hermes MemoryProvider plugin as primary integration",
      "rationale": "MemoryProvider is native; MCP is adapter only",
      "options_considered": [
        {"name": "MCP", "status": "rejected"},
        {"name": "MemoryProvider plugin", "status": "selected"}
      ],
      "tradeoffs": {"ergonomics": "better", "compatibility": "pending"},
      "status": "active",
      "reversible": true,
      "source_message_ids": ["…"],
      "confidence": 0.92
    }
  ]
}
```

## Definitions (plan §18.2.3)

- A decision is a commitment, preference, rejected path, or operational rule.
- A proposal, brainstorm, possibility, question, or temporary hypothesis is
  NOT a decision unless explicitly adopted by the user.

## Rules

1. Do not invent options not present in the window.
2. When the user says "second one" or "второй давай", reconstruct the
   referenced option or mark the decision `needs_review` (plan §31.4).
3. If confidence is below 0.6, emit status `candidate`.
4. Always cite at least one `source_message_ids` entry.
5. Never include secrets or raw credentials in `rationale` or `decision` text.

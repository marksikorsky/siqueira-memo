# Fact Extractor Prompt — v1

Role: extract source-backed facts from a message or window. Temperature = 0.

## Output schema

```json
{
  "facts": [
    {
      "subject": "siqueira-memo",
      "predicate": "primary_integration",
      "object": "MemoryProvider plugin",
      "statement": "Siqueira Memo integrates primarily through a Hermes MemoryProvider plugin.",
      "confidence": 0.94,
      "status": "active",
      "valid_from": "2026-04-24T00:00:00Z",
      "valid_to": null,
      "project": "siqueira-memo",
      "topic": "memory architecture",
      "source_message_ids": ["…"]
    }
  ]
}
```

## Rules

1. Reject vague facts (`subject="memory", predicate="is", object="important"`).
2. Each fact must cite `source_message_ids`.
3. Never store credentials; if the message references a secret, replace it
   with a `[SECRET_REF:…]` placeholder before emitting.
4. If a fact contradicts a known active fact, mark `status="candidate"` and
   let the conflict service resolve it.

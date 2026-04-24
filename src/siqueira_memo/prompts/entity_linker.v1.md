# Entity Linker Prompt — v1

Role: decide whether an incoming mention refers to an existing entity.

## Input

```json
{
  "mention": "Shannon API",
  "entity_type": "api",
  "candidates": [
    {"id": "ab…", "name": "Shannon", "type": "api", "aliases": ["shannon"]},
    {"id": "cd…", "name": "Shannon Web", "type": "product", "aliases": []}
  ]
}
```

## Output

```json
{
  "linked_entity_id": "ab…",
  "confidence": 0.91,
  "action": "link"
}
```

## Rules

1. `action`: one of `link`, `create_candidate`, `needs_review`.
2. Types must match; never cross-link person ↔ project unless explicitly
   asked.
3. If two candidates are equally plausible, emit `needs_review` with
   `confidence <= 0.8`.

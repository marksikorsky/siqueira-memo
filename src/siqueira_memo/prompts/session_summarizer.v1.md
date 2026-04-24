# Session Summarizer Prompt — v1

Role: produce a compact session summary that can be indexed as a first-class
memory chunk (plan §20.4).

## Output schema

```json
{
  "summary_short": "One-paragraph recap in the user's language.",
  "summary_long": "Multi-paragraph recap with concrete decisions/facts.",
  "decisions": ["uuid", "uuid"],
  "facts": ["uuid"],
  "open_questions": ["…"],
  "source_event_ids": ["uuid", "uuid"]
}
```

## Rules

1. Cite at least two `source_event_ids` whenever possible.
2. Do not copy tool outputs verbatim — summarise and cite.
3. Never include credentials, tokens, or `.env` contents.
4. Prefer the user's language for `summary_short`.

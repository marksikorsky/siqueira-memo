# siqueira-memo Hermes plugin

Thin integration shim that registers `SiqueiraMemoProvider` as the active
Hermes MemoryProvider.

## Enabling

Add the following to your Hermes config:

```yaml
memory:
  provider: siqueira-memo
```

Then start the Siqueira service (`uvicorn siqueira_memo.main:app`) and ensure
Hermes can reach it at `SIQUEIRA_DATABASE_URL`.

## Tools exposed

- `siqueira_memory_recall`
- `siqueira_memory_remember`
- `siqueira_memory_correct`
- `siqueira_memory_forget`
- `siqueira_memory_timeline`
- `siqueira_memory_sources`

## Hooks supported

`system_prompt_block`, `prefetch`, `queue_prefetch`, `sync_turn`, `on_pre_compress`,
`on_session_end`, `on_memory_write`, `on_delegation`, `on_turn_start`, `shutdown`.

See `../../../src/siqueira_memo/hermes_provider/system_prompt.md` for the
assistant-facing recall policy. That file is hashed into `prompt_versions` at
startup to prevent silent drift (plan §31.13).

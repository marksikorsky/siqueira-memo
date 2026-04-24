# Siqueira Memo — Hermes Plugin Install

Siqueira Memo ships a `MemoryProvider` plugin that Hermes loads via the
standard `plugins/memory` entrypoint. This doc is the operator-facing install
checklist.

## 1. Install the service

```bash
git clone <this-repo> /opt/siqueira-memo
cd /opt/siqueira-memo
python -m venv .venv && source .venv/bin/activate
pip install -e '.[postgres,queue,otel]'
cp .env.example .env
# Edit .env: SIQUEIRA_DATABASE_URL, SIQUEIRA_API_TOKEN, SIQUEIRA_EMBEDDING_PROVIDER…
alembic upgrade head
```

Verify boot:

```bash
siqueira-memo      # runs the API on SIQUEIRA_HOST:SIQUEIRA_PORT
siqueira-memo-worker &   # drains queued jobs in-process
```

## 2. Wire the plugin into Hermes

The plugin lives at `plugins/memory/siqueira-memo/` inside this repo. Expose
it on the Hermes plugin path (symlink or copy):

```bash
ln -s /opt/siqueira-memo/plugins/memory/siqueira-memo \
      ~/.hermes/plugins/memory/siqueira-memo
```

Then enable it in Hermes config:

```yaml
# ~/.hermes/config.yaml (excerpt)
memory:
  provider: siqueira-memo
  config:
    database_url: postgresql+asyncpg://siqueira:***@127.0.0.1:5432/siqueira_memo
    embedding_provider: openai
    embedding_model: text-embedding-3-large
    queue_backend: redis
    api_token: "${SIQUEIRA_API_TOKEN}"
```

Hermes will call `register(ctx)` inside the plugin package; the plugin
forwards to `SiqueiraMemoProvider` from the service package, so the service
must be importable — either install it (`pip install -e .`) or put `src/` on
`PYTHONPATH` for the Hermes process.

## 3. Tools exposed

| name                        | purpose                                      |
|-----------------------------|----------------------------------------------|
| `siqueira_memory_recall`    | hybrid recall returning a compact context pack |
| `siqueira_memory_remember`  | promote a source-backed fact or decision      |
| `siqueira_memory_correct`   | supersede/invalidate prior memory             |
| `siqueira_memory_forget`    | soft/hard deletion with cascading invalidation|
| `siqueira_memory_timeline`  | chronological timeline for a project/topic    |
| `siqueira_memory_sources`   | provenance resolver                           |

Provider-side hooks implemented: `system_prompt_block`, `prefetch`,
`queue_prefetch`, `sync_turn`, `on_pre_compress`, `on_session_end`,
`on_memory_write`, `on_delegation`, `on_turn_start`, `shutdown`.

## 4. Prompt hash parity (plan §31.13)

Every boot the service hashes both:

- `src/siqueira_memo/hermes_provider/system_prompt.md` — canonical.
- `plugins/memory/siqueira-memo/system_prompt.md` — plugin copy.

In `SIQUEIRA_ENV=production` a mismatch raises `PromptDriftError` and refuses
to start. In dev the mismatch logs a structured `prompt.hash_drift_warning`.
Smoke-check parity from the repo root:

```bash
python -c "
from siqueira_memo.services.prompt_registry import (
    assert_hermes_prompt_hash_parity, default_plugin_prompt_path)
print(assert_hermes_prompt_hash_parity(env='development',
      plugin_path=default_plugin_prompt_path()))
"
```

If you edit either copy, regenerate the other to match (or bump the `v{N}`
suffix in `src/siqueira_memo/prompts/*.v1.md` and rebuild both copies).

## 5. Readiness + smoke checklist

Local:

```bash
curl -s http://127.0.0.1:8787/healthz | jq
curl -sH "Authorization: Bearer $SIQUEIRA_API_TOKEN" \
     http://127.0.0.1:8787/readyz | jq
siqueira-memo-evals --min-pass-rate 0.8
```

Postgres-only (external infra required, not covered by pytest):

```bash
docker compose up -d postgres redis
alembic upgrade head                 # creates pgvector + HNSW + partial unique indexes
psql -c "SELECT extname FROM pg_extension WHERE extname IN ('vector','pg_trgm');"
# Expect: vector, pg_trgm
```

## 6. Failure modes operators should expect

- Service unreachable from Hermes: plugin tools return JSON-encoded error
  bodies, `sync_turn`/prefetch hooks fail soft.
- `PromptDriftError`: regenerate the plugin copy.
- `/readyz` `ok=false` with `pgvector.present=false`: migration did not run
  with pgvector available; re-run `alembic upgrade head` against a
  pgvector-enabled Postgres.
- `/readyz` reports missing current-month partitions: run the partition
  worker (see next section).

## 7. Partition automation

Plan §31.11 requires monthly partitions on `memory_events`, `messages`,
`tool_events`, `retrieval_logs`. Run `siqueira_memo.services.partition_service`
daily via cron or a scheduled worker:

```bash
python - <<'PY'
import asyncio
from siqueira_memo.db import get_session_factory
from siqueira_memo.services.partition_service import PartitionService
async def go():
    async with get_session_factory()() as session:
        report = await PartitionService().ensure_partitions_exist(session)
        await session.commit()
        print(report)
asyncio.run(go())
PY
```

SQLite is a no-op for partition DDL; production Postgres must be the real
driver.

# Siqueira Memo — Hermes Plugin Install

Siqueira Memo ships a `MemoryProvider` plugin that Hermes loads via the
standard `plugins/memory` entrypoint. This doc is the operator-facing install
checklist.

## 1. Install the service

For local operators, use the one-command Docker bootstrap:

```bash
git clone https://github.com/marksikorsky/siqueira-memo.git /opt/siqueira-memo
cd /opt/siqueira-memo
./scripts/bootstrap.sh
```

This creates `.env`, starts Postgres + pgvector + Redis, runs migrations, starts
API/worker, and verifies `/healthz` + `/readyz`.

## 2. Wire it into Hermes as the active MemoryProvider

Use the installer:

```bash
cd /opt/siqueira-memo
./scripts/install_hermes_provider.sh
```

It does the boring but easy-to-break bits:

- installs `siqueira-memo` into the same Python environment used by `hermes`;
- symlinks the provider to `~/.hermes/plugins/siqueira-memo`;
- writes `SIQUEIRA_*` connection settings into `~/.hermes/.env`;
- sets `memory.provider: siqueira-memo` in `~/.hermes/config.yaml`;
- verifies provider discovery with Hermes' plugin loader.

Important Hermes detail: user-installed memory plugins live directly under
`$HERMES_HOME/plugins/<provider-name>/`, **not** under
`$HERMES_HOME/plugins/memory/<provider-name>/`. Bundled Hermes providers live in
Hermes' own source tree under `plugins/memory/`, but external/user plugins use
the flatter user-plugin path.

Manual equivalent:

```bash
# Use the Python interpreter from the Hermes console script.
HERMES_PYTHON="$(head -n1 "$(command -v hermes)" | sed 's/^#!//')"
"$HERMES_PYTHON" -m pip install -e '/opt/siqueira-memo[postgres,queue,otel]'

mkdir -p ~/.hermes/plugins
ln -sfn /opt/siqueira-memo/plugins/memory/siqueira-memo ~/.hermes/plugins/siqueira-memo

# Add SIQUEIRA_DATABASE_URL, SIQUEIRA_REDIS_URL, SIQUEIRA_API_TOKEN, etc. to ~/.hermes/.env
hermes config set memory.provider siqueira-memo
```

Restart Hermes gateway/CLI sessions after changing the active provider.

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

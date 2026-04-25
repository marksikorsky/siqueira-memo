# Siqueira Memo

Production-grade, Hermes-native long-term memory. Canonical architecture lives
in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). This README is the
operator-facing quickstart.

## What it does

Siqueira Memo stores raw conversations, tool events, and artifacts with
per-profile provenance, redacts secrets before any embedding/LLM path, extracts
facts/decisions with canonical-key dedupe, and serves hybrid (structured +
lexical + vector) recall to Hermes through a `MemoryProvider` plugin.

## One-command install (recommended)

```bash
./scripts/bootstrap.sh
```

That script generates a local `.env`, pulls `pgvector/pgvector:pg16` + Redis,
builds the app image, runs Alembic migrations, starts API + worker, smoke
checks `/healthz` + `/readyz`, and — when Hermes is installed on the same host —
automatically installs Siqueira as the active Hermes `MemoryProvider`, updates
`~/.hermes/.env` + `~/.hermes/config.yaml`, and restarts the Hermes gateway.
No manual Postgres/pgvector setup; no embedding API key needed because local
bootstrap defaults to `SIQUEIRA_EMBEDDING_PROVIDER=mock`.

Disable Hermes wiring when you only want the service:

```bash
SIQUEIRA_INSTALL_HERMES_PROVIDER=false ./scripts/bootstrap.sh
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for operator details.

## Manual Python install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,postgres,queue,secrets,otel]'
```

`postgres` pulls `asyncpg` + `pgvector`; `queue` pulls Redis + Dramatiq;
`secrets` pulls `detect-secrets`; everything else is optional.

## Configure

```bash
cp .env.example .env
# Set at minimum:
# SIQUEIRA_DATABASE_URL=postgresql+asyncpg://siqueira:siqueira@127.0.0.1:5432/siqueira_memo
# SIQUEIRA_API_TOKEN=<random-bearer-token>
# SIQUEIRA_EMBEDDING_PROVIDER=openai
# SIQUEIRA_OPENAI_API_KEY=sk-...
```

Every setting is documented in `.env.example`. Real secrets must not land
in git.

## Database + migrations

```bash
docker compose up -d postgres redis
alembic upgrade head
```

On Postgres the migration enables `vector` + `pg_trgm`, creates HNSW indexes
on every `chunk_embeddings_*` table, GIN FTS indexes on `chunks`, and the
partial-unique indexes on active canonical keys. On SQLite the migration
creates tables only (used for local dev/tests).

Reset for dev:

```bash
python scripts/dev_reset_db.py
```

## Run the API

```bash
uvicorn siqueira_memo.main:app --host 127.0.0.1 --port 8787 --reload
# or
siqueira-memo   # uses the console_script defined in pyproject.toml
```

Routes (all POSTs require `Authorization: Bearer $SIQUEIRA_API_TOKEN`):

| method | path                                | purpose                             |
|--------|-------------------------------------|-------------------------------------|
| GET    | `/healthz`                          | liveness                            |
| GET    | `/readyz`                           | readiness + pgvector/redis status   |
| POST   | `/v1/ingest/message`                | ingest a user/assistant turn        |
| POST   | `/v1/ingest/tool-event`             | ingest a tool call + redacted output|
| POST   | `/v1/ingest/artifact`               | register a file/artifact            |
| POST   | `/v1/ingest/event`                  | raw event pass-through              |
| POST   | `/v1/ingest/delegation`             | record subagent delegation          |
| POST   | `/v1/ingest/hermes-compaction`      | record Hermes auxiliary summary     |
| POST   | `/v1/ingest/builtin-memory-mirror`  | mirror Hermes built-in memory tool  |
| POST   | `/v1/recall`                        | retrieve a context pack             |
| POST   | `/v1/memory/remember`               | promote a fact/decision             |
| POST   | `/v1/memory/correct`                | supersede/invalidate memory         |
| POST   | `/v1/memory/forget`                 | soft/hard delete                    |
| POST   | `/v1/memory/timeline`               | decisions + facts chronologically   |
| POST   | `/v1/memory/sources`                | provenance resolver                 |
| GET    | `/admin`                            | zero-build browser admin UI         |
| POST   | `/v1/admin/projects`                | project overview counts             |
| POST   | `/v1/admin/search`                  | admin search                        |
| POST   | `/v1/admin/detail`                  | detail + provenance payload         |
| POST   | `/v1/admin/export`                  | Markdown export                     |
| POST   | `/v1/admin/conflicts/scan`          | detect memory conflicts             |
| POST   | `/v1/admin/conflicts/list`          | list unresolved conflicts           |
| POST   | `/v1/admin/audit`                   | audit log viewer                    |

The `/admin` UI is a small light-themed HTML/CSS/vanilla-JS console with a
mobile bottom nav, project overview, detail drawer, recall playground, correction
flow, conflict/audit panels, sources lookup, soft delete, and Markdown export. Set
`SIQUEIRA_ADMIN_PASSWORD` and `SIQUEIRA_ADMIN_SESSION_SECRET` to enable the
built-in form login with a signed `HttpOnly`/`SameSite=Lax` session cookie. API
endpoints still accept Bearer tokens, and browser admin sessions can call the
admin/recall/memory endpoints without exposing `SIQUEIRA_API_TOKEN` to JS. Keep
`SIQUEIRA_ADMIN_COOKIE_SECURE=false` on plain HTTP tailnet URLs; flip it only
behind HTTPS.

## Run the worker

```bash
python -m siqueira_memo.workers.worker
```

The default queue backend is in-memory, which is enough for local dev. Set
`SIQUEIRA_QUEUE_BACKEND=redis` in production.

## Enable the Hermes plugin

Add to your Hermes config:

```yaml
memory:
  provider: siqueira-memo
```

Make sure the `plugins/memory/siqueira-memo` directory is on the Hermes plugin
path. The plugin entrypoint is `plugins/memory/siqueira-memo/__init__.py`
which forwards to `SiqueiraMemoProvider` and exposes the following tools:

- `siqueira_memory_recall`
- `siqueira_memory_remember`
- `siqueira_memory_correct`
- `siqueira_memory_forget`
- `siqueira_memory_timeline`
- `siqueira_memory_sources`

See `plugins/memory/siqueira-memo/system_prompt.md` for the assistant-facing
policy. The canonical prompt lives in
`src/siqueira_memo/hermes_provider/system_prompt.md`; the plugin copy points at
it. Plan §31.13 requires hash parity in production.

## Evals

Deterministic golden suite:

```bash
python -m siqueira_memo.evals.runner --min-pass-rate 0.8
```

CI:

```bash
pytest -q
pytest tests/evals -q       # deterministic
```

Blocking CI (plan §26.2): unit tests + redaction corpus + deterministic
retrieval evals + deletion cascade tests.

## Imports (migration from Hermes SQLite / Hindsight)

```bash
# Hermes session transcripts (JSONL or SQLite SessionDB).
python scripts/import_hermes_sessions.py path/to/export.jsonl --format jsonl
python scripts/import_hermes_sessions.py path/to/session.sqlite --format sqlite

# Hindsight offline import happens through siqueira_memo.services.hindsight_adapter,
# e.g. via a Dramatiq worker job; it never runs as a live fallback.
```

## Layout

```
src/siqueira_memo/
  api/              FastAPI routes and dependencies
  evals/            golden question corpus + runner
  hermes_provider/  MemoryProvider class + tool schemas + canonical system prompt
  models/           SQLAlchemy models with Postgres/SQLite compatibility
  prompts/          versioned extraction/gate/verifier prompt artifacts
  schemas/          Pydantic request/response + event payload schemas
  services/         ingest, redaction, chunking, embedding, retrieval,
                     extraction, conflict, deletion, retention, prompts,
                     entity linking, session/hindsight import
  utils/            canonical normalisation, tokenisers
  workers/          queue abstraction, job handlers, worker entrypoint
plugins/memory/siqueira-memo/
                    Hermes plugin shim: __init__.py, plugin.yaml,
                    system_prompt.md, README.md, cli.py
alembic/           schema migrations
scripts/           operational CLIs
tests/
  unit/            per-service unit tests
  integration/     FastAPI end-to-end via httpx ASGI
  evals/           deterministic golden + redaction corpus evals
  fixtures/        redaction corpora
```

## Invariants (from the plan)

1. Raw archive is the source of truth; derived memory is rebuildable.
2. Every fact/decision has `source_event_ids`.
3. No secret ever reaches embeddings or LLM inputs.
4. User corrections supersede older memory.
5. Hindsight is import-only; Siqueira is the live provider.
6. Compact memory is a bootloader, not a substitute for Siqueira recall.
7. Conflicts are surfaced, not flattened.
8. Deletion removes derived chunks + embeddings and marks derived summaries.

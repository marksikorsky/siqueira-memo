# Siqueira Memo

**Hermes-native long-term memory that remembers like a careful engineer, not like a sticky note.**

Siqueira Memo captures conversations, tool output, artifacts, decisions, corrections, and provenance; redacts secrets before anything reaches embeddings or LLMs; and gives Hermes a source-backed recall layer with facts, decisions, summaries, chunks, conflicts, and audit trails.

It is designed for one boring but important job: make future agent runs know what we already learned, while still letting live user instructions and fresh tool output win.

## Why it exists

Normal assistant memory is usually one of two bad things:

- a tiny global text blob that gets stale and impossible to audit;
- raw transcript search with no structure, provenance, or deletion story.

Siqueira sits between Hermes and Postgres/pgvector and gives memory actual shape:

- **raw events stay as the source of truth**;
- **derived facts/decisions are rebuildable**;
- **every durable memory can point back to source events/messages**;
- **corrections supersede old memory instead of silently overwriting it**;
- **conflicts are surfaced, not flattened into a fake single truth**.

## What it does

### 1. Captures the important stuff automatically

The Hermes provider records completed turns through a non-blocking queue. In aggressive capture mode it can store:

- user and assistant messages;
- tool calls and redacted tool outputs;
- artifacts/files;
- subagent delegation observations;
- Hermes context-compaction summaries;
- mirrored writes from Hermes' built-in memory tool;
- session-end summaries and pre-compression extraction markers.

A scope classifier attaches project/topic metadata where it can. A memory-capture classifier then decides whether the turn deserves promotion into structured memory.

Capture can run in two layers:

- **LLM classifier** — optional OpenAI-compatible `/chat/completions` endpoint, redacted input, JSON-only output;
- **heuristic fallback** — deterministic useful-signal matcher for preferences, decisions, infra facts, tax/accounting facts, repo analyses, Tailscale/server inventory, architecture notes, and similar durable information.

### 2. Stores structured memory, not just text

Siqueira promotes durable information into:

- **facts** — subject / predicate / object / statement / confidence / validity window;
- **decisions** — topic / decision / rationale / tradeoffs / reversibility / status;
- **summaries** — session/topic summaries for compact context;
- **chunks** — redacted searchable slices of messages, tool output, and artifacts;
- **source links** — event/message IDs backing every promoted memory;
- **conflicts** — detected disagreements between active memories.

Statuses matter: active, superseded, invalidated, deleted. The point is to keep history honest.

### 3. Recalls memory in useful packs

Recall combines structured memory with lexical/vector chunk retrieval and returns a compact `ContextPack`:

- `answer_context` — short prompt-ready summary;
- `decisions`;
- `facts`;
- `chunks`;
- `summaries`;
- `source_snippets`;
- `conflicts`;
- `confidence`;
- `warnings`;
- `token_estimate`.

Supported modes:

| mode | use when | shape |
|---|---|---|
| `fast` | lightweight hinting | smallest pack |
| `balanced` | default agent recall | prompt-safe, source-backed |
| `deep` | user asks for depth | more facts/chunks/summaries |
| `forensic` | provenance-heavy investigation | widest pack; not for automatic prefetch |

Recall supports project/topic/entity filters and logs retrieval diagnostics for later audit.

### 4. Lets the agent manage memory explicitly

Hermes gets six native tools:

| tool | purpose |
|---|---|
| `siqueira_memory_recall` | retrieve source-backed context |
| `siqueira_memory_remember` | persist a fact or decision |
| `siqueira_memory_correct` | supersede/replace stale memory |
| `siqueira_memory_forget` | soft-delete or hard-delete memory |
| `siqueira_memory_timeline` | view facts/decisions chronologically |
| `siqueira_memory_sources` | resolve memory back to source events/messages |

The provider also supports Hermes lifecycle hooks: `system_prompt_block`, `prefetch`, `queue_prefetch`, `sync_turn`, `on_pre_compress`, `on_session_end`, `on_memory_write`, `on_delegation`, `on_turn_start`, and `shutdown`.

Prefetch warming can use Redis so the next turn gets a cached context pack without blocking the current one.

### 5. Includes a built-in admin UI

Open `/admin` for a lightweight browser console — no frontend build step, just FastAPI serving HTML/CSS/vanilla JS.

The dashboard includes:

- project overview cards with fact/decision/message/summary counts;
- timeline and recent memory views;
- search across facts, decisions, messages, and summaries;
- detail drawer with provenance;
- recall playground for testing context packs;
- capture counters and recent global memories;
- manual remember/correct forms;
- conflict scan/list/resolve API support;
- audit log viewer;
- source lookup;
- soft delete;
- Markdown export by project/topic.

Admin auth is optional but built in: set `SIQUEIRA_ADMIN_PASSWORD` and `SIQUEIRA_ADMIN_SESSION_SECRET` to enable form login with a signed `HttpOnly` / `SameSite=Lax` session cookie. API clients can still use bearer tokens.

![Siqueira Memo admin dashboard](docs/assets/siqueira-admin-dashboard.jpg)

### Recommended secure setup

If you are installing this on a real server, ask Hermes to configure it end-to-end instead of hand-editing secrets and ports yourself:

```text
Install and configure Siqueira Memo for Hermes.
Generate a strong admin password and session secret.
Set SIQUEIRA_ADMIN_PASSWORD and SIQUEIRA_ADMIN_SESSION_SECRET in .env.
Enable the siqueira-memo MemoryProvider in Hermes.
Restart the services and verify /readyz and /admin/login.
Expose the admin UI only through Tailscale/tailnet access, not the public internet.
```

Security recommendation: keep the API, admin UI, Postgres, and Redis off the public internet. The sane default is localhost-only containers plus a private Tailscale URL/IP for `/admin`. If you absolutely must expose it publicly, use TLS, strong auth, and a reverse proxy — but for an agent memory database, public exposure is usually the wrong tradeoff.

## Architecture

```text
Hermes
  └─ MemoryProvider plugin
      ├─ tools: recall / remember / correct / forget / timeline / sources
      ├─ hooks: sync_turn / prefetch / session_end / delegation / memory_write
      └─ queue jobs

Siqueira API + Worker
  ├─ ingest pipeline
  │   ├─ redaction
  │   ├─ raw event archive
  │   ├─ message/tool/artifact storage
  │   ├─ chunking
  │   ├─ embeddings
  │   └─ structured memory capture
  ├─ recall pipeline
  │   ├─ facts + decisions first
  │   ├─ lexical/vector chunks
  │   ├─ summaries
  │   └─ conflict warnings
  └─ admin / audit / export

Postgres + pgvector + Redis
  ├─ source events/messages/tool events/artifacts
  ├─ facts/decisions/summaries/entities
  ├─ chunks + embedding tables
  ├─ retrieval logs
  └─ deletion/audit/conflict records
```

SQLite and in-memory queue are supported for tests/dev, but production wants Postgres + pgvector + Redis.

## One-command install

```bash
./scripts/bootstrap.sh
```

The bootstrap script:

1. generates local `.env` secrets if needed;
2. pulls `pgvector/pgvector:pg16` and `redis:7-alpine`;
3. builds the app image;
4. starts Postgres + Redis;
5. runs Alembic migrations;
6. starts API + worker;
7. checks `/healthz` and `/readyz`;
8. if Hermes is installed locally, installs Siqueira as the active `MemoryProvider`, updates `~/.hermes/.env` + `~/.hermes/config.yaml`, verifies discovery, and restarts the Hermes gateway when appropriate.

Default local install uses `SIQUEIRA_EMBEDDING_PROVIDER=mock`, so no embedding API key is required just to bring the system up.

Service-only install:

```bash
SIQUEIRA_INSTALL_HERMES_PROVIDER=false ./scripts/bootstrap.sh
```

More operator detail: [`docs/INSTALL.md`](docs/INSTALL.md).

## Manual install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,postgres,queue,secrets,otel]'
```

Optional extras:

| extra | adds |
|---|---|
| `postgres` | `asyncpg`, `pgvector` |
| `queue` | Redis + Dramatiq dependencies |
| `secrets` | `detect-secrets` redaction support |
| `otel` | OpenTelemetry packages |
| `dev` | pytest, ruff, mypy |

## Configuration

```bash
cp .env.example .env
```

Common settings:

```dotenv
SIQUEIRA_ENV=development
SIQUEIRA_API_TOKEN=change-me
SIQUEIRA_DATABASE_URL=postgresql+asyncpg://siqueira:password@127.0.0.1:5432/siqueira_memo
SIQUEIRA_REDIS_URL=redis://127.0.0.1:6379/0
SIQUEIRA_QUEUE_BACKEND=redis

# Admin UI login
SIQUEIRA_ADMIN_PASSWORD=change-me
SIQUEIRA_ADMIN_SESSION_SECRET=change-me
SIQUEIRA_ADMIN_COOKIE_SECURE=false

# Capture behavior
SIQUEIRA_MEMORY_CAPTURE_MODE=aggressive
SIQUEIRA_MEMORY_CAPTURE_TARGET_RATIO=0.8
SIQUEIRA_MEMORY_CAPTURE_SAVE_RAW_TURNS=true
SIQUEIRA_MEMORY_CAPTURE_EXTRACT_STRUCTURED=true

# Optional LLM classifier for automatic memory capture
SIQUEIRA_MEMORY_CAPTURE_LLM_ENABLED=false
SIQUEIRA_MEMORY_CAPTURE_LLM_BASE_URL=
SIQUEIRA_MEMORY_CAPTURE_LLM_API_KEY=
SIQUEIRA_MEMORY_CAPTURE_LLM_MODEL=gpt-5

# Embeddings
SIQUEIRA_EMBEDDING_PROVIDER=mock      # mock | openai | local
SIQUEIRA_EMBEDDING_MODEL=text-embedding-3-large
SIQUEIRA_EMBEDDING_DIMS=3072
```

Real secrets belong in `.env`, never in git.

## Running locally

Start infra and migrate:

```bash
docker compose up -d postgres redis
alembic upgrade head
```

Run API:

```bash
uvicorn siqueira_memo.main:app --host 127.0.0.1 --port 8787 --reload
# or
siqueira-memo
```

Run worker:

```bash
siqueira-memo-worker
# or
python -m siqueira_memo.workers.worker
```

Useful checks:

```bash
curl http://127.0.0.1:8787/healthz
curl http://127.0.0.1:8787/readyz
./scripts/smoke_compose.sh
```

Dev reset:

```bash
python scripts/dev_reset_db.py
```

## API surface

All `POST` routes require `Authorization: Bearer $SIQUEIRA_API_TOKEN`, unless called from an authenticated admin browser session.

### Health

| method | path | purpose |
|---|---|---|
| `GET` | `/healthz` | liveness |
| `GET` | `/readyz` | readiness + Postgres/pgvector/Redis status |

### Ingest

| method | path | purpose |
|---|---|---|
| `POST` | `/v1/ingest/message` | store a user/assistant message |
| `POST` | `/v1/ingest/tool-event` | store tool input/output with redaction |
| `POST` | `/v1/ingest/artifact` | register a file/artifact |
| `POST` | `/v1/ingest/event` | generic raw event |
| `POST` | `/v1/ingest/delegation` | record subagent delegation |
| `POST` | `/v1/ingest/hermes-compaction` | record Hermes compaction summary |
| `POST` | `/v1/ingest/builtin-memory-mirror` | mirror Hermes built-in memory writes |

### Recall and memory management

| method | path | purpose |
|---|---|---|
| `POST` | `/v1/recall` | retrieve a context pack |
| `POST` | `/v1/memory/remember` | promote a fact/decision |
| `POST` | `/v1/memory/correct` | supersede or replace memory |
| `POST` | `/v1/memory/forget` | soft/hard delete memory |
| `POST` | `/v1/memory/timeline` | chronological facts + decisions |
| `POST` | `/v1/memory/sources` | resolve provenance |

### Admin

| method | path | purpose |
|---|---|---|
| `GET` | `/admin` | browser dashboard |
| `POST` | `/admin/login` | admin form login |
| `POST` | `/admin/logout` | clear admin session |
| `POST` | `/v1/admin/projects` | project/topic overview counts |
| `POST` | `/v1/admin/search` | search memories/messages/summaries |
| `POST` | `/v1/admin/capture` | capture counters and recent global memories |
| `POST` | `/v1/admin/detail` | full item payload + provenance |
| `POST` | `/v1/admin/export` | Markdown export |
| `POST` | `/v1/admin/conflicts/scan` | detect conflicts |
| `POST` | `/v1/admin/conflicts/list` | list conflicts |
| `POST` | `/v1/admin/conflicts/resolve` | resolve conflict by supersession |
| `POST` | `/v1/admin/audit` | deletion/audit log viewer |

## Hermes plugin

Enable in Hermes config:

```yaml
memory:
  provider: siqueira-memo
```

The plugin shim lives in [`plugins/memory/siqueira-memo`](plugins/memory/siqueira-memo). The canonical assistant-facing policy is [`src/siqueira_memo/hermes_provider/system_prompt.md`](src/siqueira_memo/hermes_provider/system_prompt.md); startup checks hash parity with the plugin copy to prevent silent prompt drift.

See [`docs/HERMES_PLUGIN.md`](docs/HERMES_PLUGIN.md) for plugin-specific notes.

## CLI helpers

Installed console scripts:

```bash
siqueira-memo
siqueira-memo-worker
siqueira-memo-evals
siqueira-memo-import-hermes
siqueira-memo-rebuild-embeddings
siqueira-memo-export-markdown
```

Script wrappers are also available in [`scripts/`](scripts): bootstrap, provider install, smoke test, import, export, rebuild embeddings, and dev reset.

## Testing and evals

```bash
pytest -q
pytest tests/evals -q
siqueira-memo-evals --min-pass-rate 0.8
```

The test suite covers redaction, chunking, embeddings, retrieval, promotion/correction/deletion, conflicts, admin routes, Hermes provider behavior, prompt hash parity, CLI entrypoints, and security regressions.

## Repository layout

```text
src/siqueira_memo/
  api/              FastAPI routes and auth dependencies
  cli/              import/export/rebuild CLI entrypoints
  evals/            deterministic golden evals
  hermes_provider/  MemoryProvider class, tool schemas, canonical prompt
  models/           SQLAlchemy models for events, memory, chunks, retrieval, audit
  prompts/          versioned extraction/gate/verifier prompt artifacts
  schemas/          Pydantic request/response schemas
  services/         ingest, redaction, chunking, embeddings, retrieval,
                    extraction, conflicts, deletion, retention, export,
                    scope/capture classifiers, imports
  utils/            canonical normalization and token helpers
  workers/          queue abstraction, Redis backend, job handlers, worker
plugins/memory/siqueira-memo/
                    Hermes plugin shim
alembic/            migrations
docs/               install/plugin/CI docs
scripts/            operational scripts
tests/              unit, integration, eval, fixture tests
```

## Invariants

1. Raw archive is the source of truth; derived memory is rebuildable.
2. Every durable fact/decision should have source provenance.
3. Secrets are redacted before embeddings or LLM classifier paths.
4. User corrections supersede older memory.
5. Hindsight is import-only; Siqueira is the live provider.
6. Compact memory is a bootloader, not the memory system.
7. Conflicts are surfaced instead of silently flattened.
8. Forget/delete paths remove derived chunks/embeddings and leave auditable state.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

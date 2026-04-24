# Siqueira Memo Production-Ready Implementation Plan

> **For Hermes:** Use this plan as the source-of-truth for building `siqueira-memo`, a production-grade personal memory system for this Hermes assistant. Do not implement shortcuts that compromise provenance, deletion, security, or retrieval quality.

**Goal:** Build a production-ready, Hermes-native long-term memory system that stores near-complete conversational history, decisions, conclusions, project context, tool events, artifacts, and source-backed recall without polluting compact memory.

**Architecture:** `siqueira-memo` is a local-first memory service with an append-only event log, raw transcript archive, structured memory tables, pgvector-based hybrid search, async processors, and a Hermes `MemoryProvider` plugin. Hindsight is not a parallel live provider inside Hermes runtime; it is only an offline/import source during migration.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL 16, pgvector, SQLAlchemy 2.x + Alembic, Pydantic v2, Redis + Dramatiq or Celery, OpenTelemetry, pytest, Docker Compose, optional S3-compatible artifact storage, native Hermes tool integration.

---

## 0. Product Definition

### 0.1 What `siqueira-memo` must do

`Siqueira Memo` must remember substantially more than ordinary assistant memory:

- raw user/assistant messages;
- tool calls and tool outputs;
- files/artifacts created or modified;
- project-level context;
- decisions and rationale;
- facts with validity windows;
- user corrections;
- rejected ideas and superseded decisions;
- summaries by session, topic, and project;
- source snippets for every important memory;
- timelines for entities/projects/topics;
- deletion/invalidation history;
- retrieval logs and eval results.

The system must support questions like:

- “Что мы решили по памяти?”
- “Почему мы выбрали этот подход?”
- “Что последнее известно про Shannon auth?”
- “Какие решения были отменены?”
- “Покажи источники этого вывода.”
- “Забудь вот эту тему.”

### 0.2 Non-goals

Do **not** build first around MCP. MCP may be added later as an adapter, but the primary integration is Hermes-native.

Do **not** use vector search as the only memory mechanism. Vector-only memory becomes confident garbage.

Do **not** put detailed history into Hermes compact memory. Compact memory is only a bootloader.

Do **not** embed secrets, `.env` files, private keys, seed phrases, cookies, bearer tokens, database URLs, or raw credentials.

Do **not** let Hindsight and custom memory both compete as equal sources of truth.

Do **not** implement a custom `ContextEngine` replacement in v1. Siqueira Memo v1 integrates via `MemoryProvider` only; a context-engine plugin can be evaluated later if Hermes compression needs to become Siqueira-aware.

---

## 1. Core Design Principles

### 1.1 Raw archive is source of truth

All derived memory must be reproducible from the raw event log and raw messages.

Derived memory includes:

- embeddings;
- chunks;
- summaries;
- facts;
- decisions;
- topic states;
- project states;
- entity relationships.

If the extraction logic improves, derived memory should be rebuildable.

### 1.2 Every claim needs provenance

Every fact, decision, summary, or answer card must point to source events/messages.

A memory without sources is second-class and must be marked `unverified`.

### 1.3 Current/live data beats old memory

Precedence order:

1. current user instruction;
2. live tool output;
3. custom memory verified/current;
4. compact memory hard preferences;
5. Siqueira-imported Hindsight memories after verification;
6. raw session search;
7. old unverified summaries.

### 1.4 Memory must model time

Facts and decisions must support:

- `active`;
- `stale`;
- `superseded`;
- `invalidated`;
- `deleted`;
- `unverified`.

A fact from three months ago about a server is not automatically current.

### 1.5 Security is not optional

Redaction happens before embeddings and before LLM summarization unless explicitly allowed.

Sensitive raw content can be stored, but indexed content must be redacted.

### 1.6 User corrections dominate

If Mark corrects the assistant, that event has maximum priority. It can invalidate or supersede previous memory.

---

## 2. Repository Layout

Create this structure:

```text
siqueira-memo/
  README.md
  IMPLEMENTATION_PLAN.md
  pyproject.toml
  docker-compose.yml
  .env.example
  alembic.ini
  alembic/
    env.py
    versions/
  plugins/
    memory/
      siqueira-memo/
        __init__.py
        plugin.yaml
        README.md
        cli.py
        system_prompt.md
  src/
    siqueira_memo/
      __init__.py
      config.py
      logging.py
      db.py
      main.py
      api/
        __init__.py
        routes_health.py
        routes_ingest.py
        routes_recall.py
        routes_memory.py
        routes_admin.py
      models/
        __init__.py
        base.py
        events.py
        messages.py
        tools.py
        artifacts.py
        chunks.py
        entities.py
        facts.py
        decisions.py
        summaries.py
        retrieval.py
        evals.py
      schemas/
        __init__.py
        ingest.py
        recall.py
        memory.py
        admin.py
      services/
        __init__.py
        ingest_service.py
        redaction_service.py
        chunking_service.py
        embedding_service.py
        extraction_service.py
        dedupe_service.py
        retrieval_service.py
        rerank_service.py
        context_pack_service.py
        deletion_service.py
        hindsight_adapter.py
        session_importer.py
      workers/
        __init__.py
        worker.py
        jobs.py
      hermes_provider/
        __init__.py
        provider.py
        tools.py
        system_prompt.md
      evals/
        __init__.py
        golden_questions.py
        runner.py
  tests/
    unit/
    integration/
    evals/
  scripts/
    dev_reset_db.py
    import_hermes_sessions.py
    rebuild_embeddings.py
    export_markdown.py
```

---

## 3. Data Model

Use PostgreSQL + pgvector. Avoid separate graph DB until Postgres becomes a proven bottleneck.

### 3.1 `memory_events`

Append-only canonical event log.

```sql
CREATE TABLE memory_events (
  id UUID PRIMARY KEY,
  event_type TEXT NOT NULL,
  source TEXT NOT NULL,
  actor TEXT NOT NULL,
  session_id TEXT,
  trace_id TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX memory_events_type_idx ON memory_events(event_type);
CREATE INDEX memory_events_session_idx ON memory_events(session_id);
CREATE INDEX memory_events_created_idx ON memory_events(created_at);
```

Event types:

```text
message_received
assistant_message_sent
tool_called
tool_result_received
artifact_created
artifact_modified
summary_created
fact_extracted
decision_recorded
fact_invalidated
decision_superseded
memory_deleted
user_correction_received
hindsight_imported
delegation_observed
builtin_memory_mirror
hermes_auxiliary_compaction_observed
retrieval_performed_deprecated_use_retrieval_logs
```

### 3.2 `messages`

```sql
CREATE TABLE messages (
  id UUID PRIMARY KEY,
  event_id UUID REFERENCES memory_events(id),
  session_id TEXT NOT NULL,
  platform TEXT NOT NULL,
  chat_id TEXT,
  role TEXT NOT NULL,
  content_raw TEXT NOT NULL,
  content_redacted TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'live_turn',
  language TEXT,
  project TEXT,
  topic TEXT,
  entities TEXT[] DEFAULT '{}',
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL
);
```

### 3.3 `tool_events`

```sql
CREATE TABLE tool_events (
  id UUID PRIMARY KEY,
  event_id UUID REFERENCES memory_events(id),
  session_id TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  input_raw JSONB NOT NULL,
  input_redacted JSONB NOT NULL,
  output_raw TEXT,
  output_redacted TEXT,
  output_summary TEXT,
  exit_status TEXT NOT NULL,
  artifact_refs TEXT[] DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL
);
```

Policy:

- small output: store raw + redacted;
- medium output: store raw + summary + selected chunks;
- large output: store pointer + hash + summary;
- secret-like output: store raw only if encrypted/restricted, never embed.

### 3.4 `artifacts`

```sql
CREATE TABLE artifacts (
  id UUID PRIMARY KEY,
  event_id UUID REFERENCES memory_events(id),
  type TEXT NOT NULL,
  path TEXT,
  uri TEXT,
  content_hash TEXT,
  summary TEXT,
  project TEXT,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL
);
```

### 3.5 `chunks`

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
  id UUID PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_id UUID NOT NULL,
  chunk_text TEXT NOT NULL,
  embedding vector(1536),
  token_count INT NOT NULL,
  project TEXT,
  topic TEXT,
  entities TEXT[] DEFAULT '{}',
  sensitivity TEXT NOT NULL DEFAULT 'normal',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_project_idx ON chunks(project);
CREATE INDEX chunks_topic_idx ON chunks(topic);
CREATE INDEX chunks_created_idx ON chunks(created_at);
CREATE INDEX chunks_fts_idx ON chunks USING gin(to_tsvector('simple', chunk_text));
```

### 3.6 `entities`

```sql
CREATE TABLE entities (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  aliases TEXT[] NOT NULL DEFAULT '{}',
  description TEXT,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Entity types:

```text
person
project
server
repo
wallet
company
product
api
model
document
topic
```

### 3.7 `facts`

```sql
CREATE TABLE facts (
  id UUID PRIMARY KEY,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  statement TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  status TEXT NOT NULL,
  valid_from TIMESTAMPTZ,
  valid_to TIMESTAMPTZ,
  source_event_ids UUID[] NOT NULL,
  source_message_ids UUID[] NOT NULL DEFAULT '{}',
  superseded_by UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB NOT NULL DEFAULT '{}'
);
```

### 3.8 `decisions`

```sql
CREATE TABLE decisions (
  id UUID PRIMARY KEY,
  project TEXT,
  topic TEXT NOT NULL,
  decision TEXT NOT NULL,
  context TEXT NOT NULL,
  options_considered JSONB NOT NULL DEFAULT '[]',
  rationale TEXT NOT NULL,
  tradeoffs JSONB NOT NULL DEFAULT '{}',
  status TEXT NOT NULL,
  reversible BOOLEAN NOT NULL DEFAULT TRUE,
  superseded_by UUID,
  decided_at TIMESTAMPTZ NOT NULL,
  source_event_ids UUID[] NOT NULL,
  source_message_ids UUID[] NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB NOT NULL DEFAULT '{}'
);
```

Decision statuses:

```text
proposed
active
rejected
superseded
invalidated
```

### 3.9 `session_summaries`, `topic_summaries`, `project_states`

Summaries are derived, versioned artifacts.

```sql
CREATE TABLE session_summaries (
  id UUID PRIMARY KEY,
  session_id TEXT NOT NULL,
  summary_short TEXT NOT NULL,
  summary_long TEXT NOT NULL,
  decisions UUID[] DEFAULT '{}',
  facts UUID[] DEFAULT '{}',
  open_questions TEXT[] DEFAULT '{}',
  source_event_ids UUID[] NOT NULL,
  model TEXT NOT NULL,
  version INT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Project states must represent latest known project context and must cite sources.

---

## 4. API Design

### 4.1 Health

```http
GET /healthz
GET /readyz
```

`/readyz` checks DB, pgvector, Redis/worker connectivity, and migration version.

### 4.2 Ingest

```http
POST /v1/ingest/event
POST /v1/ingest/message
POST /v1/ingest/tool-event
POST /v1/ingest/artifact
```

All ingest endpoints must:

1. validate schema;
2. create `memory_events` row;
3. redact sensitive fields;
4. insert raw + redacted data;
5. enqueue async processing jobs;
6. return event/message IDs.

### 4.3 Recall

```http
POST /v1/recall
```

Request:

```json
{
  "query": "что мы решили по памяти?",
  "project": null,
  "mode": "balanced",
  "types": ["decisions", "facts", "chunks", "summaries"],
  "include_sources": true,
  "limit": 20
}
```

Modes:

```text
fast      - structured memory + top vector, target <300ms
balanced  - structured + hybrid + light rerank, target <1.5s
deep      - broad search + rerank + timeline, target <15s
forensic  - maximum provenance, raw snippets, slower but source-heavy
```

### 4.4 Remember/update/forget

```http
POST /v1/memory/remember
POST /v1/memory/correct
POST /v1/memory/forget
POST /v1/memory/timeline
POST /v1/memory/sources
```

`forget` must support:

- soft invalidation;
- hard delete;
- delete embeddings;
- delete summaries derived from deleted content;
- audit record of deletion without retaining deleted sensitive content.

---

## 5. Hermes Integration

Implement `siqueira-memo` as a Hermes **Memory Provider plugin**, not as a general plugin, MCP server, or standalone custom toolset.

Hermes' actual integration point is `agent.memory_provider.MemoryProvider`. Hermes loads exactly one external provider via `memory.provider` config, alongside the built-in compact memory files. The plugin must live under:

```text
plugins/memory/siqueira-memo/
  __init__.py      # MemoryProvider implementation + register(ctx)
  plugin.yaml
  README.md
  cli.py           # optional provider-specific CLI commands
  system_prompt.md # versioned static provider instructions
```

Registration:

```python
def register(ctx) -> None:
    ctx.register_memory_provider(SiqueiraMemoProvider())
```

### 5.1 MemoryProvider methods to implement

Required/core:

```text
name -> "siqueira-memo"
is_available() -> local config/dependency check only, no network calls
initialize(session_id, **kwargs)
get_tool_schemas()
handle_tool_call(tool_name, args, **kwargs)
get_config_schema()
save_config(values, hermes_home)
shutdown()
```

Documented hooks that must be implemented:

```text
system_prompt_block()
prefetch(query, *, session_id="")
queue_prefetch(query, *, session_id="")
sync_turn(user_content, assistant_content, *, session_id="")
on_pre_compress(messages) -> str
on_session_end(messages)
on_memory_write(action, target, content)
```

Source-inspected/version-sensitive hooks to support when present, with fallback if absent:

```text
on_delegation(task, result, *, child_session_id="", **kwargs)
on_turn_start(turn_number, message, **kwargs)
```

### 5.2 Static vs dynamic memory context

`system_prompt_block()` is for static instructions only:

- provider name/status;
- tool usage policy;
- warning that prefetched memory is contextual, not user input;
- precedence rules: current user/live tools beat memory;
- failure behavior.

It must **not** include dynamic recalled memories.

Dynamic recall belongs to:

```text
prefetch(query)       # returns cached/pre-warmed context for the current API call
queue_prefetch(query) # warms context for the next turn
siqueira_memory_recall tool # explicit same-turn/manual recall
```

### 5.3 Tools exposed through MemoryProvider

Expose these via `get_tool_schemas()`:

```text
siqueira_memory_recall
siqueira_memory_remember
siqueira_memory_correct
siqueira_memory_forget
siqueira_memory_timeline
siqueira_memory_sources
```

Prefix tools with `siqueira_` to avoid collisions with built-in `memory`, Hindsight, or future providers.

### 5.4 Prefetch path vs explicit tool path

Hermes prefetch is cache/prewarm oriented:

```text
queue_prefetch(previous_query) after a completed turn
prefetch(current_query) before each API call returns available cached/prewarmed context
```

Therefore, automatic prefetch context may lag by one turn depending on provider implementation. The plan must not assume that automatic prefetch can ingest and recall the current user message before the same response.

For same-turn needs — “запомни это”, “исправь память”, “что мы решили прямо сейчас?” — the assistant must use explicit tools:

```text
siqueira_memory_remember
siqueira_memory_correct
siqueira_memory_recall(mode="deep"|"forensic")
```

### 5.5 sync_turn is non-blocking

`sync_turn()` is called after every completed turn and must not block Hermes.

Implementation requirement:

```text
sync_turn() only enqueues raw turn ingestion into Redis/Dramatiq or a daemon thread.
No LLM extraction, embeddings, heavy DB work, or network calls inline.
```

This makes Redis/Dramatiq a hard production requirement, not over-engineering.

### 5.6 Compression hooks

`on_pre_compress(messages)` is the safest hook before Hermes discards/summarizes old context.

Use it to:

- enqueue high-priority extraction for messages about to be compressed;
- return short provider insights if the installed Hermes version actually threads return value into compressor;
- never rely solely on return-value injection unless verified by tests against the running Hermes version.

Local source check showed some Hermes builds call `on_pre_compress(messages)` but discard the return value. Therefore, Siqueira must treat this hook primarily as a persistence/extraction trigger, not the only way to influence compression.

### 5.7 Profile and identity scoping

`initialize()` receives `hermes_home`, `agent_identity`, `agent_workspace`, `platform`, `user_id`, `chat_id`, `thread_id`, and related gateway fields when available.

Every stored record must include:

```text
profile_id        # derived from agent_identity or hermes_home hash
hermes_home       # or hashed/stable representation
session_id
platform
user_id/chat_id/thread_id when available
agent_context     # primary/subagent/cron/flush
```

Providers should skip durable writes when `agent_context` is not `primary`, unless explicitly handling `on_delegation()` from the parent provider.

### 5.8 No prompt pollution

The tool and prefetch outputs must return compact context packs, not dumps.

Return format:

```json
{
  "answer_context": "short synthesized context",
  "decisions": [],
  "facts": [],
  "source_snippets": [],
  "confidence": "high|medium|low",
  "warnings": []
}
```

---

## 6. Hindsight Migration and Non-Coexistence

Hermes enforces one external live memory provider at a time. Therefore, Siqueira Memo and Hindsight cannot both be active live providers inside the same Hermes runtime.

### 6.1 Roles

```text
Compact memory: always-on built-in bootloader/hard preferences
Siqueira Memo: active external MemoryProvider and source of truth
Hindsight: offline/import-only source for migration, not live fallback
Hermes session_search: separate tool/fallback for old session transcripts when available
```

### 6.2 Hindsight import adapter

Implement `hindsight_adapter.py` only for offline import/backfill jobs:

```text
Hindsight -> external candidates -> verify/dedupe -> promote or reject in Siqueira
```

Hindsight imported data must be marked:

```json
{
  "source": "hindsight_import",
  "trust_level": "secondary",
  "requires_verification": true,
  "live_provider": false
}
```

Never query Hindsight as a live fallback from the Siqueira MemoryProvider while Hermes is running with Siqueira active. Never let Hindsight silently overwrite an active Siqueira decision.

---

## 7. Processing Pipeline

### 7.1 Async jobs

Use worker jobs for:

```text
redact_message
chunk_message
embed_chunks
extract_entities
extract_facts
extract_decisions
summarize_session
dedupe_facts
update_project_state
run_retrieval_evals
```

### 7.2 Redaction first

No chunking or embeddings before redaction.

Redaction detectors:

- API keys;
- bearer tokens;
- SSH keys;
- private keys;
- seed phrases;
- cookies;
- auth headers;
- database URLs;
- `.env` blocks;
- JWTs;
- cloud credentials.

Represent secrets as:

```text
[SECRET_REF:type/name/hash]
```

### 7.3 Extraction prompts

LLM extraction must output strict JSON with schema validation.

Extract:

- facts;
- decisions;
- rejected options;
- user corrections;
- project names;
- entity aliases;
- open questions;
- follow-up tasks.

Invalid JSON is a failed job, not silently accepted garbage.

---

## 8. Retrieval Design

### 8.1 Hybrid search

Retrieval combines:

1. exact keyword / full-text search;
2. vector similarity;
3. structured facts;
4. structured decisions;
5. project/entity filters;
6. recency weighting;
7. Siqueira-imported Hindsight candidates after verification;
8. raw Hermes session fallback if needed.

### 8.2 Reranking

Use a reranker for `balanced`, `deep`, and `forensic` modes.

Rerank criteria:

- direct relevance;
- source quality;
- recency;
- decision/fact status;
- project match;
- user-correction priority;
- explicit source quotes.

### 8.3 Conflict resolution

If results conflict, output must include:

```json
{
  "conflicts": [
    {
      "older": "...",
      "newer": "...",
      "resolution": "newer active decision supersedes older proposal"
    }
  ]
}
```

The assistant must not flatten conflicting memories into one fake certainty.

---

## 9. Security, Privacy, and Deletion

### 9.1 Security requirements

- API token for local service access.
- Bind service to localhost by default.
- Optional mTLS or reverse proxy auth if exposed remotely.
- Secrets excluded from embeddings.
- Sensitive fields encrypted at rest where practical.
- Encrypted backups.
- Audit logs for deletion/correction.

### 9.2 Deletion semantics

`hard_delete` must:

1. delete raw message/tool/artifact content selected by policy;
2. delete derived chunks and embeddings;
3. invalidate facts/decisions derived only from deleted sources;
4. regenerate affected summaries;
5. write deletion audit event without preserving deleted text.

### 9.3 Correction semantics

`memory_correct` must:

1. create `user_correction_received` event;
2. locate affected memories;
3. mark old facts/decisions `invalidated` or `superseded`;
4. create corrected fact/decision with source link to correction;
5. re-run evals for related topic.

---

## 10. Observability

Implement logs and metrics from day one.

### 10.1 Logs

Log:

- ingest event IDs;
- redaction counts, not raw secrets;
- extraction job status;
- embedding job status;
- retrieval query;
- selected source IDs;
- rejected candidates count;
- latency per stage;
- errors with trace IDs.

### 10.2 Metrics

Track:

```text
memory_ingest_events_total
memory_redaction_matches_total
memory_embedding_jobs_total
memory_extraction_jobs_total
memory_recall_latency_ms
memory_recall_candidates_total
memory_recall_conflicts_total
memory_forget_requests_total
memory_eval_pass_rate
```

### 10.3 Retrieval logs

Create `retrieval_logs` table to debug why the assistant remembered something.

---

## 11. Evaluation Suite

Memory without evals will rot.

### 11.1 Golden questions

Create golden evals for known facts/decisions:

- current memory architecture decision;
- role of Hindsight;
- whether MCP is primary or not;
- active user preferences;
- known projects;
- tax reconstruction requirements;
- Shannon/Claude token correction;
- average ETH entry fact;
- operational server rules.

Each eval must specify:

```json
{
  "question": "Что мы решили про MCP для памяти?",
  "expected_contains": ["не primary", "MemoryProvider", "можно adapter later"],
  "expected_sources_required": true
}
```

### 11.2 Eval commands

```bash
pytest tests/evals -v
python -m siqueira_memo.evals.runner --suite golden
```

Fail builds if golden recall drops below threshold.

---

## 12. Development Phases

## Phase 1: Foundations

### Task 1.1: Create project skeleton

**Files:** all base folders and empty modules.

**Verification:**

```bash
python -m compileall src
```

Expected: no syntax errors.

### Task 1.2: Add pyproject dependencies

Dependencies:

```toml
fastapi
uvicorn[standard]
sqlalchemy[asyncio]
asyncpg
alembic
pydantic-settings
pgvector
redis
dramatiq
httpx
pytest
pytest-asyncio
ruff
mypy
opentelemetry-api
opentelemetry-sdk
```

**Verification:**

```bash
pip install -e '.[dev]'
pytest --version
```

### Task 1.3: Docker Compose

Create Postgres + pgvector + Redis services.

**Verification:**

```bash
docker compose up -d
python scripts/dev_reset_db.py
```

---

## Phase 2: Database and migrations

### Task 2.1: SQLAlchemy models

Implement models for:

- events;
- messages;
- tool events;
- artifacts;
- chunks;
- entities;
- facts;
- decisions;
- summaries;
- retrieval logs.

### Task 2.2: Alembic migration

Generate migration and inspect manually. Do not trust autogenerate blindly.

**Verification:**

```bash
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

---

## Phase 3: Ingest and redaction

### Task 3.1: Message ingest endpoint

Write tests first:

- valid message creates event + message;
- content_raw stored;
- content_redacted stored;
- async jobs enqueued.

### Task 3.2: Redaction service

Test cases:

- OpenAI-style key;
- Anthropic-style key;
- bearer token;
- SSH private key;
- JWT;
- database URL;
- `.env` block;
- seed phrase-like text.

Expected: redacted content has placeholders, not secrets.

### Task 3.3: Tool event ingest

Store raw/redacted input and output, summarize medium/large outputs.

---

## Phase 4: Chunking and embeddings

### Task 4.1: Chunking service

Rules:

- preserve source IDs;
- keep message boundaries when possible;
- include metadata: project/topic/entities;
- never chunk unredacted sensitive text.

### Task 4.2: Embedding service

Support provider abstraction:

```text
openai
local
mock-for-tests
```

Tests must not call paid APIs.

### Task 4.3: Rebuild script

`python scripts/rebuild_embeddings.py --source all`

Must be idempotent.

---

## Phase 5: Structured extraction

### Task 5.1: Decision extractor

Extract:

- decision;
- context;
- rationale;
- options;
- tradeoffs;
- status;
- source IDs.

Must distinguish:

```text
proposal != decision
brainstorm != accepted choice
superseded != active
```

### Task 5.2: Fact extractor

Extract source-backed facts with confidence and validity.

### Task 5.3: Correction handler

User corrections must supersede previous facts/decisions.

---

## Phase 6: Retrieval

### Task 6.1: Structured recall

Search facts and decisions by project/topic/entity/status.

### Task 6.2: Hybrid search

Combine:

- full-text search;
- vector search;
- structured facts;
- structured decisions.

### Task 6.3: Reranking

Implement reranker abstraction with mock in tests.

### Task 6.4: Context pack builder

Output concise, source-backed context for Hermes.

Must include:

- summary;
- active decisions;
- facts;
- conflicts;
- source snippets;
- confidence;
- warnings.

---

## Phase 7: Hermes MemoryProvider plugin

### Task 7.1: Provider registration

Create `plugins/memory/siqueira-memo/__init__.py` with `SiqueiraMemoProvider(MemoryProvider)` and:

```python
def register(ctx) -> None:
    ctx.register_memory_provider(SiqueiraMemoProvider())
```

### Task 7.2: Tool schemas

Expose provider tools through `get_tool_schemas()`:

```text
siqueira_memory_recall
siqueira_memory_remember
siqueira_memory_correct
siqueira_memory_forget
siqueira_memory_timeline
siqueira_memory_sources
```

Schemas must be strict and small. Do not expose internal DB complexity to LLM.

### Task 7.3: Lifecycle hooks

Implement and test documented hooks:

```text
initialize
system_prompt_block
prefetch
queue_prefetch
sync_turn
on_pre_compress
on_session_end
on_memory_write
shutdown
```

Also support source-inspected hooks when the active Hermes version exposes them:

```text
on_turn_start
on_delegation
```

### Task 7.4: Failure behavior

If memory service is down, provider tools return structured errors and Hermes continues without hallucinating memory. `sync_turn`, `queue_prefetch`, `on_pre_compress`, and `on_session_end` must fail soft and log warnings/debug entries.

---

## Phase 8: Hindsight import adapter

### Task 8.1: Offline Hindsight import

Import Hindsight memories through an offline command/job only. Do not query Hindsight as live fallback while Siqueira is the active Hermes memory provider.

### Task 8.2: Import Hindsight memories

Import selected Hindsight data as `hindsight_imported` events with secondary trust.

### Task 8.3: Precedence tests

Verify active Siqueira memory beats imported Hindsight candidates.

---

## Phase 9: Deletion and admin

### Task 9.1: Forget endpoint

Support soft invalidation and hard deletion.

### Task 9.2: Admin search UI/API

At minimum provide API endpoints for:

- search memories;
- view sources;
- mark stale;
- invalidate;
- delete;
- export.

### Task 9.3: Markdown export

Export project/topic memory into human-readable Markdown.

---

## Phase 10: Production hardening

### Task 10.1: Auth

Require API token. Bind to localhost by default.

### Task 10.2: Backups

Implement:

- daily Postgres dump;
- artifact backup;
- restore test script.

### Task 10.3: Observability

Add OpenTelemetry traces and structured logs.

### Task 10.4: Load testing

Test with:

- 100k messages;
- 1M chunks;
- large tool outputs;
- deletion of topic with many derived memories.

---

## 13. Acceptance Criteria

The system is production-ready only when all are true:

- all raw messages are stored with redacted variants;
- no secret appears in embeddings;
- every fact/decision has source IDs;
- conflicting decisions are represented, not flattened;
- user corrections supersede older memory;
- Hindsight cannot override custom memory;
- recall modes meet latency targets;
- deletion removes derived embeddings and invalidates summaries;
- golden eval suite passes;
- backups and restore are tested;
- Hermes can use memory via native toolset;
- compact memory remains small.

---

## 14. Initial Golden Decisions to Seed

Seed these as source-backed decisions after import from current conversation:

1. `siqueira-memo` is the project name.
2. Primary integration should be a Hermes `MemoryProvider` plugin, not MCP or a general custom toolset.
3. MCP may be added later as an adapter, but not primary path.
4. Hindsight is offline/import-only for migration/backfill; it is not a live fallback while Siqueira is the active Hermes provider.
5. Compact memory remains bootloader only.
6. Production memory requires raw archive, event log, structured facts/decisions, provenance, redaction, deletion, retrieval evals, and observability.
7. The system should optimize for remembering discussions, conclusions, decisions, and rationale, not just preferences.

---

## 15. Implementation Rules for Future Agents

- Do not skip tests.
- Do not store secrets in embeddings.
- Do not accept memories without source IDs unless explicitly marked `unverified`.
- Do not implement vector-only recall.
- Do not let old memories override live tool results.
- Do not make Hindsight the source of truth.
- Do not make MCP the primary integration.
- Do not silently swallow extraction errors.
- Do not hide conflicts.
- Do not ship without deletion semantics.

---

## 16. First Commit Sequence

Recommended commits:

```bash
git init

git add README.md IMPLEMENTATION_PLAN.md pyproject.toml docker-compose.yml .env.example
git commit -m "chore: initialize siqueira-memo project"

git add src/siqueira_memo/config.py src/siqueira_memo/db.py src/siqueira_memo/main.py
git commit -m "feat: add service foundation"

git add src/siqueira_memo/models alembic tests/unit/test_models.py
git commit -m "feat: add memory database schema"

git add src/siqueira_memo/services/redaction_service.py tests/unit/test_redaction_service.py
git commit -m "feat: add secret redaction pipeline"
```

---

## 17. Final Product Shape

When complete, `siqueira-memo` should behave like this:

1. Hermes receives a Telegram message.
2. The message is captured into Siqueira Memo.
3. Sensitive data is redacted.
4. Raw and redacted content are stored.
5. Async workers extract facts, decisions, entities, summaries, chunks, and embeddings.
6. When Mark asks about past context, Hermes uses prefetched Siqueira context or calls `siqueira_memory_recall`.
7. Siqueira returns a compact context pack with sources, confidence, conflicts, and warnings.
8. Hermes answers using source-backed memory.
9. If Mark corrects something, older memory is superseded or invalidated.
10. If Mark asks to forget something, raw and derived memory are deleted or invalidated correctly.

That is the bar. Anything less is just a toy vector database with nicer branding.

---

## 18. Specification Hardening Addendum

This addendum closes the dangerous ambiguity gaps that would cause an autonomous implementation agent to build something that technically works but semantically sucks.

The original architecture is correct. The weak parts are extraction, conflict detection, entity resolution, chunking, model choices, versioning, deletion semantics, eval determinism, retention, and observability sinks.

### 18.1 Concrete production choices

Default production choices unless explicitly overridden:

```text
Primary integration: Hermes MemoryProvider plugin, not MCP and not a general custom toolset
Backend: FastAPI
Database: PostgreSQL 16 + pgvector
Queue: Redis + Dramatiq
ORM: SQLAlchemy 2.x async
Migrations: Alembic
Embeddings: OpenAI text-embedding-3-large, 3072 dimensions
Cheap/local fallback embedder: BAAI/bge-m3
Reranker default: bge-reranker-v2-m3 local if GPU/CPU acceptable, otherwise Cohere Rerank API
Gate classifier: two-stage pipeline — deterministic regex/keyword prefilter first, then Claude Haiku-class or equivalent cheap structured-output model for ambiguous messages
Message-scope extraction LLM: Claude Sonnet-class or equivalent strong structured-output model
Window/session-scope extraction LLM: Claude Sonnet 4.x-class or GPT-5-class structured-output model; prefer quality over cheapness
Conflict verifier LLM: Claude Haiku-class for simple contradiction checks, Sonnet-class only for ambiguous/high-impact conflicts
Extraction fallback: retry with same model; do not silently downgrade quality
Redaction baseline: detect-secrets + custom regex detectors + gitleaks-compatible pattern corpus
Observability MVP: structured JSON logs + retrieval_logs table
Observability production: OpenTelemetry to Grafana Tempo or local Jaeger, metrics to Prometheus/Grafana
```

Do not let an implementation agent choose these ad hoc.

### 18.2 Extraction Contract

Extraction is the highest-risk subsystem. Bad extraction poisons every downstream layer.

#### 18.2.1 Extraction scope

Do not extract facts and decisions independently from every individual message.

Use three scopes:

```text
message_scope:
  Used only for explicit user corrections, explicit “remember this”, credentials references, or direct facts.

window_scope:
  Rolling dialogue window of 10-30 messages around a topic. Used for decisions, rationale, tradeoffs, and rejected options.

session_scope:
  Full session summary extraction. Used after session/topic completion to consolidate facts, decisions, open questions, project state.
```

Decision extraction must primarily run on `window_scope` or `session_scope`, because decisions often emerge across multiple turns.

#### 18.2.2 Extraction gate before expensive extraction

Before running the expensive extractor, run a cheap classifier/gate.

Gate labels:

```text
ignore
casual_ack
tool_noise
possible_fact
possible_decision
explicit_memory_request
user_correction
project_state_update
sensitive_secret_candidate
```

Only run full extraction when one of these is present:

```text
possible_fact
possible_decision
explicit_memory_request
user_correction
project_state_update
```

Never run full extraction for:

```text
ок
спасибо
да
нет
продолжай
tool output with no semantic conclusion
assistant filler
```

#### 18.2.3 Decision definition

A `decision` is a commitment, preference, chosen architecture, rejected path, operational policy, or explicit user instruction that should affect future behavior.

A decision usually contains at least one:

- explicit language: “решили”, “выбираем”, “делаем так”, “оставляем”, “не используем”, “primary будет”, “не надо”;
- user confirmation after options were compared;
- assistant recommendation accepted by user;
- correction that changes future behavior;
- architectural choice that future implementation must obey.

A `proposal` is not a decision.

A `brainstorm` is not a decision.

A `possibility` is not a decision.

A `question` is not a decision.

A `temporary hypothesis` is not a decision unless explicitly adopted.

#### 18.2.4 Decision examples

Positive examples — extract as decisions:

```text
User: MCP не нужен, память нужна только для тебя.
Decision: Do not use MCP as primary integration for Siqueira Memo.
Status: active.
```

```text
Assistant: Я бы делал Hermes MemoryProvider plugin. Hindsight import-only because Hermes allows one external live provider.
User: Хорошо, напиши план.
Decision: Use a Hermes MemoryProvider plugin as primary memory integration; Hindsight is offline/import-only, not live fallback.
Status: active.
```

```text
User: Новые skills создавать только по явному запросу.
Decision: Do not create new skills unless explicitly requested.
Status: active.
```

```text
User: В налогах мне нужна точная реконструкция по первичным данным, грубые оценки не принимаю.
Decision: Tax/crypto work requires exact reconstruction from primary data.
Status: active.
```

Negative examples — do not extract as decisions:

```text
Assistant: Можно было бы сделать MCP сервер.
Reason: brainstorming/proposal only.
```

```text
User: А если Qdrant?
Reason: question only.
```

```text
Assistant: Наверное, можно начать с SQLite.
Reason: weak suggestion, not accepted.
```

```text
User: ок
Reason: acknowledgement only, unless previous message explicitly asked for confirmation and context proves acceptance.
```

#### 18.2.5 Fact definition

A `fact` is a source-backed statement about user, project, environment, asset, decision state, configuration, or historical event.

Facts must have:

```text
subject
predicate
object
statement
confidence
source_event_ids
status
validity
```

Do not extract vague statements as facts.

Bad fact:

```text
subject: memory
predicate: is
object: important
```

Good fact:

```text
subject: siqueira-memo
predicate: primary_integration
object: Hermes MemoryProvider plugin
statement: Siqueira Memo should integrate primarily through a Hermes MemoryProvider plugin, not MCP or a general custom toolset.
```

#### 18.2.6 Deterministic IDs and idempotency

Facts and decisions need stable semantic keys.

Fact canonical key:

```text
fact_key = sha256(normalize(subject) + "|" + normalize(predicate) + "|" + normalize(object) + "|" + project_or_global)
```

Decision canonical key:

```text
decision_key = sha256(normalize(project) + "|" + normalize(topic) + "|" + normalize(decision_summary))
```

Re-running extraction over the same source must not create duplicate active facts/decisions.

If extractor version changes, store a new extraction candidate but merge/promote only after dedupe.

Required fields:

```sql
ALTER TABLE facts ADD COLUMN canonical_key TEXT NOT NULL;
ALTER TABLE decisions ADD COLUMN canonical_key TEXT NOT NULL;
CREATE UNIQUE INDEX facts_active_key_idx ON facts(canonical_key) WHERE status = 'active';
CREATE UNIQUE INDEX decisions_active_key_idx ON decisions(canonical_key) WHERE status = 'active';
```

#### 18.2.7 Extraction metadata

Every extracted record must store:

```json
{
  "extractor_name": "decision_extractor",
  "extractor_version": "2026-04-24.v1",
  "model_provider": "...",
  "model_name": "...",
  "prompt_version": "...",
  "source_scope": "message|window|session",
  "input_event_ids": [],
  "temperature": 0,
  "schema_version": "..."
}
```

No extracted memory without extractor metadata is production-valid.

---

## 19. Entity Resolution Specification

The `entities` table is not enough. Entity resolution is a process.

### 19.1 Entity lifecycle

Entity states:

```text
candidate
active
merged
rejected
needs_review
```

### 19.2 Entity linking job

Create `entity_linking_job` that runs after ingest/extraction.

Input:

- extracted entity mentions;
- existing entities and aliases;
- project/topic context;
- fuzzy string match;
- embedding similarity over entity descriptions;
- optional LLM judgment.

Output:

```json
{
  "mention": "Shannon API",
  "linked_entity_id": "...",
  "confidence": 0.91,
  "action": "link|create_candidate|needs_review"
}
```

### 19.3 Automatic merge policy

Auto-link if:

```text
exact alias match OR normalized alias match
AND no conflicting entity type
```

Auto-merge if:

```text
same normalized name
same entity type
confidence >= 0.95
no conflicting metadata
```

Needs review if:

```text
0.75 <= confidence < 0.95
OR entity type differs
OR two plausible entities exist
```

Do not fully automate ambiguous merges for projects, servers, wallets, repos, or people.

### 19.4 Entity tables and indexes

Add explicit aliases table instead of relying only on `TEXT[]`.

```sql
CREATE TABLE entity_aliases (
  id UUID PRIMARY KEY,
  entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  alias TEXT NOT NULL,
  alias_normalized TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  source_event_ids UUID[] NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX entity_aliases_norm_idx ON entity_aliases(alias_normalized);
CREATE UNIQUE INDEX entity_aliases_norm_type_unique_idx ON entity_aliases(alias_normalized, entity_type) WHERE status = 'active';
CREATE INDEX entities_aliases_gin_idx ON entities USING gin(aliases);
```

### 19.5 Entity relationship table

```sql
CREATE TABLE entity_relationships (
  id UUID PRIMARY KEY,
  source_entity_id UUID NOT NULL REFERENCES entities(id),
  relation TEXT NOT NULL,
  target_entity_id UUID NOT NULL REFERENCES entities(id),
  confidence DOUBLE PRECISION NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  source_event_ids UUID[] NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 20. Chunking Strategy

Do not use generic LangChain defaults for dialogue memory.

### 20.1 Tokenization

Use `tiktoken` when using OpenAI embeddings. Store token counts. For local embedding models, keep tokenizer abstraction and store tokenizer name/version.

Add fields:

```sql
ALTER TABLE chunks ADD COLUMN tokenizer_name TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE chunks ADD COLUMN tokenizer_version TEXT;

-- Do not store embedding model/dimension metadata on chunks in the multi-model design.
-- Embedding metadata belongs to chunk_embeddings partitions.
```

### 20.2 Dialogue messages

Rules:

```text
Short message < 350 tokens:
  one chunk = one message

Medium message 350-1200 tokens:
  chunk by paragraphs, target 500 tokens, overlap 80 tokens

Long message > 1200 tokens:
  sliding window, target 800 tokens, overlap 120 tokens
```

For adjacent short messages in same topic:

```text
Combine up to 8 adjacent messages OR 700 tokens into dialogue_window chunks.
Preserve individual source_message_ids.
```

### 20.3 Tool outputs

Tool output chunking depends on type:

```text
JSON:
  chunk by top-level keys/items; preserve JSONPath in metadata.

Logs:
  chunk by error blocks first; include surrounding context lines; summarize boring repeated lines.

Code:
  chunk by file/function/class when possible; preserve path and line ranges.

Command output:
  store command, exit code, summary, and important stderr/stdout segments.
```

### 20.4 Summaries as chunks

Session summaries, topic summaries, project states, facts, and decisions should also be indexed as chunks with `source_type`:

```text
session_summary
topic_summary
project_state
fact
decision
```

These chunks should receive higher retrieval priority than raw conversational chunks unless the query asks for raw history.

---

## 21. Conflict Detection Algorithm

Do not implement conflict detection as “two active decisions with same topic”. That is insufficient.

### 21.1 Fact-fact conflicts

A conflict exists when:

```text
same normalized subject
same normalized predicate
different normalized object
overlapping validity windows
both statuses active or one active and one stale-but-recent
```

Example:

```text
Fact A: Shannon primary auth = API key
Fact B: Shannon primary auth = Claude OAuth token
```

If Fact B is newer and source is user correction, mark Fact A superseded.

### 21.2 Decision-decision conflicts

Conflict if:

```text
same project/topic
both active
semantic contradiction detected by rule or LLM verifier
```

Examples:

```text
Decision A: Use MCP as primary integration.
Decision B: Do not use MCP as primary integration.
```

### 21.3 Decision-fact conflicts

Conflict if a fact describes current system behavior that violates active decision.

Example:

```text
Decision: Hindsight is offline/import-only, not live provider.
Fact: Hindsight is source of truth.
```

### 21.4 User correction conflicts

User corrections override prior assistant claims unless contradicted by later user correction or live verified tool result.

### 21.5 Temporal conflicts

Use validity ranges. Two facts with different objects are not conflicting if validity windows do not overlap.

Use `tstzrange`:

```sql
ALTER TABLE facts ADD COLUMN validity tstzrange;
CREATE INDEX facts_validity_gist_idx ON facts USING gist(validity);
```

### 21.6 Conflict table

```sql
CREATE TABLE memory_conflicts (
  id UUID PRIMARY KEY,
  conflict_type TEXT NOT NULL,
  left_type TEXT NOT NULL,
  left_id UUID NOT NULL,
  right_type TEXT NOT NULL,
  right_id UUID NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  resolution TEXT,
  resolved_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at TIMESTAMPTZ
);
```

Statuses:

```text
open
auto_resolved
needs_review
ignored
```

---

## 22. Model, Prompt, and Embedding Versioning

### 22.1 Prompt registry

Create:

```text
src/siqueira_memo/prompts/
  extraction_gate.v1.md
  decision_extractor.v1.md
  fact_extractor.v1.md
  entity_linker.v1.md
  session_summarizer.v1.md
  conflict_verifier.v1.md
```

Prompts are versioned source files. Do not inline production prompts inside random functions.

### 22.2 Prompt metadata table

```sql
CREATE TABLE prompt_versions (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(name, version)
);
```

### 22.3 Embedding versioning

Do not assume all embeddings have same dimensions forever.

The authoritative production design is defined in §31.6: use one fixed-dimension physical embedding table per model/dimension plus an `embedding_indexes` registry. Do not implement a single `embedding vector` column as the production retrieval index.

Earlier single-table sketches are superseded by §31.6.

### 22.4 Re-extraction policy

Re-extract when:

- prompt major version changes;
- model family changes;
- schema changes;
- evals show regression;
- user correction invalidates related topic;
- source messages were deleted.

Do not automatically promote re-extracted memories. Store candidates, dedupe, then promote.

---

## 23. Redaction Specification

### 23.1 Baseline tools

Use layered detection:

```text
detect-secrets baseline
custom regex detectors
gitleaks-compatible patterns
entropy-based detector for high-risk strings
LLM-assisted classifier only for ambiguous cases, never as sole detector
```

### 23.2 Required custom detectors

Add detectors for:

```text
Anthropic API keys
OpenAI API keys
OpenRouter keys
GitHub tokens
Telegram bot tokens
JWTs
Bearer tokens
SSH private keys
PEM private keys
database URLs
Redis URLs
.env blocks
Claude OAuth/token material
Shannon-related token names
seed phrases
Brazil/tax document identifiers when configured
```

### 23.3 False-positive policy

Not all sensitive-looking identifiers should be destroyed.

Wallet addresses should be preserved as public identifiers unless explicitly marked secret.

Server IPs should be sensitivity-tagged, not always redacted.

Domain names and repo names should be preserved.

### 23.4 Redaction test corpus

Create:

```text
tests/fixtures/redaction/known_secrets.txt
tests/fixtures/redaction/false_positives.txt
tests/fixtures/redaction/mixed_messages.jsonl
```

Required tests:

```text
known secret recall >= 99%
false positive rate on allowed identifiers <= 2%
no known secret appears in chunk_text
no known secret appears in embeddings input mock payload
```

---

## 24. Deletion Cascade and Summary Regeneration

### 24.1 Summary regeneration thresholds

When source events are deleted:

```text
0% affected:
  no action

0-10% affected:
  mark summary stale; do not immediately regenerate

10-60% affected:
  enqueue regeneration job

>60% affected:
  mark summary invalid immediately and enqueue regeneration

100% affected:
  delete summary or mark deleted, depending on audit policy
```

### 24.2 Derived memory invalidation

If a fact/decision only cites deleted sources, mark it deleted/invalidated.

If it cites mixed sources, remove deleted source references and mark `needs_reverification`.

### 24.3 Hard delete must remove embeddings

Hard deletion must delete:

- raw content selected for deletion;
- redacted content if requested;
- chunks;
- chunk embeddings;
- retrieval snippets;
- generated summaries that solely derive from deleted content.

Audit events may preserve deletion metadata but not deleted text.

---

## 25. Retention Policy

Default retention:

```text
raw messages: keep forever until manual deletion
redacted messages: keep forever until manual deletion
tool outputs small: keep forever until manual deletion
tool outputs large: keep pointer/hash forever, raw blob configurable
chunks: rebuildable, keep while source exists
embeddings: rebuildable, can be regenerated after model changes
retrieval logs: keep 180 days by default
worker logs: keep 30 days
OpenTelemetry traces: keep 7-30 days depending backend
backups: daily 30 days, weekly 6 months, monthly 2 years
```

Add config:

```yaml
retention:
  raw_messages_days: null
  retrieval_logs_days: 180
  worker_logs_days: 30
  traces_days: 14
  backups_daily_days: 30
  backups_weekly_months: 6
  backups_monthly_years: 2
```

---

## 26. Eval Determinism

### 26.1 Eval types

Use three eval layers:

```text
unit evals:
  deterministic, mocks all LLM/embedding/rerank calls

retrieval evals:
  fixed seeded corpus, fixed mock embeddings or pinned embedding snapshots

end-to-end evals:
  real models allowed, non-blocking or rolling-average threshold
```

### 26.2 CI policy

Blocking CI:

- unit evals;
- redaction corpus;
- schema validation;
- deterministic retrieval evals;
- deletion cascade tests.

Non-blocking or scheduled:

- real LLM extraction quality;
- live embedding retrieval;
- live reranker tests;
- load tests over huge corpus.

### 26.3 Flake control

For live evals:

```text
run 3 times
store score distribution
compare rolling 7-run average
alert on regression >5-10%
do not fail a production deploy on one stochastic run unless catastrophic
```

---

## 27. Observability Backend

### 27.1 Local/dev

Use:

```text
structured JSON logs to stdout
retrieval_logs table
optional Jaeger all-in-one for traces
```

### 27.2 Production

Use one concrete stack:

```text
OpenTelemetry Collector
Grafana Tempo for traces
Prometheus for metrics
Grafana dashboards
Loki or plain JSON file logs if Loki unavailable
```

If this stack is not deployed, do not pretend tracing exists. Mark tracing as disabled and rely on structured logs.

---

## 28. Database Corrections and Indexes

### 28.1 FK constraints for supersession

```sql
ALTER TABLE facts
  ADD CONSTRAINT facts_superseded_by_fk
  FOREIGN KEY (superseded_by) REFERENCES facts(id);

ALTER TABLE decisions
  ADD CONSTRAINT decisions_superseded_by_fk
  FOREIGN KEY (superseded_by) REFERENCES decisions(id);
```

### 28.2 Source join tables

Prefer join tables over UUID arrays for critical provenance.

```sql
CREATE TABLE fact_sources (
  fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
  event_id UUID NOT NULL REFERENCES memory_events(id) ON DELETE RESTRICT,
  message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
  PRIMARY KEY (fact_id, event_id)
);

CREATE TABLE decision_sources (
  decision_id UUID NOT NULL REFERENCES decisions(id) ON DELETE CASCADE,
  event_id UUID NOT NULL REFERENCES memory_events(id) ON DELETE RESTRICT,
  message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
  PRIMARY KEY (decision_id, event_id)
);
```

Arrays may remain as denormalized cache, but join tables are authoritative.

### 28.3 FTS config

Default `simple` FTS is not enough for Russian/English mixed content.

Use separate generated columns or query-time config:

```sql
CREATE INDEX chunks_fts_ru_idx ON chunks USING gin(to_tsvector('russian', chunk_text));
CREATE INDEX chunks_fts_en_idx ON chunks USING gin(to_tsvector('english', chunk_text));
CREATE INDEX chunks_fts_simple_idx ON chunks USING gin(to_tsvector('simple', chunk_text));
```

Search should combine Russian, English, and simple configs.

### 28.4 HNSW parameters

Use explicit HNSW params:

```sql
CREATE INDEX chunks_embedding_hnsw_idx
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

At query time tune:

```sql
SET hnsw.ef_search = 80;
```

Benchmark `ef_search` values 40, 80, 120 for recall/latency.

### 28.5 Partitioning

Partition append-heavy tables by month:

```text
memory_events
messages
tool_events
retrieval_logs
```

Use monthly partitions by `created_at` from day one. Do not wait until tables are huge.

### 28.6 Metadata schemas

Every JSONB metadata field must have a Pydantic schema.

Examples:

```text
MessageMetadata
ToolEventMetadata
ChunkMetadata
FactMetadata
DecisionMetadata
RetrievalLogMetadata
```

Do not write arbitrary dicts into metadata without schema validation.

---

## 29. Hermes Prompt Artifact

The memory recall policy must be a real versioned artifact, not a comment.

Create:

```text
plugins/memory/siqueira-memo/system_prompt.md
```

It must contain:

- when to call `siqueira_memory_recall`;
- when to call `siqueira_memory_remember`;
- when to call `siqueira_memory_correct`;
- how to treat Hindsight vs Siqueira Memo;
- what to do if memory is unavailable;
- rule that live tool output beats old memory;
- rule to cite uncertainty when sources conflict.

Store prompt hash/version in `prompt_versions`.

---

## 30. Updated Acceptance Criteria

Add these acceptance criteria to the previous list:

- extraction uses gate + scoped windows, not per-message brute force;
- decision extractor has positive and negative examples;
- facts/decisions are idempotent via canonical keys;
- conflict detection covers fact-fact, decision-decision, decision-fact, correction, and temporal conflicts;
- entity linking has confidence thresholds and review path;
- chunking strategy is dialogue/tool/code/log aware;
- prompts, extraction models, embeddings, and rerankers are versioned;
- redaction test corpus proves high secret recall and controlled false positives;
- deletion cascade follows explicit summary regeneration thresholds;
- evals distinguish deterministic CI checks from stochastic live checks;
- observability has an actual backend or explicitly degrades to structured logs;
- append-heavy tables are partitioned;
- Russian and English FTS are both supported;
- provenance join tables exist and are authoritative.

---

## 31. Second Review Refinements

This section supersedes any earlier ambiguity where it conflicts. It closes the remaining sharp edges identified after the second Opus review.

### 31.1 Concrete extraction model tiers

Do not leave “strongest available model” to implementation-agent interpretation.

Use this default tiering:

```text
regex/keyword prefilter:
  local deterministic Python rules, no LLM call

gate classifier for ambiguous messages:
  Claude Haiku-class / GPT-5-mini-class structured output
  temperature = 0
  target latency <500ms

message-scope extraction:
  Claude Sonnet-class structured output
  temperature = 0

window-scope extraction:
  Claude Sonnet 4.x-class or GPT-5-class structured output
  temperature = 0
  quality over cost

session-scope extraction:
  Claude Sonnet 4.x-class or GPT-5-class structured output
  temperature = 0
  allowed to be slower because async

conflict verifier:
  rule-based precheck first
  Haiku-class for low-impact conflicts
  Sonnet-class for high-impact or ambiguous conflicts
```

If these model names are unavailable, choose the closest equivalent tier and record the exact provider/model in extraction metadata.

### 31.2 Gate classifier algorithm

The gate is two-stage, because it runs on almost every message.

#### Stage A: deterministic prefilter

Implement local rules before any LLM call.

Rules:

```text
If message normalized text is in {"ок", "окей", "да", "нет", "спасибо", "понял", "продолжай"}:
  label = casual_ack
  unless previous assistant message explicitly requested confirmation of a concrete option.

If message length < 5 chars and no context-relevant previous confirmation request:
  label = casual_ack

If message contains explicit memory verbs:
  запомни, сохрани, не забудь, забудь, удали из памяти, исправь, это неверно
  label = explicit_memory_request or user_correction

If message contains strong decision markers:
  решаем, решили, делаем так, выбираем, оставляем, не используем, primary, source of truth
  label = possible_decision

If message contains credentials/secret-like patterns:
  label = sensitive_secret_candidate

If tool output has exit_code only and no semantic summary/error:
  label = tool_noise
```

#### Stage B: cheap LLM classifier

Run only if Stage A returns uncertain/possible semantic content.

Output strict JSON:

```json
{
  "labels": ["possible_decision"],
  "confidence": 0.87,
  "reason": "User rejects MCP as primary integration and states desired architecture.",
  "requires_window_context": true
}
```

If `requires_window_context=true`, full extraction must use a dialogue window, not the single message.

### 31.3 Canonical normalization algorithm

`normalize()` must be deterministic and entity-aware.

Algorithm:

```python
def normalize_for_key(text: str) -> str:
    # 1. Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # 2. Case-fold, not lower, for better Unicode behavior
    text = text.casefold()

    # 3. Strip markdown/code punctuation that does not change identity
    text = strip_markdown(text)

    # 4. Normalize whitespace and dashes
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Remove trailing punctuation
    text = text.strip(" .,:;!?")

    return text
```

Entity-aware canonical key rule:

```text
Before hashing facts or decisions, replace recognized entity mentions with resolved entity IDs.

Example:
  "Shannon", "shannon app", "Shannon API"
  → entity:project:uuid
```

If entity resolution is uncertain:

```text
status = candidate
do not promote to active until entity is resolved or key is recomputed.
```

This prevents canonical keys from drifting after alias merges.

### 31.4 Accepted recommendation reconstruction

The extractor must handle references like “второй давай”.

Positive example:

```text
Assistant: Варианты:
1. MCP server.
2. Hermes MemoryProvider plugin.
3. Only Hindsight.

User: второй давай.

Extracted decision:
  decision: Use Hermes MemoryProvider plugin as the primary integration.
  rationale: User selected option 2 from assistant's options.
  source_scope: window
  source_message_ids: [assistant_options_msg, user_selection_msg]
```

Do not extract:

```text
decision: "второй давай"
```

If the referenced option cannot be reconstructed from the window, create `needs_review`, not an active decision.

### 31.5 Conflict verifier algorithm

Do not run LLM verifier on every active pair. Use candidate narrowing.

Pipeline:

```text
1. Rule prefilter:
   same project/topic OR same subject/predicate OR shared entity pair

2. Polarity/rule detector:
   detect negation pairs such as use vs do_not_use, primary vs secondary, enabled vs disabled

3. Temporal filter:
   ignore non-overlapping validity windows unless current query asks historical timeline

4. LLM verifier only for ambiguous candidates:
   max 20 candidate pairs per job by default

5. Store conflict result in memory_conflicts
```

Verifier prompt output:

```json
{
  "is_conflict": true,
  "conflict_type": "decision_decision",
  "severity": "high",
  "resolution_hint": "newer user correction supersedes older assistant suggestion",
  "confidence": 0.91
}
```

### 31.6 Chunk embeddings partition strategy

The multi-model embedding design is authoritative. `chunks` stores chunk text and chunking metadata only. Embeddings live in model/dimension-specific partitions.

Do not use a single unindexed `embedding vector` column for production retrieval.

Recommended schema: one fixed-dimension physical table per embedding model/dimension, plus a registry table. Do **not** use PostgreSQL partitions with per-partition extra columns; partitions cannot safely have a different vector dimension column than the parent.

```sql
CREATE TABLE embedding_indexes (
  id UUID PRIMARY KEY,
  table_name TEXT NOT NULL UNIQUE,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  dimensions INT NOT NULL,
  distance_metric TEXT NOT NULL DEFAULT 'cosine',
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunk_embeddings_openai_text_embedding_3_large (
  id UUID PRIMARY KEY,
  chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL DEFAULT 'text-embedding-3-large',
  model_version TEXT NOT NULL,
  dimensions INT NOT NULL DEFAULT 3072,
  embedding vector(3072) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(chunk_id, model_version)
);

CREATE INDEX chunk_embeddings_openai_3_large_hnsw_idx
ON chunk_embeddings_openai_text_embedding_3_large
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

For `bge-m3`, create a separate physical table with its fixed dimension and HNSW index. Query code chooses the active table via `embedding_indexes`. A SQL view may union metadata across embedding tables, but vector nearest-neighbor search must hit a concrete fixed-dimension table.

### 31.7 Candidate-to-active promotion

Do not insert extraction output directly as active memory in parallel workers.

Use lifecycle:

```text
candidate
deduped
promoted_active
rejected
needs_review
superseded
invalidated
```

Promotion algorithm:

```text
1. Insert extraction result as candidate with canonical_key.
2. Acquire PostgreSQL advisory lock on hash(canonical_key).
3. Re-check active records with same canonical_key.
4. If equivalent active exists: merge sources, mark candidate deduped.
5. If conflicting active exists: create memory_conflict, mark candidate needs_review unless user correction gives clear precedence.
6. If no active exists and confidence >= threshold: promote to active.
7. Release advisory lock.
```

Use `INSERT ... ON CONFLICT` only for idempotent candidate creation, not as the only promotion logic.

### 31.8 Extraction metadata columns

Frequently queried extraction metadata must be columns, not only JSONB.

Add to `facts`, `decisions`, and extraction candidate tables:

```sql
extractor_name TEXT NOT NULL,
extractor_version TEXT NOT NULL,
prompt_version TEXT NOT NULL,
model_provider TEXT NOT NULL,
model_name TEXT NOT NULL,
source_scope TEXT NOT NULL,
schema_version TEXT NOT NULL
```

Index:

```sql
CREATE INDEX facts_extractor_version_idx ON facts(extractor_version);
CREATE INDEX facts_prompt_version_idx ON facts(prompt_version);
CREATE INDEX decisions_extractor_version_idx ON decisions(extractor_version);
CREATE INDEX decisions_prompt_version_idx ON decisions(prompt_version);
```

JSONB metadata remains for non-query-critical details.

### 31.9 Event payload schemas

`memory_events.payload` must be schema-validated with discriminated unions by `event_type`.

Create Pydantic schemas:

```text
MessageReceivedPayload
AssistantMessageSentPayload
ToolCalledPayload
ToolResultReceivedPayload
ArtifactCreatedPayload
SummaryCreatedPayload
FactExtractedPayload
DecisionRecordedPayload
FactInvalidatedPayload
DecisionSupersededPayload
MemoryDeletedPayload
UserCorrectionReceivedPayload
HindsightImportedPayload
```

Do not write arbitrary payload dictionaries.

### 31.10 Retrieval noise policy

`retrieval_logs` is authoritative for retrieval diagnostics.

Do not write every retrieval to `memory_events`; it will pollute the append-only audit log.

Only write to `memory_events` when retrieval causes a durable state change:

```text
answer_card_created
memory_corrected_from_retrieval
user_confirmed_retrieved_context
```

### 31.11 Partition automation

Partitioning from day one requires automation.

Use one of:

```text
Preferred: pg_partman extension
Fallback: scheduled worker job ensure_partitions_exist
```

`ensure_partitions_exist` must create partitions for:

```text
current month
next month
month after next
```

Run daily. Startup readiness must fail if current partition is missing.

### 31.12 Summary regeneration thresholds are heuristics

The 10% and 60% thresholds in §24.1 are starting heuristics, not semantic truth.

Do not hard-code tests that assume these exact values forever. Store them in config:

```yaml
deletion:
  summary_stale_threshold: 0.10
  summary_invalid_threshold: 0.60
```

Tune after real data.

### 31.13 Hermes prompt runtime loading

`plugins/memory/siqueira-memo/system_prompt.md` must be loaded at MemoryProvider initialization/startup.

Boot-time check:

```text
1. Read system_prompt.md.
2. Compute SHA256 hash.
3. Look up latest prompt_versions entry for name = hermes_memory_system_prompt.
4. If no entry exists in dev: create one.
5. If hash differs in production: fail startup with explicit error.
6. If hash differs in dev: warn and require prompt version bump before commit.
```

This prevents silent prompt drift.

### 31.14 Gitleaks patterns source

Use both:

```text
gitleaks as an optional installed CLI for local/security CI scans
checked-in custom pattern corpus under tests/fixtures/redaction/patterns/
```

CI must pass even if gitleaks binary is unavailable by running the checked-in pattern corpus tests.

Security CI should run gitleaks when available.

---

## 32. Hermes Runtime Compatibility Contract

This section is based on the current Hermes MemoryProvider contract and local source inspection.

### 32.1 Integration contract is MemoryProvider, not generic plugin

Siqueira Memo must implement `agent.memory_provider.MemoryProvider` and be activated via:

```yaml
memory:
  provider: siqueira-memo
```

Do not implement the primary integration as:

- MCP server;
- general plugin hooks;
- standalone toolset injected outside MemoryManager.

Hermes routes provider tools through `MemoryManager.handle_tool_call()` and injects tool schemas from `get_tool_schemas()`.

### 32.2 Single external provider rule

Hermes allows:

```text
built-in compact memory provider: always on
one external MemoryProvider: selected by memory.provider
```

Therefore:

```text
Siqueira active => Hindsight provider inactive
Hindsight active => Siqueira provider inactive
```

Hindsight integration must be import/backfill only. The plan must not rely on live Hindsight recall while Siqueira is active.

### 32.3 Profile isolation

Hermes passes `hermes_home` and usually `agent_identity` into `initialize()`.

Add `profile_id` to all durable first-class tables:

```sql
ALTER TABLE memory_events ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE messages ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE tool_events ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE artifacts ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE chunks ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE entities ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE facts ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE decisions ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE session_summaries ADD COLUMN profile_id TEXT NOT NULL;
ALTER TABLE retrieval_logs ADD COLUMN profile_id TEXT NOT NULL;
```

Derive it as:

```text
profile_id = agent_identity if provided else sha256(hermes_home)
```

Every uniqueness constraint and canonical key must include `profile_id` unless explicitly global.

### 32.4 Agent context filtering

`initialize()` may receive:

```text
agent_context = primary | subagent | cron | flush
```

Write policy:

```text
primary: normal durable writes
subagent: no independent writes; parent observes via on_delegation
cron: skip by default unless job explicitly enables memory
flush: skip; do not ingest flush prompts as user intent
```

### 32.5 Prefetch timing

Hermes calls provider recall roughly as:

```text
prefetch_all(original_user_message) before API call
sync_all(original_user_message, final_response) after turn
queue_prefetch_all(original_user_message) after turn
```

This means automatic context injection is not a same-turn write/read guarantee.

Design implication:

```text
Automatic prefetch = best-effort contextual recall.
Explicit siqueira_memory_* tools = same-turn control path.
```

### 32.6 Compression hook caveat

The `MemoryProvider.on_pre_compress(messages) -> str` contract says returned text can be injected into compression summaries. However, local source inspection found a Hermes build where `_compress_context()` calls `self._memory_manager.on_pre_compress(messages)` and discards the return value before `context_compressor.compress()`.

Therefore implementation must include a compatibility test:

```text
test_on_pre_compress_return_value_reaches_compressor_if_supported
```

If the running Hermes version discards the return value, Siqueira must still enqueue extraction/persistence from `on_pre_compress()` and not depend on compression injection.

### 32.7 Required plugin tests against Hermes

Add integration tests or source-level contract tests:

```text
provider loads via plugins.memory.load_memory_provider('siqueira-memo')
register(ctx) calls ctx.register_memory_provider(...)
is_available() performs no network calls
initialize() receives hermes_home/session/platform/profile fields
get_tool_schemas() injects siqueira_* tools with no duplicate names
handle_tool_call() returns JSON strings
sync_turn() returns quickly and enqueues work
prefetch() returns cached/prewarmed context only
queue_prefetch() schedules next-turn work
on_pre_compress() enqueues high-priority extraction
on_memory_write() mirrors built-in memory writes
on_delegation() records parent-observed subagent result
shutdown() flushes queues within bounded timeout
```

### 32.8 System prompt artifact update

`system_prompt_block()` must load the versioned prompt artifact from:

```text
plugins/memory/siqueira-memo/system_prompt.md
```

The previous path under `src/siqueira_memo/hermes_toolset/` is deprecated because Siqueira is a MemoryProvider plugin, not a standalone toolset.

---

## 33. Additional Hermes Runtime Findings

These findings come from Hermes developer docs and source inspection after the MemoryProvider rewrite. They are compatibility requirements for Siqueira Memo.

### 33.1 Dual compression system: gateway hygiene vs agent compressor

Hermes has two independent compression paths:

```text
Gateway session hygiene:
  runs in gateway/run.py before the agent processes a message
  fixed threshold around 85% of model context
  intended for long-lived Telegram/Discord sessions that grew between turns

Agent ContextCompressor:
  runs inside the agent loop
  default threshold around 50% of context
  calls MemoryProvider.on_pre_compress(messages)
```

`on_pre_compress()` only covers the agent-level compressor. Gateway hygiene may compress a large messaging transcript before the active agent/provider gets a normal turn-level hook.

Siqueira implications:

```text
sync_turn() must ingest raw turns aggressively and non-blockingly after every completed turn.
Gateway/messaging deployments should not rely on on_pre_compress as the only preservation hook.
Long accumulated Telegram/Discord histories may be compacted by gateway hygiene before Siqueira sees a provider compression hook.
If Hermes exposes a gateway pre-hygiene hook in the future, Siqueira should subscribe to it.
```

### 33.2 Hermes auxiliary compaction summaries as non-authoritative sources

Hermes `ContextCompressor` uses an auxiliary compression model and a structured summary template with sections such as:

```text
Goal
Constraints & Preferences
Progress / Completed Actions
In Progress
Blocked
Key Decisions
Resolved/Pending Questions
Relevant Files
Remaining Work / Next Steps
Critical Context
```

These summaries may appear in compressed messages with a `[CONTEXT COMPACTION]` / `[CONTEXT COMPACTION — REFERENCE ONLY]` prefix.

Siqueira must detect these messages and ingest them as:

```text
source_type = hermes_auxiliary_compaction
source_trust = secondary
source_event_type = hermes_auxiliary_compaction_observed
```

Rules:

```text
Siqueira's own extraction/session summary supersedes Hermes auxiliary compaction when both exist.
Hermes auxiliary compaction can seed candidates and session_summaries, but it is not authoritative.
Never treat compaction summary text as new user intent.
Preserve original source links where possible; if unavailable, mark provenance as compression-derived.
```

### 33.3 ContextEngine plugin is future opportunity, not v1

Hermes also supports pluggable `ContextEngine` providers under `plugins/context_engine/`, selected via `context.engine`.

This opens a future architecture where compression itself becomes Siqueira-aware. That is explicitly out of scope for v1.

v1 rule:

```text
Siqueira Memo integrates only as MemoryProvider.
Do not replace Hermes ContextCompressor in v1.
Document any compression limitations instead of silently building a second context engine.
```

### 33.4 Hermes auxiliary compression model vs Siqueira extraction model

Hermes compression uses `auxiliary.compression` config and may use a different model than Siqueira extraction.

Therefore the system can produce two summaries for the same conversation:

```text
Hermes auxiliary compaction summary: optimized for keeping the agent alive
Siqueira extraction/session summary: optimized for durable memory/retrieval
```

Precedence:

```text
Siqueira session summary > Siqueira extracted facts/decisions > Hermes auxiliary compaction summary
```

Store Hermes auxiliary summaries as secondary sources; never let them overwrite active Siqueira facts/decisions without normal candidate promotion and conflict checks.

### 33.5 Prefetch token budget for prompt-cache efficiency

Hermes uses Anthropic prompt caching with a `system_and_3` strategy: one breakpoint on system prompt and three rolling breakpoints on the tail messages.

Long `prefetch()` payloads are injected into the current user-message context and can degrade cache efficiency and latency.

Budget rule:

```text
prefetch() fast mode: <= 1,200 tokens
prefetch() balanced mode: <= 2,000 tokens
prefetch() deep/forensic: do not inject automatically; return via explicit tool result only
source snippets in prefetch: max 3 unless explicitly requested
```

If retrieved context exceeds budget, return a synthesized context pack with source IDs and suggest explicit `siqueira_memory_recall(mode="deep")` for more.

### 33.6 initialize() kwargs required vs optional

Hermes guarantees `session_id` and `hermes_home`. Other fields depend on platform/caller.

Treat fields as:

```text
Required:
  session_id
  hermes_home

Optional:
  agent_identity
  agent_workspace
  platform
  user_id
  user_name
  chat_id
  chat_name
  chat_type
  thread_id
  gateway_session_key
  agent_context
  parent_session_id
```

`profile_id` derivation:

```text
if agent_identity present and non-empty:
  profile_id = agent_identity
elif hermes_home present:
  profile_id = sha256(normalize_path(hermes_home))
else:
  profile_id = "default"
```

Do not assume gateway fields exist in CLI mode.

### 33.7 Initial import from Hermes SQLite/FTS sessions

Hermes stores session history in its own SQLite/FTS database, separate from Siqueira Postgres. This is the main source for pre-Siqueira historical conversations.

Add an importer:

```text
scripts/import_hermes_sessions.py
```

Import sources:

```text
Hermes SessionDB / state SQLite
Hermes exported sessions JSONL when available
Hermes session_search-compatible transcript store
```

Imported rows must be marked:

```text
source = hermes_session_import
trust_level = primary_transcript_if_raw_available_else_secondary_summary
profile_id = derived from hermes_home/profile
```

### 33.8 Delegation observation event

`on_delegation(task, result, *, child_session_id, **kwargs)` is called on the parent provider when a subagent completes in Hermes builds that support it.

Persist a durable event:

```text
event_type = delegation_observed
```

Payload schema:

```json
{
  "parent_session_id": "...",
  "child_session_id": "...",
  "task": "delegation prompt",
  "result": "subagent final summary",
  "toolsets": [],
  "model": "optional",
  "created_at": "..."
}
```

Do not let subagents independently write durable memories unless they run as primary agents. Parent observation is the canonical path.

### 33.9 Ingest dedupe for protected head messages

Hermes protects the first messages during compression (`protect_first_n`, currently hardcoded around 3). These head messages can recur across compressed contexts.

`sync_turn()` and import jobs must dedupe by:

```text
profile_id
session_id
role
content_hash
created_at_bucket or original_message_id when available
```

At minimum enforce:

```sql
CREATE UNIQUE INDEX messages_ingest_dedupe_idx
ON messages(profile_id, session_id, role, content_hash)
WHERE source = 'sync_turn';
```

If two messages have same content but different real timestamps and both matter, preserve both by using platform message IDs when available.

### 33.10 Built-in memory mirror semantics

Hermes built-in memory tool supports `add`, `replace`, and `remove` for compact `memory` and `user` stores. It is tiny and orthogonal to Siqueira.

When Hermes calls `on_memory_write(action, target, content)`, Siqueira must record:

```text
event_type = builtin_memory_mirror
```

Payload:

```json
{
  "action": "add|replace|remove",
  "target": "memory|user",
  "content_redacted": "...",
  "content_hash": "...",
  "source": "hermes_builtin_memory_tool"
}
```

For `remove`, store the selector/hash, not necessarily the deleted full content unless already present and allowed by policy.

### 33.11 Embedding table deprecation policy

Because Siqueira uses one physical embedding table per model/dimension, old embedding tables will accumulate after model migrations.

Retention policy:

```text
Keep old embedding tables active during A/B and backfill.
Track query traffic per embedding table in retrieval_logs.
If an embedding table has zero production query traffic for 30 days and a newer active embedding table has full coverage, mark old table deprecated.
After 90 days deprecated, archive or drop old embedding table after backup.
```

Do not delete the source chunks; embeddings are rebuildable.

### 33.12 Advisory lock hashing

For candidate promotion locks, use a 64-bit advisory-lock key.

Preferred:

```sql
hashtextextended(canonical_key, 0)
```

Do not use `hashtext(canonical_key)` because it is effectively 32-bit and collision-prone at larger scale.

Advisory lock collisions are acceptable at single-user expected volume only if they cause conservative serialization, never incorrect promotion.

### 33.13 Contracted vs source-inspected hooks

Separate stable documented hooks from source-inspected hooks.

Documented/stable MemoryProvider hooks:

```text
system_prompt_block
prefetch
queue_prefetch
sync_turn
on_session_end
on_pre_compress
on_memory_write
shutdown
```

Source-inspected / version-sensitive hooks:

```text
on_turn_start
on_delegation
```

Implementation must:

```text
support on_turn_start/on_delegation when present;
not require them for correctness;
provide fallback through sync_turn/on_session_end;
pin or test Hermes version compatibility before relying on them.
```

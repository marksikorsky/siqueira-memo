# Siqueira Memo Memory Upgrade Implementation Plan

> **For Hermes:** Use `subagent-driven-development` to implement this plan task-by-task. Do not implement this as one giant PR.

**Goal:** Turn Siqueira Memo into an auditable operational memory for trusted agents: aggressive automatic capture, source-backed facts/decisions, controlled secret recall, conflict awareness, rollback, stronger retrieval, context-tree loading, entity cards, trust scoring, and a useful admin UI.

**Architecture:** Keep Siqueira's current strengths — source-backed facts/decisions, correction/forget, timeline, conflict handling, Hermes-native provider — and upgrade the weak spots found in competitor review. The near-term roadmap explicitly excludes Mem0-style public SDK/onboarding/product polish; that stays in backlog for later productization.

**Tech Stack:** FastAPI, SQLAlchemy/Postgres, current Siqueira worker queue, Hermes MemoryProvider hooks, OpenAI-compatible LLM capture endpoint, vanilla HTML/CSS/JS admin UI, pytest/mypy/ruff.

---

## 0. Product Direction / Non-Negotiables

Siqueira is **not** a generic memory SaaS right now.

Siqueira's niche:

> **Auditable operational memory for agents that are trusted with real work and real permissions.**

That means the system must optimize for:

- remembering enough, not under-saving;
- preserving sources/provenance;
- knowing what changed and why;
- recalling only the context needed for the task;
- storing secrets when operationally useful;
- preventing accidental secret leakage into casual prompts, UI, logs, exports, and delegated agents;
- surfacing conflicts and dangerous changes;
- allowing rollback/forget/correction.

### Explicitly rejected

Do **not** build:

- ByteRover-style mandatory human review for every fact/decision;
- security theater that pretends to protect secrets from the trusted root-level agent;
- Honcho-style psychological guesses without source evidence;
- Mem0-style product SDK/onboarding in the near-term core roadmap;
- black-box summaries that become truth without source links;
- cloud-first assumptions.

### Deferred to future productization

Mem0-inspired work is useful later, but not now:

- public Python SDK;
- public TypeScript SDK;
- polished quickstart;
- external onboarding;
- marketing examples;
- plug-and-play product packaging.

Keep this in backlog as `future-productization/mem0-dx`, not in the active core roadmap.

---

## 1. Current Baseline Confirmed From Code

Important existing files:

- `src/siqueira_memo/hermes_provider/provider.py` — Hermes MemoryProvider hooks: `sync_turn`, pre-compress/session-end/prefetch behavior.
- `src/siqueira_memo/workers/jobs.py` — worker handlers for ingest, chunking, embeddings, prefetch, extraction.
- `src/siqueira_memo/services/memory_capture_classifier.py` — current LLM capture classifier v1.
- `src/siqueira_memo/services/ingest_service.py` — durable message/event ingest.
- `src/siqueira_memo/services/retrieval_service.py` — current recall/retrieval layer.
- `src/siqueira_memo/services/context_pack_service.py` — prompt/context pack assembly.
- `src/siqueira_memo/services/conflict_service.py` — conflict detection/service layer.
- `src/siqueira_memo/services/redaction_service.py` — secret redaction/reference handling.
- `src/siqueira_memo/services/embedding_registry.py` — embedding provider metadata.
- `src/siqueira_memo/models/facts.py` — fact model.
- `src/siqueira_memo/models/decisions.py` — decision model.
- `src/siqueira_memo/models/events.py` — event/timeline model.
- `src/siqueira_memo/models/retrieval.py` — recall/retrieval request/result models.
- `src/siqueira_memo/api/routes_memory.py` — remember/correct/forget/timeline/sources endpoints.
- `src/siqueira_memo/api/routes_recall.py` — recall endpoints.
- `src/siqueira_memo/api/routes_admin.py` — admin JSON APIs.
- `src/siqueira_memo/api/routes_admin_ui.py` — current zero-build admin UI.
- `src/siqueira_memo/schemas/admin.py` — admin schemas.
- `src/siqueira_memo/schemas/memory.py` — memory API schemas.
- `tests/unit/test_hermes_provider.py` — provider/capture/classifier unit tests.
- `tests/integration/test_api_routes.py` — API integration tests.
- `tests/integration/test_admin_conflicts_audit.py` — admin conflict/audit tests.

Current strengths:

- source-backed facts and decisions;
- explicit correction/forget APIs;
- timeline/sources endpoints;
- conflict service already exists;
- Hermes-native memory provider/tools;
- raw turn ingest + chunk/embedding jobs;
- redaction service exists;
- admin UI exists;
- LLM classifier exists, but is only v1.

Current weak spots:

- capture classifier returns one memory decision per turn, not many candidates;
- secret handling is not yet designed as first-class controlled recall;
- no mature version/diff/rollback UX for memory mutations;
- retrieval is not yet full fusion: semantic + BM25 + entity + temporal + graph + reranker;
- relationship graph is not first-class;
- context tree is not first-class;
- trust/source reputation/feedback loop is underdeveloped;
- admin UI needs to become an actual memory cockpit, not just a search/debug page;
- eval/observability is not enough to know whether memory got better or silently worse.

---

## 2. Target Architecture

```text
Hermes conversation / tools / compaction / session end
  -> Hermes MemoryProvider hooks
  -> Siqueira ingest queue
  -> durable Message + MemoryEvent source layer
  -> chunks + embeddings
  -> LLM Memory Triage v2
       -> multiple candidates per turn
       -> auto-save / merge / supersede / conflict / review exception
       -> secret tagging + recall policy
  -> Fact / Decision / Entity / Relationship / Secret metadata
  -> version log + changelog + rollback snapshots
  -> retrieval fusion
       -> semantic
       -> full-text/BM25
       -> entity
       -> temporal
       -> graph expansion
       -> optional reranker
  -> context tree / context packs
  -> admin UI: capture, conflicts, sources, secrets, graph, evals, rollback
  -> Hermes tools: recall / remember / correct / forget / timeline / sources
```

---

## 3. Roadmap Overview

Active core roadmap:

1. **Capture v2** — multi-candidate LLM memory triage/extraction.
2. **Secret-aware memory** — secrets allowed, tagged, masked, controlled recall.
3. **Versioning/diff/rollback** — ByteRover idea, without mandatory review.
4. **Relationship graph** — RetainDB-style relationships between memories.
5. **Retrieval fusion** — Hindsight/RetainDB-style search quality.
6. **Context tree** — OpenViking-style hierarchical context loading.
7. **Entity cards** — Honcho-inspired project/user/server/API cards without psychology.
8. **Trust/feedback scoring** — Holographic-inspired trust/source reputation.
9. **Admin UI cockpit** — practical UI for capture, recall, conflicts, secrets, graph, evals.
10. **Observability/evals** — regression tests and dashboard for memory quality.

Deferred:

11. **Mem0-style SDK/API/onboarding** — future productization only.

---

# Phase 1 — Memory Capture v2

## Goal

Replace the current single-output classifier with a multi-candidate memory triage/extraction pipeline.

Current v1 roughly says:

```text
turn -> one MemoryCaptureDecision or skip
```

Target v2:

```text
turn -> many MemoryCandidate items
     -> auto_save / merge / supersede / conflict / needs_review / skip
```

## Contract

One conversation turn may contain:

- multiple facts;
- multiple decisions;
- user preference updates;
- project roadmap updates;
- secret values/references;
- operational details;
- conflicts with old memory;
- stale information;
- pure noise.

The classifier must return an array, not one item.

## Candidate schema

Create/extend schemas around `src/siqueira_memo/services/memory_capture_classifier.py` and likely new schema module:

- Create: `src/siqueira_memo/schemas/memory_capture.py`
- Modify: `src/siqueira_memo/services/memory_capture_classifier.py`
- Modify: `src/siqueira_memo/workers/jobs.py`
- Test: `tests/unit/test_memory_capture_classifier.py`
- Test: `tests/unit/test_hermes_provider.py`

Suggested fields:

```python
class MemoryCandidate(BaseModel):
    action: Literal[
        "auto_save",
        "skip_noise",
        "merge",
        "supersede",
        "flag_conflict",
        "needs_review",
    ]
    kind: Literal["fact", "decision", "preference", "secret", "entity", "relationship", "summary"]
    statement: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    project: str | None = None
    topic: str | None = None
    entity_names: list[str] = []
    confidence: float
    importance: float
    sensitivity: Literal["public", "internal", "private", "secret"] = "internal"
    risk: Literal["low", "medium", "high", "critical"] = "low"
    rationale: str
    source_message_ids: list[str] = []
    source_event_ids: list[str] = []
    relation_to_existing: list[MemoryRelationCandidate] = []
    review_reason: str | None = None
```

```python
class MemoryCaptureResult(BaseModel):
    candidates: list[MemoryCandidate]
    skipped_reason: str | None = None
    classifier_model: str | None = None
    prompt_version: str
```

## Classifier actions

- `auto_save` — save immediately.
- `skip_noise` — ack/small talk/tool boilerplate, do not promote.
- `merge` — combine with an existing memory.
- `supersede` — replace existing memory but keep old version.
- `flag_conflict` — save candidate and open conflict/review item.
- `needs_review` — do not auto-promote destructive/dangerous change until reviewed.

## Review policy

Review is exception-only.

Open review only when:

- candidate contradicts active decision/fact;
- candidate tries to delete/supersede important active memory;
- high importance + low confidence;
- dangerous infra/tax/security operation;
- unclear secret replacement/staleness;
- entity merge could corrupt unrelated projects/users/servers.

Do **not** review just because:

- memory is important;
- memory is a decision;
- memory contains a secret;
- classifier used LLM;
- many candidates were extracted.

## Capture audit

Never allow silent black holes.

Add audit events/counters for:

- classifier called;
- classifier failed;
- invalid JSON;
- fallback heuristic used;
- candidates extracted;
- candidates skipped;
- candidates auto-saved;
- candidates merged;
- candidates superseded;
- candidates sent to review;
- secret candidates saved;
- secret candidates masked from casual recall.

## Tests

Add tests for:

1. One turn produces multiple candidates.
2. Obvious noise produces `skip_noise` with audit event.
3. Secrets are tagged, not automatically dropped.
4. Conflict candidate opens conflict/review item.
5. Supersede creates new active memory and old version chain.
6. LLM failure falls back to deterministic heuristic.
7. Invalid JSON is audited and does not kill the worker.
8. Existing v1 behavior remains backward compatible until migration complete.

## Acceptance criteria

- `extract_turn_memory_handler()` can persist more than one structured memory per turn.
- Every skip/failure is explainable through admin/capture audit.
- Secrets are saved when useful and tagged as sensitive.
- Review queue remains small and exception-only.
- Tests prove non-marker durable information is captured semantically.

---

# Phase 2 — Secret-Aware Memory

## Goal

Support secrets as first-class operational memory without leaking them casually.

The point is **not** to protect secrets from the trusted root-level agent. The point is to prevent accidental leakage into:

- ordinary context packs;
- prompt prefetch;
- admin UI default views;
- exports;
- logs;
- delegated agents;
- Telegram answers where the secret was not needed.

## Data model additions

Likely files:

- Modify: `src/siqueira_memo/models/facts.py`
- Modify: `src/siqueira_memo/models/decisions.py`
- Possibly create: `src/siqueira_memo/models/secret_metadata.py`
- Modify: `src/siqueira_memo/schemas/memory.py`
- Modify: `src/siqueira_memo/services/redaction_service.py`
- Modify: `src/siqueira_memo/services/context_pack_service.py`
- Modify: `src/siqueira_memo/services/retrieval_service.py`
- Modify: `src/siqueira_memo/api/routes_admin.py`
- Modify: `src/siqueira_memo/api/routes_admin_ui.py`

Add metadata fields where appropriate:

```text
sensitivity: public | internal | private | secret
recall_policy: always | task_relevant | explicit_or_high_relevance | never_prefetch
masked_preview: string
secret_ref: string | null
secret_kind: api_key | token | password | connection_string | ssh | webhook | seed_phrase | other
last_secret_access_at: timestamp | null
secret_access_count: int
secret_stale_after: timestamp | null
```

Do not expose raw secret values in normal list/search UI.

## Recall policy

For `sensitivity=secret`:

- fast/balanced recall should return masked preview unless query explicitly asks to use/retrieve that secret;
- prompt prefetch should exclude secrets by default;
- delegated agent context packs should exclude secrets unless explicitly granted;
- export should default to masked/no raw secrets;
- forensic mode may show source references, but still avoid raw secret unless explicitly requested.

## UI behavior

- Secret chips/badges on records.
- Mask by default: `sk-...abcd`, never full value in table/list.
- “Reveal” requires explicit click and logs audit event.
- “Copy masked” and “Copy full” are separate actions.
- Show stale warning if secret is old or superseded.
- Show where secret was last used/recalled.
- Show blast radius notes: project/server/service linked to this secret.

## Tests

1. Secret candidate persists with `sensitivity=secret`.
2. Normal recall does not include raw secret.
3. Explicit recall can retrieve/use secret when policy allows.
4. Admin list shows masked preview, not raw value.
5. Reveal endpoint writes audit event.
6. Export defaults to no raw secrets.
7. Delegation context excludes secrets unless explicitly requested.

## Acceptance criteria

- Secrets are not treated as automatic `save=false`.
- Secrets are not casually injected into contexts.
- Every reveal/access is auditable.
- Masked UI is default.

---

# Phase 3 — Versioning, Diff, Backup, Rollback

## Goal

Take the useful part of ByteRover: memory changes are visible and reversible, without asking the user to approve everything.

## Required behavior

Every mutation of fact/decision/entity/relationship should have:

- operation id;
- actor;
- timestamp;
- old value;
- new value;
- diff;
- reason/rationale;
- source event/message ids;
- classifier/prompt/model version;
- rollback target.

Operations:

- create;
- update;
- merge;
- supersede;
- soft delete;
- restore;
- forget/hard delete request;
- conflict resolution.

## Likely files

- Create: `src/siqueira_memo/models/memory_versions.py`
- Create: `src/siqueira_memo/services/memory_version_service.py`
- Modify: `src/siqueira_memo/services/extraction_service.py` or equivalent remember path.
- Modify: `src/siqueira_memo/api/routes_memory.py`
- Modify: `src/siqueira_memo/api/routes_admin.py`
- Modify: `src/siqueira_memo/api/routes_admin_ui.py`
- Tests: `tests/unit/test_memory_versioning.py`
- Tests: `tests/integration/test_admin_conflicts_audit.py`

## UI behavior

For each memory detail drawer:

- show current active version;
- show timeline of versions;
- show diff between selected versions;
- show rollback button;
- show source snippets for each version;
- show whether this memory superseded another memory.

## Tests

1. Updating a fact creates version row.
2. Superseding a decision creates chain and deactivates old decision.
3. Rollback restores previous active version.
4. Diff endpoint returns readable before/after.
5. Admin UI contains version/diff/rollback markers.

## Acceptance criteria

- No important memory mutation is irreversible by accident.
- User can answer: “what changed in memory after this session?”
- Review is only needed for dangerous/conflicting changes, not normal saves.

---

# Phase 4 — Relationship Graph

## Goal

Make memory relational, not a pile of notes.

## Relationship types

```text
confirms
contradicts
updates
replaces
derived_from
depends_on
same_as
related_to
supersedes
invalidates
uses_secret
belongs_to_entity
```

## Data model

Create table/model:

```text
memory_relationships
  id
  source_type: fact | decision | entity | summary | message
  source_id
  relationship_type
  target_type
  target_id
  confidence
  rationale
  source_event_ids
  created_at
  created_by
  status: active | rejected | superseded
```

Likely files:

- Create: `src/siqueira_memo/models/relationships.py`
- Create: `src/siqueira_memo/services/relationship_service.py`
- Modify: `src/siqueira_memo/services/conflict_service.py`
- Modify: `src/siqueira_memo/services/retrieval_service.py`
- Modify: `src/siqueira_memo/schemas/memory.py`
- Modify: `src/siqueira_memo/api/routes_memory.py`
- Modify: `src/siqueira_memo/api/routes_admin.py`

## Capture integration

Classifier v2 should be able to emit relationship candidates:

```json
{
  "action": "auto_save",
  "kind": "relationship",
  "relationship_type": "supersedes",
  "source_statement": "Secrets may be stored when useful",
  "target_hint": "old secret skip policy",
  "confidence": 0.92
}
```

## Retrieval integration

Recall should use graph expansion:

- start from semantic/full-text hits;
- expand to directly related records;
- include superseded/conflicting records only when useful;
- explain relationship in output.

Example:

```text
Returned because this active decision supersedes earlier secret-skip policy.
```

## UI behavior

Add graph panel:

- memory node detail;
- incoming/outgoing relationships;
- conflict/supersede badges;
- simple text graph first, visual graph later;
- filters by relationship type;
- “why did this appear in recall?” trace.

## Tests

1. Create relationship between two memories.
2. Conflict service creates `contradicts` relation.
3. Supersede creates `replaces/supersedes` relation.
4. Retrieval includes related active decision.
5. UI shows relationship badges.

---

# Phase 5 — Retrieval Fusion

## Goal

Upgrade recall quality to match the best ideas from Hindsight/RetainDB.

## Search paths

Implement multiple retrieval lanes:

1. Semantic vector search.
2. Full-text/BM25-like search.
3. Entity search.
4. Project/topic exact filters.
5. Temporal search.
6. Relationship graph expansion.
7. Source/timeline search.
8. Optional reranker/cross-encoder stage.

## Fusion strategy

Use explicit scoring/fusion rather than one opaque query.

Candidate fields:

```text
source_lane: semantic | lexical | entity | temporal | graph | exact
base_score
recency_score
trust_score
project_match_score
source_quality_score
relationship_boost
final_score
explanation
```

Consider RRF-style fusion:

```text
final_score = weighted combination of lane ranks + trust/source/project boosts
```

## Likely files

- Modify: `src/siqueira_memo/services/retrieval_service.py`
- Create: `src/siqueira_memo/services/retrieval_fusion.py`
- Create: `src/siqueira_memo/services/entity_search.py`
- Modify: `src/siqueira_memo/models/retrieval.py`
- Modify: `src/siqueira_memo/services/context_pack_service.py`
- Tests: `tests/unit/test_retrieval_fusion.py`
- Tests: `tests/integration/test_api_routes.py`

## Recall output must explain itself

Return/source pack should include:

- why this was retrieved;
- which lane found it;
- confidence/trust;
- source ids;
- whether it is active/superseded/conflicting;
- whether secret content was masked or omitted.

## Tests

1. Lexical-only exact term finds memory vector missed.
2. Semantic query finds paraphrased decision.
3. Project filter prevents unrelated project pollution.
4. Graph expansion pulls superseding/related decision.
5. Temporal query finds “latest” decision.
6. Secret records are masked unless explicit.
7. Reranker optional; system works without it.

## Acceptance criteria

- Recall improves without sacrificing provenance.
- The system can explain why each item was returned.
- Secrets and superseded records are handled intentionally.

---

# Phase 6 — Context Tree / Budgeted Context Packs

## Goal

Load the right context for the task, not the entire memory soup.

## Tree model

```text
global
  user
    preferences
    work_style
    security_model
  agent
    behavior
    memory_policy

projects
  Siqueira Memo
    capture
    retrieval
    security
    ui
    roadmap
  Clawik
    infra
    deployment
    incidents
    secrets
  Brazil tax crypto
    wallets
    transactions
    reporting

entities
  servers
  wallets
  repos
  APIs
  providers
```

## Modes

- `fast` — tiny context, active decisions only.
- `balanced` — default, active facts/decisions + recent sources.
- `deep` — broader graph/timeline/context tree.
- `forensic` — source-heavy, conflicts, raw source snippets, timeline.

## Likely files

- Create: `src/siqueira_memo/models/context_tree.py`
- Create: `src/siqueira_memo/services/context_tree_service.py`
- Modify: `src/siqueira_memo/services/context_pack_service.py`
- Modify: `src/siqueira_memo/services/retrieval_service.py`
- Modify: `src/siqueira_memo/hermes_provider/provider.py`
- Tests: `tests/unit/test_context_tree_service.py`

## UI behavior

Add Context Tree tab:

- tree browser: global/project/topic/entity;
- counts per node;
- stale/conflict/secret badges;
- preview generated context pack for selected node;
- budget slider: fast/balanced/deep/forensic;
- “what will be sent to agent?” preview;
- “exclude secrets” toggle, on by default.

## Tests

1. Project-specific recall does not include unrelated project memory.
2. Global user preference still appears when relevant.
3. Secret node does not appear in ordinary context pack.
4. Context budget trims lower-priority items first.
5. UI shows tree and context preview markers.

---

# Phase 7 — Entity Cards

## Goal

Create source-backed cards for important people/projects/servers/repos/APIs/wallets.

This is inspired by Honcho, but without unsupported psychological inference.

## Entity types

```text
person
project
repo
server
service
api
wallet
company
workflow
credential
provider
document
```

## Entity card content

Each card should show:

- name;
- aliases;
- type;
- project/topic links;
- latest known facts;
- active decisions;
- related secrets, masked;
- source count;
- confidence/trust;
- last updated;
- conflicts/stale warnings;
- relationship graph links.

## Do not include

- emotional/psychological guesses;
- inferred intent without source;
- personality summaries beyond explicit user preferences;
- unsupported claims.

## Likely files

- Create: `src/siqueira_memo/models/entities.py`
- Create: `src/siqueira_memo/services/entity_service.py`
- Modify: `src/siqueira_memo/services/memory_capture_classifier.py`
- Modify: `src/siqueira_memo/services/retrieval_service.py`
- Modify: `src/siqueira_memo/api/routes_admin.py`
- Modify: `src/siqueira_memo/api/routes_admin_ui.py`
- Tests: `tests/unit/test_entity_service.py`

## UI behavior

Add Entities tab:

- list entities with type/project filters;
- entity detail card;
- aliases and merge suggestions;
- source-backed facts/decisions;
- conflicts/stale markers;
- related memories;
- “merge entities” review only for ambiguous cases.

## Tests

1. Entity is created from memory candidate.
2. Alias maps to same entity.
3. Ambiguous merge opens review, not auto-merge.
4. Entity card never includes unsupported psychological inference.
5. Recall for project/entity uses card as routing, not as sole truth.

---

# Phase 8 — Trust / Source Reputation / Feedback

## Goal

Each memory should carry a trust signal, not just text.

Inspired by Holographic-style trust/source scoring.

## Trust inputs

- source type: user/tool/agent/import/summary;
- whether source_event_ids exist;
- user-confirmed vs assistant-inferred;
- extraction confidence;
- conflict count;
- correction history;
- superseded status;
- successful recall/use count;
- last verified date;
- model/prompt version;
- whether memory came from raw message, summary, or imported transcript.

## Suggested trust score

Keep it explainable:

```text
trust_score = source_quality + confirmation + recency + low_conflict + extraction_confidence - penalties
```

Do not make this mystical.

## Likely files

- Create: `src/siqueira_memo/services/trust_service.py`
- Modify: `src/siqueira_memo/models/facts.py`
- Modify: `src/siqueira_memo/models/decisions.py`
- Modify: `src/siqueira_memo/services/retrieval_fusion.py`
- Modify: `src/siqueira_memo/api/routes_admin.py`
- Modify: `src/siqueira_memo/api/routes_admin_ui.py`
- Tests: `tests/unit/test_trust_service.py`

## UI behavior

- Trust badge on each memory.
- Explanation popover: “why trust 0.91?”
- Filters: low-trust, stale, conflicting, unverified.
- Feedback buttons: useful / wrong / stale / duplicate.
- Feedback creates audit event and can lower/raise trust.

## Tests

1. User-confirmed memory scores higher than inferred summary.
2. Corrected/superseded memory trust drops.
3. Source-backed memory scores higher than source-less import.
4. Feedback modifies score through auditable event.
5. Retrieval uses trust as boost, not absolute gate.

---

# Phase 9 — Admin UI Cockpit

## Goal

Turn `/admin` into a real memory cockpit: the place to inspect, debug, correct, rollback, and understand Siqueira.

Keep it zero-build for now: FastAPI + HTML + CSS + vanilla JS. No React/Vite/npm unless future productization needs it.

## Current UI files

- `src/siqueira_memo/api/routes_admin_ui.py`
- `src/siqueira_memo/api/routes_admin.py`
- `src/siqueira_memo/schemas/admin.py`
- `tests/integration/test_admin_conflicts_audit.py`

## UI principles

- Mobile-friendly Telegram workflow first.
- Fast enough on Tailscale/private admin.
- No raw secrets by default.
- Every table row links to sources.
- Every important mutation has diff/rollback.
- Empty dashboard is bad UX: load meaningful defaults.
- Prefer cards/drawers/tabs over raw JSON dumps.
- Admin UI must show confidence/trust/source/risk, not only text.

## Proposed UI sections

### 9.1 Overview / Health

Show:

- capture mode;
- raw turns saved today;
- structured memories extracted today;
- classifier success/failure rate;
- skip count with reasons;
- conflicts open;
- review exceptions open;
- secret records count;
- secret recalls count;
- stale records count;
- latest worker errors;
- embedding/indexing queue status.

Actions:

- run capture smoke test;
- run recall smoke test;
- view latest failed jobs;
- refresh counters.

### 9.2 Capture Audit

Show each processed turn/session:

- timestamp;
- session id;
- source user/assistant preview;
- classifier output;
- candidates count;
- saved/skipped/reviewed;
- skip reason;
- model/prompt version;
- fallback used or not;
- link to source messages.

Filters:

- project;
- topic;
- classifier model;
- action;
- sensitivity;
- skip reason;
- date range.

This is critical because silent `save=false` is how memory dies quietly.

### 9.3 Recall Playground

Controls:

- query;
- mode: fast/balanced/deep/forensic;
- project/topic/entity;
- include sources;
- include conflicts;
- allow secret recall toggle, off by default;
- max tokens/budget.

Show:

- answer context pack;
- retrieved items;
- score breakdown;
- retrieval lane: semantic/lexical/entity/temporal/graph;
- source snippets;
- masked secret notices;
- “why this was included”.

### 9.4 Memory Search

Search across:

- facts;
- decisions;
- entities;
- relationships;
- messages;
- summaries;
- events.

Each row shows:

- type;
- statement;
- project/topic;
- status;
- sensitivity;
- trust;
- confidence;
- sources count;
- last updated;
- conflict badge;
- version count.

### 9.5 Detail Drawer

For each memory item:

- full statement;
- structured fields;
- source snippets;
- active/superseded status;
- confidence/trust;
- relationships;
- version history;
- diff;
- correction form;
- soft forget;
- rollback;
- export source-backed markdown.

### 9.6 Conflicts / Review Exceptions

This replaces mandatory review.

Show only exceptional cases:

- active conflicts;
- destructive delete/supersede requests;
- low-confidence/high-importance candidates;
- dangerous infra/tax/security changes;
- ambiguous entity merges;
- secret replacement/staleness questions.

Actions:

- accept new;
- keep old;
- merge;
- mark stale;
- supersede;
- reject candidate;
- open source timeline;
- rollback.

### 9.7 Secrets Vault View

Not a hardened vault; an operational secret memory view.

Show:

- secret records masked;
- project/service/entity linked;
- secret kind;
- stale status;
- last accessed;
- access count;
- source reference;
- blast-radius note;
- related deployment/server/API.

Actions:

- reveal with audit;
- copy full with audit;
- rotate-needed marker;
- mark stale;
- supersede with new value;
- hide from all prefetch;
- view access log.

Never show raw secrets in tables.

### 9.8 Relationship Graph

First version can be text/list UI, not fancy canvas.

Show:

- selected memory node;
- incoming relationships;
- outgoing relationships;
- conflicts;
- supersedes/superseded-by;
- derived-from sources;
- related entities.

Later: visual graph.

### 9.9 Context Tree

Show hierarchy:

- global;
- user;
- projects;
- topics;
- entities;
- secrets;
- workflows.

Capabilities:

- click node -> preview context pack;
- budget/mode slider;
- show what will go to Hermes;
- show excluded secrets;
- show stale/conflict counts;
- compare context packs between modes.

### 9.10 Entity Cards

Show cards for:

- Mark;
- Siqueira Memo;
- Hermes;
- Clawik;
- DraftMotion;
- servers;
- repos;
- wallets;
- APIs;
- workflows.

Each card:

- facts;
- decisions;
- relationships;
- masked secrets;
- sources;
- latest changes;
- conflicts.

### 9.11 Timeline / Sources

Improve source inspection:

- chronological timeline;
- source event/message window;
- compaction/session-end source capture;
- tool-event source summaries;
- “show me how this memory was created”.

### 9.12 Evals / Regression Dashboard

Show:

- test query suites;
- expected memories;
- actual recall;
- pass/fail;
- classifier precision/recall samples;
- retrieval lane contribution;
- before/after prompt/model changes.

Start with small curated evals, not a huge benchmark project.

## UI acceptance criteria

- `/admin` loads on desktop and phone.
- No React/npm/build step.
- No raw secret values in initial HTML or table rows.
- Memory rows have detail drawer with sources.
- Capture audit shows skip reasons and classifier failures.
- Recall playground explains retrieval lanes/scores.
- Conflicts/review tab contains only exceptions.
- Secret reveal writes audit event.
- Rollback/diff UI exists for memory versions.
- Context tree preview shows prompt-safe context.

---

# Phase 10 — Observability and Evals

## Goal

Know whether memory is actually improving.

## Metrics

Capture metrics:

- turns ingested;
- messages saved;
- chunks created;
- embeddings created;
- classifier calls;
- classifier failures;
- candidates per turn;
- auto-save count;
- skip count by reason;
- review exception count;
- conflict count;
- secret candidate count;
- secret recall count;
- rollback count.

Retrieval metrics:

- query count;
- mode distribution;
- latency;
- lane contribution;
- empty recall rate;
- user correction rate;
- source coverage;
- secret omission/reveal count;
- stale/conflicting item inclusion count.

Quality evals:

- curated query set per project;
- expected facts/decisions;
- expected exclusions;
- secret masking tests;
- conflict surfacing tests;
- regression after prompt/model change.

## Likely files

- Create: `src/siqueira_memo/services/memory_eval_service.py`
- Create: `src/siqueira_memo/models/evals.py`
- Modify: `src/siqueira_memo/api/routes_admin.py`
- Modify: `src/siqueira_memo/api/routes_admin_ui.py`
- Tests: `tests/unit/test_memory_evals.py`
- Tests: `tests/integration/test_admin_conflicts_audit.py`

## Acceptance criteria

- Admin dashboard can explain memory health.
- A bad classifier prompt/model change is visible quickly.
- Recall regressions are testable.
- Secret recall/masking is measurable.

---

# Phase 11 — Future Productization Backlog: Mem0-Style DX

Do not implement this in near-term core work.

Keep as future tasks for when Siqueira becomes a product:

- public Python SDK;
- public TypeScript SDK;
- polished MCP server package;
- external docs;
- quickstart;
- Docker template;
- examples for LangChain/LlamaIndex/Claude Code/OpenClaw/Hermes;
- hosted demo;
- migration tools;
- public benchmark story.

Reason for deferral:

> Core memory quality matters more right now than external developer polish.

---

# Implementation Sequencing

## Milestone A — Make capture impossible to silently fail

Tasks:

1. Add capture audit events/counters for classifier skip/failure/fallback.
2. Add tests for classifier skip audit.
3. Add admin capture audit panel basics.
4. Verify worker queue drains into source + structured rows.

Why first:

If capture is blind, every later feature is guesswork.

## Milestone B — Multi-candidate capture

Tasks:

1. Add `MemoryCandidate` / `MemoryCaptureResult` schemas.
2. Update classifier prompt and parser.
3. Update worker to process candidate arrays.
4. Keep v1 compatibility temporarily.
5. Add tests for multiple facts/decisions from one turn.
6. Add tests for secret tagging and skip noise.

## Milestone C — Secret-aware policy

Tasks:

1. Add sensitivity/recall/display policy metadata.
2. Ensure context packs exclude secrets by default.
3. Add explicit secret recall path.
4. Add masked UI rendering.
5. Add reveal audit endpoint/action.
6. Add tests for masking/export/delegation behavior.

## Milestone D — Versioning and rollback

Tasks:

1. Add version model/service.
2. Wrap remember/correct/forget/supersede paths.
3. Add diff endpoint.
4. Add rollback endpoint.
5. Add admin UI version drawer.
6. Add tests.

## Milestone E — Relationships/conflicts

Tasks:

1. Add relationship model/service.
2. Connect conflict service to relationships.
3. Add relationship-aware supersede/replace.
4. Add admin relationship/conflict UI.
5. Add graph expansion to retrieval.

## Milestone F — Retrieval fusion

Tasks:

1. Extract retrieval lanes.
2. Add lexical/BM25-like lane.
3. Add entity lane.
4. Add temporal lane.
5. Add graph expansion lane.
6. Add score explanation.
7. Add optional reranker.
8. Add recall playground UI explanations.

## Milestone G — Context tree/entity cards/trust

Tasks:

1. Add context tree model/service.
2. Add entity model/service.
3. Add trust service.
4. Wire trust into retrieval ranking.
5. Add admin tabs.
6. Add tests.

## Milestone H — Evals/dashboard

Tasks:

1. Add curated eval query format.
2. Add eval runner.
3. Add admin eval dashboard.
4. Add CI/local regression command.
5. Add sample evals for Siqueira, Clawik, infra, tax, secrets.

---

# Testing Strategy

Always start with RED tests for behavior.

## Required test commands

Local:

```bash
cd /root/siqueira-memo
. .venv/bin/activate
ruff check src tests
mypy src/siqueira_memo
pytest tests -q
```

If full mypy is too noisy, type-check touched files first:

```bash
mypy \
  src/siqueira_memo/services/memory_capture_classifier.py \
  src/siqueira_memo/workers/jobs.py \
  src/siqueira_memo/services/retrieval_service.py \
  src/siqueira_memo/api/routes_admin.py \
  src/siqueira_memo/api/routes_admin_ui.py
```

Docker/live after backend changes:

```bash
cd /root/siqueira-memo
docker compose build api worker
docker compose up -d --no-deps --force-recreate api worker
```

Check booleans/status only; do not print secrets.

## Regression scenarios

1. User gives multiple durable decisions in one message.
2. User gives secret and operational instruction in one message.
3. User reverses old decision.
4. User gives vague correction.
5. Tool-heavy session contains important output.
6. Compaction happens after long session.
7. Recall asks for roadmap.
8. Recall asks for specific secret explicitly.
9. Recall asks unrelated question and must not include secret.
10. Admin UI loads on mobile.
11. Export excludes raw secrets.
12. Rollback restores prior decision.

---

# Security Model

Trusted operator mode.

## What we accept

- Agent has root/terminal in Mark's workflow.
- Telegram can contain secrets when Mark chooses.
- Siqueira may store secrets when operationally useful.
- Docker/user isolation is not the main safety boundary here if it wrecks UX.

## What we protect against

- accidental secret leakage into ordinary context packs;
- accidental secret display in UI;
- accidental secret output in Telegram answers;
- delegated agents receiving secrets without need;
- exports/logs containing raw secrets;
- stale secrets being used as current;
- source-less false memories becoming truth;
- conflicts being silently flattened;
- destructive memory changes without rollback.

## Rule

> Do not pretend secrets are safe because we refused to store them. Make them usable, tagged, masked by default, access-controlled by context, and auditable.

---

# Definition of Done for the Whole Roadmap

Siqueira is “good enough” when:

- most useful turns are saved automatically;
- a turn can produce many memories;
- secrets can be saved and used without casual leakage;
- every important memory has sources;
- conflicts are visible;
- old decisions can be superseded cleanly;
- memory changes can be diffed and rolled back;
- recall explains why items were returned;
- context packs are scoped and budgeted;
- admin UI shows capture health, conflicts, secrets, sources, versions, graph, evals;
- evals catch capture/retrieval regressions;
- Mem0-style product DX remains deferred until core quality is solid.

---

# First PR Recommendation

Start with the smallest high-leverage slice:

**PR 1: Capture Audit + Multi-Candidate Schema Skeleton**

Include:

1. `MemoryCandidate` / `MemoryCaptureResult` schemas.
2. Classifier parser that can accept array output.
3. Worker loop over candidates, initially supporting only `auto_save`, `skip_noise`, `needs_review`.
4. Capture audit events for skip/failure/fallback.
5. Admin capture audit counters.
6. Tests proving:
   - multi-candidate extraction;
   - secret candidate is tagged, not dropped;
   - skip is audited;
   - fallback still works.

Do **not** start with UI polish or SDKs. Start with making memory capture truthful and observable.

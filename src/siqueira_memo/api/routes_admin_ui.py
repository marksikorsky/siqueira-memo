"""Small built-in admin UI for local Siqueira operators."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_ADMIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Siqueira Memo Admin</title>
  <style>
    :root {
      --bg: #fbfaf8;
      --surface: #ffffff;
      --surface-soft: #f4f2ee;
      --text: #22211f;
      --muted: #68635d;
      --faint: #9c968f;
      --line: rgba(34, 33, 31, 0.12);
      --accent: #1d6fd8;
      --accent-dark: #1658ad;
      --good: #168466;
      --warn: #a65f00;
      --danger: #b42318;
      --shadow: rgba(0,0,0,0.04) 0 16px 42px, rgba(0,0,0,0.03) 0 4px 14px;
      --radius: 18px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html { background: var(--bg); color: var(--text); font-family: var(--sans); }
    body { margin: 0; min-height: 100vh; }
    button, input, select, textarea { font: inherit; }
    button { cursor: pointer; }
    .shell { max-width: 1180px; margin: 0 auto; padding: 22px; }
    .topbar { display: flex; gap: 16px; justify-content: space-between; align-items: center; margin-bottom: 22px; }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .logo { width: 42px; height: 42px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); box-shadow: var(--shadow); display: grid; place-items: center; font-weight: 760; letter-spacing: -0.04em; }
    h1 { font-size: clamp(30px, 6vw, 56px); line-height: .96; letter-spacing: -0.055em; margin: 0; }
    .subtitle { color: var(--muted); margin: 8px 0 0; max-width: 740px; line-height: 1.45; }
    .badge { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 650; background: #eef6ff; color: var(--accent); border: 1px solid rgba(29,111,216,.13); white-space: nowrap; }
    .grid { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 18px; align-items: start; }
    .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
    .card.pad { padding: 18px; }
    .card-title { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 0 0 14px; }
    h2 { font-size: 19px; line-height: 1.15; letter-spacing: -0.02em; margin: 0; }
    label { display: block; color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; margin: 14px 0 7px; }
    input, select, textarea { width: 100%; border: 1px solid var(--line); background: #fff; color: var(--text); border-radius: 12px; padding: 12px 12px; outline: none; min-height: 44px; }
    textarea { min-height: 94px; resize: vertical; line-height: 1.4; }
    input:focus, select:focus, textarea:focus { border-color: rgba(29,111,216,.55); box-shadow: 0 0 0 4px rgba(29,111,216,.12); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
    .btn { border: 1px solid var(--line); border-radius: 12px; min-height: 42px; padding: 10px 13px; background: var(--surface); color: var(--text); font-weight: 680; }
    .btn.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .btn.primary:hover { background: var(--accent-dark); }
    .btn.danger { color: var(--danger); }
    .btn:disabled { opacity: .55; cursor: not-allowed; }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.45; }
    .status { min-height: 22px; color: var(--muted); font-size: 13px; margin-top: 12px; }
    .tabs { display: flex; gap: 8px; padding: 10px; border-bottom: 1px solid var(--line); background: var(--surface-soft); overflow-x: auto; }
    .tab { border: 1px solid transparent; background: transparent; color: var(--muted); border-radius: 999px; padding: 9px 12px; white-space: nowrap; font-weight: 700; }
    .tab.active { background: var(--surface); color: var(--text); border-color: var(--line); box-shadow: rgba(0,0,0,.03) 0 2px 8px; }
    .panel { display: none; padding: 18px; }
    .panel.active { display: block; }
    .results { display: grid; gap: 10px; }
    .item { border: 1px solid var(--line); border-radius: 14px; padding: 13px; background: #fff; }
    .item-head { display: flex; gap: 8px; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
    .type { font: 700 11px/1 var(--mono); text-transform: uppercase; letter-spacing: .08em; color: var(--accent); background: #f1f7ff; border-radius: 999px; padding: 5px 7px; }
    .meta { color: var(--faint); font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
    .preview { white-space: pre-wrap; line-height: 1.45; overflow-wrap: anywhere; }
    .empty { color: var(--muted); text-align: center; border: 1px dashed var(--line); border-radius: 16px; padding: 36px 18px; background: rgba(255,255,255,.62); }
    .pillline { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .small { font-size: 12px; color: var(--muted); }
    .mono { font-family: var(--mono); }
    .split { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; }
    .footer { color: var(--faint); font-size: 12px; text-align: center; padding: 24px 0 4px; }
    @media (max-width: 720px) {
      .shell { padding: 14px; }
      .topbar { align-items: flex-start; }
      .grid { grid-template-columns: 1fr; }
      .row, .split { grid-template-columns: 1fr; }
      .brand { align-items: flex-start; }
      .logo { width: 38px; height: 38px; flex: 0 0 auto; }
      .badge { display: none; }
      .card.pad, .panel { padding: 14px; }
      .tabs { padding: 8px; }
      h1 { font-size: 38px; }
      .btn { width: 100%; }
      .actions { display: grid; grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="logo">S</div>
        <div>
          <h1>Siqueira Memo</h1>
          <p class="subtitle">Lightweight memory admin: search facts, inspect decisions, open provenance, and clean stale records. No build step, no npm zoo.</p>
        </div>
      </div>
      <span class="badge">Local admin UI</span>
    </header>

    <section class="grid">
      <aside class="card pad">
        <div class="card-title"><h2>Connection</h2><span id="connection-badge" class="badge">Not checked</span></div>
        <label for="token">API token</label>
        <input id="token" type="password" autocomplete="off" placeholder="Paste SIQUEIRA_API_TOKEN">
        <p class="hint">Stored only in this browser via localStorage. The backend APIs still require Bearer auth.</p>
        <div class="actions">
          <button class="btn primary" id="save-token">Save token</button>
          <button class="btn" id="check-ready">Check ready</button>
        </div>
        <div id="connection-status" class="status"></div>

        <label for="profile">Profile</label>
        <input id="profile" placeholder="default" value="default">

        <label for="project">Project filter</label>
        <input id="project" placeholder="e.g. brazil-tax-crypto">

        <label for="topic">Topic filter</label>
        <input id="topic" placeholder="optional">
      </aside>

      <section class="card">
        <nav class="tabs" aria-label="Admin sections">
          <button class="tab active" data-tab="search">Search</button>
          <button class="tab" data-tab="timeline">Timeline</button>
          <button class="tab" data-tab="remember">Remember</button>
          <button class="tab" data-tab="sources">Sources</button>
        </nav>

        <section id="search" class="panel active">
          <div class="card-title"><h2>Search memory</h2><span class="small">facts, decisions, messages, summaries</span></div>
          <div class="row">
            <div><label for="target-type">Type</label><select id="target-type"><option value="fact">Facts</option><option value="decision">Decisions</option><option value="message">Messages</option><option value="summary">Summaries</option></select></div>
            <div><label for="status-filter">Status</label><select id="status-filter"><option value="">Any</option><option value="active">Active</option><option value="superseded">Superseded</option><option value="invalidated">Invalidated</option></select></div>
          </div>
          <label for="query">Query</label>
          <div class="split"><input id="query" placeholder="Search text"><button class="btn primary" id="run-search">Search</button></div>
          <div id="search-status" class="status"></div>
          <div id="search-results" class="results"><div class="empty">Run a search to inspect memory.</div></div>
        </section>

        <section id="timeline" class="panel">
          <div class="card-title"><h2>Timeline</h2><span class="small">chronological facts + decisions</span></div>
          <div class="actions"><button class="btn primary" id="load-timeline">Load timeline</button></div>
          <div id="timeline-status" class="status"></div>
          <div id="timeline-results" class="results"><div class="empty">Load a project/topic timeline.</div></div>
        </section>

        <section id="remember" class="panel">
          <div class="card-title"><h2>Remember manually</h2><span class="small">promote a fact or decision</span></div>
          <div class="row">
            <div><label for="remember-kind">Kind</label><select id="remember-kind"><option value="fact">Fact</option><option value="decision">Decision</option></select></div>
            <div><label for="remember-confidence">Confidence</label><input id="remember-confidence" type="number" min="0" max="1" step="0.05" value="0.9"></div>
          </div>
          <div id="fact-fields">
            <label for="fact-subject">Subject</label><input id="fact-subject" placeholder="Mark / project / wallet">
            <label for="fact-predicate">Predicate</label><input id="fact-predicate" placeholder="prefers / uses / cost basis">
            <label for="fact-object">Object</label><input id="fact-object" placeholder="concise object value">
          </div>
          <div id="decision-fields" style="display:none">
            <label for="decision-topic">Decision topic</label><input id="decision-topic" placeholder="architecture / tax / deployment">
            <label for="decision-rationale">Rationale</label><input id="decision-rationale" placeholder="why this is the choice">
          </div>
          <label for="remember-statement">Statement</label><textarea id="remember-statement" placeholder="Write the durable memory exactly as it should be recalled."></textarea>
          <div class="actions"><button class="btn primary" id="save-memory">Save memory</button></div>
          <div id="remember-status" class="status"></div>
        </section>

        <section id="sources" class="panel">
          <div class="card-title"><h2>Sources & cleanup</h2><span class="small">inspect provenance or soft-delete a memory</span></div>
          <div class="row">
            <div><label for="source-type">Type</label><select id="source-type"><option value="fact">Fact</option><option value="decision">Decision</option><option value="summary">Summary</option></select></div>
            <div><label for="source-id">ID</label><input id="source-id" class="mono" placeholder="uuid"></div>
          </div>
          <div class="actions"><button class="btn primary" id="load-sources">Load sources</button><button class="btn danger" id="soft-delete">Soft delete</button></div>
          <div id="sources-status" class="status"></div>
          <div id="sources-results" class="results"><div class="empty">Paste an ID from search/timeline.</div></div>
        </section>
      </section>
    </section>
    <div class="footer">Designed for localhost use. Keep the token private; do not expose this service publicly without real auth in front.</div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const state = {
      token: localStorage.getItem('siqueira.apiToken') || '',
      profile: localStorage.getItem('siqueira.profile') || 'default'
    };
    $('token').value = state.token;
    $('profile').value = state.profile;

    function authHeaders() {
      const token = $('token').value.trim();
      const profile = $('profile').value.trim() || 'default';
      const headers = { 'Content-Type': 'application/json', 'X-Profile-Id': profile };
      if (token) headers.Authorization = 'Bearer ' + token;
      return headers;
    }
    function filters() {
      return { profile_id: $('profile').value.trim() || 'default', project: $('project').value.trim() || null, topic: $('topic').value.trim() || null };
    }
    function setStatus(id, text, tone='muted') {
      const el = $(id); el.textContent = text || ''; el.style.color = tone === 'danger' ? 'var(--danger)' : tone === 'good' ? 'var(--good)' : 'var(--muted)';
    }
    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
    }
    async function post(path, body) {
      const res = await fetch(path, { method: 'POST', headers: authHeaders(), body: JSON.stringify(body) });
      const text = await res.text();
      let data; try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
      if (!res.ok) throw new Error(data.detail || res.statusText || 'Request failed');
      return data;
    }
    function renderHits(target, hits) {
      if (!hits.length) { $(target).innerHTML = '<div class="empty">Nothing found.</div>'; return; }
      $(target).innerHTML = hits.map((hit) => `
        <article class="item">
          <div class="item-head"><span class="type">${escapeHtml(hit.target_type || hit.kind)}</span><span class="meta mono">${escapeHtml(hit.id)}</span></div>
          <div class="preview">${escapeHtml(hit.preview || hit.statement || hit.decision || hit.title || '')}</div>
          <div class="pillline">
            ${hit.project ? `<span class="badge">${escapeHtml(hit.project)}</span>` : ''}
            ${hit.topic ? `<span class="badge">${escapeHtml(hit.topic)}</span>` : ''}
            ${hit.status ? `<span class="badge">${escapeHtml(hit.status)}</span>` : ''}
          </div>
          <div class="meta">${escapeHtml(hit.created_at || hit.decided_at || '')}</div>
        </article>`).join('');
    }

    document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {
      document.querySelectorAll('.tab,.panel').forEach((el) => el.classList.remove('active'));
      tab.classList.add('active'); $(tab.dataset.tab).classList.add('active');
    }));

    $('save-token').onclick = () => {
      localStorage.setItem('siqueira.apiToken', $('token').value.trim());
      localStorage.setItem('siqueira.profile', $('profile').value.trim() || 'default');
      setStatus('connection-status', 'Saved locally.', 'good');
    };
    $('check-ready').onclick = async () => {
      try {
        const res = await fetch('/readyz');
        const body = await res.json();
        $('connection-badge').textContent = body.ok ? 'Ready' : 'Not ready';
        setStatus('connection-status', body.ok ? 'API ready. Database reachable.' : 'API responded but is not ready.', body.ok ? 'good' : 'danger');
      } catch (err) { setStatus('connection-status', err.message, 'danger'); }
    };
    $('run-search').onclick = async () => {
      setStatus('search-status', 'Searching…');
      try {
        const body = await post('/v1/admin/search', { ...filters(), query: $('query').value.trim() || null, target_type: $('target-type').value, status: $('status-filter').value || null, limit: 50 });
        setStatus('search-status', `${body.total} result(s).`, 'good');
        renderHits('search-results', body.hits || []);
      } catch (err) { setStatus('search-status', err.message, 'danger'); }
    };
    $('load-timeline').onclick = async () => {
      setStatus('timeline-status', 'Loading…');
      try {
        const body = await post('/v1/memory/timeline', { ...filters(), limit: 50 });
        setStatus('timeline-status', `${body.entries.length} timeline item(s).`, 'good');
        renderHits('timeline-results', (body.entries || []).map(e => ({...e, target_type: e.kind, preview: e.preview})));
      } catch (err) { setStatus('timeline-status', err.message, 'danger'); }
    };
    $('remember-kind').onchange = () => {
      const isFact = $('remember-kind').value === 'fact';
      $('fact-fields').style.display = isFact ? '' : 'none';
      $('decision-fields').style.display = isFact ? 'none' : '';
    };
    $('save-memory').onclick = async () => {
      setStatus('remember-status', 'Saving…');
      try {
        const kind = $('remember-kind').value;
        const payload = { ...filters(), kind, statement: $('remember-statement').value.trim(), confidence: Number($('remember-confidence').value || 0.9) };
        if (kind === 'fact') Object.assign(payload, { subject: $('fact-subject').value.trim(), predicate: $('fact-predicate').value.trim(), object: $('fact-object').value.trim() });
        else Object.assign(payload, { topic: $('decision-topic').value.trim() || $('topic').value.trim() || 'manual', rationale: $('decision-rationale').value.trim() });
        const body = await post('/v1/memory/remember', payload);
        setStatus('remember-status', `Saved ${body.kind}: ${body.id}`, 'good');
      } catch (err) { setStatus('remember-status', err.message, 'danger'); }
    };
    $('load-sources').onclick = async () => {
      setStatus('sources-status', 'Loading…');
      try {
        const body = await post('/v1/memory/sources', { profile_id: $('profile').value.trim() || 'default', target_type: $('source-type').value, target_id: $('source-id').value.trim() });
        setStatus('sources-status', `${body.sources.length} source(s).`, 'good');
        $('sources-results').innerHTML = body.sources.length ? body.sources.map(s => `<article class="item"><div class="preview mono">${escapeHtml(JSON.stringify(s, null, 2))}</div></article>`).join('') : '<div class="empty">No sources found.</div>';
      } catch (err) { setStatus('sources-status', err.message, 'danger'); }
    };
    $('soft-delete').onclick = async () => {
      if (!confirm('Soft-delete this memory?')) return;
      setStatus('sources-status', 'Deleting…');
      try {
        const body = await post('/v1/memory/forget', { profile_id: $('profile').value.trim() || 'default', target_type: $('source-type').value, target_id: $('source-id').value.trim(), mode: 'soft', reason: 'admin-ui' });
        setStatus('sources-status', `Soft-deleted. Event ${body.event_id}`, 'good');
      } catch (err) { setStatus('sources-status', err.message, 'danger'); }
    };
    $('query').addEventListener('keydown', (event) => { if (event.key === 'Enter') $('run-search').click(); });
  </script>
</body>
</html>
"""


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_ui() -> HTMLResponse:
    """Serve the zero-build browser admin interface."""
    return HTMLResponse(_ADMIN_HTML)


@router.get("/admin/", response_class=HTMLResponse, include_in_schema=False)
async def admin_ui_slash() -> HTMLResponse:
    """Serve the same UI with a trailing slash."""
    return HTMLResponse(_ADMIN_HTML)

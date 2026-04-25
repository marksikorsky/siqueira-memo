"""Small built-in admin UI for local Siqueira operators."""

from __future__ import annotations

from html import escape
from typing import cast
from urllib.parse import parse_qs

from fastapi import APIRouter, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from siqueira_memo.api.deps import (
    ADMIN_SESSION_COOKIE,
    admin_auth_enabled,
    create_admin_session_token,
    request_has_admin_session,
    verify_admin_password,
)
from siqueira_memo.config import Settings

_LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Siqueira Memo Admin Login</title>
  <style>
    :root { --bg: #fbfaf8; --card: #fff; --text: #22211f; --muted: #6d675f; --line: rgba(34,33,31,.13); --accent: #1d6fd8; --danger: #a73522; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 22px; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(100%, 420px); background: var(--card); border: 1px solid var(--line); border-radius: 24px; padding: 28px; box-shadow: rgba(0,0,0,.05) 0 18px 50px, rgba(0,0,0,.03) 0 4px 16px; }
    .eyebrow { margin: 0 0 10px; color: var(--accent); font-size: 12px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 7vw, 38px); letter-spacing: -.05em; }
    p { margin: 0 0 24px; color: var(--muted); line-height: 1.5; }
    label { display: block; margin: 0 0 8px; font-size: 13px; font-weight: 750; }
    input { width: 100%; min-height: 46px; border: 1px solid var(--line); border-radius: 14px; padding: 0 13px; font: inherit; background: #fff; color: var(--text); }
    input:focus { outline: 3px solid rgba(29,111,216,.15); border-color: rgba(29,111,216,.65); }
    button { width: 100%; min-height: 46px; margin-top: 14px; border: 0; border-radius: 14px; background: var(--accent); color: white; font: inherit; font-weight: 800; cursor: pointer; }
    .error { margin: 0 0 16px; padding: 10px 12px; border-radius: 14px; background: rgba(167,53,34,.09); color: var(--danger); border: 1px solid rgba(167,53,34,.18); }
    @media (max-width: 720px) { body { padding: 12px; align-items: start; padding-top: 12vh; } main { border-radius: 20px; padding: 22px; } }
  </style>
</head>
<body>
  <main>
    <p class="eyebrow">Private dashboard</p>
    <h1>Siqueira Memo Admin</h1>
    <p>Enter the admin password to open the memory dashboard.</p>
    {error}
    <form method="post" action="/admin/login" autocomplete="on">
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required autofocus>
      <button type="submit">Log in</button>
    </form>
  </main>
</body>
</html>"""


def _login_html(error: str = "") -> str:
    if not error:
        return _LOGIN_HTML.replace("{error}", "")
    return _LOGIN_HTML.replace("{error}", f'<p class="error">{escape(error)}</p>')


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
      --surface-soft: #f3f1ed;
      --text: #22211f;
      --muted: #68635d;
      --faint: #9b958d;
      --line: rgba(34, 33, 31, 0.12);
      --accent: #1d6fd8;
      --accent-soft: #eef6ff;
      --good: #16775f;
      --warn: #a45f00;
      --danger: #b42318;
      --shadow: rgba(0,0,0,0.035) 0 10px 30px, rgba(0,0,0,0.025) 0 2px 8px;
      --radius: 18px;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }
    * { box-sizing: border-box; }
    html { background: var(--bg); color: var(--text); font-family: var(--sans); }
    body { margin: 0; min-height: 100vh; }
    button, input, select, textarea { font: inherit; }
    button { cursor: pointer; }
    .shell { max-width: 1220px; margin: 0 auto; padding: 16px 18px 32px; }
    .topbar { display: flex; justify-content: space-between; gap: 14px; align-items: center; margin-bottom: 14px; }
    .brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
    .logo { width: 38px; height: 38px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); display: grid; place-items: center; box-shadow: var(--shadow); font-weight: 800; letter-spacing: -0.05em; }
    h1 { margin: 0; font-size: clamp(26px, 5vw, 42px); line-height: .98; letter-spacing: -0.055em; }
    .subtitle { margin: 3px 0 0; color: var(--muted); font-size: 14px; line-height: 1.35; }
    .layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 14px; align-items: start; }
    .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
    .pad { padding: 14px; }
    .sidebar { position: sticky; top: 14px; }
    h2 { margin: 0; font-size: 18px; line-height: 1.15; letter-spacing: -0.02em; }
    h3 { margin: 0; font-size: 15px; line-height: 1.25; }
    label { display: block; margin: 12px 0 6px; color: var(--muted); font-size: 11px; font-weight: 760; letter-spacing: .05em; text-transform: uppercase; }
    input, select, textarea { width: 100%; min-height: 42px; border: 1px solid var(--line); border-radius: 12px; background: #fff; color: var(--text); padding: 10px 11px; outline: none; }
    textarea { min-height: 88px; resize: vertical; line-height: 1.42; }
    input:focus, select:focus, textarea:focus { border-color: rgba(29,111,216,.55); box-shadow: 0 0 0 4px rgba(29,111,216,.12); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .three { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .btn { border: 1px solid var(--line); border-radius: 12px; min-height: 38px; padding: 9px 12px; background: var(--surface); color: var(--text); font-weight: 720; }
    .btn.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .btn.danger { color: var(--danger); }
    .btn.ghost { background: transparent; }
    .btn.small { min-height: 32px; padding: 7px 9px; font-size: 12px; }
    .badge { display: inline-flex; align-items: center; gap: 5px; border-radius: 999px; padding: 4px 8px; font-size: 11px; font-weight: 760; background: var(--accent-soft); color: var(--accent); border: 1px solid rgba(29,111,216,.12); white-space: nowrap; }
    .muted { color: var(--muted); }
    .small { font-size: 12px; color: var(--muted); }
    .mono { font-family: var(--mono); }
    .tabs { display: flex; gap: 7px; padding: 8px; background: var(--surface-soft); border-bottom: 1px solid var(--line); overflow-x: auto; scrollbar-width: thin; }
    .tab { border: 1px solid transparent; border-radius: 999px; background: transparent; color: var(--muted); padding: 8px 11px; font-weight: 760; white-space: nowrap; }
    .tab.active { background: #fff; color: var(--text); border-color: var(--line); box-shadow: rgba(0,0,0,.025) 0 2px 8px; }
    .panel { display: none; padding: 14px; }
    .panel.active { display: block; }
    .overview { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .project-card, .item { border: 1px solid var(--line); border-radius: 15px; background: #fff; padding: 12px; }
    .project-card { cursor: pointer; transition: transform .12s ease, border-color .12s ease; }
    .project-card:hover { transform: translateY(-1px); border-color: rgba(29,111,216,.35); }
    .statline { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
    .stat { border-radius: 999px; background: #f6f5f2; padding: 4px 7px; font-size: 11px; color: var(--muted); }
    .results { display: grid; gap: 9px; margin-top: 12px; }
    .item-head { display: flex; justify-content: space-between; gap: 8px; align-items: flex-start; margin-bottom: 8px; }
    .preview { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.42; }
    .meta { color: var(--faint); font-size: 11px; line-height: 1.35; overflow-wrap: anywhere; }
    .empty { color: var(--muted); text-align: center; border: 1px dashed var(--line); border-radius: 15px; padding: 28px 14px; background: rgba(255,255,255,.62); }
    .status { min-height: 20px; margin-top: 9px; color: var(--muted); font-size: 12px; }
    .top-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .top-actions form { margin: 0; }
    .drawer-backdrop { position: fixed; inset: 0; background: rgba(34,33,31,.18); display: none; z-index: 30; }
    .drawer-backdrop.open { display: block; }
    .drawer { position: fixed; right: 0; top: 0; height: 100vh; width: min(620px, 100%); background: var(--surface); border-left: 1px solid var(--line); box-shadow: rgba(0,0,0,.12) -18px 0 50px; z-index: 40; transform: translateX(105%); transition: transform .16s ease; overflow: auto; }
    .drawer.open { transform: translateX(0); }
    .drawer-header { position: sticky; top: 0; background: rgba(255,255,255,.96); border-bottom: 1px solid var(--line); padding: 12px 14px; display: flex; justify-content: space-between; gap: 10px; align-items: center; }
    .drawer-body { padding: 14px; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f5f2; border: 1px solid var(--line); border-radius: 12px; padding: 10px; font-size: 12px; line-height: 1.45; }
    .bottom-nav { display: none; }
    @media (max-width: 720px) {
      .shell { padding: 10px 10px calc(78px + env(safe-area-inset-bottom)); }
      .topbar { align-items: flex-start; }
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .row, .three, .overview { grid-template-columns: 1fr; }
      .panel { padding: 12px; }
      .tabs { display: none; }
      .subtitle { display: none; }
      .logo { width: 34px; height: 34px; }
      .actions .btn { flex: 1 1 auto; }
      .bottom-nav { display: grid; grid-template-columns: repeat(5, 1fr); position: fixed; left: 8px; right: 8px; bottom: calc(8px + env(safe-area-inset-bottom)); z-index: 20; background: rgba(255,255,255,.96); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); padding: 6px; gap: 4px; }
      .bottom-nav button { border: 0; background: transparent; border-radius: 12px; min-height: 42px; font-size: 11px; font-weight: 800; color: var(--muted); }
      .bottom-nav button.active { background: var(--accent-soft); color: var(--accent); }
      .drawer { width: 100%; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><div class="logo">S</div><div><h1>Siqueira Memo</h1><p class="subtitle">Project overview, memory search, recall playground, corrections, conflicts, audit, and export.</p></div></div>
      <div class="top-actions"><span class="badge">Light admin</span><button class="btn small" id="check-ready" type="button">Check ready</button><form method="post" action="/admin/logout"><button class="btn small" type="submit">Log out</button></form></div>
    </header>

    <section class="layout">
      <aside class="card pad sidebar">
        <h2>Scope</h2>
        <label for="token">API token</label>
        <input id="token" type="password" autocomplete="off" placeholder="Optional behind public proxy">
        <label for="profile">Profile</label>
        <input id="profile" value="default">
        <label for="project">Project</label>
        <input id="project" value="siqueira-memo" placeholder="project id">
        <div class="actions">
          <button class="btn small" id="load-siqueira" type="button">Siqueira project</button>
          <button class="btn small" id="load-tax" type="button">Brazil tax</button>
        </div>
        <label for="topic">Topic</label>
        <input id="topic" placeholder="optional">
        <div class="actions"><button class="btn primary" id="save-token">Save scope</button><button class="btn" id="refresh-all">Refresh</button></div>
        <div id="connection-status" class="status"></div>
      </aside>

      <section class="card">
        <nav class="tabs" aria-label="Admin sections">
          <button class="tab active" data-tab="dashboard">Dashboard</button>
          <button class="tab" data-tab="search">Search</button>
          <button class="tab" data-tab="recall">Recall</button>
          <button class="tab" data-tab="write">Write</button>
          <button class="tab" data-tab="ops">Ops</button>
        </nav>

        <section id="dashboard" class="panel active">
          <div class="item-head"><div><h2>Project overview</h2><p class="small">Counts by project and topic. Tap a project to inspect it.</p></div><button class="btn small primary" id="load-projects">Reload</button></div>
          <div id="projects-status" class="status"></div>
          <div id="project-overview" class="overview"><div class="empty">Loading projects…</div></div>
          <div class="row" style="margin-top:12px">
            <div><h2>Timeline</h2><div id="timeline-status" class="status"></div><div id="timeline-results" class="results"><div class="empty">Timeline loads automatically.</div></div></div>
            <div><h2>Recent search</h2><div id="search-status" class="status"></div><div id="search-results" class="results"><div class="empty">Search loads automatically.</div></div></div>
          </div>
        </section>

        <section id="search" class="panel">
          <div class="item-head"><div><h2>Search</h2><p class="small">Facts, decisions, messages, summaries. Click any card for Detail drawer.</p></div></div>
          <div class="row"><div><label for="target-type">Type</label><select id="target-type"><option value="fact">Facts</option><option value="decision">Decisions</option><option value="message">Messages</option><option value="summary">Summaries</option></select></div><div><label for="status-filter">Status</label><select id="status-filter"><option value="">Any</option><option value="active">Active</option><option value="superseded">Superseded</option><option value="invalidated">Invalidated</option></select></div></div>
          <label for="query">Query</label><div class="row"><input id="query" placeholder="Search text"><button class="btn primary" id="run-search">Search</button></div>
        </section>

        <section id="recall" class="panel">
          <h2>Recall playground</h2><p class="small">Ask Siqueira like the agent does. Shows answer_context, facts, decisions, conflicts.</p>
          <div class="row"><div><label for="recall-mode">Mode</label><select id="recall-mode"><option value="balanced">balanced</option><option value="deep">deep</option><option value="forensic">forensic</option></select></div><div><label for="recall-limit">Limit</label><input id="recall-limit" type="number" min="1" max="80" value="20"></div></div>
          <label for="recall-query">Question</label><textarea id="recall-query" placeholder="What do we know about Siqueira Memo?"></textarea>
          <div class="actions"><button class="btn primary" id="run-recall">Run recall</button></div>
          <div id="recall-status" class="status"></div><div id="recall-results" class="results"><div class="empty">Run recall to inspect context packs.</div></div>
        </section>

        <section id="write" class="panel">
          <div class="row">
            <div><h2>Remember</h2><div class="row"><div><label for="remember-kind">Kind</label><select id="remember-kind"><option value="fact">Fact</option><option value="decision">Decision</option></select></div><div><label for="remember-confidence">Confidence</label><input id="remember-confidence" type="number" min="0" max="1" step="0.05" value="0.9"></div></div><div id="fact-fields"><label for="fact-subject">Subject</label><input id="fact-subject"><label for="fact-predicate">Predicate</label><input id="fact-predicate"><label for="fact-object">Object</label><input id="fact-object"></div><div id="decision-fields" style="display:none"><label for="decision-topic">Decision topic</label><input id="decision-topic"><label for="decision-rationale">Rationale</label><input id="decision-rationale"></div><label for="remember-statement">Statement</label><textarea id="remember-statement"></textarea><button class="btn primary" id="save-memory">Save memory</button><div id="remember-status" class="status"></div></div>
            <div><h2>Correct</h2><p class="small">Detail drawer can prefill this. Use replacement to supersede old memory.</p><label for="correct-type">Type</label><select id="correct-type"><option value="fact">Fact</option><option value="decision">Decision</option></select><label for="correct-id">Target ID</label><input id="correct-id" class="mono"><label for="correction-text">Correction text</label><textarea id="correction-text"></textarea><label for="replacement-statement">Replacement statement</label><textarea id="replacement-statement"></textarea><button class="btn primary" id="submit-correction">Submit correction</button><div id="correct-status" class="status"></div></div>
          </div>
        </section>

        <section id="ops" class="panel">
          <div class="three">
            <div><h2>Conflicts</h2><div class="actions"><button class="btn primary" id="scan-conflicts">Scan</button><button class="btn" id="list-conflicts">List</button></div><div id="conflicts-status" class="status"></div><div id="conflicts-results" class="results"></div></div>
            <div><h2>Audit</h2><button class="btn primary" id="load-audit">Load audit</button><div id="audit-status" class="status"></div><div id="audit-results" class="results"></div></div>
            <div><h2>Export Markdown</h2><p class="small">Download current project/topic facts and decisions.</p><button class="btn primary" id="export-markdown">Export Markdown</button><div id="export-status" class="status"></div><h2 style="margin-top:18px">Sources</h2><label for="source-type">Type</label><select id="source-type"><option value="fact">Fact</option><option value="decision">Decision</option><option value="summary">Summary</option></select><label for="source-id">ID</label><input id="source-id" class="mono"><div class="actions"><button class="btn" id="load-sources">Load</button><button class="btn danger" id="soft-delete">Soft delete</button></div><div id="sources-status" class="status"></div><div id="sources-results" class="results"></div></div>
          </div>
        </section>
      </section>
    </section>
  </main>

  <div id="drawer-backdrop" class="drawer-backdrop"></div>
  <aside id="detail-drawer" class="drawer" aria-label="Detail drawer"><div class="drawer-header"><div><h2>Detail drawer</h2><div id="drawer-subtitle" class="small"></div></div><button class="btn small" id="close-drawer">Close</button></div><div id="drawer-body" class="drawer-body"><div class="empty">Select a memory card.</div></div></aside>
  <nav class="bottom-nav"><button class="active" data-tab="dashboard">Home</button><button data-tab="search">Search</button><button data-tab="recall">Recall</button><button data-tab="write">Write</button><button data-tab="ops">Ops</button></nav>

  <script>
    const $ = (id) => document.getElementById(id);
    const params = new URLSearchParams(window.location.search);
    const state = { token: localStorage.getItem('siqueira.apiToken') || '', profile: params.get('profile') || localStorage.getItem('siqueira.profile') || 'default', project: params.get('project') || localStorage.getItem('siqueira.project') || 'siqueira-memo' };
    $('token').value = state.token; $('profile').value = state.profile; $('project').value = state.project;
    const escapeHtml = (v) => String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
    const setStatus = (id, text, tone='muted') => { const el=$(id); if (!el) return; el.textContent=text||''; el.style.color = tone === 'danger' ? 'var(--danger)' : tone === 'good' ? 'var(--good)' : 'var(--muted)'; };
    function authHeaders() { const headers = { 'Content-Type': 'application/json', 'X-Profile-Id': $('profile').value.trim() || 'default' }; const token = $('token').value.trim(); if (token) headers.Authorization = 'Bearer ' + token; return headers; }
    function filters() { return { profile_id: $('profile').value.trim() || 'default', project: $('project').value.trim() || null, topic: $('topic').value.trim() || null }; }
    async function post(path, body) { const res = await fetch(path, { method:'POST', headers:authHeaders(), body:JSON.stringify(body || {}) }); const text = await res.text(); let data; try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; } if (!res.ok) throw new Error(data.detail || res.statusText || 'Request failed'); return data; }
    function activateTab(name) { document.querySelectorAll('.tab,.panel,.bottom-nav button').forEach(el => el.classList.remove('active')); document.querySelectorAll(`[data-tab="${name}"]`).forEach(el => el.classList.add('active')); $(name).classList.add('active'); }
    document.querySelectorAll('[data-tab]').forEach(tab => tab.addEventListener('click', () => activateTab(tab.dataset.tab)));
    function cardHtml(hit) { const type = hit.target_type || hit.kind; const title = hit.topic || hit.project || type; return `<article class="item" data-type="${escapeHtml(type)}" data-id="${escapeHtml(hit.id)}"><div class="item-head"><div><span class="badge">${escapeHtml(type)}</span> <strong>${escapeHtml(title)}</strong></div><span class="meta mono">${escapeHtml(hit.id)}</span></div><div class="preview">${escapeHtml(hit.preview || hit.statement || hit.decision || hit.title || '')}</div><div class="statline">${hit.project ? `<span class="stat">${escapeHtml(hit.project)}</span>` : ''}${hit.topic ? `<span class="stat">${escapeHtml(hit.topic)}</span>` : ''}${hit.status ? `<span class="stat">${escapeHtml(hit.status)}</span>` : ''}</div></article>`; }
    function bindDetailClicks(rootId) { $(rootId).querySelectorAll('.item[data-id]').forEach(el => el.addEventListener('click', () => openDetail(el.dataset.type, el.dataset.id))); }
    function renderHits(target, hits) { $(target).innerHTML = hits.length ? hits.map(cardHtml).join('') : '<div class="empty">Nothing found.</div>'; bindDetailClicks(target); }
    async function loadProjects() { setStatus('projects-status','Loading…'); try { const body = await post('/v1/admin/projects', { profile_id: $('profile').value.trim() || 'default' }); setStatus('projects-status', `${body.projects.length} project(s).`, 'good'); $('project-overview').innerHTML = body.projects.length ? body.projects.map(p => `<article class="project-card" data-project="${escapeHtml(p.project)}"><div class="item-head"><h3>${escapeHtml(p.project)}</h3><span class="badge">${p.total}</span></div><div class="statline"><span class="stat">${p.facts} facts</span><span class="stat">${p.decisions} decisions</span><span class="stat">${p.messages} messages</span><span class="stat">${p.summaries} summaries</span></div><div class="small" style="margin-top:8px">${(p.topics||[]).slice(0,4).map(t => escapeHtml(t.topic)).join(' · ') || 'No topics yet'}</div></article>`).join('') : '<div class="empty">No projects yet.</div>'; document.querySelectorAll('.project-card').forEach(el => el.addEventListener('click', () => loadDefaultProject(el.dataset.project))); } catch (err) { setStatus('projects-status', err.message, 'danger'); } }
    async function runSearch() { setStatus('search-status','Searching…'); try { const body = await post('/v1/admin/search', { ...filters(), query: $('query').value.trim() || null, target_type: $('target-type').value, status: $('status-filter').value || null, limit: 50 }); setStatus('search-status', `${body.total} result(s) for ${$('project').value.trim() || 'all projects'}.`, 'good'); renderHits('search-results', body.hits || []); } catch (err) { setStatus('search-status', err.message, 'danger'); } }
    async function loadTimeline() { setStatus('timeline-status','Loading…'); try { const body = await post('/v1/memory/timeline', { ...filters(), limit: 50 }); setStatus('timeline-status', `${body.entries.length} timeline item(s).`, 'good'); renderHits('timeline-results', (body.entries||[]).map(e => ({ ...e, target_type:e.kind, preview:e.preview }))); } catch (err) { setStatus('timeline-status', err.message, 'danger'); } }
    function loadDefaultProject(project) { $('project').value = project; $('topic').value = ''; $('query').value = ''; $('target-type').value = 'fact'; localStorage.setItem('siqueira.project', project); runSearch(); loadTimeline(); }
    async function openDetail(type, id) { $('detail-drawer').classList.add('open'); $('drawer-backdrop').classList.add('open'); $('drawer-subtitle').textContent = `${type} ${id}`; $('drawer-body').innerHTML = '<div class="empty">Loading detail…</div>'; $('source-type').value = type === 'decision' ? 'decision' : 'fact'; $('source-id').value = id; $('correct-type').value = type === 'decision' ? 'decision' : 'fact'; $('correct-id').value = id; try { const body = await post('/v1/admin/detail', { target_type:type, target_id:id, profile_id:$('profile').value.trim() || 'default' }); const item = body.item || {}; $('drawer-body').innerHTML = `<h2>${escapeHtml(item.topic || item.subject || item.target_type)}</h2><pre>${escapeHtml(JSON.stringify(item, null, 2))}</pre><h2>Sources</h2><pre>${escapeHtml(JSON.stringify(body.sources || [], null, 2))}</pre><div class="actions"><button class="btn primary" id="prefill-correction">Correct this</button><button class="btn danger" id="drawer-soft-delete">Soft delete</button></div>`; $('prefill-correction').onclick = () => { activateTab('write'); $('correction-text').focus(); }; $('drawer-soft-delete').onclick = () => softDelete(type, id); } catch (err) { $('drawer-body').innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`; } }
    function closeDrawer() { $('detail-drawer').classList.remove('open'); $('drawer-backdrop').classList.remove('open'); }
    async function runRecall() { setStatus('recall-status','Recalling…'); try { const body = await post('/v1/recall', { ...filters(), query:$('recall-query').value.trim() || `What do we know about ${$('project').value.trim() || 'this project'}?`, mode:$('recall-mode').value, limit:Number($('recall-limit').value || 20) }); const pack = body.context_pack || {}; setStatus('recall-status', `Returned ${(pack.facts||[]).length} facts and ${(pack.decisions||[]).length} decisions.`, 'good'); $('recall-results').innerHTML = `<article class="item"><h3>Answer context</h3><div class="preview">${escapeHtml(pack.answer_context || '')}</div></article><article class="item"><h3>Raw pack</h3><pre>${escapeHtml(JSON.stringify(pack, null, 2))}</pre></article>`; } catch (err) { setStatus('recall-status', err.message, 'danger'); } }
    async function saveMemory() { setStatus('remember-status','Saving…'); try { const kind = $('remember-kind').value; const payload = { ...filters(), kind, statement:$('remember-statement').value.trim(), confidence:Number($('remember-confidence').value || 0.9) }; if (kind === 'fact') Object.assign(payload, { subject:$('fact-subject').value.trim(), predicate:$('fact-predicate').value.trim(), object:$('fact-object').value.trim() }); else Object.assign(payload, { topic:$('decision-topic').value.trim() || $('topic').value.trim() || 'manual', rationale:$('decision-rationale').value.trim() }); const body = await post('/v1/memory/remember', payload); setStatus('remember-status', `Saved ${body.kind}: ${body.id}`, 'good'); loadProjects(); runSearch(); } catch (err) { setStatus('remember-status', err.message, 'danger'); } }
    async function submitCorrection() { setStatus('correct-status','Correcting…'); try { const targetType = $('correct-type').value; const replacement = { kind: targetType, statement:$('replacement-statement').value.trim(), project:$('project').value.trim() || null, topic:$('topic').value.trim() || 'manual' }; if (targetType === 'fact') Object.assign(replacement, { subject:'corrected memory', predicate:'states', object:$('replacement-statement').value.trim() }); const body = await post('/v1/memory/correct', { target_type:targetType, target_id:$('correct-id').value.trim(), correction_text:$('correction-text').value.trim(), replacement }); setStatus('correct-status', `Corrected. Replacement: ${body.replacement_id || 'none'}`, 'good'); runSearch(); } catch (err) { setStatus('correct-status', err.message, 'danger'); } }
    async function loadSources() { setStatus('sources-status','Loading…'); try { const body = await post('/v1/memory/sources', { profile_id:$('profile').value.trim() || 'default', target_type:$('source-type').value, target_id:$('source-id').value.trim() }); setStatus('sources-status', `${body.sources.length} source(s).`, 'good'); $('sources-results').innerHTML = body.sources.length ? body.sources.map(s => `<article class="item"><pre>${escapeHtml(JSON.stringify(s, null, 2))}</pre></article>`).join('') : '<div class="empty">No sources.</div>'; } catch (err) { setStatus('sources-status', err.message, 'danger'); } }
    async function softDelete(type, id) { if (!confirm('Soft-delete this memory?')) return; setStatus('sources-status','Deleting…'); try { const body = await post('/v1/memory/forget', { target_type:type || $('source-type').value, target_id:id || $('source-id').value.trim(), mode:'soft', reason:'admin-ui' }); setStatus('sources-status', `Soft-deleted. Event ${body.event_id}`, 'good'); closeDrawer(); runSearch(); loadTimeline(); } catch (err) { setStatus('sources-status', err.message, 'danger'); } }
    async function loadConflicts(path) { setStatus('conflicts-status','Loading…'); try { const body = await post(path, {}); setStatus('conflicts-status', `${body.detected} conflict(s).`, 'good'); $('conflicts-results').innerHTML = (body.conflicts||[]).length ? body.conflicts.map(c => `<article class="item"><pre>${escapeHtml(JSON.stringify(c, null, 2))}</pre></article>`).join('') : '<div class="empty">No conflicts.</div>'; } catch (err) { setStatus('conflicts-status', err.message, 'danger'); } }
    async function loadAudit() { setStatus('audit-status','Loading…'); try { const body = await post('/v1/admin/audit', { profile_id:$('profile').value.trim() || 'default', limit:50 }); setStatus('audit-status', `${body.entries.length} audit event(s).`, 'good'); $('audit-results').innerHTML = body.entries.length ? body.entries.map(e => `<article class="item"><pre>${escapeHtml(JSON.stringify(e, null, 2))}</pre></article>`).join('') : '<div class="empty">No audit events.</div>'; } catch (err) { setStatus('audit-status', err.message, 'danger'); } }
    async function exportMarkdown() { setStatus('export-status','Exporting…'); try { const res = await fetch('/v1/admin/export', { method:'POST', headers:authHeaders(), body:JSON.stringify({ ...filters(), format:'markdown' }) }); if (!res.ok) throw new Error(await res.text()); const blob = await res.blob(); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `siqueira-${$('project').value.trim() || 'memory'}.md`; a.click(); URL.revokeObjectURL(url); setStatus('export-status','Downloaded markdown.', 'good'); } catch (err) { setStatus('export-status', err.message, 'danger'); } }
    async function refreshAll() { await loadProjects(); await runSearch(); await loadTimeline(); }
    $('save-token').onclick = () => { localStorage.setItem('siqueira.apiToken', $('token').value.trim()); localStorage.setItem('siqueira.profile', $('profile').value.trim() || 'default'); localStorage.setItem('siqueira.project', $('project').value.trim() || ''); setStatus('connection-status','Saved locally.','good'); };
    $('check-ready').onclick = async () => { try { const res = await fetch('/readyz'); const body = await res.json(); setStatus('connection-status', body.ok ? 'API ready. Database reachable.' : 'API responded but is not ready.', body.ok ? 'good' : 'danger'); } catch (err) { setStatus('connection-status', err.message, 'danger'); } };
    $('refresh-all').onclick = refreshAll; $('load-projects').onclick = loadProjects; $('run-search').onclick = runSearch; $('load-siqueira').onclick = () => loadDefaultProject('siqueira-memo'); $('load-tax').onclick = () => loadDefaultProject('brazil-tax-crypto'); $('run-recall').onclick = runRecall; $('save-memory').onclick = saveMemory; $('submit-correction').onclick = submitCorrection; $('load-sources').onclick = loadSources; $('soft-delete').onclick = () => softDelete(); $('scan-conflicts').onclick = () => loadConflicts('/v1/admin/conflicts/scan'); $('list-conflicts').onclick = () => loadConflicts('/v1/admin/conflicts/list'); $('load-audit').onclick = loadAudit; $('export-markdown').onclick = exportMarkdown; $('close-drawer').onclick = closeDrawer; $('drawer-backdrop').onclick = closeDrawer;
    $('remember-kind').onchange = () => { const fact = $('remember-kind').value === 'fact'; $('fact-fields').style.display = fact ? '' : 'none'; $('decision-fields').style.display = fact ? 'none' : ''; };
    $('query').addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
    window.addEventListener('load', refreshAll);
  </script>
</body>
</html>
"""


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _redirect_to_login() -> RedirectResponse:
    return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)


def _serve_admin_or_redirect(request: Request) -> Response:
    settings = _settings(request)
    if admin_auth_enabled(settings) and not request_has_admin_session(request):
        return _redirect_to_login()
    return HTMLResponse(_ADMIN_HTML)


@router.get("/admin/login", response_class=HTMLResponse, include_in_schema=False)
async def admin_login(request: Request) -> Response:
    """Serve the app-level admin login page instead of a browser Basic Auth prompt."""
    settings = _settings(request)
    if not admin_auth_enabled(settings) or request_has_admin_session(request):
        return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
    return HTMLResponse(_login_html())


@router.post("/admin/login", include_in_schema=False)
async def admin_login_submit(request: Request) -> Response:
    """Validate the admin password and set a signed HttpOnly session cookie."""
    settings = _settings(request)
    form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    password = form.get("password", [""])[0]
    if not verify_admin_password(settings, password):
        return HTMLResponse(_login_html("Invalid password"), status_code=status.HTTP_401_UNAUTHORIZED)
    response = RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        ADMIN_SESSION_COOKIE,
        create_admin_session_token(settings),
        max_age=settings.admin_session_ttl_seconds,
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.admin_cookie_secure,
    )
    return response


@router.post("/admin/logout", include_in_schema=False)
async def admin_logout() -> RedirectResponse:
    """Clear the browser admin session cookie."""
    response = RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(ADMIN_SESSION_COOKIE, path="/")
    return response


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_ui(request: Request) -> Response:
    """Serve the zero-build browser admin interface."""
    return _serve_admin_or_redirect(request)


@router.get("/admin/", response_class=HTMLResponse, include_in_schema=False)
async def admin_ui_slash(request: Request) -> Response:
    """Serve the same UI with a trailing slash."""
    return _serve_admin_or_redirect(request)

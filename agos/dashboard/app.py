"""Dashboard — FastAPI + WebSocket real-time agent monitoring.

`agos dashboard` launches this server at localhost:8420.
Provides REST endpoints and a WebSocket for live event streaming.
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from agos.events.bus import Event

dashboard_app = FastAPI(title="agos dashboard", version="0.1.0")

# These get injected at startup by the CLI command
_runtime = None
_event_bus = None
_audit_trail = None
_policy_engine = None
_tracer = None
_loom = None


def configure(
    runtime=None,
    event_bus=None,
    audit_trail=None,
    policy_engine=None,
    tracer=None,
    loom=None,
) -> None:
    """Inject live subsystem references into the dashboard."""
    global _runtime, _event_bus, _audit_trail, _policy_engine, _tracer, _loom
    _runtime = runtime
    _event_bus = event_bus
    _audit_trail = audit_trail
    _policy_engine = policy_engine
    _tracer = tracer
    _loom = loom


# ── REST Endpoints ─────────────────────────────────────────────────


@dashboard_app.get("/")
async def index() -> HTMLResponse:
    """Dashboard landing page."""
    return HTMLResponse(_DASHBOARD_HTML)


@dashboard_app.get("/api/agents")
async def list_agents() -> list[dict]:
    if _runtime is None:
        return []
    return _runtime.list_agents()


@dashboard_app.get("/api/events")
async def list_events(topic: str = "*", limit: int = 50) -> list[dict]:
    if _event_bus is None:
        return []
    events = _event_bus.history(topic_filter=topic, limit=limit)
    return [e.model_dump(mode="json") for e in events]


@dashboard_app.get("/api/audit")
async def list_audit(agent_id: str = "", action: str = "", limit: int = 50) -> list[dict]:
    if _audit_trail is None:
        return []
    entries = await _audit_trail.query(agent_id=agent_id, action=action, limit=limit)
    return [e.model_dump(mode="json") for e in entries]


@dashboard_app.get("/api/audit/violations")
async def list_violations(limit: int = 20) -> list[dict]:
    if _audit_trail is None:
        return []
    entries = await _audit_trail.violations(limit=limit)
    return [e.model_dump(mode="json") for e in entries]


@dashboard_app.get("/api/policies")
async def list_policies() -> list[dict]:
    if _policy_engine is None:
        return []
    return _policy_engine.list_policies()


@dashboard_app.get("/api/traces")
async def list_traces(limit: int = 20) -> list[dict]:
    if _tracer is None:
        return []
    traces = _tracer.list_traces(limit=limit)
    return [t.model_dump(mode="json") for t in traces]


@dashboard_app.get("/api/status")
async def system_status() -> dict:
    agents = _runtime.list_agents() if _runtime else []
    return {
        "version": "0.1.0",
        "agents_total": len(agents),
        "agents_running": sum(1 for a in agents if a["state"] == "running"),
        "agents_completed": sum(1 for a in agents if a["state"] == "completed"),
        "event_subscribers": _event_bus.subscriber_count if _event_bus else 0,
        "ws_connections": _event_bus.ws_connection_count if _event_bus else 0,
        "audit_entries": await _audit_trail.count() if _audit_trail else 0,
        "policies": len(_policy_engine.list_policies()) if _policy_engine else 0,
        "active_spans": _tracer.active_span_count if _tracer else 0,
        "knowledge_available": _loom is not None,
    }


@dashboard_app.get("/api/knowledge/recall")
async def knowledge_recall(q: str = "", limit: int = 10) -> list[dict]:
    if _loom is None or not q:
        return []
    threads = await _loom.recall(q, limit=limit)
    return [
        {
            "id": t.id,
            "kind": t.kind,
            "content": t.content[:500],
            "tags": t.tags,
            "confidence": t.confidence,
            "created_at": t.created_at.isoformat(),
        }
        for t in threads
    ]


@dashboard_app.get("/api/knowledge/timeline")
async def knowledge_timeline(limit: int = 20) -> list[dict]:
    if _loom is None:
        return []
    threads = await _loom.timeline(limit=limit)
    return [
        {
            "id": t.id,
            "kind": t.kind,
            "content": t.content[:300],
            "created_at": t.created_at.isoformat(),
        }
        for t in threads
    ]


@dashboard_app.get("/api/knowledge/graph")
async def knowledge_graph(entity: str = "", limit: int = 20) -> dict:
    if _loom is None:
        return {"entities": [], "edges": []}
    graph = _loom.graph
    if entity:
        edges = await graph.connections(entity)
        return {
            "entity": entity,
            "edges": [
                {"source": e.source, "relation": e.relation, "target": e.target}
                for e in edges[:limit]
            ],
        }
    return {"entities": [], "edges": []}


@dashboard_app.post("/api/knowledge/remember")
async def knowledge_remember(body: dict) -> dict:
    if _loom is None:
        return {"ok": False, "error": "Knowledge system not initialized"}
    content = body.get("content", "")
    if not content:
        return {"ok": False, "error": "No content provided"}
    thread = await _loom.remember(content)
    return {"ok": True, "id": thread.id}


# ── WebSocket — live event stream ─────────────────────────────────


@dashboard_app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()

    async def send_event(event: Event) -> None:
        try:
            await websocket.send_json(event.model_dump(mode="json"))
        except Exception:
            pass

    if _event_bus:
        _event_bus.add_ws_connection(send_event)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if _event_bus:
            _event_bus.remove_ws_connection(send_event)


# ── Full Dashboard HTML ──────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>agos — Dashboard</title>
<style>
:root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --border: #30363d; --text: #c9d1d9; --text2: #8b949e;
    --blue: #58a6ff; --green: #3fb950; --yellow: #d29922;
    --red: #f85149; --purple: #bc8cff; --cyan: #39d2c0;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }

/* ── Header ── */
header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; }
header h1 { font-size: 20px; font-weight: 600; }
header h1 span { color: var(--blue); }
.header-stats { display: flex; gap: 20px; font-size: 13px; color: var(--text2); }
.header-stats .stat { display: flex; align-items: center; gap: 6px; }
.header-stats .dot { width: 8px; height: 8px; border-radius: 50%; }
.dot-green { background: var(--green); }
.dot-blue { background: var(--blue); }
.dot-yellow { background: var(--yellow); }

/* ── Nav ── */
nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 0 24px; display: flex; gap: 0; }
nav button { background: none; border: none; color: var(--text2); padding: 10px 16px; font-size: 13px; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.15s; }
nav button:hover { color: var(--text); }
nav button.active { color: var(--blue); border-bottom-color: var(--blue); }

/* ── Main ── */
main { padding: 20px 24px; max-width: 1400px; margin: 0 auto; }
.page { display: none; }
.page.active { display: block; }

/* ── Cards ── */
.grid { display: grid; gap: 16px; }
.grid-4 { grid-template-columns: repeat(4, 1fr); }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.card h3 { font-size: 12px; text-transform: uppercase; color: var(--text2); margin-bottom: 8px; letter-spacing: 0.5px; }
.card .value { font-size: 28px; font-weight: 700; }
.card .sub { font-size: 12px; color: var(--text2); margin-top: 4px; }

/* ── Tables ── */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 12px; color: var(--text2); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
td { padding: 8px 12px; border-bottom: 1px solid var(--bg3); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
.badge-running { background: rgba(63,185,80,0.15); color: var(--green); }
.badge-completed { background: rgba(139,148,158,0.15); color: var(--text2); }
.badge-ready { background: rgba(88,166,255,0.15); color: var(--blue); }
.badge-error, .badge-terminated { background: rgba(248,81,73,0.15); color: var(--red); }
.badge-fact { background: rgba(57,210,192,0.15); color: var(--cyan); }
.badge-interaction { background: rgba(188,140,255,0.15); color: var(--purple); }
.empty { color: var(--text2); text-align: center; padding: 40px; font-size: 14px; }

/* ── Events feed ── */
.event-feed { max-height: 400px; overflow-y: auto; }
.event-item { padding: 8px 12px; border-bottom: 1px solid var(--bg3); font-size: 13px; display: flex; gap: 10px; align-items: baseline; }
.event-topic { color: var(--blue); font-weight: 500; min-width: 140px; }
.event-data { color: var(--text2); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.event-time { color: var(--text2); font-size: 11px; opacity: 0.6; }

/* ── Knowledge search ── */
.search-bar { display: flex; gap: 8px; margin-bottom: 16px; }
.search-bar input { flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 6px; font-size: 14px; outline: none; }
.search-bar input:focus { border-color: var(--blue); }
.search-bar button { background: var(--blue); color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; }
.search-bar button:hover { opacity: 0.9; }

.knowledge-item { background: var(--bg); border: 1px solid var(--bg3); border-radius: 6px; padding: 12px; margin-bottom: 8px; }
.knowledge-item .meta { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
.knowledge-item .content { font-size: 13px; line-height: 1.5; color: var(--text); }
.knowledge-item .tags { margin-top: 6px; }
.knowledge-item .tags span { display: inline-block; background: var(--bg3); color: var(--text2); padding: 1px 6px; border-radius: 3px; font-size: 11px; margin-right: 4px; }

/* ── Audit ── */
.audit-item { display: flex; gap: 12px; padding: 8px 12px; border-bottom: 1px solid var(--bg3); font-size: 13px; }
.audit-actor { color: var(--cyan); min-width: 100px; font-weight: 500; }
.audit-action { color: var(--yellow); min-width: 120px; }
.audit-time { color: var(--text2); font-size: 11px; min-width: 130px; }

/* ── Policy ── */
.policy-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.policy-item { background: var(--bg); border: 1px solid var(--bg3); border-radius: 6px; padding: 12px; }
.policy-item .label { font-size: 11px; text-transform: uppercase; color: var(--text2); margin-bottom: 4px; }
.policy-item .val { font-size: 16px; font-weight: 600; }
.val-ok { color: var(--green); }
.val-warn { color: var(--yellow); }
.val-deny { color: var(--red); }

@media (max-width: 900px) {
    .grid-4 { grid-template-columns: repeat(2, 1fr); }
    .grid-3, .grid-2 { grid-template-columns: 1fr; }
    .policy-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<header>
    <h1><span>agos</span> dashboard</h1>
    <div class="header-stats">
        <div class="stat"><div class="dot dot-green"></div><span id="h-running">0</span> running</div>
        <div class="stat"><div class="dot dot-blue"></div><span id="h-total">0</span> agents</div>
        <div class="stat"><div class="dot dot-yellow"></div><span id="h-events">0</span> events</div>
        <div class="stat" style="color:var(--text2);font-size:11px" id="h-version">v0.1.0</div>
    </div>
</header>

<nav>
    <button class="active" onclick="showPage('overview')">Overview</button>
    <button onclick="showPage('agents')">Agents</button>
    <button onclick="showPage('knowledge')">Knowledge</button>
    <button onclick="showPage('events')">Events</button>
    <button onclick="showPage('audit')">Audit</button>
    <button onclick="showPage('policy')">Policy</button>
</nav>

<main>

<!-- OVERVIEW -->
<div class="page active" id="page-overview">
    <div class="grid grid-4" style="margin-bottom:20px">
        <div class="card"><h3>Agents Running</h3><div class="value" style="color:var(--green)" id="s-running">0</div><div class="sub" id="s-total-sub">0 total</div></div>
        <div class="card"><h3>Events</h3><div class="value" style="color:var(--blue)" id="s-events">0</div><div class="sub">live stream</div></div>
        <div class="card"><h3>Audit Entries</h3><div class="value" style="color:var(--yellow)" id="s-audit">0</div><div class="sub">actions logged</div></div>
        <div class="card"><h3>Policies</h3><div class="value" style="color:var(--purple)" id="s-policies">0</div><div class="sub">active rules</div></div>
    </div>
    <div class="grid grid-2">
        <div class="card">
            <h3>Recent Agents</h3>
            <table><thead><tr><th>Name</th><th>Role</th><th>State</th><th>Tokens</th></tr></thead><tbody id="ov-agents"></tbody></table>
            <div class="empty" id="ov-agents-empty" style="display:none">No agents yet. Run <code>agos "do something"</code></div>
        </div>
        <div class="card">
            <h3>Live Events</h3>
            <div class="event-feed" id="ov-events"></div>
            <div class="empty" id="ov-events-empty">Waiting for events...</div>
        </div>
    </div>
</div>

<!-- AGENTS -->
<div class="page" id="page-agents">
    <div class="card">
        <h3>All Agents</h3>
        <table><thead><tr><th>ID</th><th>Name</th><th>Role</th><th>State</th><th>Tokens</th><th>Turns</th></tr></thead><tbody id="ag-table"></tbody></table>
        <div class="empty" id="ag-empty">No agents spawned yet</div>
    </div>
</div>

<!-- KNOWLEDGE -->
<div class="page" id="page-knowledge">
    <div class="card" style="margin-bottom:16px">
        <h3>Search Knowledge</h3>
        <div class="search-bar">
            <input type="text" id="k-search" placeholder="Search memories, facts, interactions..." onkeydown="if(event.key==='Enter')searchKnowledge()">
            <button onclick="searchKnowledge()">Search</button>
        </div>
        <div id="k-results"></div>
        <div class="empty" id="k-empty">Type a query to search the knowledge system</div>
    </div>
    <div class="card">
        <h3>Timeline</h3>
        <div id="k-timeline"></div>
    </div>
</div>

<!-- EVENTS -->
<div class="page" id="page-events">
    <div class="card">
        <h3>Event Stream</h3>
        <div class="event-feed" id="ev-feed" style="max-height:600px"></div>
        <div class="empty" id="ev-empty">Waiting for events...</div>
    </div>
</div>

<!-- AUDIT -->
<div class="page" id="page-audit">
    <div class="card">
        <h3>Audit Trail</h3>
        <div id="au-feed"></div>
        <div class="empty" id="au-empty">No audit entries yet</div>
    </div>
</div>

<!-- POLICY -->
<div class="page" id="page-policy">
    <div class="card">
        <h3>Active Policies</h3>
        <div class="policy-grid" id="po-grid"></div>
        <div class="empty" id="po-empty">No policies configured</div>
    </div>
</div>

</main>

<script>
// ── Navigation ──
function showPage(name) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.getElementById('page-' + name).classList.add('active');
    document.querySelectorAll('nav button').forEach(b => {
        if (b.textContent.toLowerCase() === name) b.classList.add('active');
    });
    if (name === 'knowledge') loadTimeline();
    if (name === 'audit') loadAudit();
    if (name === 'policy') loadPolicies();
}

// ── Data fetching ──
async function fetchJSON(url) {
    try { return await (await fetch(url)).json(); }
    catch { return null; }
}

// ── Overview + Agents refresh ──
async function refresh() {
    const status = await fetchJSON('/api/status');
    if (status) {
        document.getElementById('s-running').textContent = status.agents_running;
        document.getElementById('s-events').textContent = eventCount;
        document.getElementById('s-audit').textContent = status.audit_entries;
        document.getElementById('s-policies').textContent = status.policies;
        document.getElementById('s-total-sub').textContent = status.agents_total + ' total';
        document.getElementById('h-running').textContent = status.agents_running;
        document.getElementById('h-total').textContent = status.agents_total;
        document.getElementById('h-events').textContent = eventCount;
        document.getElementById('h-version').textContent = 'v' + status.version;
    }

    const agents = await fetchJSON('/api/agents') || [];
    // Overview table
    const ovBody = document.getElementById('ov-agents');
    const ovEmpty = document.getElementById('ov-agents-empty');
    if (agents.length) {
        ovEmpty.style.display = 'none';
        ovBody.innerHTML = agents.slice(0, 8).map(a => `
            <tr>
                <td>${esc(a.name)}</td>
                <td style="color:var(--text2)">${esc(a.role)}</td>
                <td><span class="badge badge-${a.state}">${a.state}</span></td>
                <td>${(a.tokens_used||0).toLocaleString()}</td>
            </tr>`).join('');
    } else {
        ovEmpty.style.display = '';
        ovBody.innerHTML = '';
    }

    // Full agents table
    const agBody = document.getElementById('ag-table');
    const agEmpty = document.getElementById('ag-empty');
    if (agents.length) {
        agEmpty.style.display = 'none';
        agBody.innerHTML = agents.map(a => `
            <tr>
                <td style="font-family:monospace;font-size:12px;color:var(--text2)">${a.id.slice(0,10)}</td>
                <td>${esc(a.name)}</td>
                <td style="color:var(--text2)">${esc(a.role)}</td>
                <td><span class="badge badge-${a.state}">${a.state}</span></td>
                <td>${(a.tokens_used||0).toLocaleString()}</td>
                <td>${a.turns||0}</td>
            </tr>`).join('');
    } else {
        agEmpty.style.display = '';
        agBody.innerHTML = '';
    }
}

// ── Knowledge ──
async function searchKnowledge() {
    const q = document.getElementById('k-search').value.trim();
    if (!q) return;
    const results = await fetchJSON('/api/knowledge/recall?q=' + encodeURIComponent(q) + '&limit=15');
    const div = document.getElementById('k-results');
    const empty = document.getElementById('k-empty');
    if (!results || !results.length) {
        div.innerHTML = '';
        empty.style.display = '';
        empty.textContent = 'No results for "' + q + '"';
        return;
    }
    empty.style.display = 'none';
    div.innerHTML = results.map(r => `
        <div class="knowledge-item">
            <div class="meta">
                <span class="badge badge-${r.kind}">${r.kind}</span>
                <span style="font-size:11px;color:var(--text2)">${r.created_at.slice(0,16).replace('T',' ')}</span>
                <span style="font-size:11px;color:var(--text2)">confidence: ${(r.confidence*100).toFixed(0)}%</span>
            </div>
            <div class="content">${esc(r.content)}</div>
            ${r.tags.length ? '<div class="tags">' + r.tags.map(t => '<span>'+esc(t)+'</span>').join('') + '</div>' : ''}
        </div>`).join('');
}

async function loadTimeline() {
    const items = await fetchJSON('/api/knowledge/timeline?limit=20');
    const div = document.getElementById('k-timeline');
    if (!items || !items.length) {
        div.innerHTML = '<div class="empty">No timeline events</div>';
        return;
    }
    div.innerHTML = items.map(t => `
        <div class="event-item">
            <span class="event-topic">${t.kind}</span>
            <span class="event-data">${esc(t.content.slice(0,120))}</span>
            <span class="event-time">${t.created_at.slice(0,16).replace('T',' ')}</span>
        </div>`).join('');
}

// ── Events (WebSocket) ──
let eventCount = 0;
function addEvent(event) {
    eventCount++;
    const feeds = ['ov-events', 'ev-feed'];
    feeds.forEach(id => {
        const div = document.getElementById(id);
        const empty = document.getElementById(id === 'ov-events' ? 'ov-events-empty' : 'ev-empty');
        if (empty) empty.style.display = 'none';
        const el = document.createElement('div');
        el.className = 'event-item';
        el.innerHTML = `
            <span class="event-topic">${esc(event.topic)}</span>
            <span class="event-data">${esc(JSON.stringify(event.data).slice(0,120))}</span>
            <span class="event-time">${(event.timestamp||'').slice(11,19)}</span>`;
        div.prepend(el);
        while (div.children.length > 100) div.lastChild.remove();
    });
}

try {
    const ws = new WebSocket('ws://' + location.host + '/ws/events');
    ws.onmessage = (e) => addEvent(JSON.parse(e.data));
} catch(e) {}

// Load historical events
(async () => {
    const events = await fetchJSON('/api/events?limit=30');
    if (events && events.length) {
        events.reverse().forEach(addEvent);
    }
})();

// ── Audit ──
async function loadAudit() {
    const entries = await fetchJSON('/api/audit?limit=50');
    const div = document.getElementById('au-feed');
    const empty = document.getElementById('au-empty');
    if (!entries || !entries.length) {
        div.innerHTML = '';
        empty.style.display = '';
        return;
    }
    empty.style.display = 'none';
    div.innerHTML = entries.map(e => `
        <div class="audit-item">
            <span class="audit-time">${(e.timestamp||'').slice(0,19).replace('T',' ')}</span>
            <span class="audit-actor">${esc(e.actor || e.agent_name || e.agent_id?.slice(0,10) || '?')}</span>
            <span class="audit-action">${esc(e.action)}</span>
            <span style="color:var(--text2)">${esc((e.detail||'').slice(0,80))}</span>
        </div>`).join('');
}

// ── Policy ──
async function loadPolicies() {
    const status = await fetchJSON('/api/status');
    const grid = document.getElementById('po-grid');
    const empty = document.getElementById('po-empty');
    if (!status) { empty.style.display = ''; return; }
    empty.style.display = 'none';
    const items = [
        { label: 'Version', val: status.version, cls: '' },
        { label: 'Agents Total', val: status.agents_total, cls: '' },
        { label: 'Agents Running', val: status.agents_running, cls: status.agents_running > 0 ? 'val-ok' : '' },
        { label: 'Event Subscribers', val: status.event_subscribers, cls: '' },
        { label: 'WebSocket Clients', val: status.ws_connections, cls: status.ws_connections > 0 ? 'val-ok' : '' },
        { label: 'Audit Entries', val: status.audit_entries, cls: '' },
        { label: 'Active Policies', val: status.policies, cls: '' },
        { label: 'Active Spans', val: status.active_spans, cls: '' },
        { label: 'Knowledge', val: status.knowledge_available ? 'Online' : 'Offline', cls: status.knowledge_available ? 'val-ok' : 'val-deny' },
    ];
    grid.innerHTML = items.map(i => `
        <div class="policy-item">
            <div class="label">${i.label}</div>
            <div class="val ${i.cls}">${i.val}</div>
        </div>`).join('');
}

// ── Helpers ──
function esc(s) { const d = document.createElement('div'); d.textContent = s||''; return d.innerHTML; }

// ── Start ──
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""

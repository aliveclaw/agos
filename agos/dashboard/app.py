"""Dashboard — FastAPI + WebSocket real-time agent monitoring.

`agos dashboard` launches this server at localhost:8420.
Provides REST endpoints and a WebSocket for live event streaming.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

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


def configure(
    runtime=None,
    event_bus=None,
    audit_trail=None,
    policy_engine=None,
    tracer=None,
) -> None:
    """Inject live subsystem references into the dashboard."""
    global _runtime, _event_bus, _audit_trail, _policy_engine, _tracer
    _runtime = runtime
    _event_bus = event_bus
    _audit_trail = audit_trail
    _policy_engine = policy_engine
    _tracer = tracer


# ── REST Endpoints ─────────────────────────────────────────────────


@dashboard_app.get("/")
async def index() -> HTMLResponse:
    """Dashboard landing page."""
    return HTMLResponse(_LANDING_HTML)


@dashboard_app.get("/api/agents")
async def list_agents() -> list[dict]:
    """List all agents and their status."""
    if _runtime is None:
        return []
    return _runtime.list_agents()


@dashboard_app.get("/api/events")
async def list_events(topic: str = "*", limit: int = 50) -> list[dict]:
    """Get recent events."""
    if _event_bus is None:
        return []
    events = _event_bus.history(topic_filter=topic, limit=limit)
    return [e.model_dump(mode="json") for e in events]


@dashboard_app.get("/api/audit")
async def list_audit(agent_id: str = "", action: str = "", limit: int = 50) -> list[dict]:
    """Query the audit trail."""
    if _audit_trail is None:
        return []
    entries = await _audit_trail.query(agent_id=agent_id, action=action, limit=limit)
    return [e.model_dump(mode="json") for e in entries]


@dashboard_app.get("/api/audit/violations")
async def list_violations(limit: int = 20) -> list[dict]:
    """Get recent policy violations."""
    if _audit_trail is None:
        return []
    entries = await _audit_trail.violations(limit=limit)
    return [e.model_dump(mode="json") for e in entries]


@dashboard_app.get("/api/policies")
async def list_policies() -> list[dict]:
    """List all assigned policies."""
    if _policy_engine is None:
        return []
    return _policy_engine.list_policies()


@dashboard_app.get("/api/traces")
async def list_traces(limit: int = 20) -> list[dict]:
    """Get recent execution traces."""
    if _tracer is None:
        return []
    traces = _tracer.list_traces(limit=limit)
    return [t.model_dump(mode="json") for t in traces]


@dashboard_app.get("/api/status")
async def system_status() -> dict:
    """System-wide status overview."""
    agents = _runtime.list_agents() if _runtime else []
    return {
        "agents_total": len(agents),
        "agents_running": sum(1 for a in agents if a["state"] == "running"),
        "event_subscribers": _event_bus.subscriber_count if _event_bus else 0,
        "ws_connections": _event_bus.ws_connection_count if _event_bus else 0,
        "audit_entries": await _audit_trail.count() if _audit_trail else 0,
        "policies": len(_policy_engine.list_policies()) if _policy_engine else 0,
        "active_spans": _tracer.active_span_count if _tracer else 0,
    }


# ── WebSocket — live event stream ─────────────────────────────────


@dashboard_app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """Stream all events in real-time over WebSocket."""
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
            # Keep connection alive, read any incoming pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if _event_bus:
            _event_bus.remove_ws_connection(send_event)


# ── Minimal Landing Page ──────────────────────────────────────────

_LANDING_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>agos dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        .section { margin-bottom: 24px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; }
        .section h2 { color: #8b949e; font-size: 14px; text-transform: uppercase; margin-bottom: 12px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #21262d; }
        th { color: #8b949e; font-size: 12px; }
        .status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
        .running { background: #3fb950; }
        .completed { background: #8b949e; }
        .error { background: #f85149; }
        #events { max-height: 300px; overflow-y: auto; font-size: 13px; }
        .event-line { padding: 4px 0; border-bottom: 1px solid #21262d; }
        .event-topic { color: #58a6ff; }
        .event-time { color: #484f58; font-size: 11px; }
    </style>
</head>
<body>
    <h1>agos dashboard</h1>

    <div class="section">
        <h2>Agents</h2>
        <table id="agents-table">
            <thead><tr><th>ID</th><th>Name</th><th>State</th><th>Tokens</th><th>Turns</th></tr></thead>
            <tbody></tbody>
        </table>
    </div>

    <div class="section">
        <h2>Live Events</h2>
        <div id="events"></div>
    </div>

    <div class="section">
        <h2>System Status</h2>
        <div id="status"></div>
    </div>

    <script>
        async function refresh() {
            // Agents
            const agents = await (await fetch('/api/agents')).json();
            const tbody = document.querySelector('#agents-table tbody');
            tbody.innerHTML = agents.map(a => `
                <tr>
                    <td>${a.id}</td>
                    <td>${a.name}</td>
                    <td><span class="status ${a.state}"></span>${a.state}</td>
                    <td>${a.tokens_used.toLocaleString()}</td>
                    <td>${a.turns}</td>
                </tr>
            `).join('');

            // Status
            const status = await (await fetch('/api/status')).json();
            document.getElementById('status').innerHTML = Object.entries(status)
                .map(([k, v]) => `<div>${k}: <strong>${v}</strong></div>`).join('');
        }

        // WebSocket for live events
        const ws = new WebSocket(`ws://${location.host}/ws/events`);
        ws.onmessage = (e) => {
            const event = JSON.parse(e.data);
            const div = document.getElementById('events');
            const line = document.createElement('div');
            line.className = 'event-line';
            line.innerHTML = `<span class="event-topic">${event.topic}</span> ${JSON.stringify(event.data)} <span class="event-time">${event.timestamp}</span>`;
            div.prepend(line);
            while (div.children.length > 100) div.lastChild.remove();
        };

        refresh();
        setInterval(refresh, 3000);
    </script>
</body>
</html>"""

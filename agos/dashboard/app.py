"""Dashboard — FastAPI + WebSocket real-time OS monitoring.

`agos dashboard` launches this server at localhost:8420.
Provides REST endpoints, real system intelligence, and a live WebSocket stream.
"""

from __future__ import annotations

import os
import time
import pathlib
import subprocess

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agos.events.bus import Event
from agos.config import settings

dashboard_app = FastAPI(title="AGenticOS dashboard", version="0.1.0")

_runtime = None
_event_bus = None
_audit_trail = None
_policy_engine = None
_tracer = None
_loom = None
_evolution_state = None
_start_time = time.time()


def configure(runtime=None, event_bus=None, audit_trail=None,
              policy_engine=None, tracer=None, loom=None,
              evolution_state=None) -> None:
    global _runtime, _event_bus, _audit_trail, _policy_engine, _tracer, _loom, _evolution_state
    _runtime = runtime
    _event_bus = event_bus
    _audit_trail = audit_trail
    _policy_engine = policy_engine
    _tracer = tracer
    _loom = loom
    _evolution_state = evolution_state


# ── Original endpoints (kept) ────────────────────────────────────

@dashboard_app.get("/")
async def index() -> HTMLResponse:
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
        "uptime_s": int(time.time() - _start_time),
    }


# ── Settings: API Key ────────────────────────────────────────────

class ApiKeyPayload(BaseModel):
    api_key: str


class GitHubTokenPayload(BaseModel):
    github_token: str


class FederatedTogglePayload(BaseModel):
    enabled: bool
    interval: int = 3


@dashboard_app.get("/api/settings")
async def get_settings() -> dict:
    has_key = bool(settings.anthropic_api_key)
    has_gh = bool(settings.github_token)
    return {
        "has_api_key": has_key,
        "api_key_preview": settings.anthropic_api_key[:8] + "..." if has_key else "",
        "model": settings.default_model,
        "has_github_token": has_gh,
        "github_token_preview": settings.github_token[:8] + "..." if has_gh else "",
        "auto_share_every": settings.auto_share_every,
        "is_contributor": bool(settings.github_token and settings.auto_share_every > 0),
    }


@dashboard_app.post("/api/settings/apikey")
async def set_api_key(payload: ApiKeyPayload) -> dict:
    key = payload.api_key.strip()
    if not key:
        return {"ok": False, "error": "API key cannot be empty"}
    settings.anthropic_api_key = key
    return {"ok": True, "preview": key[:8] + "..."}


@dashboard_app.post("/api/settings/github-token")
async def set_github_token(payload: GitHubTokenPayload) -> dict:
    token = payload.github_token.strip()
    if not token:
        return {"ok": False, "error": "Token cannot be empty"}
    settings.github_token = token
    return {"ok": True, "preview": token[:8] + "..."}


@dashboard_app.post("/api/settings/federated")
async def set_federated(payload: FederatedTogglePayload) -> dict:
    if payload.enabled:
        settings.auto_share_every = max(1, payload.interval)
    else:
        settings.auto_share_every = 0
    return {
        "ok": True,
        "auto_share_every": settings.auto_share_every,
        "is_contributor": bool(settings.github_token and settings.auto_share_every > 0),
    }


# ── Evolution state + community sharing ──────────────────────────

@dashboard_app.get("/api/evolution/state")
async def evolution_state_endpoint() -> dict:
    if _evolution_state is None:
        return {"available": False}
    d = _evolution_state.data
    return {
        "available": True,
        "instance_id": d.instance_id,
        "cycles_completed": d.cycles_completed,
        "last_saved": d.last_saved,
        "strategies_applied": [s.model_dump() for s in d.strategies_applied],
        "discovered_patterns": [p.model_dump() for p in d.discovered_patterns],
        "parameters": d.parameters,
    }


class SharePayload(BaseModel):
    github_token: str = ""


@dashboard_app.post("/api/evolution/share")
async def share_evolution(payload: SharePayload) -> dict:
    if _evolution_state is None:
        return {"ok": False, "error": "Evolution state not available"}
    token = payload.github_token.strip() or settings.github_token
    if not token:
        return {"ok": False, "error": "GitHub token required — set in Settings or AGOS_GITHUB_TOKEN env var"}
    try:
        from agos.evolution.contribute import share_learnings
        contribution = _evolution_state.export_contribution()
        result = await share_learnings(contribution, token)
        return {"ok": True, "pr_url": result["pr_url"], "branch": result["branch"]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ── NEW: Real system vitals ──────────────────────────────────────

@dashboard_app.get("/api/vitals")
async def system_vitals() -> dict:
    """Real CPU, memory, disk stats from the container."""
    vitals = {"cpu_percent": 0.0, "mem_total_mb": 0, "mem_used_mb": 0,
              "mem_percent": 0.0, "disk_total_gb": 0.0, "disk_used_gb": 0.0,
              "disk_percent": 0.0, "load_avg": [0, 0, 0], "processes": 0,
              "uptime_s": int(time.time() - _start_time)}
    try:
        # CPU from /proc/stat
        with open("/proc/stat") as f:
            parts = f.readline().split()
            total = sum(int(x) for x in parts[1:])
            idle = int(parts[4])
            vitals["cpu_percent"] = round(100 * (1 - idle / max(total, 1)), 1)
        # Memory from /proc/meminfo
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                meminfo[k.strip()] = int(v.strip().split()[0])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        used_kb = total_kb - avail_kb
        vitals["mem_total_mb"] = total_kb // 1024
        vitals["mem_used_mb"] = used_kb // 1024
        vitals["mem_percent"] = round(100 * used_kb / max(total_kb, 1), 1)
        # Disk
        st = os.statvfs("/")
        total_b = st.f_blocks * st.f_frsize
        free_b = st.f_bfree * st.f_frsize
        used_b = total_b - free_b
        vitals["disk_total_gb"] = round(total_b / 1e9, 1)
        vitals["disk_used_gb"] = round(used_b / 1e9, 1)
        vitals["disk_percent"] = round(100 * used_b / max(total_b, 1), 1)
        # Load
        vitals["load_avg"] = [round(x, 2) for x in os.getloadavg()]
        # Processes
        vitals["processes"] = len([p for p in os.listdir("/proc") if p.isdigit()])
    except Exception:
        pass
    return vitals


# ── NEW: Real codebase analysis ──────────────────────────────────

@dashboard_app.get("/api/codebase")
async def codebase_analysis() -> dict:
    """Scan the actual source tree and return real metrics."""
    src = pathlib.Path("/app/agos")
    if not src.exists():
        src = pathlib.Path("agos")
    result = {"total_files": 0, "total_lines": 0, "total_bytes": 0,
              "python_files": 0, "modules": [], "file_types": {},
              "largest_files": [], "todos": [], "imports": set(),
              "classes": 0, "functions": 0, "health_score": 0}
    todos = []
    largest = []
    modules = set()
    classes = 0
    functions = 0
    imports = set()

    for f in src.rglob("*"):
        if f.is_dir() or "__pycache__" in str(f):
            continue
        result["total_files"] += 1
        size = f.stat().st_size
        result["total_bytes"] += size
        ext = f.suffix or "(none)"
        result["file_types"][ext] = result["file_types"].get(ext, 0) + 1

        if ext == ".py":
            result["python_files"] += 1
            # Track module
            parts = f.relative_to(src).parts
            if len(parts) > 1:
                modules.add(parts[0])
            try:
                lines = f.read_text(errors="ignore").splitlines()
                result["total_lines"] += len(lines)
                largest.append({"file": str(f.relative_to(src.parent)), "lines": len(lines)})
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if "TODO" in stripped or "FIXME" in stripped or "HACK" in stripped:
                        todos.append({
                            "file": str(f.relative_to(src.parent)),
                            "line": i,
                            "text": stripped[:120],
                        })
                    if stripped.startswith("class ") and "(" in stripped:
                        classes += 1
                    if stripped.startswith("def ") or stripped.startswith("async def "):
                        functions += 1
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        mod = stripped.split()[1].split(".")[0]
                        if mod not in ("__future__",):
                            imports.add(mod)
            except Exception:
                pass

    largest.sort(key=lambda x: x["lines"], reverse=True)
    result["largest_files"] = largest[:10]
    result["todos"] = todos[:30]
    result["modules"] = sorted(modules)
    result["classes"] = classes
    result["functions"] = functions
    result["imports"] = sorted(imports)

    # Health score: 100 minus penalties
    score = 100
    if len(todos) > 10:
        score -= 10
    if len(todos) > 20:
        score -= 10
    if result["total_lines"] > 0 and result["python_files"] > 0:
        avg = result["total_lines"] / result["python_files"]
        if avg > 200:
            score -= 5
    result["health_score"] = max(0, min(100, score))

    return result


# ── NEW: Dependency health ───────────────────────────────────────

@dashboard_app.get("/api/deps")
async def dependency_health() -> list[dict]:
    """List real installed packages."""
    deps = []
    try:
        out = subprocess.check_output(
            ["pip", "list", "--format=json"], text=True, timeout=10
        )
        import json
        for pkg in json.loads(out):
            deps.append({"name": pkg["name"], "version": pkg["version"]})
    except Exception:
        pass
    return deps


# ── WebSocket — live event stream ────────────────────────────────

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


# ── Full Dashboard HTML — Draggable Widget Grid ──────────────────

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AGenticOS</title>
<style>
:root {
    --bg: #0a0e14; --bg2: #12171f; --bg3: #1a2030;
    --border: #232d3f; --text: #d4dce8; --text2: #6b7a90;
    --blue: #4facfe; --blue2: #00f2fe; --green: #43e97b; --green2: #38f9d7;
    --yellow: #f5af19; --red: #f85149; --purple: #a855f7; --cyan: #22d3ee;
    --glow-blue: rgba(79,172,254,0.3); --glow-green: rgba(67,233,123,0.3);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); overflow-x: hidden; }

/* ── Header ── */
header { background: linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%); border-bottom: 1px solid var(--border); padding: 14px 28px; display: flex; align-items: center; justify-content: space-between; }
header h1 { font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }
.brand-a { background: linear-gradient(135deg, var(--blue2), var(--blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
.brand-g { background: linear-gradient(135deg, var(--blue), var(--blue2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
.brand-entic { color: var(--text); font-weight: 300; }
.brand-os { background: linear-gradient(135deg, var(--blue), var(--cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
.uptime { color: var(--text2); font-size: 12px; font-family: monospace; }
.header-right { display: flex; align-items: center; gap: 16px; }
.pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--glow-green); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.reset-btn { background: var(--bg3); border: 1px solid var(--border); color: var(--text2); padding: 5px 12px; border-radius: 6px; font-size: 11px; cursor: pointer; transition: all 0.2s; }
.reset-btn:hover { border-color: var(--blue); color: var(--text); }

/* ── Widget Grid ── */
.widget-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; padding: 20px 24px; max-width: 1600px; margin: 0 auto; }

/* ── Widget ── */
.widget { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; transition: border-color 0.3s, box-shadow 0.3s, opacity 0.2s, transform 0.3s; min-height: 120px; display: flex; flex-direction: column; }
.widget:hover { border-color: rgba(79,172,254,0.2); }
.widget.size-1 { grid-column: span 1; }
.widget.size-2 { grid-column: span 2; }
.widget.size-3 { grid-column: span 3; }
.widget.size-4 { grid-column: span 4; }
.widget.dragging { opacity: 0.4; border-color: var(--blue); transform: scale(0.98); }
.widget.drag-over { border-color: var(--blue); box-shadow: 0 0 24px var(--glow-blue), inset 0 0 24px rgba(79,172,254,0.05); }
.widget.swap-anim { animation: swapIn 0.35s ease; }
@keyframes swapIn { 0% { transform: scale(0.92); opacity: 0.3; } 100% { transform: scale(1); opacity: 1; } }

/* ── Widget Header ── */
.widget-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--border); cursor: grab; user-select: none; flex-shrink: 0; }
.widget-header:active { cursor: grabbing; }
.widget-title { display: flex; align-items: center; gap: 8px; font-size: 11px; text-transform: uppercase; color: var(--text2); letter-spacing: 1px; font-weight: 600; }
.grip { font-size: 14px; opacity: 0.4; line-height: 1; }
.widget-header:hover .grip { opacity: 0.8; }

/* ── Widget Controls ── */
.widget-controls { display: flex; align-items: center; gap: 4px; }
.wbtn { width: 26px; height: 22px; border: 1px solid var(--border); border-radius: 5px; background: var(--bg3); color: var(--text2); font-size: 12px; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.15s; font-family: inherit; }
.wbtn:hover { border-color: var(--blue); color: var(--text); background: rgba(79,172,254,0.1); }
.sz-btn { width: 20px; height: 18px; border: 1px solid var(--border); border-radius: 4px; background: transparent; color: var(--text2); font-size: 9px; font-weight: 700; cursor: pointer; transition: all 0.15s; font-family: monospace; padding: 0; }
.sz-btn:hover { border-color: var(--blue); color: var(--text); }
.sz-btn.active { background: var(--blue); color: #fff; border-color: var(--blue); }
.sz-divider { width: 1px; height: 16px; background: var(--border); margin: 0 4px; }

/* ── Widget Body ── */
.widget-body { padding: 16px; overflow-y: auto; flex: 1; }
.widget-body::-webkit-scrollbar { width: 4px; }
.widget-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Mini Stats ── */
.mini-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }
.mini-stat { text-align: center; padding: 10px 6px; background: var(--bg3); border-radius: 8px; }
.mini-val { font-size: 22px; font-weight: 800; line-height: 1.2; }
.mini-lbl { font-size: 10px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

/* ── Ring gauges ── */
.gauge-row { display: flex; gap: 16px; justify-content: space-around; align-items: center; padding: 8px 0; }
.gauge { position: relative; width: 100px; height: 100px; }
.gauge svg { transform: rotate(-90deg); }
.gauge-bg { fill: none; stroke: var(--bg3); stroke-width: 8; }
.gauge-fill { fill: none; stroke-width: 8; stroke-linecap: round; transition: stroke-dashoffset 1s ease; }
.gauge-label { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; }
.gauge-val { font-size: 18px; font-weight: 700; }
.gauge-name { font-size: 9px; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; }
.gauge-details { display: flex; justify-content: space-around; font-size: 11px; color: var(--text2); padding: 8px 0; }

/* ── Tables ── */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 10px; color: var(--text2); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--border); }
td { padding: 8px 10px; border-bottom: 1px solid rgba(35,45,63,0.5); }
tr:hover td { background: rgba(255,255,255,0.02); }
.sys-tbl td:first-child { color: var(--text2); font-size: 12px; }
.sys-tbl td:last-child { font-family: monospace; font-size: 12px; text-align: right; }

/* ── Badges ── */
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge-running { background: rgba(67,233,123,0.12); color: var(--green); }
.badge-completed { background: rgba(107,122,144,0.15); color: var(--text2); }
.badge-ready { background: rgba(79,172,254,0.12); color: var(--blue); }
.badge-error { background: rgba(248,81,73,0.12); color: var(--red); }

/* ── Feed ── */
.feed { max-height: 400px; overflow-y: auto; }
.feed::-webkit-scrollbar { width: 4px; }
.feed::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.feed-item { padding: 8px 10px; border-bottom: 1px solid rgba(35,45,63,0.4); font-size: 13px; display: flex; gap: 10px; align-items: baseline; animation: fadeIn 0.3s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
.feed-topic { color: var(--blue); font-weight: 600; min-width: 130px; font-size: 12px; }
.feed-data { color: var(--text2); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: monospace; font-size: 12px; }
.feed-time { color: var(--text2); font-size: 11px; opacity: 0.5; font-family: monospace; }
.empty { color: var(--text2); text-align: center; padding: 30px; font-size: 13px; }

/* ── Health ring ── */
.health-block { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
.health-ring { width: 120px; height: 120px; flex-shrink: 0; }
.score-text { font-size: 10px; color: var(--text2); text-align: center; text-transform: uppercase; letter-spacing: 1px; margin-top: -38px; }

/* ── Section labels ── */
.section-label { font-size: 10px; text-transform: uppercase; color: var(--text2); letter-spacing: 1px; font-weight: 600; margin-bottom: 8px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }

/* ── TODO / File breakdown ── */
.todo-item { padding: 6px 8px; border-bottom: 1px solid rgba(35,45,63,0.4); font-size: 11px; font-family: monospace; }
.todo-file { color: var(--cyan); }
.todo-line { color: var(--yellow); }
.todo-text { color: var(--text2); }
.breakdown-item { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; }
.breakdown-item .ext { min-width: 45px; color: var(--cyan); font-family: monospace; font-weight: 600; }
.breakdown-item .cnt { min-width: 28px; text-align: right; color: var(--text2); }
.stat-bar { height: 5px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
.stat-bar-fill { height: 100%; border-radius: 3px; transition: width 1s ease; }

/* ── Deps ── */
.dep-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.dep-pill { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 11px; }
.dep-pill .dv { color: var(--blue); margin-left: 4px; }

/* ── Responsive ── */
@media (max-width: 1100px) { .widget-grid { grid-template-columns: repeat(2, 1fr); } .widget.size-3, .widget.size-4 { grid-column: span 2; } }
@media (max-width: 700px) { .widget-grid { grid-template-columns: 1fr; } .widget.size-1, .widget.size-2, .widget.size-3, .widget.size-4 { grid-column: span 1; } .mini-stats { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>

<header>
    <h1><span class="brand-a">A</span><span class="brand-g">G</span><span class="brand-entic">entic</span><span class="brand-os">OS</span></h1>
    <div class="header-right">
        <button class="reset-btn" onclick="resetLayout()" title="Reset panel layout">Reset Layout</button>
        <button class="reset-btn" onclick="openSettings()" title="Settings" id="settings-btn">&#9881; Settings</button>
        <span class="uptime" id="h-uptime">00:00:00</span>
        <div class="pulse" id="key-pulse"></div>
        <span style="color:var(--text2);font-size:12px">v0.1.0</span>
    </div>
</header>

<div class="widget-grid" id="widget-grid">

<!-- ═══ SYSTEM VITALS ═══ -->
<div class="widget size-2" id="w-vitals" data-wid="vitals">
    <div class="widget-header" draggable="true">
        <div class="widget-title"><span class="grip">&#x2801;&#x2801;</span> System Vitals</div>
        <div class="widget-controls">
            <button class="sz-btn" data-sz="1" onclick="setSize('vitals',1)">1</button>
            <button class="sz-btn active" data-sz="2" onclick="setSize('vitals',2)">2</button>
            <button class="sz-btn" data-sz="3" onclick="setSize('vitals',3)">3</button>
            <button class="sz-btn" data-sz="4" onclick="setSize('vitals',4)">4</button>
        </div>
    </div>
    <div class="widget-body">
        <div class="mini-stats">
            <div class="mini-stat"><div class="mini-val" style="color:var(--green)" id="v-agents">0</div><div class="mini-lbl">Agents<br><span style="font-size:9px;opacity:0.6" id="v-agents-sub">0 total</span></div></div>
            <div class="mini-stat"><div class="mini-val" style="color:var(--blue)" id="v-events">0</div><div class="mini-lbl">Events</div></div>
            <div class="mini-stat"><div class="mini-val" style="color:var(--yellow)" id="v-audit">0</div><div class="mini-lbl">Audit</div></div>
            <div class="mini-stat"><div class="mini-val" style="color:var(--cyan)" id="v-uptime">0s</div><div class="mini-lbl">Uptime</div></div>
        </div>
        <div class="gauge-row">
            <div class="gauge">
                <svg viewBox="0 0 120 120"><circle class="gauge-bg" cx="60" cy="60" r="52"/><circle class="gauge-fill" cx="60" cy="60" r="52" stroke="url(#grad-blue)" stroke-dasharray="327" stroke-dashoffset="327" id="g-cpu-fill"/><defs><linearGradient id="grad-blue"><stop offset="0%" stop-color="#4facfe"/><stop offset="100%" stop-color="#00f2fe"/></linearGradient></defs></svg>
                <div class="gauge-label"><div class="gauge-val" id="g-cpu-val">0%</div><div class="gauge-name">CPU</div></div>
            </div>
            <div class="gauge">
                <svg viewBox="0 0 120 120"><circle class="gauge-bg" cx="60" cy="60" r="52"/><circle class="gauge-fill" cx="60" cy="60" r="52" stroke="url(#grad-green)" stroke-dasharray="327" stroke-dashoffset="327" id="g-mem-fill"/><defs><linearGradient id="grad-green"><stop offset="0%" stop-color="#43e97b"/><stop offset="100%" stop-color="#38f9d7"/></linearGradient></defs></svg>
                <div class="gauge-label"><div class="gauge-val" id="g-mem-val">0%</div><div class="gauge-name">Memory</div></div>
            </div>
            <div class="gauge">
                <svg viewBox="0 0 120 120"><circle class="gauge-bg" cx="60" cy="60" r="52"/><circle class="gauge-fill" cx="60" cy="60" r="52" stroke="url(#grad-purple)" stroke-dasharray="327" stroke-dashoffset="327" id="g-disk-fill"/><defs><linearGradient id="grad-purple"><stop offset="0%" stop-color="#a855f7"/><stop offset="100%" stop-color="#ec4899"/></linearGradient></defs></svg>
                <div class="gauge-label"><div class="gauge-val" id="g-disk-val">0%</div><div class="gauge-name">Disk</div></div>
            </div>
        </div>
        <div class="gauge-details">
            <span>CPU: <span id="v-cpu-detail">-</span></span>
            <span>RAM: <span id="v-mem-detail">-</span></span>
            <span>Disk: <span id="v-disk-detail">-</span></span>
        </div>
        <table class="sys-tbl" style="margin-top:8px">
            <tr><td>Load Average</td><td id="v-load">-</td></tr>
            <tr><td>Processes</td><td id="v-procs">-</td></tr>
            <tr><td>Python Files</td><td id="v-pyfiles">-</td></tr>
            <tr><td>Lines of Code</td><td id="v-loc">-</td></tr>
            <tr><td>Classes</td><td id="v-classes">-</td></tr>
            <tr><td>Functions</td><td id="v-funcs">-</td></tr>
            <tr><td>Health Score</td><td id="v-health" style="font-weight:700">-</td></tr>
        </table>
    </div>
</div>

<!-- ═══ AGENTS ═══ -->
<div class="widget size-2" id="w-agents" data-wid="agents">
    <div class="widget-header" draggable="true">
        <div class="widget-title"><span class="grip">&#x2801;&#x2801;</span> Agents</div>
        <div class="widget-controls">
            <button class="sz-btn" data-sz="1" onclick="setSize('agents',1)">1</button>
            <button class="sz-btn active" data-sz="2" onclick="setSize('agents',2)">2</button>
            <button class="sz-btn" data-sz="3" onclick="setSize('agents',3)">3</button>
            <button class="sz-btn" data-sz="4" onclick="setSize('agents',4)">4</button>
        </div>
    </div>
    <div class="widget-body">
        <table><thead><tr><th>ID</th><th>Name</th><th>Role</th><th>State</th><th>Tokens</th><th>Turns</th></tr></thead><tbody id="ag-table"></tbody></table>
        <div class="empty" id="ag-empty">No agents spawned yet</div>
    </div>
</div>

<!-- ═══ LIVE EVENTS ═══ -->
<div class="widget size-2" id="w-events" data-wid="events">
    <div class="widget-header" draggable="true">
        <div class="widget-title"><span class="grip">&#x2801;&#x2801;</span> Live Events <span style="font-size:10px;color:var(--green);margin-left:6px">&#x25CF; LIVE</span></div>
        <div class="widget-controls">
            <button class="sz-btn" data-sz="1" onclick="setSize('events',1)">1</button>
            <button class="sz-btn active" data-sz="2" onclick="setSize('events',2)">2</button>
            <button class="sz-btn" data-sz="3" onclick="setSize('events',3)">3</button>
            <button class="sz-btn" data-sz="4" onclick="setSize('events',4)">4</button>
        </div>
    </div>
    <div class="widget-body" style="padding:0">
        <div class="feed" id="ev-feed" style="max-height:500px"></div>
        <div class="empty" id="ev-empty">Waiting for events...</div>
    </div>
</div>

<!-- ═══ CODEBASE INTEL ═══ -->
<div class="widget size-2" id="w-codebase" data-wid="codebase">
    <div class="widget-header" draggable="true">
        <div class="widget-title"><span class="grip">&#x2801;&#x2801;</span> Codebase Intel</div>
        <div class="widget-controls">
            <button class="sz-btn" data-sz="1" onclick="setSize('codebase',1)">1</button>
            <button class="sz-btn active" data-sz="2" onclick="setSize('codebase',2)">2</button>
            <button class="sz-btn" data-sz="3" onclick="setSize('codebase',3)">3</button>
            <button class="sz-btn" data-sz="4" onclick="setSize('codebase',4)">4</button>
        </div>
    </div>
    <div class="widget-body">
        <div class="health-block">
            <div>
                <div class="health-ring">
                    <svg viewBox="0 0 140 140"><circle fill="none" stroke="var(--bg3)" stroke-width="10" cx="70" cy="70" r="60"/><circle id="health-ring-fill" fill="none" stroke-width="10" stroke-linecap="round" cx="70" cy="70" r="60" stroke-dasharray="377" stroke-dashoffset="377" style="transform:rotate(-90deg);transform-origin:center;transition:stroke-dashoffset 1.5s ease"/></svg>
                    <div style="position:relative;top:-82px;text-align:center"><span id="cb-score" style="font-size:26px;font-weight:800">-</span></div>
                </div>
                <div class="score-text">Health</div>
            </div>
            <div style="flex:1;min-width:180px">
                <div class="section-label">File Types</div>
                <div id="cb-types"></div>
            </div>
        </div>
        <div class="section-label" style="margin-top:14px">Largest Files</div>
        <div id="cb-largest" style="font-size:11px;font-family:monospace"></div>
        <div class="section-label" style="margin-top:14px">TODOs / FIXMEs</div>
        <div class="feed" id="cb-todos" style="max-height:180px"></div>
        <div class="empty" id="cb-todos-empty" style="display:none;padding:12px">Clean code!</div>
        <div class="section-label" style="margin-top:14px">Modules</div>
        <div id="cb-modules"></div>
    </div>
</div>

<!-- ═══ AUDIT TRAIL ═══ -->
<div class="widget size-2" id="w-audit" data-wid="audit">
    <div class="widget-header" draggable="true">
        <div class="widget-title"><span class="grip">&#x2801;&#x2801;</span> Audit Trail</div>
        <div class="widget-controls">
            <button class="sz-btn" data-sz="1" onclick="setSize('audit',1)">1</button>
            <button class="sz-btn active" data-sz="2" onclick="setSize('audit',2)">2</button>
            <button class="sz-btn" data-sz="3" onclick="setSize('audit',3)">3</button>
            <button class="sz-btn" data-sz="4" onclick="setSize('audit',4)">4</button>
        </div>
    </div>
    <div class="widget-body" style="padding:0 16px 16px 16px">
        <table>
            <thead><tr><th>Time</th><th>Agent</th><th>Action</th><th>Detail</th><th>OK</th></tr></thead>
            <tbody id="au-table"></tbody>
        </table>
        <div class="empty" id="au-empty">No audit entries yet</div>
    </div>
</div>

<!-- ═══ DEPENDENCIES ═══ -->
<div class="widget size-2" id="w-deps" data-wid="deps">
    <div class="widget-header" draggable="true">
        <div class="widget-title"><span class="grip">&#x2801;&#x2801;</span> Dependencies</div>
        <div class="widget-controls">
            <button class="sz-btn" data-sz="1" onclick="setSize('deps',1)">1</button>
            <button class="sz-btn active" data-sz="2" onclick="setSize('deps',2)">2</button>
            <button class="sz-btn" data-sz="3" onclick="setSize('deps',3)">3</button>
            <button class="sz-btn" data-sz="4" onclick="setSize('deps',4)">4</button>
        </div>
    </div>
    <div class="widget-body">
        <div class="dep-grid" id="dep-grid"></div>
    </div>
</div>

<!-- ═══ EVOLUTION STATE ═══ -->
<div class="widget size-2" id="w-evolution" data-wid="evolution">
    <div class="widget-header" draggable="true">
        <div class="widget-title"><span class="grip">&#x2801;&#x2801;</span> Evolution State <span style="font-size:10px;color:var(--purple);margin-left:6px">&#x2B50;</span></div>
        <div class="widget-controls">
            <button class="sz-btn" data-sz="1" onclick="setSize('evolution',1)">1</button>
            <button class="sz-btn active" data-sz="2" onclick="setSize('evolution',2)">2</button>
            <button class="sz-btn" data-sz="3" onclick="setSize('evolution',3)">3</button>
            <button class="sz-btn" data-sz="4" onclick="setSize('evolution',4)">4</button>
        </div>
    </div>
    <div class="widget-body">
        <div class="mini-stats">
            <div class="mini-stat"><div class="mini-val" style="color:var(--purple)" id="evo-cycles">0</div><div class="mini-lbl">Cycles</div></div>
            <div class="mini-stat"><div class="mini-val" style="color:var(--green)" id="evo-strategies">0</div><div class="mini-lbl">Strategies</div></div>
            <div class="mini-stat"><div class="mini-val" style="color:var(--cyan)" id="evo-patterns">0</div><div class="mini-lbl">Patterns</div></div>
            <div class="mini-stat"><div class="mini-val" style="color:var(--text2);font-size:12px" id="evo-saved">-</div><div class="mini-lbl">Last Saved</div></div>
        </div>
        <div class="section-label">Applied Strategies</div>
        <div id="evo-strat-list" style="max-height:180px;overflow-y:auto"></div>
        <div class="empty" id="evo-strat-empty" style="padding:12px">No strategies yet</div>
        <div class="section-label" style="margin-top:14px">Discovered Patterns</div>
        <div id="evo-pat-list" style="max-height:140px;overflow-y:auto"></div>
        <div class="empty" id="evo-pat-empty" style="padding:12px">No patterns yet</div>
        <div style="margin-top:16px;padding-top:14px;border-top:1px solid var(--border)">
            <div class="section-label">Federated Learning</div>
            <div id="fed-status" style="margin-top:8px;font-size:12px;padding:8px 10px;background:var(--bg3);border-radius:8px">
                <span style="color:var(--text2)">Checking...</span>
            </div>
            <div id="fed-last-pr" style="margin-top:6px;font-size:12px;color:var(--text2)"></div>
            <div style="display:flex;gap:8px;margin-top:10px">
                <button onclick="shareLearnings()" style="flex:1;background:linear-gradient(135deg,var(--purple),var(--blue));border:none;border-radius:8px;padding:8px 16px;color:#fff;font-weight:700;font-size:12px;cursor:pointer">Share Now</button>
                <button onclick="openSettings()" style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:8px 14px;color:var(--text2);font-size:12px;cursor:pointer">Configure</button>
            </div>
            <div id="share-status" style="margin-top:6px;font-size:12px;color:var(--text2)"></div>
        </div>
    </div>
</div>

</div><!-- /widget-grid -->

<!-- ═══ SETTINGS MODAL ═══ -->
<div id="settings-modal" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.6);z-index:1000;backdrop-filter:blur(4px);display:none;align-items:center;justify-content:center">
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:16px;width:440px;max-width:90vw;box-shadow:0 24px 80px rgba(0,0,0,0.5)">
        <div style="padding:18px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between">
            <span style="font-size:14px;font-weight:700;letter-spacing:0.5px">&#9881; Settings</span>
            <button onclick="closeSettings()" style="background:none;border:none;color:var(--text2);font-size:20px;cursor:pointer;padding:0 4px;line-height:1">&times;</button>
        </div>
        <div style="padding:22px">
            <div style="margin-bottom:18px">
                <label style="display:block;font-size:11px;text-transform:uppercase;color:var(--text2);letter-spacing:1px;margin-bottom:8px;font-weight:600">Anthropic API Key</label>
                <div style="display:flex;gap:8px">
                    <input id="api-key-input" type="password" placeholder="sk-ant-api03-..." style="flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px 14px;color:var(--text);font-family:monospace;font-size:13px;outline:none;transition:border-color 0.2s" onfocus="this.style.borderColor='var(--blue)'" onblur="this.style.borderColor='var(--border)'"/>
                    <button onclick="toggleKeyVis()" style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:0 12px;color:var(--text2);cursor:pointer;font-size:14px" title="Show/hide">&#128065;</button>
                </div>
                <div id="api-key-status" style="margin-top:8px;font-size:12px;color:var(--text2)"></div>
            </div>
            <button onclick="saveApiKey()" style="width:100%;padding:10px;background:linear-gradient(135deg,var(--blue),var(--blue2));border:none;border-radius:8px;color:#0a0e14;font-weight:700;font-size:13px;cursor:pointer;letter-spacing:0.5px;transition:opacity 0.2s" onmouseover="this.style.opacity='0.85'" onmouseout="this.style.opacity='1'">Save API Key</button>

            <!-- GitHub Token -->
            <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border)">
                <label style="display:block;font-size:11px;text-transform:uppercase;color:var(--text2);letter-spacing:1px;margin-bottom:8px;font-weight:600">GitHub Token</label>
                <div style="display:flex;gap:8px">
                    <input id="gh-token-input" type="password" placeholder="ghp_..." style="flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:10px 14px;color:var(--text);font-family:monospace;font-size:13px;outline:none;transition:border-color 0.2s" onfocus="this.style.borderColor='var(--purple)'" onblur="this.style.borderColor='var(--border)'"/>
                    <button onclick="saveGHToken()" style="background:linear-gradient(135deg,var(--purple),var(--blue));border:none;border-radius:8px;padding:0 16px;color:#fff;font-weight:700;font-size:12px;cursor:pointer">Save</button>
                </div>
                <div id="gh-token-status" style="margin-top:8px;font-size:12px;color:var(--text2)"></div>
            </div>

            <!-- Federated Learning Toggle -->
            <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border)">
                <label style="display:block;font-size:11px;text-transform:uppercase;color:var(--text2);letter-spacing:1px;margin-bottom:10px;font-weight:600">Federated Learning</label>
                <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:var(--bg3);border-radius:8px;margin-bottom:10px">
                    <div>
                        <div style="font-size:13px;font-weight:600">Auto-share learnings</div>
                        <div style="font-size:11px;color:var(--text2);margin-top:2px">Contribute evolution strategies to the community</div>
                    </div>
                    <label style="position:relative;display:inline-block;width:44px;height:24px;cursor:pointer">
                        <input type="checkbox" id="fed-toggle" onchange="toggleFederated()" style="opacity:0;width:0;height:0">
                        <span id="fed-slider" style="position:absolute;top:0;left:0;right:0;bottom:0;background:var(--border);border-radius:12px;transition:0.3s"></span>
                        <span id="fed-dot" style="position:absolute;height:18px;width:18px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:0.3s"></span>
                    </label>
                </div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <span style="font-size:12px;color:var(--text2)">Share every</span>
                    <select id="fed-interval" onchange="toggleFederated()" style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:5px 10px;color:var(--text);font-size:12px;outline:none">
                        <option value="1">1 cycle</option>
                        <option value="2">2 cycles</option>
                        <option value="3" selected>3 cycles</option>
                        <option value="5">5 cycles</option>
                        <option value="10">10 cycles</option>
                    </select>
                </div>
                <div id="fed-reciprocity" style="font-size:11px;padding:8px 10px;background:var(--bg);border-radius:6px;border:1px solid var(--border)"></div>
            </div>

            <div style="margin-top:16px;padding-top:14px;border-top:1px solid var(--border)">
                <div style="font-size:11px;color:var(--text2)">
                    <span style="color:var(--yellow)">&#x26A0;</span> Runtime settings are stored in memory only.
                    Set <code style="color:var(--cyan);background:var(--bg3);padding:2px 6px;border-radius:4px;font-size:11px">AGOS_GITHUB_TOKEN</code> and <code style="color:var(--cyan);background:var(--bg3);padding:2px 6px;border-radius:4px;font-size:11px">AGOS_AUTO_SHARE_EVERY</code> as env vars for persistence.
                </div>
            </div>
        </div>
    </div>
</div>

<script>
/* ═══════════════════════════════════════════════════════════════════
   LAYOUT ENGINE — drag-to-swap, resize, persist
   ═══════════════════════════════════════════════════════════════════ */
const WIDGETS = ['vitals','agents','events','codebase','audit','deps','evolution'];
const DEFAULT_SIZES = {vitals:2, agents:2, events:2, codebase:2, audit:2, deps:2, evolution:2};
let layout = { order: [...WIDGETS], sizes: {...DEFAULT_SIZES} };

function loadLayout() {
    try {
        const s = localStorage.getItem('agos-layout');
        if (s) {
            const l = JSON.parse(s);
            if (l.order && l.sizes) {
                // Add any new widgets missing from saved layout
                WIDGETS.forEach(w => {
                    if (!l.order.includes(w)) { l.order.push(w); l.sizes[w] = DEFAULT_SIZES[w]; }
                });
                // Remove stale widgets no longer in WIDGETS
                l.order = l.order.filter(w => WIDGETS.includes(w));
                layout = l;
            }
        }
    } catch {}
}
function saveLayout() { localStorage.setItem('agos-layout', JSON.stringify(layout)); }

function applyLayout() {
    layout.order.forEach((wid, i) => {
        const el = document.getElementById('w-' + wid);
        if (!el) return;
        el.style.order = i;
        // Update size class
        el.className = 'widget size-' + layout.sizes[wid];
        // Update size button indicators
        el.querySelectorAll('.sz-btn').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.sz) === layout.sizes[wid]);
        });
    });
}

function setSize(wid, sz) {
    layout.sizes[wid] = sz;
    saveLayout();
    applyLayout();
}

function resetLayout() {
    layout = { order: [...WIDGETS], sizes: {...DEFAULT_SIZES} };
    saveLayout();
    applyLayout();
    // Flash all widgets
    WIDGETS.forEach(wid => {
        const el = document.getElementById('w-' + wid);
        if (el) { el.classList.add('swap-anim'); setTimeout(() => el.classList.remove('swap-anim'), 400); }
    });
}

/* ── Drag and Drop ── */
let dragSrcId = null;

document.querySelectorAll('.widget-header[draggable]').forEach(header => {
    header.addEventListener('dragstart', function(e) {
        const widget = this.closest('.widget');
        dragSrcId = widget.dataset.wid;
        widget.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', dragSrcId);
        // Use the whole widget as drag image
        try { e.dataTransfer.setDragImage(widget, 50, 20); } catch {}
    });
    header.addEventListener('dragend', function() {
        document.querySelectorAll('.widget').forEach(w => w.classList.remove('dragging', 'drag-over'));
        dragSrcId = null;
    });
});

document.querySelectorAll('.widget').forEach(widget => {
    widget.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        if (this.dataset.wid !== dragSrcId) this.classList.add('drag-over');
    });
    widget.addEventListener('dragleave', function() {
        this.classList.remove('drag-over');
    });
    widget.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('drag-over');
        const srcId = e.dataTransfer.getData('text/plain');
        const tgtId = this.dataset.wid;
        if (srcId && srcId !== tgtId) {
            // Swap positions in layout order
            const srcIdx = layout.order.indexOf(srcId);
            const tgtIdx = layout.order.indexOf(tgtId);
            if (srcIdx !== -1 && tgtIdx !== -1) {
                [layout.order[srcIdx], layout.order[tgtIdx]] = [layout.order[tgtIdx], layout.order[srcIdx]];
                saveLayout();
                applyLayout();
                // Animate both
                [srcId, tgtId].forEach(wid => {
                    const el = document.getElementById('w-' + wid);
                    if (el) { el.classList.add('swap-anim'); setTimeout(() => el.classList.remove('swap-anim'), 400); }
                });
            }
        }
    });
});

// Boot layout
loadLayout();
applyLayout();


/* ═══════════════════════════════════════════════════════════════════
   DATA FETCHING
   ═══════════════════════════════════════════════════════════════════ */
const CIRC = 2 * Math.PI * 52;
const HCIRC = 2 * Math.PI * 60;

async function fetchJSON(url) {
    try { return await (await fetch(url)).json(); } catch { return null; }
}
function setGauge(id, pct) {
    const off = CIRC * (1 - pct / 100);
    const el = document.getElementById(id);
    if (el) el.setAttribute('stroke-dashoffset', off);
}
function fmtUptime(s) {
    const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
    return [h,m,sec].map(v => String(v).padStart(2,'0')).join(':');
}
function esc(s) { const d = document.createElement('div'); d.textContent = s||''; return d.innerHTML; }

/* ── Vitals + Status + Agents (every 2s) ── */
async function refreshFast() {
    const [status, vitals, agents] = await Promise.all([
        fetchJSON('/api/status'), fetchJSON('/api/vitals'), fetchJSON('/api/agents')
    ]);
    if (status) {
        document.getElementById('v-agents').textContent = status.agents_running;
        document.getElementById('v-agents-sub').textContent = status.agents_total + ' total';
        document.getElementById('v-events').textContent = eventCount;
        document.getElementById('v-audit').textContent = status.audit_entries;
        document.getElementById('v-uptime').textContent = fmtUptime(status.uptime_s || 0);
        document.getElementById('h-uptime').textContent = fmtUptime(status.uptime_s || 0);
    }
    if (vitals) {
        setGauge('g-cpu-fill', vitals.cpu_percent);
        setGauge('g-mem-fill', vitals.mem_percent);
        setGauge('g-disk-fill', vitals.disk_percent);
        document.getElementById('g-cpu-val').textContent = vitals.cpu_percent + '%';
        document.getElementById('g-mem-val').textContent = vitals.mem_percent + '%';
        document.getElementById('g-disk-val').textContent = vitals.disk_percent + '%';
        document.getElementById('v-cpu-detail').textContent = 'Load ' + (vitals.load_avg||[]).join(', ');
        document.getElementById('v-mem-detail').textContent = vitals.mem_used_mb + '/' + vitals.mem_total_mb + ' MB';
        document.getElementById('v-disk-detail').textContent = vitals.disk_used_gb + '/' + vitals.disk_total_gb + ' GB';
        document.getElementById('v-load').textContent = (vitals.load_avg||[]).join('  ');
        document.getElementById('v-procs').textContent = vitals.processes;
    }
    // Agents table
    const agBody = document.getElementById('ag-table');
    const agEmpty = document.getElementById('ag-empty');
    if (agents && agents.length) {
        agEmpty.style.display = 'none';
        agBody.innerHTML = agents.slice(-20).reverse().map(a => '<tr>' +
            '<td style="font-family:monospace;font-size:11px;color:var(--text2)">' + (a.id||'').slice(0,10) + '</td>' +
            '<td style="font-weight:600">' + esc(a.name) + '</td>' +
            '<td style="color:var(--text2)">' + esc(a.role) + '</td>' +
            '<td><span class="badge badge-' + a.state + '">' + a.state + '</span></td>' +
            '<td style="font-family:monospace">' + (a.tokens_used||0).toLocaleString() + '</td>' +
            '<td style="font-family:monospace">' + (a.turns||0) + '</td>' +
        '</tr>').join('');
    } else { if (agEmpty) agEmpty.style.display = ''; if (agBody) agBody.innerHTML = ''; }
}

/* ── Codebase (every 30s) ── */
async function refreshCodebase() {
    const code = await fetchJSON('/api/codebase');
    if (!code) return;
    // System info in vitals widget
    const el = (id) => document.getElementById(id);
    if (el('v-pyfiles')) el('v-pyfiles').textContent = code.python_files;
    if (el('v-loc')) el('v-loc').textContent = code.total_lines.toLocaleString();
    if (el('v-classes')) el('v-classes').textContent = code.classes;
    if (el('v-funcs')) el('v-funcs').textContent = code.functions;
    const hs = code.health_score;
    const hEl = el('v-health');
    if (hEl) { hEl.textContent = hs + '/100'; hEl.style.color = hs >= 80 ? 'var(--green)' : hs >= 60 ? 'var(--yellow)' : 'var(--red)'; }
    // Health ring
    const off = HCIRC * (1 - hs / 100);
    const ring = el('health-ring-fill');
    if (ring) {
        ring.setAttribute('stroke-dashoffset', off);
        ring.setAttribute('stroke', hs >= 80 ? 'url(#hg)' : hs >= 60 ? '#f5af19' : '#f85149');
        if (!document.getElementById('hg')) {
            const svg = ring.parentElement;
            const defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
            defs.innerHTML = '<linearGradient id="hg"><stop offset="0%" stop-color="#43e97b"/><stop offset="100%" stop-color="#38f9d7"/></linearGradient>';
            svg.prepend(defs);
        }
    }
    if (el('cb-score')) el('cb-score').textContent = hs;
    // File types
    const types = Object.entries(code.file_types).sort((a,b) => b[1]-a[1]);
    if (el('cb-types')) el('cb-types').innerHTML = types.map(([ext, cnt]) =>
        '<div class="breakdown-item"><span class="ext">' + esc(ext) + '</span><span class="cnt">' + cnt + '</span>' +
        '<div style="flex:1"><div class="stat-bar"><div class="stat-bar-fill" style="width:' + (100*cnt/Math.max(types[0][1],1)) + '%;background:linear-gradient(90deg,var(--blue),var(--cyan))"></div></div></div></div>').join('');
    // Largest files
    if (el('cb-largest')) el('cb-largest').innerHTML = code.largest_files.map(f =>
        '<div style="padding:3px 0;display:flex;justify-content:space-between"><span style="color:var(--cyan)">' + esc(f.file) + '</span><span style="color:var(--text2)">' + f.lines + ' ln</span></div>').join('');
    // TODOs
    const todosDiv = el('cb-todos');
    const todosEmpty = el('cb-todos-empty');
    if (code.todos.length) {
        if (todosEmpty) todosEmpty.style.display = 'none';
        if (todosDiv) todosDiv.innerHTML = code.todos.map(t =>
            '<div class="todo-item"><span class="todo-file">' + esc(t.file) + '</span><span class="todo-line">:' + t.line + '</span> <span class="todo-text">' + esc(t.text) + '</span></div>').join('');
    } else { if (todosEmpty) todosEmpty.style.display = ''; if (todosDiv) todosDiv.innerHTML = ''; }
    // Modules
    if (el('cb-modules')) el('cb-modules').innerHTML = code.modules.map(m =>
        '<div style="padding:5px 8px;margin:3px 0;background:var(--bg3);border-radius:6px;font-family:monospace;font-size:12px;color:var(--cyan)">agos/' + esc(m) + '/</div>').join('');
}

/* ── Audit (every 10s) ── */
async function refreshAudit() {
    const entries = await fetchJSON('/api/audit?limit=50');
    const body = document.getElementById('au-table');
    const empty = document.getElementById('au-empty');
    if (!entries || !entries.length) { if (empty) empty.style.display = ''; if (body) body.innerHTML = ''; return; }
    if (empty) empty.style.display = 'none';
    if (body) body.innerHTML = entries.map(e => '<tr>' +
        '<td style="font-family:monospace;font-size:11px;color:var(--text2)">' + ((e.timestamp||'').slice(11,19)) + '</td>' +
        '<td style="color:var(--cyan);font-weight:600">' + esc(e.agent_name||((e.agent_id||'?').slice(0,10))) + '</td>' +
        '<td><span class="badge" style="background:rgba(245,175,25,0.1);color:var(--yellow)">' + esc(e.action) + '</span></td>' +
        '<td style="color:var(--text2);font-size:12px">' + esc((e.detail||'').slice(0,80)) + '</td>' +
        '<td>' + (e.success ? '<span style="color:var(--green)">&#10003;</span>' : '<span style="color:var(--red)">&#10007;</span>') + '</td>' +
    '</tr>').join('');
}

/* ── Dependencies (every 60s) ── */
async function refreshDeps() {
    const deps = await fetchJSON('/api/deps');
    if (!deps) return;
    const grid = document.getElementById('dep-grid');
    if (grid) grid.innerHTML = deps.map(d =>
        '<div class="dep-pill">' + esc(d.name) + '<span class="dv">' + esc(d.version) + '</span></div>').join('');
}

/* ── Live events via WebSocket ── */
let eventCount = 0;
function addEvent(event) {
    eventCount++;
    const div = document.getElementById('ev-feed');
    const empty = document.getElementById('ev-empty');
    if (empty) empty.style.display = 'none';
    const el = document.createElement('div');
    el.className = 'feed-item';
    el.innerHTML = '<span class="feed-topic">' + esc(event.topic) + '</span><span class="feed-data">' + esc(JSON.stringify(event.data).slice(0,140)) + '</span><span class="feed-time">' + ((event.timestamp||'').slice(11,19)) + '</span>';
    div.prepend(el);
    while (div.children.length > 200) div.lastChild.remove();
}
try { const ws = new WebSocket('ws://'+location.host+'/ws/events'); ws.onmessage = e => { const ev = JSON.parse(e.data); addEvent(ev); if (ev.topic === 'evolution.auto_share_success' && ev.data && ev.data.pr_url) { lastSharePR = ev.data.pr_url; } }; } catch(err) {}
(async () => { const ev = await fetchJSON('/api/events?limit=30'); if (ev && ev.length) ev.reverse().forEach(addEvent); })();

/* ── Settings Modal ── */
function openSettings() {
    const m = document.getElementById('settings-modal');
    m.style.display = 'flex';
    fetchJSON('/api/settings').then(s => {
        if (!s) return;
        // API key
        const akStatus = document.getElementById('api-key-status');
        if (s.has_api_key) {
            akStatus.innerHTML = '<span style="color:var(--green)">&#10003; Key configured:</span> <code style="color:var(--cyan)">' + esc(s.api_key_preview) + '</code>';
            document.getElementById('api-key-input').placeholder = s.api_key_preview;
        } else {
            akStatus.innerHTML = '<span style="color:var(--yellow)">&#x26A0; No API key set</span>';
        }
        // GitHub token
        const ghStatus = document.getElementById('gh-token-status');
        if (s.has_github_token) {
            ghStatus.innerHTML = '<span style="color:var(--green)">&#10003; Token configured:</span> <code style="color:var(--cyan)">' + esc(s.github_token_preview) + '</code>';
            document.getElementById('gh-token-input').placeholder = s.github_token_preview;
        } else {
            ghStatus.innerHTML = '<span style="color:var(--text2)">Required for federated learning</span>';
        }
        // Federated toggle
        const toggle = document.getElementById('fed-toggle');
        const interval = document.getElementById('fed-interval');
        const isOn = s.auto_share_every > 0;
        toggle.checked = isOn;
        updateToggleUI(isOn);
        if (isOn) interval.value = String(s.auto_share_every);
        // Reciprocity info
        updateReciprocityInfo(s.is_contributor);
    });
}
function updateToggleUI(on) {
    const slider = document.getElementById('fed-slider');
    const dot = document.getElementById('fed-dot');
    if (on) { slider.style.background = 'var(--purple)'; dot.style.transform = 'translateX(20px)'; }
    else { slider.style.background = 'var(--border)'; dot.style.transform = 'translateX(0)'; }
}
function updateReciprocityInfo(isContributor) {
    const el = document.getElementById('fed-reciprocity');
    if (isContributor) {
        el.innerHTML = '<span style="color:var(--green)">&#10003; Contributor</span> — you get <b style="color:var(--text)">real-time</b> community strategies on every boot. Your instance helps evolve the OS for everyone.';
    } else {
        el.innerHTML = '<span style="color:var(--yellow)">&#x26A0; Observer only</span> — you get community updates <b style="color:var(--text)">weekly</b> (bundled with releases). Enable sharing to unlock real-time strategies from all instances.';
    }
}
function closeSettings() { document.getElementById('settings-modal').style.display = 'none'; }
document.getElementById('settings-modal').addEventListener('click', function(e) { if (e.target === this) closeSettings(); });
function toggleKeyVis() {
    const inp = document.getElementById('api-key-input');
    inp.type = inp.type === 'password' ? 'text' : 'password';
}
async function saveApiKey() {
    const key = document.getElementById('api-key-input').value.trim();
    if (!key) { document.getElementById('api-key-status').innerHTML = '<span style="color:var(--red)">Please enter an API key</span>'; return; }
    const resp = await fetch('/api/settings/apikey', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({api_key: key}) });
    const data = await resp.json();
    const status = document.getElementById('api-key-status');
    if (data.ok) {
        status.innerHTML = '<span style="color:var(--green)">&#10003; Key saved:</span> <code style="color:var(--cyan)">' + esc(data.preview) + '</code>';
        document.getElementById('api-key-input').value = '';
        document.getElementById('api-key-input').placeholder = data.preview;
        document.getElementById('key-pulse').style.background = 'var(--green)';
    } else {
        status.innerHTML = '<span style="color:var(--red)">' + esc(data.error) + '</span>';
    }
}
async function saveGHToken() {
    const token = document.getElementById('gh-token-input').value.trim();
    if (!token) { document.getElementById('gh-token-status').innerHTML = '<span style="color:var(--red)">Please enter a token</span>'; return; }
    const resp = await fetch('/api/settings/github-token', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({github_token: token}) });
    const data = await resp.json();
    const st = document.getElementById('gh-token-status');
    if (data.ok) {
        st.innerHTML = '<span style="color:var(--green)">&#10003; Token saved:</span> <code style="color:var(--cyan)">' + esc(data.preview) + '</code>';
        document.getElementById('gh-token-input').value = '';
        document.getElementById('gh-token-input').placeholder = data.preview;
        // Re-check contributor status
        const s = await fetchJSON('/api/settings');
        if (s) updateReciprocityInfo(s.is_contributor);
    } else {
        st.innerHTML = '<span style="color:var(--red)">' + esc(data.error) + '</span>';
    }
}
async function toggleFederated() {
    const on = document.getElementById('fed-toggle').checked;
    const interval = parseInt(document.getElementById('fed-interval').value) || 3;
    updateToggleUI(on);
    const resp = await fetch('/api/settings/federated', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({enabled: on, interval: interval}) });
    const data = await resp.json();
    if (data.ok) updateReciprocityInfo(data.is_contributor);
}
// Check key status on load
fetchJSON('/api/settings').then(s => {
    if (s && !s.has_api_key) document.getElementById('key-pulse').style.background = 'var(--yellow)';
});

/* ── Evolution State (every 10s) ── */
let lastSharePR = '';
async function refreshEvolution() {
    const [data, sett] = await Promise.all([
        fetchJSON('/api/evolution/state'),
        fetchJSON('/api/settings'),
    ]);
    if (!data || !data.available) return;
    document.getElementById('evo-cycles').textContent = data.cycles_completed;
    document.getElementById('evo-strategies').textContent = data.strategies_applied.length;
    document.getElementById('evo-patterns').textContent = data.discovered_patterns.length;
    if (data.last_saved) {
        const ts = data.last_saved.slice(11, 19);
        document.getElementById('evo-saved').textContent = ts || '-';
    }
    const stratList = document.getElementById('evo-strat-list');
    const stratEmpty = document.getElementById('evo-strat-empty');
    if (data.strategies_applied.length) {
        stratEmpty.style.display = 'none';
        stratList.innerHTML = data.strategies_applied.map(s =>
            '<div style="padding:6px 8px;border-bottom:1px solid rgba(35,45,63,0.4);font-size:12px">' +
            '<div style="display:flex;justify-content:space-between"><span style="color:var(--green);font-weight:600">' + esc(s.name.slice(0,55)) + '</span>' +
            '<span style="color:var(--text2);font-size:10px">' + (s.applied_count||1) + 'x</span></div>' +
            '<div style="color:var(--text2);font-size:11px;margin-top:2px">Module: <span style="color:var(--cyan)">' + esc(s.module) + '</span>' +
            (s.sandbox_passed ? ' <span style="color:var(--green)">&#10003; sandbox</span>' : '') + '</div></div>'
        ).join('');
    } else { stratEmpty.style.display = ''; stratList.innerHTML = ''; }
    const patList = document.getElementById('evo-pat-list');
    const patEmpty = document.getElementById('evo-pat-empty');
    if (data.discovered_patterns.length) {
        patEmpty.style.display = 'none';
        patList.innerHTML = data.discovered_patterns.map(p =>
            '<div style="padding:5px 8px;border-bottom:1px solid rgba(35,45,63,0.4);font-size:12px">' +
            '<span style="color:var(--cyan)">' + esc(p.name) + '</span> <span style="color:var(--text2)">(' + esc(p.module) + ')</span></div>'
        ).join('');
    } else { patEmpty.style.display = ''; patList.innerHTML = ''; }
    // Federated status
    const fedEl = document.getElementById('fed-status');
    if (sett && fedEl) {
        const hasGH = sett.has_github_token;
        const interval = sett.auto_share_every || 0;
        if (!hasGH) {
            fedEl.innerHTML = '<span style="color:var(--yellow)">&#x26A0; No GitHub token</span> — set <code style="color:var(--cyan);font-size:11px">AGOS_GITHUB_TOKEN</code> to enable auto-sharing';
        } else if (interval <= 0) {
            fedEl.innerHTML = '<span style="color:var(--text2)">Auto-share disabled</span> — set <code style="color:var(--cyan);font-size:11px">AGOS_AUTO_SHARE_EVERY=3</code>';
        } else {
            const nextShare = interval - (data.cycles_completed % interval);
            fedEl.innerHTML = '<span style="color:var(--green)">&#10003; Active</span> — sharing every <b>' + interval + '</b> cycles · next in <b>' + nextShare + '</b> cycle' + (nextShare > 1 ? 's' : '');
        }
    }
    const prEl = document.getElementById('fed-last-pr');
    if (prEl && lastSharePR) {
        prEl.innerHTML = 'Last PR: <a href="' + esc(lastSharePR) + '" target="_blank" style="color:var(--cyan)">' + esc(lastSharePR) + '</a>';
    }
}

async function shareLearnings() {
    const status = document.getElementById('share-status');
    status.innerHTML = '<span style="color:var(--yellow)">Sharing learnings via PR...</span>';
    try {
        const resp = await fetch('/api/evolution/share', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({github_token: ''})
        });
        const data = await resp.json();
        if (data.ok) {
            status.innerHTML = '<span style="color:var(--green)">&#10003; PR created:</span> <a href="' + esc(data.pr_url) + '" target="_blank" style="color:var(--cyan)">' + esc(data.pr_url) + '</a>';
            lastSharePR = data.pr_url;
        } else {
            status.innerHTML = '<span style="color:var(--red)">' + esc(data.error) + '</span>';
        }
    } catch(e) {
        status.innerHTML = '<span style="color:var(--red)">Network error</span>';
    }
}

/* ── Boot ── */
refreshFast();
refreshCodebase();
refreshAudit();
refreshDeps();
refreshEvolution();
setInterval(refreshFast, 2000);
setInterval(refreshCodebase, 30000);
setInterval(refreshAudit, 10000);
setInterval(refreshDeps, 60000);
setInterval(refreshEvolution, 10000);
</script>
</body>
</html>"""

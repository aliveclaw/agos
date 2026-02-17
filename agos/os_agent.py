"""OS Agent — the agentic brain of AGOS.

The single entry point for ALL user interaction. Uses Claude to reason
about ANY request, then executes it using tools and sub-agents.

The OS agent can:
- Run any shell command, install any package, write any file
- Spawn sub-agents for specialized work (security, code analysis, etc.)
- Build entire applications from scratch
- Debug, fix, deploy — anything a senior engineer can do
- Manage the system: processes, resources, networking

Everything else in agos is a sub-agent or subsystem that the OS agent
can call on when needed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agos.llm.base import BaseLLMProvider, LLMMessage
from agos.tools.schema import ToolSchema, ToolParameter
from agos.tools.registry import ToolRegistry
from agos.events.bus import EventBus
from agos.policy.audit import AuditTrail, AuditEntry

MAX_TURNS = 40
MAX_TOKENS = 200_000

SYSTEM_PROMPT = """\
You are AGOS, an agentic operating system running in a Linux container.
You are not a chatbot. You are an OS that DOES things.

You have full root access. Python 3.13, Node 20, Go 1.22 are installed.
You can install anything else with apt-get, pip, npm, cargo, etc.

CAPABILITIES — you can do ALL of these:
- Write code in any language, build and run it
- Install any package or software
- Create, read, edit, delete files and directories
- Run shell commands, scripts, background processes
- Make HTTP requests, call APIs, scrape the web
- Manage system processes, resources, networking
- Spawn sub-agents: give a task + persona to a new agent that works in parallel
- Access the knowledge system (TheLoom) for memory/recall
- Analyze the codebase, run tests, deploy services
- Use tools from connected MCP servers (external databases, APIs, file systems) — prefixed with mcp_

RULES:
1. DO things. Use tools. Don't just explain what you could do.
2. Think step by step for complex tasks. Break them down.
3. If something fails, debug it. Read errors. Try another approach.
4. Verify your work — run the code, check the output.
5. Be concise in your final response. Show what you accomplished.
6. For big tasks, spawn sub-agents to work in parallel.

{context}"""


class OSAgent:
    """The brain. Handles ANY request via Claude + tools + sub-agents."""

    def __init__(
        self,
        event_bus: EventBus,
        audit_trail: AuditTrail,
        agent_registry=None,
        process_manager=None,
        policy_engine=None,
        llm: BaseLLMProvider | None = None,
        approval_gate=None,
        sandbox_config=None,
    ) -> None:
        self._bus = event_bus
        self._audit = audit_trail
        self._registry = agent_registry
        self._pm = process_manager
        self._policy = policy_engine
        self._llm = llm
        self._approval = approval_gate
        self._inner_registry = ToolRegistry()
        self._start_time = time.time()
        self._sub_agents: dict[str, dict] = {}
        self._register_tools()

        # Wrap with sandbox if configured
        if sandbox_config is not None:
            from agos.sandbox.executor import SandboxedToolExecutor
            self._tools = SandboxedToolExecutor(
                inner_registry=self._inner_registry,
                config=sandbox_config,
            )
        else:
            self._tools = self._inner_registry

    def set_llm(self, llm: BaseLLMProvider) -> None:
        self._llm = llm

    async def execute(self, command: str) -> dict[str, Any]:
        """Execute ANY natural language command."""
        if not self._llm:
            return _reply(False, "error",
                          "No API key. Set your Anthropic key in Settings.")

        command = command.strip()
        if not command:
            return _reply(False, "error", "No command.")

        await self._bus.emit("os.command", {"command": command}, source="os_agent")

        # Build live context
        ctx_parts = []
        if self._registry:
            agents = self._registry.list_agents()
            if agents:
                lines = [f"  {a['name']} [{a['runtime']}] {a['status']}" for a in agents]
                ctx_parts.append("INSTALLED AGENTS:\n" + "\n".join(lines))
        if self._sub_agents:
            lines = [f"  {k}: {v['status']}" for k, v in self._sub_agents.items()]
            ctx_parts.append("RUNNING SUB-AGENTS:\n" + "\n".join(lines))
        ctx_parts.append(f"UPTIME: {int(time.time() - self._start_time)}s")

        system = SYSTEM_PROMPT.format(context="\n\n".join(ctx_parts))
        messages: list[LLMMessage] = [LLMMessage(role="user", content=command)]
        tools = self._tools.get_anthropic_tools()

        steps: list[dict] = []
        tokens = 0
        turns = 0
        final_text = ""

        try:
            while turns < MAX_TURNS and tokens < MAX_TOKENS:
                turns += 1
                resp = await self._llm.complete(
                    messages=messages, system=system,
                    tools=tools, max_tokens=4096,
                )
                tokens += resp.input_tokens + resp.output_tokens

                if resp.content:
                    await self._bus.emit("os.thinking", {
                        "turn": turns, "text": resp.content[:500],
                    }, source="os_agent")

                # Done — no tool calls
                if not resp.tool_calls:
                    final_text = resp.content or ""
                    messages.append(LLMMessage(role="assistant", content=final_text))
                    break

                # Append assistant message with tool_use blocks
                asst: list[dict] = []
                if resp.content:
                    asst.append({"type": "text", "text": resp.content})
                for tc in resp.tool_calls:
                    asst.append({
                        "type": "tool_use", "id": tc.id,
                        "name": tc.name, "input": tc.arguments,
                    })
                messages.append(LLMMessage(role="assistant", content=asst))

                # Execute tools (with optional approval gate)
                results: list[dict] = []
                for tc in resp.tool_calls:
                    await self._bus.emit("os.tool_call", {
                        "turn": turns, "tool": tc.name,
                        "args": _trunc_args(tc.arguments),
                    }, source="os_agent")

                    # Approval gate: check if human needs to approve
                    if self._approval:
                        approved = await self._approval.check(tc.name, tc.arguments)
                        if not approved:
                            out = f"Tool '{tc.name}' rejected by human operator."
                            results.append({
                                "type": "tool_result", "tool_use_id": tc.id,
                                "content": out, "is_error": True,
                            })
                            steps.append({
                                "tool": tc.name,
                                "args": _trunc_args(tc.arguments),
                                "ok": False, "preview": out, "ms": 0,
                            })
                            continue

                    res = await self._tools.execute(tc.name, tc.arguments)
                    out = str(res.result) if res.success else str(res.error)
                    if len(out) > 8000:
                        out = out[:4000] + "\n...[truncated]...\n" + out[-2000:]

                    results.append({
                        "type": "tool_result", "tool_use_id": tc.id,
                        "content": out, "is_error": not res.success,
                    })
                    steps.append({
                        "tool": tc.name,
                        "args": _trunc_args(tc.arguments),
                        "ok": res.success,
                        "preview": out[:200],
                        "ms": res.execution_time_ms,
                    })

                    await self._bus.emit("os.tool_result", {
                        "turn": turns, "tool": tc.name,
                        "ok": res.success, "preview": out[:200],
                    }, source="os_agent")

                messages.append(LLMMessage(role="user", content=results))

            # Audit
            try:
                await self._audit.record(AuditEntry(
                    agent_id="os_agent", agent_name="OSAgent",
                    action="execute",
                    detail=f"{command[:80]} | turns={turns} tokens={tokens}",
                    success=True,
                ))
            except Exception:
                pass

            await self._bus.emit("os.complete", {
                "command": command[:200], "turns": turns,
                "tokens": tokens, "steps": len(steps),
            }, source="os_agent")

            return {
                "ok": True, "action": "execute",
                "message": final_text,
                "data": {"turns": turns, "tokens_used": tokens, "steps": steps},
            }

        except Exception as e:
            await self._bus.emit("os.error", {
                "command": command[:200], "error": str(e)[:300],
            }, source="os_agent")
            return _reply(False, "error", f"Failed: {e}")

    # ── Tool registration ────────────────────────────────────────

    def _register_tools(self) -> None:
        T, P = ToolSchema, ToolParameter
        reg = self._inner_registry

        reg.register(T(
            name="shell",
            description="Run any shell command. You have root. Use for: apt-get, pip, npm, git, ls, ps, curl, make, gcc, etc.",
            parameters=[
                P(name="command", description="Shell command to execute"),
                P(name="timeout", type="integer", description="Timeout seconds (default 60)", required=False),
            ],
        ), _shell)

        reg.register(T(
            name="read_file",
            description="Read a file or list a directory.",
            parameters=[P(name="path", description="File or directory path")],
        ), _read_file)

        reg.register(T(
            name="write_file",
            description="Write content to a file. Creates parent dirs.",
            parameters=[
                P(name="path", description="File path"),
                P(name="content", description="Content to write"),
            ],
        ), _write_file)

        reg.register(T(
            name="http",
            description="HTTP request. Use for APIs, web scraping, downloads.",
            parameters=[
                P(name="url", description="URL"),
                P(name="method", description="GET/POST/PUT/DELETE", required=False),
                P(name="body", description="Request body", required=False),
                P(name="headers", description="JSON headers string", required=False),
            ],
        ), _http)

        reg.register(T(
            name="python",
            description="Run Python code. Use print() for output.",
            parameters=[P(name="code", description="Python code")],
        ), _python)

        reg.register(T(
            name="spawn_agent",
            description=(
                "Spawn a sub-agent for a specialized task. The agent runs in the background "
                "with its own tools (shell, files, http, python). Use for parallelizing work: "
                "e.g. one agent researches while another writes code."
            ),
            parameters=[
                P(name="name", description="Short name for the agent (e.g. 'researcher', 'coder')"),
                P(name="task", description="What the agent should do — be specific"),
                P(name="persona", description="Who the agent is (e.g. 'senior Python developer', 'security auditor')", required=False),
            ],
        ), self._spawn_agent)

        reg.register(T(
            name="check_agent",
            description="Check status or get result of a spawned sub-agent.",
            parameters=[
                P(name="name", description="Name of the sub-agent to check"),
            ],
        ), self._check_agent)

        # Agent management if registry available
        if self._registry:
            agent_reg = self._registry
            reg.register(T(
                name="list_agents",
                description="List installed agents on this system.",
                parameters=[],
            ), _make_list_agents(agent_reg))

            reg.register(T(
                name="manage_agent",
                description="Manage installed agents: setup/start/stop/restart/uninstall/status.",
                parameters=[
                    P(name="action", description="setup|start|stop|restart|uninstall|status"),
                    P(name="name", description="Agent name"),
                    P(name="github_url", description="GitHub URL (for setup)", required=False),
                ],
            ), _make_manage_agent(agent_reg))

    # ── Sub-agent spawning ───────────────────────────────────────

    async def _spawn_agent(self, name: str, task: str, persona: str = "") -> str:
        """Spawn a sub-agent that works on a task independently."""
        if not self._llm:
            return "Error: No LLM available"

        agent_id = f"sub_{name}_{int(time.time()) % 10000}"
        self._sub_agents[name] = {"id": agent_id, "task": task, "status": "running", "result": None}

        await self._bus.emit("os.sub_agent.spawned", {
            "name": name, "task": task[:200],
        }, source="os_agent")

        # Run in background
        asyncio.create_task(self._run_sub_agent(name, task, persona))
        return f"Sub-agent '{name}' spawned and working on: {task[:100]}"

    async def _run_sub_agent(self, name: str, task: str, persona: str) -> None:
        """Run a sub-agent's task using its own Claude loop."""
        sub_system = f"""You are a sub-agent of AGOS named '{name}'.
{f'You are a {persona}.' if persona else ''}
Your task: {task}

You have the same tools as the OS (shell, files, http, python).
Focus on your task. Be thorough. Report your findings/results clearly.
Working directory: /app"""

        messages = [LLMMessage(role="user", content=task)]
        tools = self._tools.get_anthropic_tools()
        # Remove spawn/check tools to prevent recursion
        tools = [t for t in tools if t["name"] not in ("spawn_agent", "check_agent")]

        try:
            for turn in range(15):
                resp = await self._llm.complete(
                    messages=messages, system=sub_system,
                    tools=tools, max_tokens=4096,
                )

                if not resp.tool_calls:
                    self._sub_agents[name]["status"] = "done"
                    self._sub_agents[name]["result"] = resp.content or "(no output)"
                    break

                asst: list[dict] = []
                if resp.content:
                    asst.append({"type": "text", "text": resp.content})
                for tc in resp.tool_calls:
                    asst.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                messages.append(LLMMessage(role="assistant", content=asst))

                results: list[dict] = []
                for tc in resp.tool_calls:
                    # Approval gate for sub-agents too
                    if self._approval:
                        approved = await self._approval.check(tc.name, tc.arguments)
                        if not approved:
                            results.append({
                                "type": "tool_result", "tool_use_id": tc.id,
                                "content": f"Tool '{tc.name}' rejected by human operator.",
                                "is_error": True,
                            })
                            continue

                    res = await self._tools.execute(tc.name, tc.arguments)
                    out = str(res.result) if res.success else str(res.error)
                    if len(out) > 6000:
                        out = out[:3000] + "\n...[truncated]...\n" + out[-1500:]
                    results.append({
                        "type": "tool_result", "tool_use_id": tc.id,
                        "content": out, "is_error": not res.success,
                    })
                messages.append(LLMMessage(role="user", content=results))
            else:
                self._sub_agents[name]["status"] = "done"
                self._sub_agents[name]["result"] = resp.content or "(max turns reached)"

            await self._bus.emit("os.sub_agent.done", {
                "name": name, "result": (self._sub_agents[name]["result"] or "")[:300],
            }, source="os_agent")

        except Exception as e:
            self._sub_agents[name]["status"] = "error"
            self._sub_agents[name]["result"] = f"Error: {e}"

    async def _check_agent(self, name: str) -> str:
        """Check a sub-agent's status."""
        if name not in self._sub_agents:
            return f"No sub-agent named '{name}'. Active: {list(self._sub_agents.keys())}"
        agent = self._sub_agents[name]
        if agent["status"] == "running":
            return f"Sub-agent '{name}' is still working on: {agent['task'][:100]}"
        result = agent.get("result", "(no result)")
        return f"Sub-agent '{name}' finished ({agent['status']}).\n\nResult:\n{result}"


# ── Standalone tool implementations ──────────────────────────────


async def _shell(command: str, timeout: int = 60) -> str:
    import subprocess as _sp
    try:
        proc = await asyncio.create_subprocess_shell(
            command, stdout=_sp.PIPE, stderr=_sp.PIPE, cwd="/app",
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        parts = [f"exit={proc.returncode}"]
        if stdout:
            parts.append(stdout.decode(errors="replace")[:6000])
        if stderr:
            parts.append(f"stderr: {stderr.decode(errors='replace')[:3000]}")
        return "\n".join(parts)
    except asyncio.TimeoutError:
        return f"Timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


async def _read_file(path: str) -> str:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return f"Not found: {path}"
    if p.is_dir():
        entries = sorted(p.iterdir())
        lines = []
        for e in entries[:100]:
            kind = "DIR " if e.is_dir() else "FILE"
            sz = e.stat().st_size if e.is_file() else 0
            lines.append(f"  {kind} {e.name:40s} {sz:>10,}b")
        return f"{path} ({len(entries)} entries)\n" + "\n".join(lines)
    try:
        c = p.read_text(encoding="utf-8", errors="replace")
        if len(c) > 10000:
            return c[:5000] + f"\n...[{len(c)} chars total]...\n" + c[-3000:]
        return c
    except Exception as e:
        return f"Error: {e}"


async def _write_file(path: str, content: str) -> str:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


async def _http(url: str, method: str = "GET", body: str = "", headers: str = "") -> str:
    import httpx
    import json
    try:
        hdrs = json.loads(headers) if headers else {}
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.request(method, url, content=body or None, headers=hdrs)
            return f"HTTP {r.status_code}\n{r.text[:8000]}"
    except Exception as e:
        return f"Error: {e}"


async def _python(code: str) -> str:
    import subprocess as _sp
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", code, stdout=_sp.PIPE, stderr=_sp.PIPE, cwd="/app",
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        out = ""
        if stdout:
            out += stdout.decode(errors="replace")[:6000]
        if stderr:
            out += f"\nstderr: {stderr.decode(errors='replace')[:3000]}"
        return out or "(no output)"
    except asyncio.TimeoutError:
        return "Timed out after 60s"
    except Exception as e:
        return f"Error: {e}"


def _make_list_agents(registry):
    async def _fn() -> str:
        agents = registry.list_agents()
        if not agents:
            return "No agents installed."
        lines = [f"  {a['name']} [{a['runtime']}] {a['status']}" for a in agents]
        return "\n".join(lines)
    return _fn


def _make_manage_agent(registry):
    async def _fn(action: str, name: str, github_url: str = "") -> str:
        try:
            if action == "setup":
                a = await registry.setup(name, github_url=github_url)
                return f"Setup {a.display_name}: {a.status.value}"
            agent = registry.get_agent_by_name(name)
            if not agent:
                return f"Agent '{name}' not found."
            if action == "start":
                a = await registry.start(agent.id)
                return f"Started {a.display_name}: {a.status.value}"
            elif action == "stop":
                a = await registry.stop(agent.id)
                return f"Stopped {a.display_name}."
            elif action == "restart":
                if agent.status.value == "running":
                    await registry.stop(agent.id)
                a = await registry.start(agent.id)
                return f"Restarted {a.display_name}."
            elif action == "uninstall":
                await registry.uninstall(agent.id)
                return f"Uninstalled {name}."
            elif action == "status":
                return f"{agent.display_name} [{agent.runtime}] {agent.status.value} mem={agent.memory_limit_mb}MB"
            return f"Unknown action: {action}"
        except Exception as e:
            return f"Error: {e}"
    return _fn


def _reply(ok: bool, action: str, message: str, data: dict | None = None) -> dict:
    return {"ok": ok, "action": action, "message": message, "data": data or {}}


def _trunc_args(args: dict) -> dict:
    return {k: (str(v)[:100] + "..." if len(str(v)) > 100 else str(v)) for k, v in args.items()}

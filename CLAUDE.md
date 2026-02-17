# AGOS — Agentic Operating System

## Identity

AGOS is an **operating system**, not an application. Think of it like Linux, not like a todo app.

- It manages agents like an OS manages processes
- It has a kernel (AgentRuntime), a shell (OSAgent), a file system (TheLoom), an event bus, audit trail, and policy engine
- Agents are first-class citizens — they get spawned, scheduled, supervised, killed, and resource-limited
- The OS agent is the brain — it uses Claude to reason about ANY command and executes it with real tools (shell, files, HTTP, python, sub-agents)
- Everything else (evolution engine, security scanner, code analyst, etc.) are sub-agents or system services

## Architecture

- **Kernel**: `agos/kernel/` — AgentRuntime, Agent, state machine
- **OS Agent**: `agos/os_agent.py` — Claude-powered brain, handles all user commands
- **Knowledge**: `agos/knowledge/` — TheLoom (episodic, semantic, graph memory)
- **Processes**: `agos/processes/` — ProcessManager, WorkloadDiscovery, AgentRegistry
- **Evolution**: `agos/evolution/` — self-improving via arxiv papers, code generation, sandbox testing
- **Dashboard**: `agos/dashboard/app.py` — FastAPI web UI at port 8420
- **Serve**: `agos/serve.py` — Docker entry point, boots the full OS

## Key Principles

1. The OS agent can do ANYTHING — it has shell, file, HTTP, python, and sub-agent tools
2. Sub-agents run their own Claude loops in parallel
3. The evolution engine is always running — scanning arxiv, testing improvements, evolving the OS
4. All actions are audited. All events flow through the EventBus.
5. Docker is the deployment target — `docker compose up` boots the entire OS
6. Look for old code base that is no longer required. This is OS and need to be frugal and not a bloated engine. User still has to do tasks using their agents. System level token consumption, compute consumption should be less.

## Evolution: MUST BE REAL, NEVER COSMETIC

This is the most important rule for evolution:

- **Evolved code MUST actually execute.** No placeholder `apply()` methods that just set `self._applied = True`. If the evolution engine generates code from a paper, that code must run and change real system behavior.
- **Every evolved strategy `apply()` must call the pattern code** — hook it into the actual component (knowledge, intent, policy, orchestration) so it modifies live behavior.
- **No fake versioning** — if v2 of a pattern is the same code as v1 with a different paper ID, that's not evolution, that's waste. Only write a new version if the code actually changed.
- **Verify with real assertions** — after applying an evolved strategy, the system must prove the behavior changed (e.g., call the new function, check the output differs from before).
- **Parameter mutations are the gold standard** — MetaEvolver directly modifies running components via real method calls. Code evolution must match this standard.
- **Delete what doesn't work** — if a pattern fails sandbox or health check, remove it. Don't accumulate dead files.
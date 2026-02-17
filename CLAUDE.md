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
6. look for old code base that is no longer required. This is OS and need to be frugal and not a bloated engin. user still has to do tasks using their agents, System level token consumption, compute consumption should be less.
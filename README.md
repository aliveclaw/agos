# agos — Agentic Operating System

[![PyPI version](https://img.shields.io/pypi/v/agos)](https://pypi.org/project/agos/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/github/actions/workflow/status/aliveclaw/agenticOS/ci.yml?label=tests)](https://github.com/aliveclaw/agenticOS/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Your personal AI team. An intelligence layer, not a library.**

agos is not another agent framework. It's an operating system for intelligence. You speak naturally, it reasons, plans, spawns agents, executes, remembers, and evolves.

```bash
agos "why is my API slow?"
# → Spawns analyst agent, profiles endpoints, checks logs, reports findings

agos "write a REST API for user management with tests and docs"
# → Assembles a team: architect → coder → [tester + documenter in parallel]
```

## Install

### pip (recommended)

```bash
pip install agos
```

### Windows Executable

Download `agos.exe` from the [latest release](https://github.com/aliveclaw/agenticOS/releases/latest).

### From Source

```bash
git clone https://github.com/aliveclaw/agenticOS.git
cd agenticOS
pip install -e ".[dev]"
```

## Quick Start

```bash
# Set your API key
export AGOS_ANTHROPIC_API_KEY=your-key-here

# Initialize workspace
agos init

# Talk to it
agos "analyze my codebase and summarize what each module does"
agos "find all TODO comments and prioritize them"
agos "review this function for bugs" < src/main.py
```

## Architecture

```
+------------------------------------------------------------------+
|                            agos                                   |
|                                                                   |
|   INTERFACE    Natural Language CLI  |  Dashboard  |  SDK         |
|                         |                                         |
|   SOUL         Intent Engine (understand → plan → execute)        |
|                         |                                         |
|   BRAIN        Agent Kernel (lifecycle, state, budget)            |
|                         |                                         |
|   MEMORY       Knowledge System (episodic + semantic + graph)     |
|                         |                                         |
|   BODY         Tool Bus (file, shell, HTTP, Python, web search)   |
|                         |                                         |
|   SENSES       Triggers (file watch, cron, webhooks)              |
|                         |                                         |
|   SOCIAL       Coordination (channels, teams, debate protocol)    |
|                         |                                         |
|   IMMUNE       Policy Engine + Audit Trail                        |
|                         |                                         |
|   EVOLUTION    Self-Improving R&D (arxiv → analyze → integrate)   |
+------------------------------------------------------------------+
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `agos "<intent>"` | Natural language — the OS figures out what to do |
| `agos ps` | List running agents |
| `agos recall "<topic>"` | Search knowledge system |
| `agos timeline` | View event history |
| `agos watch <path> "<intent>"` | Watch files, trigger agent on changes |
| `agos schedule <interval> "<intent>"` | Run agent on a schedule |
| `agos team "<task>"` | Multi-agent team execution |
| `agos evolve` | Run R&D cycle (scan arxiv, analyze, propose) |
| `agos evolve --proposals` | View pending evolution proposals |
| `agos ambient --start` | Start background watchers |
| `agos proactive --scan` | Run pattern detection |
| `agos audit` | View audit trail |
| `agos policy` | Configure safety policies |
| `agos dashboard` | Launch web monitoring UI |
| `agos update` | Check for updates and self-update |
| `agos version` | Show version |

## Configuration

All settings via environment variables with `AGOS_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGOS_ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `AGOS_DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `AGOS_WORKSPACE_DIR` | `.agos` | Local workspace directory |
| `AGOS_MAX_CONCURRENT_AGENTS` | `50` | Max agents running at once |
| `AGOS_DASHBOARD_PORT` | `8420` | Dashboard web UI port |
| `AGOS_LOG_LEVEL` | `INFO` | Logging level |

## Self-Evolution

agos continuously improves itself by scanning the latest AI research:

1. **Scout** — Searches arxiv for papers on agentic AI, memory systems, coordination
2. **Analyze** — Claude extracts actionable techniques from each paper
3. **Extract** — Finds implementation code from linked GitHub repos
4. **Test** — Runs code patterns in a sandboxed environment
5. **Propose** — Creates evolution proposals with risk assessment
6. **Integrate** — Applies accepted proposals with snapshot/rollback support

```bash
agos evolve                    # Run a full R&D cycle
agos evolve --proposals        # Review what it found
agos evolve --accept <id>      # Accept a proposal
agos evolve --apply <id>       # Apply it (with auto-rollback on failure)
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v          # 400+ tests
ruff check agos/ tests/   # lint
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

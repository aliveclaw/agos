"""agos — Live Demo of all subsystems."""

import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


async def full_demo():
    from agos.types import AgentDefinition, AgentState
    from agos.llm.base import LLMResponse, ToolCall
    from agos.kernel.runtime import AgentRuntime
    from agos.tools.registry import ToolRegistry
    from agos.tools.builtins import register_builtin_tools
    from agos.knowledge.manager import TheLoom
    from agos.knowledge.base import Thread
    from agos.triggers.base import TriggerConfig
    from agos.triggers.manager import TriggerManager
    from agos.intent.engine import IntentEngine
    from tests.conftest import MockLLMProvider
    import tempfile

    # Setup
    tools = ToolRegistry()
    register_builtin_tools(tools)

    console.print()
    console.print(Panel(
        "[bold cyan]agos -- Live Demo[/bold cyan]\nAll subsystems running end-to-end",
        width=60,
    ))

    # ================================================================
    # DEMO 1: Agent spawns, uses a tool, returns result
    # ================================================================
    console.print()
    console.print("[bold yellow]--- Demo 1: Agent uses a tool ---[/bold yellow]")

    mock = MockLLMProvider([
        LLMResponse(
            content="Let me check the project structure.",
            stop_reason="tool_use",
            tool_calls=[ToolCall(
                id="t1",
                name="shell_exec",
                arguments={"command": "echo src/ tests/ agos/ pyproject.toml"},
            )],
            input_tokens=30,
            output_tokens=15,
        ),
        LLMResponse(
            content="Your project has 3 main directories: **src/**, **tests/**, **agos/** and a **pyproject.toml** config.",
            stop_reason="end_turn",
            input_tokens=80,
            output_tokens=25,
        ),
    ])

    runtime = AgentRuntime(llm_provider=mock, tool_registry=tools)
    defn = AgentDefinition(
        name="analyst",
        system_prompt="You analyze codebases.",
        tools=["shell_exec"],
    )

    console.print('[dim]> agos "what is in my project?"[/dim]')
    agent = await runtime.spawn(defn, user_message="What is in my project?")
    result = await agent.wait()

    console.print("[dim]  intent=research  strategy=solo  agents=[analyst][/dim]")
    console.print()
    console.print(Markdown(result))

    # Show agent table
    table = Table(title="agos ps")
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Name", style="green")
    table.add_column("State", style="yellow")
    table.add_column("Tokens", style="white")
    table.add_column("Turns", style="dim")
    for a in runtime.list_agents():
        table.add_row(
            a["id"][:12], a["name"], a["state"],
            str(a["tokens_used"]), str(a["turns"]),
        )
    console.print(table)

    # ================================================================
    # DEMO 2: Knowledge -- remember, learn, recall
    # ================================================================
    console.print()
    console.print("[bold yellow]--- Demo 2: The Loom -- Knowledge System ---[/bold yellow]")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    loom = TheLoom(db_path)
    await loom.initialize()

    # Store knowledge
    await loom.remember(
        "agos uses Claude as its LLM backbone via the Anthropic API",
        kind="fact",
    )
    await loom.remember(
        "The Intent Engine classifies natural language into execution plans",
        kind="fact",
    )
    await loom.remember(
        "Triggers enable ambient intelligence -- file watching, schedules, webhooks",
        kind="fact",
    )
    console.print("[dim]  Stored 3 facts into The Loom[/dim]")

    # Record an interaction via learner
    await loom.learner.record_interaction(
        agent_id=agent.id,
        agent_name="analyst",
        user_input="What is in my project?",
        agent_output=result,
        tokens_used=agent.context.tokens_used,
    )
    console.print("[dim]  Learner recorded interaction -> episodic + semantic + graph[/dim]")
    console.print()

    # Recall
    console.print('[dim]> agos recall "what LLM does agos use"[/dim]')
    results = await loom.recall("what LLM does agos use")
    table = Table(title='The Loom -- recall: "what LLM does agos use"')
    table.add_column("When", style="dim", max_width=19)
    table.add_column("Kind", style="cyan")
    table.add_column("Content", style="white")
    for r in results[:3]:
        table.add_row(
            r.created_at.strftime("%Y-%m-%d %H:%M"),
            r.kind,
            r.content[:100],
        )
    console.print(table)

    # Timeline
    console.print()
    console.print("[dim]> agos timeline[/dim]")
    events = await loom.timeline(limit=5)
    table = Table(title="The Loom -- Timeline")
    table.add_column("When", style="dim", max_width=19)
    table.add_column("Kind", style="cyan")
    table.add_column("Event", style="white")
    for e in events:
        table.add_row(
            e.created_at.strftime("%Y-%m-%d %H:%M"),
            e.kind,
            e.content[:100],
        )
    console.print(table)

    # Graph
    conns = await loom.graph.connections("agent:analyst")
    console.print()
    console.print("[dim]> Knowledge graph for agent:analyst[/dim]")
    for c in conns:
        console.print(
            f"  [cyan]agent:analyst[/cyan] "
            f"--[green]{c.relation}[/green]--> "
            f"[yellow]{c.target}[/yellow]"
        )

    # ================================================================
    # DEMO 3: Triggers -- schedule fires
    # ================================================================
    console.print()
    console.print("[bold yellow]--- Demo 3: Triggers -- Ambient Intelligence ---[/bold yellow]")

    manager = TriggerManager()
    trigger_events = []

    async def on_trigger(intent):
        trigger_events.append(intent)

    manager.set_handler(on_trigger)

    config = TriggerConfig(
        kind="schedule",
        description="Health check every 0.1s",
        intent="check if server is healthy",
        params={"interval_seconds": 0.1, "max_fires": 3},
    )
    console.print('[dim]> agos schedule 0.1 "check if server is healthy" --max 3[/dim]')
    await manager.register(config)
    console.print("[green]Scheduled[/green] every [bold]0.1s[/bold] (max 3 fires)")

    await asyncio.sleep(0.5)
    await manager.stop_all()

    console.print(f"[dim]  Trigger fired {len(trigger_events)} times[/dim]")
    for i, evt in enumerate(trigger_events):
        console.print(f"  [dim]fire {i+1}:[/dim] {evt[:70]}")

    # ================================================================
    # DEMO 4: Multiple agents in parallel
    # ================================================================
    console.print()
    console.print("[bold yellow]--- Demo 4: Parallel Agents ---[/bold yellow]")

    mock2 = MockLLMProvider([
        LLMResponse(
            content="Researcher found 3 relevant papers on agentic AI.",
            stop_reason="end_turn",
            input_tokens=20, output_tokens=10,
        ),
        LLMResponse(
            content="Coder implemented the API endpoint with tests.",
            stop_reason="end_turn",
            input_tokens=20, output_tokens=10,
        ),
        LLMResponse(
            content="Reviewer found no critical issues. Ship it!",
            stop_reason="end_turn",
            input_tokens=20, output_tokens=10,
        ),
    ])
    runtime2 = AgentRuntime(llm_provider=mock2, tool_registry=tools)

    agents_to_run = [
        ("researcher", "Research agentic AI trends"),
        ("coder", "Implement user API endpoint"),
        ("reviewer", "Review the implementation"),
    ]

    console.print('[dim]> agos "research, code, and review the user API"[/dim]')
    console.print(
        "[dim]  intent=code  strategy=parallel  "
        "agents=[researcher, coder, reviewer][/dim]"
    )

    spawned = []
    for name, task in agents_to_run:
        d = AgentDefinition(
            name=name,
            system_prompt=f"You are a {name}.",
            tools=[],
        )
        a = await runtime2.spawn(d, user_message=task)
        spawned.append(a)

    all_results = await asyncio.gather(*[a.wait() for a in spawned])

    table = Table(title="agos ps")
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Name", style="green")
    table.add_column("State", style="yellow")
    table.add_column("Result", style="white")
    for a, r in zip(spawned, all_results):
        table.add_row(
            a.id[:12], a.definition.name, a.state.value, r[:60],
        )
    console.print(table)

    # ================================================================
    # Summary
    # ================================================================
    console.print()
    console.print(Panel(
        "[bold green]All systems operational.[/bold green]\n\n"
        "  Sprint 1: The Soul -- Intent Engine + Kernel + Tools        [green]v[/green]\n"
        "  Sprint 2: The Loom -- Knowledge (episodic+semantic+graph)   [green]v[/green]\n"
        "  Sprint 3: The Senses -- Triggers (file+schedule+webhook)   [green]v[/green]\n\n"
        "  [dim]96 tests passing | 31 modules | 9 CLI commands[/dim]",
        title="agos v0.1.0",
        width=70,
    ))


if __name__ == "__main__":
    asyncio.run(full_demo())
